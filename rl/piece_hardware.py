"""
rl/piece_hardware.py

HardwareExecutor: runs PieceResidualEnv strokes on the real UR5e.

Everything that touches the robot is PiecePlayer's own verified machinery —
apply_verified_installation, zero_ft_at_config_safe, safe_moveL through
pose_at/apply_depth/lifted — so an RL stroke is commanded exactly the way a
baseline performance stroke is. This file only adds:

    - a continuous microphone stream, sliced per stroke into the ~500 ms
      peak-normalized window the classifier was trained on
    - physical params per stroke from the StateLogger summary, in the same
      slot order dataset.build_physical_features uses
    - episode plumbing: set the bow down before the first stroke, honour
      inter-note gaps and retakes, lift off at episode end

Timing: training does not chase the absolute performance timeline. The gap
before each note (rests, retake room) is slept, but classifier inference
time simply delays the next stroke instead of being "caught up", so rhythm
WITHIN each stroke is exact and the piece as a whole may run a little slow
during training. Use play_piece.py --perform for a timeline-faithful run.

Requires: robot reachable, mic connected, .venv311.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rl.piece_env import PMP, ExecutorBase, ExecStroke, StrokeResult

WINDOW_SEC = 0.5          # classifier window (SoundClassifier convention)
PRE_ROLL_SEC = 0.05       # skip the attack transient when the stroke is long


def _rec():
    """The recording module play_midi_pieces already loaded, for its verified
    audio-device discovery and channel probing."""
    return PMP._rec


class HardwareExecutor(ExecutorBase):
    """
    One connection for the whole training run; one bow placement per episode.
    """

    def __init__(self, audio_device=None, audio_channel: int = 1,
                 retake_speed: float = PMP.RETAKE_SPEED,
                 save_dir: Path | None = None):
        import rtde_control
        if getattr(rtde_control, "_IS_CELLO_RL_STUB", False):
            raise RuntimeError(
                "This environment has no real ur_rtde build (rtde_control is "
                "a mock-mode stub) — hardware runs need the venv with the "
                "compiled ur_rtde binaries.")
        import sounddevice as sd
        self._sd = sd

        # PiecePlayer owns the controller, verified installation, logger.
        self.player = PMP.PiecePlayer(record_audio=False,
                                      retake_speed=retake_speed)
        self.logger = self.player.logger
        self.rtde_r = self.player.rtde_r

        self.sample_rate = int(PMP.SAMPLE_RATE)

        # Audio input must be resolved the SAME way recording_a_only resolved
        # it for dataset_a_final, or the classifier is fed out-of-distribution
        # audio — or, worse, an empty socket's noise floor. That script's own
        # comment records a whole session lost at -90 dBFS because the mic had
        # been moved to input 2 while the code still took input 1, so the
        # channel is probed and proven live before any stroke is played.
        # A training run is hours of robot time; discovering silence at the end
        # of it is not acceptable.
        if audio_device is None:
            audio_device = _rec().find_focusrite_device()
        self.audio_device = audio_device

        if not audio_channel:
            channel, levels = _rec().select_audio_channel(audio_device)
            _rec().report_audio_channels(levels, channel)
            if channel == 0:
                raise RuntimeError(
                    f"No input channel on device {audio_device} carries signal "
                    f"(all below {_rec().AUDIO_LIVE_DBFS} dBFS). Check the mic "
                    f"is connected and phantom power/gain is up — training "
                    f"against silence would score every stroke identically.")
            audio_channel = channel
        self.audio_channel = audio_channel
        print(f"Audio: device {self.audio_device}, channel {self.audio_channel}")
        self._stream = None
        self._chunks: list[np.ndarray] = []
        self._audio_t0 = None

        self.save_dir = save_dir
        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
        self._episode = 0
        self._stroke_n = 0

    # ── audio ─────────────────────────────────────────────────────

    def _audio_callback(self, indata, frames, t, status):
        self._chunks.append(indata[:, self.audio_channel - 1].copy()
                            if indata.ndim > 1 else indata.copy())

    def _start_audio(self):
        self._chunks = []
        self._stream = self._sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.audio_channel,
            device=self.audio_device,
            dtype="float32",
            callback=self._audio_callback,
        )
        self._stream.start()
        self._audio_t0 = time.time()

    def _stop_audio(self):
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def _slice_window(self, t_start: float, t_end: float) -> np.ndarray | None:
        """~500 ms peak-normalized window from the middle of [t_start, t_end]."""
        if self._audio_t0 is None or not self._chunks:
            return None
        audio = np.concatenate(self._chunks)
        sr = self.sample_rate
        dur = t_end - t_start
        if dur > WINDOW_SEC + 2 * PRE_ROLL_SEC:
            centre = t_start + PRE_ROLL_SEC + (dur - PRE_ROLL_SEC) / 2.0
        else:
            centre = (t_start + t_end) / 2.0
        lo = int((centre - self._audio_t0 - WINDOW_SEC / 2.0) * sr)
        hi = lo + int(WINDOW_SEC * sr)
        lo = max(lo, 0)
        if hi > len(audio):
            hi = len(audio)
        window = audio[lo:hi]
        if len(window) == 0:
            return None
        peak = float(np.max(np.abs(window)))
        if peak > 1e-6:
            window = window / peak
        return window.astype(np.float32)

    # ── episode plumbing ──────────────────────────────────────────

    def begin_episode(self, first: ExecStroke):
        self._episode += 1
        self._stroke_n = 0
        if self._episode == 1:
            # F/T tare + set the bow down, PiecePlayer's own prepare().
            self.player.prepare(_as_plan_stroke(first))
        else:
            # Bow is lifted from the previous episode's end: traverse in the
            # air back to the first stroke's start and set down again.
            self._retake_to(first.u_start, first.depth)
        self._start_audio()
        self.logger.start()
        self._t_zero = time.time()
        self._free_from = 0.0

    def _retake_to(self, u_to: float, depth: float):
        there = PMP.apply_depth(PMP.pose_at(u_to), PMP.CFG, depth)
        PMP.safe_moveL(self.player.controller, PMP.lifted(there),
                       self.player.retake_speed, self.player.retake_accel,
                       what=f"episode reset: traverse above u={u_to:.3f}")
        PMP.safe_moveL(self.player.controller, there,
                       self.player.retake_speed, self.player.retake_accel,
                       what=f"episode reset: set down at u={u_to:.3f}")

    def execute(self, stroke: ExecStroke) -> StrokeResult:
        self._stroke_n += 1

        # Honour the silence the score writes before this note (bounded so a
        # long rest does not stall training), then any planned retake.
        gap = min(stroke.gap_before, 2.0)
        if stroke.retake_from is not None:
            self.player._retake(stroke.retake_from, stroke.u_start, stroke.depth)
        elif gap > 0.02:
            time.sleep(gap)

        t_start = time.time()
        self.player._play(_as_plan_stroke(stroke))
        t_end = time.time()

        # Physical params in dataset.build_physical_features slot order.
        summary = self.logger.get_summary(t_start, t_end) or {}
        physical = np.array([
            stroke.depth,                                   # depth_or_force
            0.0,                                            # force_deviation_or_zero
            summary.get("speed_mean", stroke.mean_speed),   # bow_speed
            (stroke.u_start + stroke.u_end) / 2.0,          # bow_position
            summary.get("torque_mag_mean", 0.0),            # torque_or_lateral
            summary.get("torque_mag_max", 0.0),             # torque_max_or_torque
        ], dtype=np.float32)

        # Wait for the tail of the window to be captured before slicing.
        time.sleep(0.06)
        audio = self._slice_window(t_start, t_end)

        if self.save_dir and audio is not None:
            import soundfile as sf
            sf.write(Path(self.save_dir) /
                     f"ep{self._episode:04d}_s{self._stroke_n:03d}.wav",
                     audio, self.sample_rate)

        measured_speed = summary.get("speed_mean", stroke.mean_speed)
        return StrokeResult(
            audio=audio,
            physical=physical,
            measured_mean_speed=float(measured_speed),
            # The command in u-space is authoritative for the bow budget
            # (moveL either completes or safe_moveL raises); the logger's
            # bow_pos projection is diagnostic only.
            achieved_u_end=float(stroke.u_end),
        )

    def end_episode(self):
        self.logger.stop()
        self._stop_audio()
        # Lift clear so the string is not damped between episodes and the arm
        # is not left loaded against the instrument.
        pose = np.asarray(self.rtde_r.getActualTCPPose(), dtype=float)
        PMP.safe_moveL(self.player.controller, PMP.lifted(pose),
                       PMP.RESET_SPEED, PMP.MOVE_ACCEL,
                       what="lift off at episode end")

    def close(self):
        self._stop_audio()
        try:
            self.player.close()
        except Exception:
            pass


def _as_plan_stroke(s: ExecStroke):
    """Adapter: ExecStroke -> the PMP.Stroke shape PiecePlayer._play expects."""
    return PMP.Stroke(
        note_index=s.note_index,
        onset=s.onset,
        duration=s.duration,
        direction=s.direction,
        direction_source="rl",
        u_start=s.u_start,
        u_end=s.u_end,
        length=s.length,
        mean_speed=s.mean_speed,
        depth=s.depth,
        volume_target=s.volume_target,
        volume_actual=s.volume_target,
        dynamic=PMP.velocity_to_dynamic(PMP.volume_to_velocity(s.volume_target)),
        segments=list(s.segments),
        retake_from=s.retake_from,
    )
