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
from rl.loudness import measure_dbfs

WINDOW_SEC = 0.5          # classifier window (SoundClassifier convention)
PRE_ROLL_SEC = 0.05       # skip the attack transient when the stroke is long

# Shortest window worth analysing. Below this the hand-engineered features
# genuinely run out of signal — at 50 ms an open A gives only ~11 periods, and
# F0 stability and HNR need more than that to mean anything.
MIN_ANALYSIS_SEC = 0.08

# Cut the analysis window to the NOTE, not to a fixed 500 ms.
#
# A fixed window centred on an 80 ms note contains 80 ms of that note and
# 420 ms of its neighbours, so the score is mostly about the wrong notes. On
# challengepiece that is 86% of the piece. Averaging across notes to fill a
# window is not a defensible way to score a note.
#
# The model can take it: scoring the standard-config recordings at different
# window lengths gives rho 0.824 at 0.10 s against 0.835 at 0.50 s, so ranking
# barely degrades as the window shrinks. The trunk pools time with mean+std, so
# any frame count works, and a note-length window means every frame it sees
# belongs to the note being judged.
#
# For notes too short even for MIN_ANALYSIS_SEC, the window is the note plus
# whatever release follows it — and the reward already grades those mostly on
# attack_quality, which is a human-labelled judgement about exactly that part
# of a stroke and is the only sound estimate available at that length.
NOTE_MATCHED_WINDOW = True


def _rec():
    """The recording module play_midi_pieces already loaded, for its verified
    audio-device discovery and channel probing."""
    return PMP._rec


class HardwareExecutor(ExecutorBase):
    """
    One connection for the whole training run; one bow placement per episode.
    """

    # audio_channel=0 means "probe and pick the input that actually has
    # signal", the same convention (and the same default) recording_a_only
    # uses. Defaulting to 1 is exactly the mistake its comment warns about:
    # channel 1 on this rig is an empty socket at -91.9 dBFS, so a run would
    # record silence and score every stroke identically.
    def __init__(self, audio_device=None, audio_channel: int = 0,
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
        self._recording = False

        self.save_dir = save_dir
        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
        self._episode = 0
        self._stroke_n = 0

    # ── audio ─────────────────────────────────────────────────────

    def _audio_callback(self, indata, frames, t, status):
        if not self._recording:
            return
        self._chunks.append(indata[:, self.audio_channel - 1].copy()
                            if indata.ndim > 1 else indata.copy())

    def _start_audio(self):
        # One stream for the whole process, gated by _recording. Tearing the
        # stream down per episode is not safe on macOS: CoreAudio's IO thread
        # can deliver one more buffer to the just-closed stream's callback,
        # which lands in freed memory (crashed reproducibly on the 3rd
        # open/close cycle, 2026-08-18). The flag gives the same
        # chunks-only-between-start-and-stop semantics without ever closing.
        self._chunks = []
        if self._stream is None:
            self._stream = self._sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.audio_channel,
                device=self.audio_device,
                dtype="float32",
                callback=self._audio_callback,
            )
            self._stream.start()
        self._audio_t0 = time.time()
        self._recording = True

    def _stop_audio(self):
        self._recording = False

    @staticmethod
    def _window_bounds(t_start: float, t_end: float) -> tuple[float, float]:
        """
        Absolute (start, end) of the analysis window, matched to the note.

        LONG note (longer than the window plus room to skip the attack): take a
        steady WINDOW_SEC from the middle, past the onset transient. This is
        the sustained-tone case the classifier was trained on.

        SHORT note: take the NOTE, whatever length it is. Nothing is padded and
        no neighbour is included, so the score is about this note alone. A note
        below MIN_ANALYSIS_SEC is extended to that minimum, which runs into its
        own release rather than into the next note.
        """
        dur = t_end - t_start
        if dur > WINDOW_SEC + 2 * PRE_ROLL_SEC:
            centre = t_start + PRE_ROLL_SEC + (dur - PRE_ROLL_SEC) / 2.0
            return centre - WINDOW_SEC / 2.0, centre + WINDOW_SEC / 2.0
        return t_start, t_start + max(dur, MIN_ANALYSIS_SEC)

    def _slice_window(self, t_start: float, t_end: float) -> np.ndarray | None:
        """Note-matched, peak-normalised window (see _window_bounds)."""
        if self._audio_t0 is None or not self._chunks:
            return None, None
        audio = np.concatenate(self._chunks)
        sr = self.sample_rate
        w_start, w_end = self._window_bounds(t_start, t_end)
        lo = max(int((w_start - self._audio_t0) * sr), 0)
        hi = min(int((w_end - self._audio_t0) * sr), len(audio))
        window = audio[lo:hi]
        if len(window) == 0:
            return None, None

        # Absolute level FIRST. The classifier wants a peak-normalised window
        # (that is what it was trained on), but normalising is exactly what
        # destroys the loudness the dynamics reward needs to measure.
        dbfs = measure_dbfs(window)

        peak = float(np.max(np.abs(window)))
        if peak > 1e-6:
            window = window / peak
        return window.astype(np.float32), dbfs

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
        try:
            self.player._play(_as_plan_stroke(stroke))
        except PMP.RobotFaultStop:
            # "moveL returned False" on its own says nothing about WHY, and by
            # the time anyone reads the traceback the controller state that
            # caused it is gone. Snapshot it here, while it is still true.
            self._report_fault(stroke)
            raise
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

        # Wait for the tail of the window to actually be captured before
        # slicing. A fixed pause is not enough: the window is centred on the
        # stroke, so a SHORT note needs audio from well past its end, and
        # slicing early returns a truncated window that the classifier then
        # zero-pads -- silence the model never saw in training. Wait for the
        # real end of the window, with a bound so a stalled input stream can
        # never hang the run.
        window_end = self._window_bounds(t_start, t_end)[1]
        deadline = min(window_end + 0.02, t_end + 0.5)
        while time.time() < deadline:
            time.sleep(0.01)
        audio, dbfs = self._slice_window(t_start, t_end)

        if self.save_dir and audio is not None:
            import soundfile as sf
            sf.write(Path(self.save_dir) /
                     f"ep{self._episode:04d}_s{self._stroke_n:03d}.wav",
                     audio, self.sample_rate)

        measured_speed = summary.get("speed_mean", stroke.mean_speed)
        return StrokeResult(
            audio=audio,
            physical=physical,
            measured_dbfs=dbfs,
            measured_mean_speed=float(measured_speed),
            # The command in u-space is authoritative for the bow budget
            # (moveL either completes or safe_moveL raises); the logger's
            # bow_pos projection is diagnostic only.
            achieved_u_end=float(stroke.u_end),
        )

    def _report_fault(self, stroke: ExecStroke):
        """
        Dump controller and contact state at the moment a move failed.

        moveL returns False for several unrelated reasons — protective stop,
        the control script having died, an unreachable target — and they need
        different fixes. These are the readings that tell them apart.
        """
        print("\n" + "=" * 70)
        print(f"ROBOT FAULT during note {stroke.note_index} ({stroke.direction})")
        print(f"  commanded: u {stroke.u_start:.3f}->{stroke.u_end:.3f}  "
              f"{stroke.length*1000:.1f} mm  {stroke.speed:.3f} m/s  "
              f"accel {stroke.accel:.2f}  depth {stroke.depth*1000:+.2f} mm")
        r = self.rtde_r
        for label, fn in (
            ("robot_mode", "getRobotMode"),
            ("safety_mode", "getSafetyMode"),
            ("safety_status_bits", "getSafetyStatusBits"),
            ("runtime_state", "getRuntimeState"),
        ):
            try:
                print(f"  {label}: {getattr(r, fn)()}")
            except Exception as e:
                print(f"  {label}: unavailable ({type(e).__name__})")
        try:
            force = np.asarray(r.getActualTCPForce(), dtype=float)
            print(f"  |F| at fault: {np.linalg.norm(force[:3]):.3f} N   "
                  f"F={[round(v, 3) for v in force[:3]]}")
            print(f"  TCP pose:  {[round(v, 4) for v in r.getActualTCPPose()]}")
        except Exception as e:
            print(f"  force/pose unavailable ({type(e).__name__})")
        try:
            print(f"  control script running: "
                  f"{self.player.rtde_c.isProgramRunning()}")
        except Exception as e:
            print(f"  control script state unavailable ({type(e).__name__})")
        print("=" * 70 + "\n")

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
        # Without this the player's pre-start lead defaults to 0 and every
        # retaken note begins a full retake late (writeup-aug17 bug).
        retake_time=getattr(s, "retake_time", 0.0),
    )
