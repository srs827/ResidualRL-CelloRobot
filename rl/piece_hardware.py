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

# The process-wide audio input stream, shared by every HardwareExecutor
# instance (see _start_audio for why it is opened once and never closed).
# "owner" is the executor currently recording; the callback routes chunks
# to it and drops audio when nobody is.
_AUDIO = {"stream": None, "key": None, "owner": None}


def _shared_audio_callback(indata, frames, t, status):
    ex = _AUDIO["owner"]
    if ex is None or not ex._recording:
        return
    block = (indata[:, ex.audio_channel - 1].copy()
             if indata.ndim > 1 else indata.copy())
    if not ex._chunks:
        # Re-stamp _audio_t0 from when audio ACTUALLY starts.
        #
        # _start_audio sets _audio_t0 = time.time(), which is right only when
        # it also creates the stream: the first block is then captured from
        # that instant and delivered one block later. But the stream is shared
        # and never restarted, so from the SECOND episode onward a block is
        # already in flight when _chunks is cleared, and its sample 0 was
        # captured up to a full block BEFORE _audio_t0.
        #
        # Measured 2026-08-19 on the Scarlett at 44100: first block arrives
        # +92.6 ms after _audio_t0 on a fresh stream, +16.3 ms on an
        # already-running one, block 4096 samples = 92.9 ms. So sample 0 sits
        # ~76 ms before _audio_t0, and _slice_window's (t - _audio_t0)*sr
        # lands that far EARLY in the note. On a 0.5 s steady-middle window
        # that is invisible; on a 0.111 s yunpiece note it is 68% of the note,
        # so the window mostly captures the previous note's tail.
        #
        # _start_audio runs once per episode (reset -> begin_episode), so this
        # affected 103 of 104 episodes of the 8/19 run while the single-episode
        # calibration stayed aligned -- calibration and training measuring on
        # different timebases, which is exactly the cancellation that must not
        # break.
        #
        # len(block)/sr rather than PortAudio's inputBufferAdcTime: the ADC
        # clock has no fixed relationship to time.time(), while block length
        # is exact and the delivery instant is what the rest of the class
        # already works in.
        ex._audio_t0 = time.time() - len(block) / float(ex.sample_rate)
    ex._chunks.append(block)

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
        self._chunks: list[np.ndarray] = []
        self._audio_t0 = None
        self._recording = False

        self.save_dir = save_dir
        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
        self._episode = 0
        self._stroke_n = 0

    # ── audio ─────────────────────────────────────────────────────

    def _start_audio(self):
        # One stream for the whole PROCESS, gated by _recording. Tearing the
        # stream down is not safe on macOS: CoreAudio's IO thread can deliver
        # one more buffer to the just-closed stream's callback, which lands
        # in freed memory (crashed reproducibly on the 3rd open/close cycle,
        # 2026-08-18). The flag gives the same chunks-only-between-start-and-
        # stop semantics without ever closing.
        #
        # The stream is shared across HardwareExecutor INSTANCES, not owned
        # by one: driver_eval-style code constructs a fresh executor per grid
        # cell, and with close() no longer closing, per-instance streams
        # would leak one open input stream per cell (writeup-aug18 Part 7).
        # The module-level callback routes chunks to whichever executor is
        # currently recording.
        self._chunks = []
        key = (self.audio_device, self.audio_channel, self.sample_rate)
        if _AUDIO["stream"] is None or _AUDIO["key"] != key:
            if _AUDIO["stream"] is not None:
                # Config changed mid-process (different device/channel/rate):
                # the old stream must go. This is the one remaining close and
                # it carries the documented teardown race; it never happens in
                # any current caller (config is fixed per process).
                try:
                    _AUDIO["stream"].stop()
                    _AUDIO["stream"].close()
                except Exception:
                    pass
                _AUDIO["stream"] = None
            stream = self._sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.audio_channel,
                device=self.audio_device,
                dtype="float32",
                callback=_shared_audio_callback,
            )
            stream.start()
            _AUDIO["stream"], _AUDIO["key"] = stream, key
        _AUDIO["owner"] = self
        self._audio_t0 = time.time()
        self._recording = True

    def _stop_audio(self):
        self._recording = False
        if _AUDIO["owner"] is self:
            _AUDIO["owner"] = None

    def _wait_for_window(self, window_end: float, t_end: float) -> None:
        """
        Block until the audio buffer covers window_end, then return.

        The input arrives in ~93 ms blocks (4096 samples at 44.1 kHz), and
        the old fixed 20 ms grace lost the race to that quantum on most
        strokes: the window's final block was still in flight when the slice
        was cut, so the judge scored a truncated window zero-padded with
        silence it never saw in training. Poll for ARRIVAL instead -- return
        as soon as the buffered samples reach window_end (average wait ~half
        a block, worst one block) -- with a deadline as the backstop so a
        stalled stream can never hang the run: 0.15 s past the window covers
        one full block period, and the t_end bound still caps how long a
        short note's centred window can hold the loop.
        """
        deadline = min(window_end + 0.15, t_end + 0.5)
        sr = float(self.sample_rate)
        while time.time() < deadline:
            t0 = self._audio_t0
            if t0 is not None:
                n = sum(len(c) for c in self._chunks)
                if t0 + n / sr >= window_end:
                    return
            time.sleep(0.005)

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
        # slicing -- see _wait_for_window for why a fixed grace period lost
        # the race to the sound card's block quantum.
        window_end = self._window_bounds(t_start, t_end)[1]
        self._wait_for_window(window_end, t_end)
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
