"""
rl/piece_env.py

Residual RL over a whole piece, one bow stroke per RL step.

The baseline is BaselineControls/play_midi_pieces.py: its score parsing
(MIDI / MusicXML, bowings, slurs, dynamics) and its BowPlanner produce the
strokes the piece *should* be played with. This environment lets a policy
apply a small residual to each stroke — a speed scale and a depth offset,
the two knobs the one-stroke RL (other_github/rl/sac) plugs in — and is
rewarded primarily by the sound classifier's tone-quality score.

The score's intent is never overridden, only shaded:
    - bow DIRECTIONS (the written/planned bowings) are kept exactly
    - note ONSETS and DURATIONS are kept exactly (rhythm is sacred)
    - mean speed may move ±SPEED_RESIDUAL_FRAC around the planned value,
      which by the dataset calibration is about ±1.6 dB of shading
    - depth may move ±DEPTH_RESIDUAL_M, clamped to the recording script's
      safety envelope [-MAX_OUTWARD_DEPTH, +MAX_INWARD_DEPTH]

Because speed × duration = bow length, a speed residual changes how much bow
a stroke consumes, so the environment re-tracks the bow position u stroke by
stroke and clamps any stroke that would run past U_MIN/U_MAX (with a reward
penalty), exactly mirroring the planner's own budget discipline. Retakes
re-anchor the bow at their planned landing point, so drift cannot compound
across a rest.

Episode  = one pass through the piece.
Step     = one stroke:  obs -> action (residual) -> execute -> classify -> reward.

Executors:
    MockExecutor      no hardware; synthesizes physical params. Pair with
                      MockScorer, whose sweet spot in (speed, depth) gives
                      SAC a learnable signal for end-to-end testing —
                      the same idea as the partner code's MockSoundClassifier,
                      extended to both knobs.
    HardwareExecutor  rl/piece_hardware.py — real UR5e via PiecePlayer's
                      verified motion primitives + microphone capture.

Scorers:
    RealScorer        SoundClassifier/classifier.py BowingQualityClassifier:
                      predict(audio_500ms, physical_params_6, string) -> [0,1].
                      Falls back to its own heuristic when no checkpoint.
    MockScorer        physics-only stand-in for training without a mic.

Run from the repository environment created by scripts/setup.sh or
scripts/setup.ps1 (importing the baseline pulls in sounddevice; ur_rtde is
stubbed in mock mode when the optional hardware profile is not installed).
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import gymnasium as gym
from gymnasium import spaces

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ══════════════════════════════════════════════════════════════════
# Baseline import — the planner IS the baseline, loaded, not copied
# ══════════════════════════════════════════════════════════════════

def _stub_rtde_if_missing():
    """
    Mock-mode fallback: baseline_controller does `import rtde_control` /
    `import rtde_receive` at module level, but never touches the robot until
    a controller is constructed. On a machine without the ur_rtde build the
    mock pipeline is still perfectly usable, so install loud stub modules
    rather than dying at import. Anything that actually reaches for the robot
    raises immediately, and piece_hardware refuses to run against a stub.
    """
    import types
    stubbed = []
    for name in ("rtde_control", "rtde_receive", "rtde_io"):
        try:
            __import__(name)
        except ImportError:
            module = types.ModuleType(name)
            module._IS_CELLO_RL_STUB = True

            def _raise(*_a, _name=name, **_k):
                raise RuntimeError(
                    f"{_name} is a mock-mode stub — the real ur_rtde build is "
                    f"not importable in this environment, so the robot cannot "
                    f"be used. Mock training is unaffected.")

            def _getattr(attr, _r=_raise):
                # Behave like a normal module for dunder probes (__file__,
                # __spec__, ...) so inspect/pickle machinery is not confused;
                # only robot-API lookups get the raiser.
                if attr.startswith("__") and attr.endswith("__"):
                    raise AttributeError(attr)
                return _r

            module.__getattr__ = _getattr
            sys.modules[name] = module
            stubbed.append(name)
    if stubbed:
        print(f"(ur_rtde not importable — stubbed {', '.join(stubbed)}; "
              f"mock mode only, hardware runs will refuse to start)")


def load_baseline_module():
    """
    Load BaselineControls/play_midi_pieces.py by path (it is not a package).

    This transitively imports the verified setup from recording_a_only
    (waypoints, depth convention, safe_moveL, sounddevice), which is
    the point: the RL baseline cannot drift from what was verified on the
    pendant. Without the ur_rtde build only mock mode is available.
    """
    _stub_rtde_if_missing()
    path = REPO_ROOT / "BaselineControls" / "play_midi_pieces.py"
    spec = importlib.util.spec_from_file_location("play_midi_pieces", path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except ImportError as e:
        raise ImportError(
            f"Could not import the baseline planner ({e}).\n"
            f"It pulls in ur_rtde and sounddevice — run with\n"
            f"  python rl/play_piece.py ..."
        ) from e
    sys.modules.setdefault("play_midi_pieces", module)
    return module


PMP = load_baseline_module()


# ══════════════════════════════════════════════════════════════════
# Residual bounds
# ══════════════════════════════════════════════════════════════════

# ±20% on mean bow speed ≈ ±1.6 dB by the dataset_a_final calibration:
# audible as shading, never as a wrong dynamic. Matches BOW_SLACK, the
# planner's own bow-distribution allowance.
SPEED_RESIDUAL_FRAC = 0.20

# ±0.5 mm of depth around the planned value, always re-clamped to the
# recording script's safety envelope (-1.5 mm .. +2.0 mm). Depth is the
# timbral knob: worth <1 dB of loudness but it changes how the string speaks.
DEPTH_RESIDUAL_M = 0.0005

DEPTH_LO = -PMP.MAX_OUTWARD_DEPTH   # -1.5 mm (lighter)
DEPTH_HI = PMP.MAX_INWARD_DEPTH     # +2.0 mm (heavier)

# Reward weights. Quality is the headline; dynamic accuracy anchors the
# policy to the written dynamics so "best tone" cannot be bought by playing
# everything at one comfortable loudness.
W_QUALITY = 0.50
W_DYNAMIC = 0.25
W_BOW     = 0.15
W_SMOOTH  = 0.10

# A dynamic error of this many dB zeroes the dynamic reward term.
DYNAMIC_FULL_ERR_DB = 3.0

OBS_DIM = 18
ACTION_DIM = 2      # [speed residual, depth residual], each in [-1, 1]


# ══════════════════════════════════════════════════════════════════
# Executor / scorer interfaces
# ══════════════════════════════════════════════════════════════════

@dataclass
class ExecStroke:
    """One stroke as actually commanded (baseline plan + residual applied)."""
    note_index: int
    direction: str                  # 'down' | 'up' — from the plan, never changed
    u_start: float
    u_end: float
    length: float                   # m
    duration: float                 # s, from the score (rhythm preserved)
    mean_speed: float               # m/s — what sets the loudness
    speed: float                    # m/s commanded peak
    accel: float                    # m/s^2 commanded
    depth: float                    # m signed, clamped to safety envelope
    volume_target: float            # 0..1, what the score asked for
    retake_from: float | None = None
    onset: float = 0.0
    gap_before: float = 0.0         # s of silence before this stroke
    segments: list = field(default_factory=list)   # PMP.Segment list


@dataclass
class StrokeResult:
    """What came back from executing one stroke."""
    audio: np.ndarray | None        # ~500 ms peak-normalized window, or None (mock)
    physical: np.ndarray            # (6,) in dataset.PHYSICAL_FEATURE_NAMES order
    measured_mean_speed: float
    achieved_u_end: float


class ExecutorBase:
    def begin_episode(self, first: ExecStroke):
        pass

    def execute(self, stroke: ExecStroke) -> StrokeResult:
        raise NotImplementedError

    def end_episode(self):
        pass

    def close(self):
        pass


class MockExecutor(ExecutorBase):
    """
    No hardware. Physical params are synthesized from the command:

        [depth_m, 0, mean_speed, mid_u, torque_est, torque_est*1.3]

    which mirrors dataset.build_physical_features for records without a
    force_contact block (slot 0 = commanded depth). Torque grows with depth
    so a depth-sensitive classifier sees a consistent signal.
    """

    def __init__(self, noise: float = 0.02, rng: np.random.Generator | None = None):
        self.noise = noise
        self.rng = rng or np.random.default_rng()

    def execute(self, stroke: ExecStroke) -> StrokeResult:
        n = lambda s: 1.0 + self.noise * self.rng.standard_normal() * s
        mean_speed = stroke.mean_speed * n(1.0)
        mid_u = (stroke.u_start + stroke.u_end) / 2.0
        # crude contact-torque model: rises with depth into the string
        torque = max(0.0, 0.08 + 60.0 * (stroke.depth - DEPTH_LO)) * n(2.0)
        physical = np.array([
            stroke.depth, 0.0, mean_speed, mid_u, torque, torque * 1.3,
        ], dtype=np.float32)
        return StrokeResult(
            audio=None,
            physical=physical,
            measured_mean_speed=mean_speed,
            achieved_u_end=stroke.u_end,
        )


class MockScorer:
    """
    Stand-in for the sound classifier: tone quality peaks at a sweet spot in
    (depth, speed) and degrades away from it, with noise. Gives SAC a real
    gradient to climb so the whole pipeline can be validated hardware-free.
    """

    DEPTH_STAR = 0.0004     # m — slightly into the string
    DEPTH_SIGMA = 0.0007
    SPEED_BREAKUP = 0.07    # m/s — tone breaks up below this

    def __init__(self, noise: float = 0.03, rng: np.random.Generator | None = None):
        self.noise = noise
        self.rng = rng or np.random.default_rng()

    def score(self, audio, physical, window_pos=0.5, string="A") -> float:
        depth, speed = float(physical[0]), float(physical[2])
        q = np.exp(-0.5 * ((depth - self.DEPTH_STAR) / self.DEPTH_SIGMA) ** 2)
        if speed < self.SPEED_BREAKUP:
            q *= max(0.0, speed / self.SPEED_BREAKUP) ** 2
        q += self.noise * self.rng.standard_normal()
        return float(np.clip(q, 0.0, 1.0))


# Preferred RL reward checkpoint. SoundClassifier's own DEFAULT_CHECKPOINT is
# quality_cnn.pt, which is trained on pseudo_heuristic_v1 PSEUDO-LABELS — its
# near-perfect metrics (val_mse 0.0003, rho 0.99) are it re-learning a
# deterministic heuristic, not human judgement of tone. Optimising a policy
# against it would maximise that heuristic, which is not what "best tone"
# means here. The human-annotated checkpoint (399 ratings from 2 annotators,
# recording-level Spearman rho 0.70) is the honest reward signal, so it is
# preferred when present.
HUMAN_CHECKPOINT = (REPO_ROOT / "SoundClassifier" / "checkpoints"
                    / "quality_cnn_human_current.pt")


class RealScorer:
    """
    Wraps SoundClassifier/classifier.py's BowingQualityClassifier — the real
    reward source. Without a checkpoint the classifier falls back to its own
    scalar-feature heuristic.

    Checkpoint choice is reported loudly, and pseudo-label checkpoints are
    warned about, because which one is loaded decides what the policy is
    actually being trained to maximise.
    """

    def __init__(self, checkpoint_path=None, device="cpu"):
        sc_dir = REPO_ROOT / "SoundClassifier"
        if str(sc_dir) not in sys.path:
            sys.path.insert(0, str(sc_dir))
        from classifier import BowingQualityClassifier, DEFAULT_CHECKPOINT

        if checkpoint_path is None:
            checkpoint_path = (HUMAN_CHECKPOINT if HUMAN_CHECKPOINT.exists()
                               else DEFAULT_CHECKPOINT)
        self.checkpoint_path = Path(checkpoint_path)
        self.clf = BowingQualityClassifier(checkpoint_path=checkpoint_path,
                                           device=device)

        self.label_source = "unknown"
        try:
            import torch
            meta = torch.load(checkpoint_path, map_location="cpu",
                              weights_only=False)
            self.label_source = meta.get("label_source", "unknown")
            if meta.get("pseudo_labels"):
                print(f"##  WARNING: {self.checkpoint_path.name} is trained on "
                      f"PSEUDO-LABELS ({self.label_source}). A policy trained "
                      f"against it maximises a heuristic, not human-judged "
                      f"tone. Prefer {HUMAN_CHECKPOINT.name}.")
            else:
                print(f"Reward checkpoint: {self.checkpoint_path.name} "
                      f"(label_source={self.label_source})")
        except Exception:
            pass

    def score(self, audio, physical, window_pos=0.5, string="A") -> float:
        return float(self.clf.predict(audio, physical, string=string,
                                      window_pos=window_pos))


# ══════════════════════════════════════════════════════════════════
# Score loading + baseline planning
# ══════════════════════════════════════════════════════════════════

def load_piece(path: str, tempo_scale: float = 1.0,
               bowing_rule: str | None = None):
    """
    Parse a piece the same way play_midi_pieces.py does and plan it.

    Returns (notes, meta, strokes) where strokes is the baseline plan the
    residual policy shades. .mxl/.musicxml/.xml files carry written bowings,
    slurs and dynamics; .mid carries rhythm + velocities (+ CC2 hairpins).
    """
    p = Path(path)
    if p.suffix.lower() in (".mxl", ".musicxml", ".xml"):
        notes, meta = PMP.parse_musicxml(str(p), tempo_scale=tempo_scale)
    else:
        notes, meta = PMP.parse_midi(str(p), tempo_scale=tempo_scale)

    if bowing_rule:
        PMP.assign_bowings(notes, bowing_rule)

    notes, n_merged, n_portato = PMP.merge_slurs(notes)
    meta["n_slurs_merged"] = n_merged
    meta["dynamics"] = PMP.describe_dynamic_range(notes)

    planner = PMP.BowPlanner()
    strokes = planner.plan(notes)
    return notes, meta, strokes


# ══════════════════════════════════════════════════════════════════
# Environment
# ══════════════════════════════════════════════════════════════════

class PieceResidualEnv(gym.Env):
    """
    Observation (18,), roughly normalized to [-1, 1]:
        0  planned duration / 2 s
        1  planned mean speed / SPEED_MAX
        2  planned depth / DEPTH_HI
        3  direction (+1 down, -1 up)
        4  current bow position u
        5  available bow fraction in this stroke's direction
        6  volume target (0..1)
        7  volume at note end (== target when no swell)
        8  1 if this stroke starts with a retake
        9  1 if this stroke is part of a split note
        10 next stroke volume (0 at end)
        11 next stroke duration / 2 s
        12 next stroke direction (0 at end)
        13 last stroke's quality score
        14 EMA of quality this episode
        15 progress through the piece (0..1)
        16 previous action[0]
        17 previous action[1]

    Action (2,) in [-1, 1]:
        0  speed residual  -> mean speed × (1 + a0 · SPEED_RESIDUAL_FRAC)
        1  depth residual  -> depth + a1 · DEPTH_RESIDUAL_M (clamped to envelope)

    Reward per stroke:
        W_QUALITY · classifier score
      + W_DYNAMIC · (1 - |achieved-vs-written dB error| / 3 dB)
      - W_BOW     · bow-budget violation (shortfall + limit proximity)
      - W_SMOOTH  · action jerk between consecutive strokes
    """

    metadata = {"render_modes": []}

    def __init__(self, piece_path: str, executor: ExecutorBase | None = None,
                 scorer=None, tempo_scale: float = 1.0,
                 bowing_rule: str | None = None, ema_alpha: float = 0.7,
                 string: str = "A"):
        super().__init__()
        self.piece_path = str(piece_path)
        self.executor = executor or MockExecutor()
        self.scorer = scorer or MockScorer()
        self.string = string
        self.ema_alpha = ema_alpha

        self.notes, self.meta, self.plan = load_piece(
            piece_path, tempo_scale=tempo_scale, bowing_rule=bowing_rule)
        if not self.plan:
            raise ValueError(f"{piece_path}: the planner produced no strokes")

        self.observation_space = spaces.Box(-np.inf, np.inf, (OBS_DIM,),
                                            dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (ACTION_DIM,),
                                       dtype=np.float32)

        self._i = 0
        self._u = self.plan[0].u_start
        self._ema = 0.5
        self._last_q = 0.5
        self._prev_action = np.zeros(ACTION_DIM, dtype=np.float32)
        self._episode_log: list[dict] = []

    # ── residual application ──────────────────────────────────────

    def _build_exec_stroke(self, plan_stroke, action) -> tuple[ExecStroke, float]:
        """
        Apply the residual to a planned stroke from the CURRENT bow position.

        Returns (exec_stroke, shortfall_frac) where shortfall_frac is how much
        of the desired length had to be cut to stay inside [U_MIN, U_MAX].
        """
        s = plan_stroke
        direction = s.direction
        sign = 1.0 if direction == "down" else -1.0

        # Retakes re-anchor the bow at the planned landing point.
        retake_from = None
        u_start = self._u
        if s.retake_from is not None:
            retake_from = self._u
            u_start = s.u_start

        speed_scale = 1.0 + float(action[0]) * SPEED_RESIDUAL_FRAC
        depth = float(np.clip(s.depth + float(action[1]) * DEPTH_RESIDUAL_M,
                              DEPTH_LO, DEPTH_HI))

        desired = s.length * speed_scale
        if direction == "down":
            available = max(0.0, (PMP.U_MAX - u_start) * PMP.BOW_LENGTH)
        else:
            available = max(0.0, (u_start - PMP.U_MIN) * PMP.BOW_LENGTH)
        length = float(np.clip(desired, 1e-4, available))
        shortfall = (desired - length) / max(desired, 1e-9)

        solution = PMP.solve_stroke(length, s.duration)
        length = solution.length
        u_end = float(np.clip(u_start + sign * length / PMP.BOW_LENGTH,
                              PMP.U_MIN, PMP.U_MAX))

        # Segments: scale the planned spans onto the new length, keep the
        # swell's relative depth shape, shift it by the depth residual.
        segments = []
        if len(s.segments) > 1 and abs(s.u_end - s.u_start) > 1e-9:
            scale = (u_end - u_start) / (s.u_end - s.u_start)
            u = u_start
            for seg in s.segments:
                span = (seg.u_end - seg.u_start) * scale
                sub = PMP.solve_stroke(abs(span) * PMP.BOW_LENGTH, seg.duration)
                segments.append(PMP.Segment(
                    u_start=u, u_end=u + span, speed=sub.speed, accel=sub.accel,
                    depth=float(np.clip(seg.depth + float(action[1]) * DEPTH_RESIDUAL_M,
                                        DEPTH_LO, DEPTH_HI)),
                    duration=sub.duration))
                u += span
        else:
            segments.append(PMP.Segment(
                u_start=u_start, u_end=u_end, speed=solution.speed,
                accel=solution.accel, depth=depth, duration=solution.duration))

        prev_end = (self.plan[self._i - 1].onset + self.plan[self._i - 1].duration
                    if self._i > 0 else 0.0)
        exec_stroke = ExecStroke(
            note_index=s.note_index,
            direction=direction,
            u_start=u_start,
            u_end=u_end,
            length=length,
            duration=s.duration,
            mean_speed=solution.mean_speed,
            speed=solution.speed,
            accel=solution.accel,
            depth=depth,
            volume_target=s.volume_target,
            retake_from=retake_from,
            onset=s.onset,
            gap_before=max(0.0, s.onset - prev_end),
            segments=segments,
        )
        return exec_stroke, float(shortfall)

    # ── reward ────────────────────────────────────────────────────

    def _reward(self, plan_stroke, exec_stroke, result, shortfall,
                action) -> tuple[float, dict]:
        window_pos = 0.5
        quality = float(self.scorer.score(result.audio, result.physical,
                                          window_pos=window_pos,
                                          string=self.string))

        # Dynamic accuracy: achieved mean speed vs what the SCORE asked for
        # (the plan's own volume_actual may already be shaded by the planner;
        # the score's volume_target is the ground truth).
        target_speed = PMP.volume_to_speed(plan_stroke.volume_target)
        err_db = abs(20.0 * np.log10(
            max(result.measured_mean_speed, 1e-3) / max(target_speed, 1e-3)))
        r_dynamic = float(np.clip(1.0 - err_db / DYNAMIC_FULL_ERR_DB, 0.0, 1.0))

        # Bow budget: cutting a stroke short, or parking near a hard limit,
        # is the residual policy's fault — the baseline plan always fits.
        u_end = result.achieved_u_end
        edge = min(u_end - PMP.U_MIN, PMP.U_MAX - u_end)
        r_bow = -float(np.clip(3.0 * shortfall, 0.0, 1.0))
        if edge < 0.03:
            r_bow -= 0.5 * (1.0 - edge / 0.03)

        r_smooth = -0.5 * float(np.mean(np.abs(action - self._prev_action)))

        total = (W_QUALITY * quality + W_DYNAMIC * r_dynamic
                 + W_BOW * r_bow + W_SMOOTH * r_smooth)
        components = {
            "quality": quality, "r_dynamic": r_dynamic, "r_bow": r_bow,
            "r_smooth": r_smooth, "err_db": float(err_db),
            "shortfall": shortfall, "total": float(total),
        }
        return float(total), components

    # ── observation ───────────────────────────────────────────────

    def _obs(self) -> np.ndarray:
        if self._i < len(self.plan):
            s = self.plan[self._i]
        else:
            s = self.plan[-1]
        sign = 1.0 if s.direction == "down" else -1.0
        u = self._u if s.retake_from is None else s.u_start
        if s.direction == "down":
            avail = (PMP.U_MAX - u) / (PMP.U_MAX - PMP.U_MIN)
        else:
            avail = (u - PMP.U_MIN) / (PMP.U_MAX - PMP.U_MIN)

        if self._i + 1 < len(self.plan):
            nxt = self.plan[self._i + 1]
            nxt_vol, nxt_dur = nxt.volume_target, nxt.duration
            nxt_dir = 1.0 if nxt.direction == "down" else -1.0
        else:
            nxt_vol, nxt_dur, nxt_dir = 0.0, 0.0, 0.0

        vol_end = s.volume_target
        note = self.notes[s.note_index] if s.note_index < len(self.notes) else None
        if note is not None and note.volume_end is not None:
            vol_end = float(np.clip(note.volume_end, 0.0, 1.0))

        return np.array([
            s.duration / 2.0,
            s.mean_speed / PMP.SPEED_MAX,
            s.depth / DEPTH_HI,
            sign,
            u,
            float(np.clip(avail, 0.0, 1.0)),
            s.volume_target,
            vol_end,
            1.0 if s.retake_from is not None else 0.0,
            1.0 if s.part_of is not None else 0.0,
            nxt_vol,
            nxt_dur / 2.0,
            nxt_dir,
            self._last_q,
            self._ema,
            self._i / max(len(self.plan), 1),
            float(self._prev_action[0]),
            float(self._prev_action[1]),
        ], dtype=np.float32)

    # ── Gymnasium API ─────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._i = 0
        self._u = self.plan[0].u_start
        self._ema = 0.5
        self._last_q = 0.5
        self._prev_action = np.zeros(ACTION_DIM, dtype=np.float32)
        self._episode_log = []

        first, _ = self._build_exec_stroke(self.plan[0],
                                           np.zeros(ACTION_DIM, dtype=np.float32))
        self.executor.begin_episode(first)
        return self._obs(), {"piece": self.piece_path, "n_strokes": len(self.plan)}

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32).flatten()
        plan_stroke = self.plan[self._i]

        exec_stroke, shortfall = self._build_exec_stroke(plan_stroke, action)
        result = self.executor.execute(exec_stroke)
        reward, components = self._reward(plan_stroke, exec_stroke, result,
                                          shortfall, action)

        self._u = result.achieved_u_end
        self._last_q = components["quality"]
        self._ema = self.ema_alpha * self._ema + (1 - self.ema_alpha) * self._last_q
        self._prev_action = action.copy()

        self._episode_log.append({
            "note_index": exec_stroke.note_index,
            "direction": exec_stroke.direction,
            "u_start": exec_stroke.u_start,
            "u_end": result.achieved_u_end,
            "mean_speed": exec_stroke.mean_speed,
            "depth_mm": exec_stroke.depth * 1000.0,
            "action": action.tolist(),
            **components,
        })

        self._i += 1
        terminated = self._i >= len(self.plan)
        if terminated:
            self.executor.end_episode()

        info = {"stroke": self._episode_log[-1]}
        if terminated:
            info["episode_log"] = self._episode_log
            info["mean_quality"] = float(np.mean(
                [s["quality"] for s in self._episode_log]))

        return self._obs(), reward, terminated, False, info

    def close(self):
        self.executor.close()
