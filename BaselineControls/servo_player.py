"""
servo_player.py — stream a bow stroke with servoL so it can be changed mid-note.

WHY THIS EXISTS
───────────────
Every stroke in play_midi_pieces.py is one moveL: a blocking, point-to-point
move with its speed and acceleration fixed before it starts. That is why the
RL can only act once per stroke, and why a swell has to be pre-computed as a
blended path rather than reacted to. You cannot change a moveL in flight.

servoL is the opposite: a servo target you send repeatedly, typically at
125 Hz. The controller interpolates toward whatever pose arrived last, so the
speed and depth of a stroke can be changed WHILE it sounds — from a script, or
from a policy reading the microphone.

WHAT THIS DOES NOT CHANGE
─────────────────────────
The geometry. Poses are still apply_depth(pose_at(u), CFG, depth) for u inside
[U_MIN, U_MAX], so the taught frog->tip line, the axis-angle canonicalisation
that stops the wrist flipping, and the depth safety envelope all carry over
untouched. Only the way poses reach the controller changes.


SAFETY — READ THIS
──────────────────
servoL BYPASSES safe_moveL, and with it the fault checking every other motion
in this project relies on. safe_moveL raises if moveL returns False or the
control script stopped; a servo loop just keeps streaming. So this module has
to provide its own guards, and does:

  - u is clamped to [U_MIN, U_MAX] every single tick, so no command can ever
    leave the usable bow however wrong the speed setpoint is
  - depth is clamped to the recording script's envelope every tick
  - the loop is bounded in wall-clock time (never runs longer than the note it
    was asked for, times MAX_OVERRUN), so a stalled setpoint cannot leave the
    bow travelling
  - isProgramRunning() is polled (not every tick — it is an RTDE round trip),
    and the stroke aborts if the control script has died
  - servoStop() runs in a finally block, always

A stalled Python loop is the specific failure servoL makes possible: the
controller keeps servoing toward the last target it received. Because every
target here is an ABSOLUTE pose on the bow line rather than a velocity, a
stall means the bow stops at that pose — it does not run on. That is the main
reason to prefer servoL over speedL for this.

Inference must NOT run inside the loop. A classifier forward pass is ~100 ms,
which at 125 Hz is twelve missed ticks and an audible stutter. Update the
setpoint from another thread and let the loop read whatever is current.
"""

from __future__ import annotations

import threading
import time

import numpy as np


# ── Servo loop parameters ─────────────────────────────────────────
# Nominal control period, used only as a fallback. THE LOOP DOES NOT RUN AT
# THIS RATE: initPeriod()/waitPeriod() pace at the RTDE control interface's own
# frequency (500 Hz on a UR5e), whatever dt is passed to servoL. Measured on
# this rig, the loop ticked every ~2.4 ms (~410 Hz).
#
# That is why the bow position is integrated with the MEASURED interval rather
# than this constant. Assuming 8 ms while the loop really ran at 2.4 ms made
# every stroke 3.3x too fast — a commanded 0.090 m/s came out at 0.285.
SERVO_DT = 1.0 / 125.0

# servoL's speed/accel arguments are LIMITS on the servo's own tracking, not
# the speed of the stroke: the stroke's speed comes from how fast successive
# targets advance. They must stay above anything a stroke can ask for, or they
# silently become the binding constraint.
#
# The acceleration limit was 2.0 and that was WRONG. Covering L metres in T
# seconds needs 4L/T^2, so a 0.12 s note over 17.7 mm needs 4.9 m/s^2 — and at
# 2.0 the bow reached only 31% of the commanded speed. Measured in air:
#
#     dur    length   accel needed   measured lag @ limit 2.0
#     1.00s  149.9mm      0.6            -5%
#     0.50s   74.8mm      1.2           -11%
#     0.25s   37.3mm      2.4           -13%
#     0.12s   17.7mm      4.9           -61%
#
# ACCEL_MAX in play_midi_pieces is 4.0, and solve_stroke already shortens any
# stroke needing more than that, so matching it keeps the servo path from
# being tighter than the planner it is meant to execute.
SERVO_SPEED_LIMIT = 0.5      # m/s, above the 0.25 calibration ceiling
SERVO_ACCEL_LIMIT = 4.0      # m/s^2, matches play_midi_pieces.ACCEL_MAX

# lookahead_time smooths the trajectory: the controller aims slightly ahead of
# the target it was given. Low values track tightly and can judder if the
# stream is jittery; high values are smooth but lag. 0.1 s is a reasonable
# starting point for a bow — tune it by ear.
SERVO_LOOKAHEAD = 0.10       # s, valid range roughly 0.03..0.2

# Proportional gain on the servo. Higher tracks harder, and past ~1000 tends to
# be audible as mechanical noise on a light tool. Tune with lookahead.
SERVO_GAIN = 300             # valid range roughly 100..2000

# Note length at and above which play_auto() streams with servoL instead of
# handing the stroke to moveL. Measured in air, servoL tracks to -4% at 1.0 s
# and -8% at 0.5 s, but degrades to -13% at 0.25 s and collapses below that,
# because a servo stream can only approximate the trapezoid moveL's controller
# plans exactly.
#
# 0.5 s is the EXECUTION threshold. Being able to USE mid-stroke feedback is a
# separate, higher bar: a 0.5 s classifier window plus ~0.1 s of inference means
# no correction can land before ~0.6 s, so a note needs roughly 1 s to be
# adjusted on the basis of its own sound. On challengepiece that is 2% of notes;
# on string_crossings, 93%.
HYBRID_THRESHOLD = 0.5

# Hard bound on how long a stroke may run relative to what was asked for. A
# stalled or nonsensical setpoint hits this and the stroke ends.
MAX_OVERRUN = 1.5

# How often to check that the control script is still alive, in ticks. Every
# tick would add an RTDE round trip to a ~2.4 ms budget. At the measured
# ~410 Hz this is roughly four checks a second.
FAULT_CHECK_EVERY = 100


class ServoFaultStop(RuntimeError):
    """Raised when a servo stroke has to be abandoned. Same intent as
    recording_a_only.RobotFaultStop: stop, do not carry on regardless."""


class Setpoint:
    """
    The (speed, depth) the control loop reads each tick.

    Deliberately tiny and lock-guarded: the loop reads it 125 times a second
    and a policy thread writes it whenever it has a new opinion, which may be
    every few hundred milliseconds. Neither should block the other for long.
    """

    def __init__(self, speed: float, depth: float):
        self._lock = threading.Lock()
        self._speed = float(speed)
        self._depth = float(depth)
        self.updates = 0

    def read(self) -> tuple[float, float]:
        with self._lock:
            return self._speed, self._depth

    def write(self, speed: float | None = None, depth: float | None = None):
        with self._lock:
            if speed is not None:
                self._speed = float(speed)
            if depth is not None:
                self._depth = float(depth)
            self.updates += 1


class ServoStrokePlayer:
    """
    Plays strokes by streaming servoL targets along the taught bow line.

    `geom` is the loaded play_midi_pieces module — passed in rather than
    imported so this file has no opinion about how that module gets loaded
    (it lives outside a package and is normally loaded by path).
    """

    def __init__(self, controller, geom, dt: float = SERVO_DT,
                 lookahead: float = SERVO_LOOKAHEAD, gain: int = SERVO_GAIN,
                 lift_m: float = 0.0, accel_limit: float = SERVO_ACCEL_LIMIT):
        self.controller = controller
        self.rtde_c = controller.rtde_c
        self.rtde_r = controller.rtde_r
        self.geom = geom
        self.dt = dt
        self.lookahead = lookahead
        self.gain = gain
        # Raise every pose clear of the string by this much, along the same +z
        # the taught safe points use. lift_m = LIFT_M runs the whole stroke in
        # the AIR, which is how the loop should be commissioned: it exercises
        # the timing, the lookahead/gain tuning and Python's jitter with no
        # acoustic risk and nothing touching the instrument.
        self.lift_m = float(lift_m)
        self.accel_limit = float(accel_limit)

    # ── geometry helpers, all clamped ─────────────────────────────

    def _pose(self, u: float, depth: float) -> list[float]:
        g = self.geom
        u = float(np.clip(u, g.U_MIN, g.U_MAX))
        depth = float(np.clip(depth, -g.MAX_OUTWARD_DEPTH, g.MAX_INWARD_DEPTH))
        pose = np.asarray(g.apply_depth(g.pose_at(u), g.CFG, depth), dtype=float)
        if self.lift_m:
            pose[:3] = pose[:3] + self.lift_m * g.press_direction(g.CFG)
        return list(pose)

    # ── the loop ──────────────────────────────────────────────────

    def play_stroke(self, u_start: float, direction: str, duration: float,
                    setpoint: Setpoint, u_limit: float | None = None,
                    on_tick=None) -> dict:
        """
        Stream one stroke and return what actually happened.

        u_start    where the bow is now (the caller is responsible for it
                   being true — call place() first, or chain from the previous
                   stroke's u_end)
        direction  'down' (u increases) or 'up' (u decreases)
        duration   seconds to sound for; the loop ends at this, or earlier if
                   u_limit is reached
        setpoint   read every tick; this is what makes mid-stroke change work
        u_limit    optional hard stop along the bow, e.g. the planned u_end.
                   The stroke ends when it is passed, so a raised speed
                   setpoint shortens the note rather than overrunning the bow
        on_tick    optional callback(elapsed_s, u, speed, depth). Must be
                   FAST — it runs inside the 8 ms budget. Anything slow
                   (a classifier) belongs on another thread writing `setpoint`

        Returns a dict of what was achieved, including the true mean speed,
        which is what actually sets the loudness.
        """
        g = self.geom
        sign = 1.0 if direction == "down" else -1.0
        u = float(np.clip(u_start, g.U_MIN, g.U_MAX))

        # A stroke starting and ending at rest follows at best a triangular
        # velocity profile, whose MEAN is half its peak. Covering L in T
        # therefore needs 4L/T^2 of acceleration, so the fastest mean speed
        # available is accel_limit * T / 4.
        #
        # Commanding past that does not make the bow go faster, it just makes
        # the command a fiction the servo silently falls short of: measured in
        # air, a 0.12 s note asked for 0.15 m/s (ceiling 0.12) and delivered
        # 0.069 — a 54% shortfall that looked like a tracking bug and was not.
        #
        # solve_stroke does exactly this for the moveL path, shortening the
        # stroke rather than missing the beat. Matching it here keeps the two
        # paths honest about the same physics.
        speed_ceiling = self.accel_limit * duration / 4.0

        deadline = duration * MAX_OVERRUN
        t0 = time.time()
        t_prev = t0
        ticks = 0
        speed_capped = False
        stopped_because = "duration"
        # Tick timing is the thing most likely to go wrong in Python at
        # 125 Hz: GC pauses and any slow work inside the loop show up here as
        # jitter, and jitter is audible as judder.
        tick_times: list[float] = []

        try:
            while True:
                period = self.rtde_c.initPeriod()
                elapsed = time.time() - t0

                if elapsed >= duration:
                    break
                if elapsed >= deadline:
                    stopped_because = "overrun"
                    break

                speed, depth = setpoint.read()
                if speed > speed_ceiling:
                    speed = speed_ceiling
                    speed_capped = True

                # Advance along the bow by the time that ACTUALLY passed, not
                # by a nominal dt.
                #
                # initPeriod()/waitPeriod() pace the loop at the RTDE control
                # interface's own frequency (500 Hz on a UR5e), NOT at whatever
                # dt is passed to servoL. Integrating with a nominal 1/125 s
                # while the loop really ticked every ~2.4 ms made every stroke
                # run 3.3x too fast — measured 0.285 m/s for a commanded 0.090.
                # Using the real interval makes the speed correct at whatever
                # rate the loop happens to run.
                now = time.time()
                tick_dt = now - t_prev
                t_prev = now

                # DO NOT slew-limit the commanded velocity here. It was tried
                # (ramping toward the setpoint at accel_limit, on the theory
                # that a velocity step is unfollowable) and measured WORSE at
                # every duration:
                #
                #     dur     step command    slew-limited
                #     1.00s      -4.1%           -5.7%
                #     0.50s      -7.7%          -12.7%
                #     0.25s     -13.3%          -24.6%
                #     0.12s     -52.2%          -67.2%
                #
                # The servo already smooths the step — that is what lookahead
                # and gain do. Ramping the command as well puts two lags in
                # series and roughly halves the effective acceleration. The
                # step command is the right thing to send; the residual lag is
                # the servo's own, and belongs to lookahead tuning.
                u = float(np.clip(u + sign * speed * tick_dt / g.BOW_LENGTH,
                                  g.U_MIN, g.U_MAX))

                if u_limit is not None and (
                        (sign > 0 and u >= u_limit) or (sign < 0 and u <= u_limit)):
                    u = u_limit
                    stopped_because = "u_limit"
                    self._servo(u, depth, tick_dt)
                    break

                if u in (g.U_MIN, g.U_MAX):
                    stopped_because = "bow_end"
                    self._servo(u, depth, tick_dt)
                    break

                self._servo(u, depth, tick_dt)

                ticks += 1
                if ticks % FAULT_CHECK_EVERY == 0 and not self.rtde_c.isProgramRunning():
                    raise ServoFaultStop(
                        f"control script stopped {elapsed:.2f}s into a "
                        f"{direction} stroke at u={u:.3f}. STOPPING.")

                if on_tick is not None:
                    on_tick(elapsed, u, speed, depth)

                tick_times.append(time.time())
                self.rtde_c.waitPeriod(period)
        finally:
            # Always. A servo left running keeps tracking its last target.
            self.rtde_c.servoStop()

        achieved = time.time() - t0
        travelled = abs(u - u_start) * g.BOW_LENGTH
        gaps = np.diff(np.asarray(tick_times)) if len(tick_times) > 2 else np.array([])
        return {
            "u_start": u_start,
            "u_end": u,
            "length_m": travelled,
            "duration": achieved,
            "mean_speed": travelled / achieved if achieved > 0 else 0.0,
            "ticks": ticks,
            "setpoint_updates": setpoint.updates,
            "stopped_because": stopped_because,
            # True when the note was too short for the speed asked of it, so
            # the stroke was shortened rather than the timing missed. The
            # caller should expect it to sound quieter, exactly as the moveL
            # planner's CAP event means.
            "speed_capped": speed_capped,
            "speed_ceiling": speed_ceiling,
            # Loop health. dt_max well above dt is a dropped tick, and dropped
            # ticks are what judder sounds like.
            "dt_mean": float(gaps.mean()) if gaps.size else float("nan"),
            "dt_max": float(gaps.max()) if gaps.size else float("nan"),
            "dt_jitter": float(gaps.std()) if gaps.size else float("nan"),
        }

    def _servo(self, u: float, depth: float, dt: float | None = None):
        # servoL's `time` argument is how long the controller should take to
        # reach this target. It has to match the interval at which targets
        # actually arrive, or the controller is permanently aiming at a
        # deadline that does not match reality.
        ok = self.rtde_c.servoL(self._pose(u, depth), SERVO_SPEED_LIMIT,
                                self.accel_limit, dt or self.dt,
                                self.lookahead, self.gain)
        if ok is False:
            raise ServoFaultStop(
                f"servoL rejected a target at u={u:.3f}, depth={depth*1000:+.2f}mm. "
                f"STOPPING.")

    # ── hybrid dispatch ───────────────────────────────────────────

    def play_auto(self, u_start: float, direction: str, duration: float,
                  speed: float, depth: float, setpoint: Setpoint | None = None,
                  threshold: float = None) -> dict:
        """
        moveL for short notes, servoL for long ones — each where it wins.

        moveL hands the whole stroke to the controller, which plans an optimal
        trapezoid internally. A servo stream can only approximate one from
        outside, and measured in air that approximation costs:

            1.00 s  -4.1%      0.25 s  -13.3%
            0.50 s  -7.7%      0.12 s  -52%   (and capped)

        So below the threshold moveL is simply better, and nothing is lost:
        mid-stroke control is useless on a short note anyway. The feedback loop
        is a 0.5 s classifier window plus ~0.1 s of inference, so a correction
        cannot arrive until ~0.6 s in — after a 0.25 s note has finished.

        HYBRID_THRESHOLD defaults to 0.5 s, where servoL starts tracking
        cleanly. Note that is the EXECUTION threshold; a note needs to be
        roughly 1 s before a mid-stroke correction can actually act on what it
        heard, which on most repertoire is a small minority of notes.
        """
        g = self.geom
        threshold = HYBRID_THRESHOLD if threshold is None else threshold

        if duration >= threshold:
            sp = setpoint if setpoint is not None else Setpoint(speed, depth)
            sign = 1.0 if direction == "down" else -1.0
            u_limit = float(np.clip(u_start + sign * speed * duration / g.BOW_LENGTH,
                                    g.U_MIN, g.U_MAX))
            out = self.play_stroke(u_start, direction, duration, sp,
                                   u_limit=u_limit)
            out["mode"] = "servoL"
            return out

        # Short note: let the controller plan it. solve_stroke supplies the
        # same speed/accel the moveL pipeline would have used, including its
        # length cap when the note is too short for the bow it asked for.
        sign = 1.0 if direction == "down" else -1.0
        solution = g.solve_stroke(speed * duration, duration,
                                  accel_max=self.accel_limit)
        u_end = float(np.clip(u_start + sign * solution.length / g.BOW_LENGTH,
                              g.U_MIN, g.U_MAX))
        target = g.apply_depth(g.pose_at(u_end), g.CFG, depth)
        if self.lift_m:
            target = np.asarray(target, dtype=float)
            target[:3] = target[:3] + self.lift_m * g.press_direction(g.CFG)
        t0 = time.time()
        g.safe_moveL(self.controller, target, solution.speed, solution.accel,
                     what=f"short note ({duration*1000:.0f} ms, {direction})")
        achieved = time.time() - t0
        return {
            "mode": "moveL",
            "u_start": u_start,
            "u_end": u_end,
            "length_m": solution.length,
            "duration": achieved,
            "mean_speed": solution.length / achieved if achieved > 0 else 0.0,
            "ticks": 0,
            "setpoint_updates": 0,
            "stopped_because": "moveL",
            "speed_capped": solution.length_capped,
            "speed_ceiling": self.accel_limit * duration / 4.0,
            "dt_mean": float("nan"), "dt_max": float("nan"),
            "dt_jitter": float("nan"),
        }

    # ── verification without touching the string ──────────────────

    def dry_trajectory(self, u_start: float, direction: str, duration: float,
                       speed: float, depth: float) -> dict:
        """
        Compute the pose stream WITHOUT sending it. Use before any hardware
        run to confirm the trajectory stays on the bow line and inside the
        limits — cheap, and it catches sign and clamping errors on the desk.
        """
        g = self.geom
        sign = 1.0 if direction == "down" else -1.0
        u = float(np.clip(u_start, g.U_MIN, g.U_MAX))
        poses, us = [], [u]
        for _ in range(int(duration / self.dt)):
            u = float(np.clip(u + sign * speed * self.dt / g.BOW_LENGTH,
                              g.U_MIN, g.U_MAX))
            poses.append(self._pose(u, depth))
            us.append(u)
        poses = np.asarray(poses)
        us = np.asarray(us)
        # Every commanded pose must be exactly the pose its own u implies —
        # compared against _pose() itself so the depth offset and any lift are
        # accounted for. (Measuring distance from the bare frog->tip line would
        # just re-report the lift, which is useless in air mode.)
        offsets = [np.linalg.norm(np.asarray(p[:3])
                                  - np.asarray(self._pose(u, depth)[:3]))
                   for p, u in zip(poses, us[1:])]
        return {
            "n_ticks": len(poses),
            "u_range": (float(us.min()), float(us.max())),
            "within_limits": bool(us.min() >= g.U_MIN - 1e-9
                                  and us.max() <= g.U_MAX + 1e-9),
            "max_offset_from_bow_line_mm": float(np.max(offsets) * 1000.0),
            "length_m": float(abs(us[-1] - us[0]) * g.BOW_LENGTH),
        }
