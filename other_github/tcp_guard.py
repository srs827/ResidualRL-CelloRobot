"""
tcp_guard.py — verify (and repair) the robot's active TCP before a session trusts any pose.

WHY: two stacks share this UR5e. Ours works in the BOW-TIP TCP (~141mm from the flange;
baseline_controller / teach_tip_on_axis set it). Samantha's SoundClassifier stack works in
the FLANGE TCP ([0,0,0,0,0,0]; recording_a_only.apply_verified_installation sets it).
setTcp PERSISTS on the controller after a script disconnects, so whichever stack ran LAST
silently decides the frame the next pose read/command happens in — a ~141mm ambush
(check_tcp.py documents the original 06-21 incident).

WHAT: ensure_bowtip_tcp(rtde_c, rtde_r=None) detects which frame was active, forces the
bow-tip TCP, and prints a loud banner if the previous frame was NOT ours — so you know the
F/T tare and any poses captured since the other stack ran are suspect.

Detection is two-tier because our pinned ur_rtde may not expose getTCPOffset() (check_tcp.py
already wraps it):
  1. getTCPOffset() read-back when available;
  2. otherwise the pose-jump trick (same as check_tcp.py): read getActualTCPPose before/after
     setTcp — the same physical pose "jumps" by the TCP delta iff the previous TCP differed.

Desk-tested with mocks only (run:  python tcp_guard.py --selftest);
NOT yet exercised against the live robot — watch its output on the next lab visit.
Added 2026-07-30 (RL stack side) for the two-stacks-one-robot era.
"""
import time

import numpy as np

# Canonical bow-tip TCP — must match baseline_controller.TCP_OFFSET / check_tcp.TCP_OFFSET.
# Callers that own their own copy should pass it via `expected` (the caller's constant wins),
# so a future re-teach only has to update the caller.
BOWTIP_TCP = [0.028210348281514253, -0.09610723587300697, -0.09969041498611403, 0.0, 0.0, 0.0]

FLANGE_TOL_MM = 1.0   # |xyz offset| below this ~= flange TCP (the other stack)
MATCH_TOL_M = 1e-4    # xyz agreement for "already correct"
JUMP_WARN_MM = 5.0    # reported-pose jump above this = previous TCP was different
_SETTLE_S = 0.3


def _banner(lines):
    bar = "!" * 78
    print("\n" + bar)
    for ln in lines:
        print("!! " + ln)
    print(bar + "\n")


def _warn_wrong_frame(tag, prev=None, jump_mm=None):
    lines = [f"ACTIVE TCP WAS NOT OURS{tag} — corrected to the bow-tip TCP just now."]
    if prev is not None:
        off_mm = float(np.linalg.norm(np.asarray(prev, float)[:3]) * 1000.0)
        which = "FLANGE [0,0,0] — the recording stack ran last" if off_mm < FLANGE_TOL_MM \
            else f"unknown frame (|xyz offset| {off_mm:.1f}mm)"
        lines.append(f"previous offset: {[round(float(x), 5) for x in prev]}  -> {which}.")
    if jump_mm is not None:
        lines.append(f"same physical pose jumped {jump_mm:.0f}mm when the bow-tip TCP was applied.")
    lines += [
        "Everything since the other stack ran is SUSPECT:",
        "  - F/T tare + payload are still THEIRS -> before trusting forces: (1) set OUR",
        "    payload (tare_force_sensor.py does this), (2) lift the bow CLEAR of the string,",
        "    (3) re-zero the F/T sensor;",
        "  - any poses captured/logged in that window are in the WRONG frame.",
        "The bow-tip TCP is active from here on; safe to continue.",
    ]
    _banner(lines)


def ensure_bowtip_tcp(rtde_c, rtde_r=None, expected=None, label=""):
    """Force the bow-tip TCP; report whether the previously-active TCP already matched.

    rtde_c: RTDEControlInterface (required — setTcp/getTCPOffset live here).
    rtde_r: RTDEReceiveInterface (optional — enables the pose-jump fallback).
    expected: 6-vector TCP offset; defaults to BOWTIP_TCP.
    Returns {"was_correct": True/False/None, "jump_mm": float|None, "method": str}.
    Detection failures never raise; the one guarantee is that setTcp(expected) gets called.
    """
    exp = np.asarray(expected if expected is not None else BOWTIP_TCP, float)
    tag = f" [{label}]" if label else ""

    # Tier 1: direct read-back (not available on every ur_rtde build)
    prev = None
    try:
        prev = np.asarray(list(rtde_c.getTCPOffset()), float)
    except Exception:
        prev = None

    if prev is not None and prev.shape == (6,):
        was_correct = bool(np.allclose(prev[:3], exp[:3], atol=MATCH_TOL_M)
                           and np.allclose(prev[3:], exp[3:], atol=1e-3))
        if was_correct:
            print(f"[tcp_guard]{tag} active TCP already bow-tip (getTCPOffset) - ok")
        else:
            rtde_c.setTcp(list(exp))
            time.sleep(_SETTLE_S)
            _warn_wrong_frame(tag, prev=prev)
        return {"was_correct": was_correct, "jump_mm": None, "method": "getTCPOffset"}

    # Tier 2: pose-jump (check_tcp.py trick) — needs a receive interface
    if rtde_r is not None:
        try:
            p0 = np.asarray(rtde_r.getActualTCPPose(), float)
            rtde_c.setTcp(list(exp))
            time.sleep(_SETTLE_S)
            p1 = np.asarray(rtde_r.getActualTCPPose(), float)
            jump_mm = float(np.linalg.norm(p1[:3] - p0[:3]) * 1000.0)
            was_correct = jump_mm < JUMP_WARN_MM
            if was_correct:
                print(f"[tcp_guard]{tag} active TCP already bow-tip "
                      f"(pose-jump {jump_mm:.1f}mm) - ok")
            else:
                _warn_wrong_frame(tag, jump_mm=jump_mm)
            return {"was_correct": was_correct, "jump_mm": jump_mm, "method": "pose-jump"}
        except Exception as e:
            print(f"[tcp_guard]{tag} pose-jump detection failed ({e}); setting TCP blind.")

    # Tier 3: no way to detect — set blind, say so
    rtde_c.setTcp(list(exp))
    time.sleep(_SETTLE_S)
    print(f"[tcp_guard]{tag} bow-tip TCP set (previous frame UNVERIFIED — no getTCPOffset, "
          f"no receive interface). If forces/poses look ~141mm odd, run check_tcp.py.")
    return {"was_correct": None, "jump_mm": None, "method": "blind-set"}


# ------------------------------- self-test (no robot) -------------------------------

class _FakeC:
    def __init__(self, offset, has_getter=True):
        self.offset = list(offset)
        self.has_getter = has_getter
        self.set_calls = []

    def getTCPOffset(self):
        if not self.has_getter:
            raise RuntimeError("getTCPOffset not available in this ur_rtde build")
        return list(self.offset)

    def setTcp(self, v):
        self.set_calls.append(list(v))
        self.offset = list(v)


class _FakeR:
    """Reports a pose that shifts by the TCP-offset delta once setTcp is called."""

    def __init__(self, fake_c, initial_offset):
        self._c = fake_c
        self._initial = np.asarray(initial_offset, float)
        self._base = np.array([0.4, 0.5, 0.1, -1.4, -2.2, 1.4])

    def getActualTCPPose(self):
        cur = np.asarray(self._c.offset, float)
        p = self._base.copy()
        p[:3] += (cur[:3] - self._initial[:3])  # crude but right order of magnitude
        return p.tolist()


def _selftest():
    global _SETTLE_S
    _SETTLE_S = 0.0
    flange = [0.0] * 6

    c = _FakeC(BOWTIP_TCP)
    out = ensure_bowtip_tcp(c, None, label="t1-correct")
    assert out["was_correct"] is True and not c.set_calls, out

    c = _FakeC(flange)
    out = ensure_bowtip_tcp(c, None, label="t2-flange-getter")
    assert out["was_correct"] is False and c.set_calls, out

    c = _FakeC(flange, has_getter=False)
    r = _FakeR(c, flange)
    out = ensure_bowtip_tcp(c, r, label="t3-flange-jump")
    assert out["was_correct"] is False and out["jump_mm"] > 100, out

    c = _FakeC(BOWTIP_TCP, has_getter=False)
    r = _FakeR(c, BOWTIP_TCP)
    out = ensure_bowtip_tcp(c, r, label="t4-correct-jump")
    assert out["was_correct"] is True and out["jump_mm"] < 1, out

    c = _FakeC(flange, has_getter=False)
    out = ensure_bowtip_tcp(c, None, label="t5-blind")
    assert out["was_correct"] is None and c.set_calls, out

    print("\ntcp_guard selftest: 5/5 PASS")


if __name__ == "__main__":
    import sys as _sys
    if "--selftest" in _sys.argv:
        _selftest()
    else:
        print(__doc__)
        print("(desk use:  python tcp_guard.py --selftest ;  lab use: imported by "
              "baseline_controller / reteach_waypoints, or run check_tcp.py manually)")
