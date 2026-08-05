"""
tilt_probe.py — measure the TILT of the taught contact line vs the real string (2026-07-20).

Symptom (Zixian, live): at low depths the stroke sounds in the first half and dies in the
second — the tip end sits shallower than the frog end. This probes the contact depth at
three bow fractions: force-guarded descent at each until |F| >= CONTACT_N, report the depth
where contact was reached. The SPREAD across fractions = the line tilt in mm.

No bowing — descend/lift only, same safety envelope as collect_tone.reset_to.
Run:  .venv-rtde/bin/python tilt_probe.py            # fracs 0.25 / 0.50 / 0.75
"""
import sys
import time

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "Baseline-Runners")
import recording_grid as R                                       # noqa: E402
from baseline_controller import TargetStringBaselineController   # noqa: E402
from collect_tone import (pose, fmag, safe, retract, CLEAR,      # noqa: E402
                          FORCE_ABORT, TARE_MAX_N, STEP, SLOW_BAND)

CONTACT_N = 0.8         # "contact reached" threshold (above tare noise, below playing force)
D_START = -0.004        # begin the slow search 4mm ABOVE the taught line (looser bow = shallower)
D_MAX = 0.008           # give up below this (something is very wrong — call it out)
FRACS = (0.25, 0.50, 0.75)


def probe_frac(ctrl, frac):
    """Force-guarded descent at `frac` until CONTACT_N; returns contact depth in mm."""
    try:
        ctrl.rtde_c.servoStop(); time.sleep(0.1)
    except Exception:
        pass
    ctrl.rtde_c.moveL(pose(frac, -CLEAR).tolist(), 0.10, 0.4)
    if not safe(ctrl):
        raise RuntimeError("not safe at hover")
    ctrl.rtde_c.zeroFtSensor(); time.sleep(0.3)
    f0 = fmag(ctrl)
    if f0 > TARE_MAX_N:
        raise RuntimeError(f"free-air |F|={f0:.2f}N > {TARE_MAX_N} after tare")
    # fast through air to just above the search band, then slow force-guarded steps
    fast_to = D_START - SLOW_BAND
    if not ctrl.rtde_c.moveL(pose(frac, fast_to).tolist(), 0.10, 0.4) or not safe(ctrl):
        raise RuntimeError("fast approach failed / not safe")
    d = fast_to
    while d <= D_MAX + 1e-9:
        ctrl.execute_timestep(pose(frac, d), ctrl.BASELINE_FORCE_A); time.sleep(0.12)
        f = fmag(ctrl)
        if f > FORCE_ABORT or not safe(ctrl):
            retract(ctrl)
            raise RuntimeError(f"FORCE/safety abort at frac {frac} d={d*1000:+.1f}mm |F|={f:.1f}N")
        if f >= CONTACT_N:
            print(f"  frac {frac:.2f}: contact ({CONTACT_N}N) at d = {d*1000:+.2f} mm   (|F|={f:.2f}N)")
            retract(ctrl)
            return d * 1000.0
        d += STEP
    retract(ctrl)
    print(f"  frac {frac:.2f}: NO contact down to {D_MAX*1000:+.1f} mm — line is deeply stale here")
    return None


def main():
    ctrl = TargetStringBaselineController([])
    try:
        try:
            ctrl.rtde_c.setPayload(float(R.FT_PAYLOAD_KG), list(R.FT_PAYLOAD_COG))
        except Exception:
            pass
        input(f">>> TILT PROBE: descend-only at fracs {FRACS} (no bowing). HAND ON THE E-STOP. ENTER...")
        depths = {}
        for fr in FRACS:
            depths[fr] = probe_frac(ctrl, fr)
        got = {f: d for f, d in depths.items() if d is not None}
        if len(got) >= 2:
            fs, ds = list(got.keys()), list(got.values())
            tilt = max(ds) - min(ds)
            slope = np.polyfit(fs, ds, 1)[0] if len(got) >= 2 else 0.0
            print(f"\n  TILT = {tilt:.2f} mm across the probed span "
                  f"(slope {slope:+.2f} mm per unit frac; positive = tip end SHALLOWER)")
            print("  <1 mm: proceed, log it.  1-2 mm: proceed but flag for anchor re-teach.  "
                  ">2 mm: re-teach anchors before science runs.")
    finally:
        retract(ctrl)
        try:
            ctrl.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
