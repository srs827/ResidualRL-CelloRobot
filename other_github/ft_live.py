"""
ft_live.py — live RAW wrist F/T stream for a hand-press diagnosis.

READ-ONLY: connects with RTDEReceiveInterface ONLY (no control interface), so this
script CANNOT command the robot to move. YOU move/press the bow BY HAND and watch
which axis responds.

PURPOSE — separate two failure modes we found today:
  - "built-in F/T sensor is too coarse to feel ~1-3N bow force"  (=> need external sensor)
  - "robot is just pressing the wrong direction / payload off"   (=> fixable in software)

PROCEDURE:
  1. Keep hands OFF -> it captures a 2s free-air baseline (the noise floor).
  2. Press the bow into the string / push the bow mount BY HAND, a few N, various dirs.
  3. Watch per-axis delta vs the noise floor. Ctrl+C for a summary verdict.

READING IT:
  - A firm hand-press (a few N) shows a clear delta >> noise on some axis
        -> sensor IS sensitive enough; problem is press direction / payload (fixable).
  - Even a firm press barely exceeds the noise
        -> built-in sensor too coarse -> we likely need an external force sensor.

Run:  .venv-rtde/bin/python ft_live.py            (default ip 192.168.1.100)
Author: Claude (for Zixian, 2026-06-04).
"""

import argparse
import time

import numpy as np
from rtde_receive import RTDEReceiveInterface

AXES = ["fx", "fy", "fz", "tx", "ty", "tz"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", default="192.168.1.100")
    ap.add_argument("--hz", type=float, default=50.0, help="sample rate")
    args = ap.parse_args()

    print(f"connecting (READ-ONLY) to {args.ip} ...")
    r = RTDEReceiveInterface(args.ip)
    print("connected. READ-ONLY — this script will NOT move the robot.\n")

    # --- free-air baseline / noise floor ---
    print("Capturing 2s free-air baseline — KEEP HANDS OFF THE BOW ...")
    base = []
    t0 = time.time()
    while time.time() - t0 < 2.0:
        base.append(r.getActualTCPForce())
        time.sleep(1.0 / args.hz)
    base = np.array(base)
    ref = base.mean(0)
    noise = base.std(0)
    print("baseline (mean  |  noise std), Newtons / Nm:")
    for i, a in enumerate(AXES):
        print(f"  {a}: {ref[i]:+.3f}   ± {noise[i]:.3f}")
    print("\nNow PRESS the bow into the string BY HAND (try several directions).")
    print("Watch the deltas. Ctrl+C to stop and see the verdict.\n")

    maxabs = np.zeros(6)
    period = 1.0 / args.hz
    disp_every = max(1, int(args.hz / 5))   # ~5 Hz on-screen
    k = 0
    try:
        while True:
            f = np.array(r.getActualTCPForce())
            d = f - ref
            maxabs = np.maximum(maxabs, np.abs(d))
            k += 1
            if k % disp_every == 0:
                j = int(np.argmax(np.abs(d[:3])))           # biggest force-axis delta
                fxyz = float(np.linalg.norm(f[:3] - ref[:3]))
                cells = []
                for i in range(3):
                    star = "*" if i == j else " "
                    cells.append(f"{AXES[i]} {d[i]:+5.2f}{star}")
                pk = int(np.argmax(maxabs[:3]))
                print("  ".join(cells) + f"  |dF|={fxyz:4.2f}N   "
                      f"(peak so far: {AXES[pk]} {maxabs[pk]:.2f}N)")
            time.sleep(period)
    except KeyboardInterrupt:
        print("\n\n=== SUMMARY ===")
        print("axis :  noise_std   max_press_delta   ratio")
        for i, a in enumerate(AXES):
            ratio = maxabs[i] / (noise[i] + 1e-9)
            print(f"  {a}  :  {noise[i]:.3f}       {maxabs[i]:.3f}          {ratio:.1f}x")
        bf = int(np.argmax(maxabs[:3]))
        print(f"\nBiggest FORCE response: {AXES[bf]} = {maxabs[bf]:.2f} N "
              f"({maxabs[bf] / (noise[bf] + 1e-9):.1f}x noise)")
        print("Rule of thumb:  >~5x noise on some axis = sensor feels it "
              "(fix press direction / payload).")
        print("                <~2x = too coarse for bow force -> need an external sensor.")


if __name__ == "__main__":
    main()
