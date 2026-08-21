import json, sys
J = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "rl/gate.json"))
print(f"piece {J['piece']}")
terms = ["total", "quality", "r_dynamic", "r_defect", "r_onset", "r_envelope"]
print(f"{'stroke':>8}" + "".join(f"{t:>12}" for t in terms))
for sid, S in J["strokes"].items():
    row = f"{sid:>8}"
    for t in terms:
        rsd = S["repeat"].get(t, [0, 0])[1]; asd = S["probe"].get(t, [0, 0])[1]
        row += f"{(asd / rsd if rsd > 1e-9 else float('inf')):>12.2f}" if asd or rsd else f"{'-':>12}"
    print(row)
print("\nSNR = action sd / repeat sd.  <1 invisible, <2 weak, >=2 usable.")
print("The one that matters for TONE is 'quality'. If r_dynamic carries the")
print("mean and quality is under 1, the run optimises loudness, not tone.")

# Why this exists: reward_noise.py prints "mean SNR on total reward" and that
# headline can pass while the TONE term is invisible. Measured on
# twinkle-short 2026-08-20: total 1.75/3.36/2.97 (mean 2.7, "usable") with
# quality 0.91/0.65/0.89 -- under 1 on every probe stroke. The total was
# carried by r_dynamic (3.74) and r_defect (5.08), so the policy could only
# see loudness and defects. That run finished BELOW its baseline. Check the
# column you actually care about before spending 1.8 h of robot time.
