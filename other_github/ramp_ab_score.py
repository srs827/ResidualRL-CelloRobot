"""
ramp_ab_score.py — attack-ramp v1' acceptance scoring (desk tool, no robot).

Compares two groups of strokes (ramp OFF vs ramp ON, recorded via
  trainer --mode zero --nominal-mm <D>            # 8x, ramp off
  trainer --mode zero --nominal-mm <D> --attack-ramp   # 8x, ramp on
whose wavs land at repo root as oneshot_<ts>_d..mm_s...wav / ..._ramp.wav) on the
three attack instruments (six-head diagnosis 2026-08-02 + lit check §三):
  - attack_quality head (stroke mean)     — her CNN's attack judge
  - FIRST-500ms-window tone_quality score — where the attack lives
  - full-stroke tone_quality score        — did the ramp cost sustain quality?

PASS hint: ramp-on first-window & attack head >= ramp-off, full-stroke tq not lower.

Usage (args are wav PATHS — let the SHELL expand the globs, do not quote them):
  ./.venv-rtde/bin/python ramp_ab_score.py oneshot_*.wav
  (auto-split: files with '_ramp' in the name = ON group, the rest = OFF group)
  ./.venv-rtde/bin/python ramp_ab_score.py off_dir/*.wav --vs-- on_dir/*.wav

Added 2026-08-02 (RL stack side).
"""
import os
import re
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "classifier_pilot"))
from feature_tone_reward import _SamanthaCNN     # noqa: E402  (lazy torch import inside)

FN_RE = re.compile(r"[dD](-?\d+(?:\.\d+)?)mm_[sS](\d+(?:\.\d+)?)")


def score_group(name, wavs, cnn_tq, cnn_at):
    rows = []
    for p in sorted(wavs):
        a, sr = sf.read(p)
        m = FN_RE.search(os.path.basename(p))
        dmm, spd = (float(m.group(1)), float(m.group(2))) if m else (None, None)
        tq_full = cnn_tq.score_stroke(a, sr, depth_mm=dmm, speed_ms=spd)
        tq_first = cnn_tq.last_window_scores[0] if cnn_tq.last_window_scores else float("nan")
        at_full = cnn_at.score_stroke(a, sr, depth_mm=dmm, speed_ms=spd)
        rows.append((os.path.basename(p), at_full, tq_first, tq_full))
    print(f"\n--- {name} (n={len(rows)}) ---")
    print(f"{'file':52s} {'attack':>7s} {'tq_win1':>8s} {'tq_full':>8s}")
    for fn, at, t1, tf in rows:
        print(f"{fn:52s} {at:7.3f} {t1:8.3f} {tf:8.3f}")
    arr = np.array([(r[1], r[2], r[3]) for r in rows], dtype=float)
    if len(arr):
        print(f"{'MEAN':52s} {arr[:,0].mean():7.3f} {arr[:,1].mean():8.3f} {arr[:,2].mean():8.3f}")
    return arr


def main(argv):
    if "--vs--" in argv:
        i = argv.index("--vs--")
        off, on = argv[:i], argv[i + 1:]
    else:
        # heuristic: files containing '_ramp' are the ON group
        off = [p for p in argv if "_ramp" not in os.path.basename(p)]
        on = [p for p in argv if "_ramp" in os.path.basename(p)]
    off = [p for p in off if p.endswith(".wav")]
    on = [p for p in on if p.endswith(".wav")]
    if not off or not on:
        raise SystemExit("need both groups: <off wavs...> [--vs--] <on wavs...> "
                         f"(got off={len(off)}, on={len(on)})")
    # disjointness guard (review 2026-08-02): a sloppy first glob that also matches the _ramp
    # files would put them in BOTH groups and bias every delta toward 0
    inter = {os.path.basename(p) for p in off} & {os.path.basename(p) for p in on}
    if inter:
        raise SystemExit(f"OFF and ON groups overlap ({len(inter)} files, e.g. "
                         f"{sorted(inter)[0]}) — fix the globs")
    stray = [os.path.basename(p) for p in off if "_ramp" in os.path.basename(p)]
    if stray:
        raise SystemExit(f"OFF group contains _ramp files ({stray[0]}...) — fix the globs")

    # judges pinned to the DEPLOYED payload's sha + depth offset (review 2026-08-02: the
    # verdict must come from the same-configured judge the trainer will actually use);
    # the two HEADS stay fixed by design — they are the acceptance instruments.
    import joblib
    from feature_tone_reward import DEFAULT_PATH
    _dep = joblib.load(DEFAULT_PATH)
    _off_mm = float(_dep.get("depth_offset_mm", 0.0)) if isinstance(_dep, dict) else 0.0
    _sha = _dep.get("ckpt_sha256") if isinstance(_dep, dict) else None
    cnn_tq = _SamanthaCNN(expected_sha256=_sha, head="tone_quality", depth_offset_mm=_off_mm)
    cnn_at = _SamanthaCNN(expected_sha256=_sha, head="attack_quality", depth_offset_mm=_off_mm)

    a_off = score_group("RAMP OFF", off, cnn_tq, cnn_at)
    a_on = score_group("RAMP ON", on, cnn_tq, cnn_at)

    d_at = a_on[:, 0].mean() - a_off[:, 0].mean()
    d_t1 = a_on[:, 1].mean() - a_off[:, 1].mean()
    d_tf = a_on[:, 2].mean() - a_off[:, 2].mean()
    print(f"\nDELTA (on - off):  attack {d_at:+.3f}   tq_win1 {d_t1:+.3f}   tq_full {d_tf:+.3f}")
    # V0 finding 2026-08-02: window-0 tq is position-inflated/saturated (~1.0 on scratchy AND
    # clean attacks) -> tq_win1 is INFORMATIONAL ONLY; the verdict leans on the attack head.
    verdict = "PASS" if (d_at >= 0 and d_tf > -0.02) else "NOT PASSED"
    print(f"VERDICT hint: {verdict}  (PASS = attack head up & full-stroke tq not down; tq_win1 "
          f"is informational — window 0 saturates; the EAR has the final vote)")


if __name__ == "__main__":
    main(sys.argv[1:])
