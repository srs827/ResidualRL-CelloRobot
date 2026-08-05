"""
g2_check.py — G2 lab check: score strokes with BOTH rewards side by side
(deployed v6 vs samantha_cnn_v1) before switching the live reward.

Usage (from the repo root, after strokes are recorded by collect_tone):
  ./.venv-rtde/bin/python g2_check.py drift_visits/20260731_g2/audio/*.wav
  ./.venv-rtde/bin/python g2_check.py tone_2d_live/audio/*.wav      # desk sanity, 42 cells

Depth/speed are parsed from the collect_tone filename convention (…_D2.0mm_S0.11_r00.wav)
and fed to samantha_cnn_v1 (its CNN takes physical inputs — G1 variant c). Files that don't
match the pattern still score, using the checkpoint's training means (variant b).
If an ear_labels.json sits next to the audio dir, ear labels are shown and Spearman printed.

The samantha scorer goes through the REAL FeatureToneReward dispatch (a temp sha-pinned
payload), so this exercises the exact load->verify->gate->CNN->ramp path the trainer will use.
Lab bar for G2: orderings agree with your ears; no stroke where the two rewards violently
disagree without an audible reason.

Added 2026-07-31 (G2 wiring, RL stack side).
"""
import hashlib
import json
import os
import re
import sys
import tempfile

import numpy as np
import joblib
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
CLF = os.path.join(os.path.dirname(HERE), "classifier_pilot")
for _p in (HERE, CLF):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from feature_tone_reward import FeatureToneReward, SAMANTHA_CKPT, DEFAULT_PATH   # noqa: E402

FN_RE = re.compile(r"D(-?\d+(?:\.\d+)?)mm_S(\d+(?:\.\d+)?)")


def main(paths):
    wavs = [p for p in paths if p.endswith(".wav")]
    if not wavs:
        raise SystemExit("no .wav files given (shell should expand the glob)")

    dep = FeatureToneReward()                     # whatever is DEPLOYED right now
    print(f"deployed reward: feature_set={dep.feature_set}  ({os.path.basename(DEFAULT_PATH)})")

    # candidate payload MIRRORS the deployed head/offset when deployed is samantha, so
    # "self-comparison" is decided on the FULL config, not just the feature_set (review
    # 2026-08-02: after the tq deploy, feature_set alone would falsely claim same-scorer)
    dep_payload = joblib.load(DEFAULT_PATH)
    dep_cfg = ((dep_payload.get("head", "overall"), dep_payload.get("depth_offset_mm", 0.0))
               if isinstance(dep_payload, dict) and
               dep_payload.get("feature_set") == "samantha_cnn_v1" else None)
    cand_head, cand_off = dep_cfg if dep_cfg else ("overall", 0.0)
    self_cmp = dep_cfg is not None
    if self_cmp:
        print(f"!! NOTE: the deployed reward IS samantha_cnn_v1 (head={cand_head}, "
              f"offset={cand_off:+.1f}mm) — candidate mirrors it, so the two columns are the "
              "same scorer. This run only shows samantha-vs-ear.")

    sha = hashlib.sha256(open(SAMANTHA_CKPT, "rb").read()).hexdigest()
    tmp = tempfile.NamedTemporaryFile(suffix=".joblib", delete=False)
    import atexit
    atexit.register(lambda p=tmp.name: os.path.exists(p) and os.unlink(p))  # no leak on crash
    joblib.dump({"feature_set": "samantha_cnn_v1", "ckpt_sha256": sha, "stamp": "g2check",
                 "head": cand_head, "depth_offset_mm": cand_off}, tmp.name)
    sam = FeatureToneReward(tmp.name)             # real dispatch: load -> sha verify -> score
    print(f"samantha reward: ckpt sha {sha[:16]}...  head={cand_head} offset={cand_off:+.1f}mm "
          f"(gate_v6 + ramp wrap the CNN)\n")

    # per-DIRECTORY ear files (fix 2026-08-02: the old code read only the FIRST wav's dir, so
    # mixed-dir invocations silently dropped every other dir's labels)
    _ear_cache = {}

    def ear_label(wav_path):
        d = os.path.dirname(os.path.dirname(os.path.abspath(wav_path)))
        if d not in _ear_cache:
            ep = os.path.join(d, "ear_labels.json")
            labs = json.load(open(ep)) if os.path.exists(ep) else {}
            # 0 = "skip" in the labeling convention (same filter as train_interim_reward)
            _ear_cache[d] = {k: v for k, v in labs.items() if v not in (None, 0)}
        return _ear_cache[d].get(os.path.basename(wav_path))

    rows = []
    print(f"{'file':44s} {'d_mm':>5s} {'spd':>5s} {'deployed':>8s} {'sam':>6s} {'ear':>4s}")
    for p in sorted(wavs):
        a, sr = sf.read(p)
        m = FN_RE.search(os.path.basename(p))
        dmm, spd = (float(m.group(1)), float(m.group(2))) if m else (None, None)
        s_dep = float(dep.score(a, sr, depth_mm=dmm, speed_ms=spd))
        s_sam = float(sam.score(a, sr, depth_mm=dmm, speed_ms=spd))
        e = ear_label(p)
        rows.append((s_dep, s_sam, e))
        print(f"{os.path.basename(p):44s} {dmm if dmm is not None else '?':>5} "
              f"{spd if spd is not None else '?':>5} {s_dep:8.3f} {s_sam:6.3f} "
              f"{e if e is not None else '-':>4}")

    if len(rows) >= 5:
        from scipy.stats import spearmanr
        a6 = np.array([r[0] for r in rows]); asam = np.array([r[1] for r in rows])
        if not self_cmp:
            print(f"\nSpearman(deployed, samantha) = {spearmanr(a6, asam)[0]:+.3f}   (n={len(rows)})")
        ears = np.array([r[2] if r[2] is not None else np.nan for r in rows], dtype=float)
        ok = ~np.isnan(ears)
        if ok.sum() >= 5:
            if not self_cmp:
                print(f"Spearman(deployed, ear)      = {spearmanr(a6[ok], ears[ok])[0]:+.3f}")
            print(f"Spearman(samantha, ear)= {spearmanr(asam[ok], ears[ok])[0]:+.3f}  (n={int(ok.sum())})")
    if os.path.exists(tmp.name):
        os.unlink(tmp.name)


if __name__ == "__main__":
    main(sys.argv[1:])
