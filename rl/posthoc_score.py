"""
rl/posthoc_score.py

Score a recorded performance (a rl/checkpoints_piece/perform_* dir) with the
SAME judge inputs the training env uses, so the number lands on the same
scale as the stroke log's "tone" and select_best's ranking:

    audio     sliced per the EXECUTED timeline (actual onset/duration)
    physical  rebuilt from the 100 Hz state log: measured |bow_speed| mean,
              measured torque magnitude mean/max over the stroke window
              (piece_hardware.py builds the env's vector the same way)
    tone      the length-aware head mix piece_env._reward computes
              (TONE_HEAD_WEIGHTS, fill-shifted for short notes)

Why measured phys is non-negotiable: the judge is a joint (audio, physical)
model. Scoring the 2026-08-20 A/B takes with COMMANDED speed and constant
torque said "no difference" (baseline 0.558 vs policy 0.545); the same takes
with measured phys said policy +0.062, p=4e-6 — and only the measured-phys
version reproduces the env's own logged scores (per-stroke corr 0.89-0.97 on
run_20260820_221310 episodes 1 and 100; the commanded-phys version missed
ep-1 by 0.11 and failed validation). A ~-0.06 level offset vs in-env scoring
remains (audio-chain details), so compare offline numbers only with other
offline numbers.

Usage:
    .venv/bin/python rl/posthoc_score.py PERFORM_DIR
    .venv/bin/python rl/posthoc_score.py DIR_A DIR_B      # paired A/B verdict

Writes posthoc.json into each dir. ab_compare imports score_dir() to fill
its summary column after the passes.

Zixian Liu, 2026-08-21.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_JUDGE = None


def _judge():
    global _JUDGE
    if _JUDGE is None:
        from rl.piece_env import RealScorer
        _JUDGE = RealScorer()
    return _JUDGE


def _mix_tone(detail: dict, fill: float) -> float:
    """The head mix piece_env._reward uses (keep the two in step)."""
    from rl.piece_env import TONE_HEAD_WEIGHTS
    w = dict(TONE_HEAD_WEIGHTS)
    if fill < 1.0:
        shift = 1.0 - fill
        w["attack_quality"] += 0.30 * shift
        w["release_quality"] += 0.10 * shift
        w["tone_quality"] = max(0.0, w["tone_quality"] - 0.30 * shift)
        w["overall"] = max(0.0, w["overall"] - 0.10 * shift)
    usable = {h: v for h, v in w.items() if h in detail and v > 0}
    return float(sum(detail[h] * v for h, v in usable.items())
                 / sum(usable.values()))


def score_stem(stem: Path) -> dict | None:
    """Score one take: STEM.wav + STEM_performance.json + STEM_state.npy."""
    import soundfile as sf
    from rl.piece_env import CLASSIFIER_WINDOW_SEC

    perf_p = Path(str(stem) + "_performance.json")
    wav_p = Path(str(stem) + ".wav")
    state_p = Path(str(stem) + "_state.npy")
    if not (perf_p.exists() and wav_p.exists()):
        return None
    if not state_p.exists():
        # Refusing beats silently falling back to commanded phys — that
        # fallback is exactly the retracted-verdict mistake this tool exists
        # to prevent.
        print(f"  !! {state_p.name} missing — cannot rebuild measured phys, "
              f"not scoring {stem.name}")
        return None

    perf = json.loads(perf_p.read_text())
    lead = float(perf.get("audio_lead_in_s", 0.0))
    audio, sr = sf.read(wav_p, dtype="float32")
    if audio.ndim > 1:
        audio = audio[:, 0]
    S = np.load(state_p, allow_pickle=True)
    T = np.array([row["t"] for row in S])
    # Validated on the 08-20 takes: the state log spans exactly the
    # performance (len == wall_s), so its first sample is performance t=0.
    wall0 = T[0]

    judge = _judge()
    per_note, skipped = [], 0
    for t in perf["timeline"]:
        on, dur = float(t["actual_onset"]), float(t["actual_duration"])
        lo = int((lead + on) * sr)
        seg = audio[lo:lo + int(dur * sr)]
        m = (T >= wall0 + on) & (T <= wall0 + on + dur)
        if len(seg) < sr // 20 or not m.any():
            skipped += 1
            continue
        spd = float(np.mean(np.abs([row["bow_speed"] for row in S[m]])))
        tq = np.sqrt([row["ft_tx"] ** 2 + row["ft_ty"] ** 2
                      + row["ft_tz"] ** 2 for row in S[m]])
        phys = np.array([t["depth_m"], 0.0, spd,
                         (t["u_start"] + t["u_end"]) / 2.0,
                         float(tq.mean()), float(tq.max())], dtype=np.float32)
        detail = judge.score_detailed(seg, phys)
        fill = float(min(dur / CLASSIFIER_WINDOW_SEC, 1.0))
        per_note.append({"note_index": t.get("note_index"),
                         "tone": round(_mix_tone(detail, fill), 4),
                         "overall": round(float(detail["overall"]), 4),
                         "torque_mean": round(float(tq.mean()), 4),
                         "speed_meas": round(spd, 4)})
    if not per_note:
        return None
    tones = np.array([n["tone"] for n in per_note])
    overalls = np.array([n["overall"] for n in per_note])
    return {"stem": stem.name, "n": len(per_note), "skipped": skipped,
            "mean_tone": float(tones.mean()),
            "mean_overall": float(overalls.mean()),
            "per_note": per_note}


def score_dir(perform_dir) -> dict | None:
    """Score the newest take in a perform dir; write posthoc.json there."""
    d = Path(perform_dir)
    perfs = sorted(d.glob("*_performance.json"), key=lambda p: p.stat().st_mtime)
    if not perfs:
        return None
    stem = Path(str(perfs[-1])[: -len("_performance.json")])
    res = score_stem(stem)
    if res:
        (d / "posthoc.json").write_text(json.dumps(res, indent=2))
    return res


def main() -> None:
    dirs = [Path(a) for a in sys.argv[1:]]
    if not dirs:
        sys.exit("usage: posthoc_score.py PERFORM_DIR [PERFORM_DIR_B]")
    results = []
    for d in dirs:
        res = score_dir(d)
        if res is None:
            sys.exit(f"nothing scoreable in {d}")
        results.append(res)
        print(f"{d.name}/{res['stem']}: n={res['n']}  "
              f"tone {res['mean_tone']:.3f}  overall {res['mean_overall']:.3f}"
              + (f"  ({res['skipped']} skipped)" if res["skipped"] else ""))

    if len(results) == 2:
        a, b = results
        n = min(a["n"], b["n"])
        ta = np.array([x["tone"] for x in a["per_note"][:n]])
        tb = np.array([x["tone"] for x in b["per_note"][:n]])
        diff = float(tb.mean() - ta.mean())
        wins = int((tb > ta).sum())
        try:
            from scipy import stats
            p = float(stats.ttest_rel(tb, ta).pvalue)
            ptxt = f"  p={p:.2g}"
        except ImportError:
            ptxt = ""
        print(f"\npaired (B - A), n={n}: diff {diff:+.3f}{ptxt}  "
              f"B wins {wins}/{n}")
        print("One pair of takes: treat |diff| under the take-to-take sd "
              "(~0.03/note on twinkle-short) as noise.")


if __name__ == "__main__":
    main()
