"""
make_rl_figure.py — depth-RL result figure (Paper A core): learning curve + open-loop vs closed-loop.

Left:  per-episode tone reward + per-episode mean depth (twin axis) over the run = the learning curve.
Right: open-loop baseline (the controller at the nominal depth, no RL = tone_smooth d=2 strokes) vs the
       RL-converged strokes (last N episodes) tone reward = the "closed-loop > open-loop" comparison.

Run (after a longer RL run):
  classifier_pilot/.venv/bin/python make_rl_figure.py --run rl_runs/<dir> [--baseline-mm 2] [--conv 8]
Saves <run>/rl_result.png. Needs matplotlib (pip install matplotlib into the venv first).

Author: Claude (for Zixian, 2026-06-29).
"""
import glob
import json
import os
import re
import sys

import numpy as np
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "classifier_pilot"))
from feature_tone_reward import FeatureToneReward  # noqa: E402

import matplotlib                                   # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                     # noqa: E402


def arg(flag, default):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


def main():
    run = arg("--run", sorted(glob.glob(os.path.join(HERE, "rl_runs", "2026*")))[-1])
    baseline_mm = arg("--baseline-mm", "2")
    baseline_dir = arg("--baseline-dir", "tone_smooth")   # MUST be a sweep from the SAME chain/domain
                                                          # as the run (old tone_smooth scores 0.000
                                                          # under new-era rewards -> absurd figure)
    n_conv = int(arg("--conv", "8"))
    allow_silent_baseline = "--allow-silent-baseline" in sys.argv   # an honest nominal-0 baseline
                                                                    # legitimately scores ~0

    # reward-version cross-check (review reward-tooling-2): the run's strokes were scored by the
    # reward in ITS manifest; this figure re-scores the baseline with the CURRENT joblib. If they
    # differ, the two bars use two different judges -- warn loudly.
    mani_p = os.path.join(run, "manifest.json")
    nominal_mm = 2.0            # pre-manifest-era runs were all trained at the old nominal 2mm
    if os.path.exists(mani_p):
        import hashlib
        _mani = json.load(open(mani_p)) or {}
        if _mani.get("nominal_mm") is not None:
            nominal_mm = float(_mani["nominal_mm"])
        run_sha = _mani.get("reward_sha256_16")
        from feature_tone_reward import DEFAULT_PATH as _RP
        cur_sha = hashlib.sha256(open(_RP, "rb").read()).hexdigest()[:16]
        if run_sha and run_sha != cur_sha:
            print(f"  !! REWARD VERSION MISMATCH: run scored under {run_sha}, current deployed "
                  f"{cur_sha} -- the run's curve and this figure's baseline use DIFFERENT judges. "
                  f"Redeploy the run's reward version before making paper figures.")
    else:
        try:                        # pre-manifest run: the (backfilled) ckpt meta still knows the nominal
            import torch
            _cm = torch.load(os.path.join(run, "ppo_latest.pt"), map_location="cpu").get("meta", {})
            nominal_mm = float(_cm["nominal_mm"])
            print(f"  (no manifest.json: nominal {nominal_mm:g}mm read from ppo_latest.pt meta)")
        except Exception:
            print("  (no manifest.json / no ckpt meta: assuming legacy nominal 2mm for the "
                  "depth reconstruction/annotation)")

    # learning curve — follow the WARM-START chain (manifest.warm_start_from) so a policy
    # trained across several sessions shows its FULL curve, not just the last run's slice
    def load_run_curve(rd):
        eps, rew, dep_logged, sp, lh = [], [], [], set(), []
        with open(os.path.join(rd, "episodes.jsonl")) as fh:
            for line in fh:
                d = json.loads(line)
                eps.append(d["ep"]); rew.append(d.get("stroke_score", d.get("mean_reward")))
                dep_logged.append(d.get("depth_mm"))
                if d.get("bow_speed") is not None:
                    sp.add(round(d["bow_speed"], 4))
        # logged truth where present; legacy per-EPISODE-NUMBER reconstruction (positional zip
        # breaks when an aborted episode leaves a gap), clipped like the robot did
        recon = {}
        for f in glob.glob(os.path.join(rd, "ep*_transitions.npy")):
            m = re.search(r"ep(\d+)_transitions", f)
            t = np.load(f, allow_pickle=True)
            a = np.clip(np.array([r[1][0] for r in t], dtype=float), -1.0, 1.0)
            recon[int(m.group(1))] = float(np.mean(np.clip(nominal_mm + a * 5.0, -6.0, 8.0)))
        dep = [dl if dl is not None else recon.get(e) for e, dl in zip(eps, dep_logged)]
        assert all(x is not None for x in dep), f"{rd}: episode depth missing from BOTH episodes.jsonl and transitions"
        lh_p = os.path.join(rd, "learned_depth_history.json")
        if os.path.exists(lh_p):
            lh = json.load(open(lh_p))
        return eps, rew, dep, sp, lh

    chain, _seen, _cur = [run], {os.path.realpath(run)}, run
    while True:                       # walk lineage backwards: child -> warm-start parent
        _mp = os.path.join(_cur, "manifest.json")
        _ws = (json.load(open(_mp)) or {}).get("warm_start_from") if os.path.exists(_mp) else None
        if not _ws:
            break
        _parent = os.path.dirname(os.path.abspath(_ws))
        if (os.path.realpath(_parent) in _seen
                or not os.path.exists(os.path.join(_parent, "episodes.jsonl"))):
            break
        chain.insert(0, _parent); _seen.add(os.path.realpath(_parent)); _cur = _parent
    eps, rew, dep, sp_train, lh, boundaries = [], [], [], set(), [], []
    for rd in chain:
        _e, _r, _d, _s, _l = load_run_curve(rd)
        off = eps[-1] if eps else 0
        eps += [e + off for e in _e]; rew += _r; dep += _d; sp_train |= _s
        lh += [{**r, "ep": r["ep"] + off} for r in _l]
        if rd is not chain[-1]:
            boundaries.append(eps[-1])
    if len(chain) > 1:
        print(f"  stitched {len(chain)} warm-start-linked runs ({len(eps)} episodes): "
              + " -> ".join(os.path.basename(r) for r in chain))

    # open-loop baseline = tone_smooth strokes at the nominal depth, scored by the SAME reward
    r = FeatureToneReward()
    base = []
    pat = f"*_D{float(baseline_mm):.1f}mm_*"          # accepts --baseline-mm 2 / 2.0 / 2.5
    base_dir = os.path.join(HERE, baseline_dir, "audio")
    for f in glob.glob(os.path.join(base_dir, pat)):
        a, sr = sf.read(f); base.append(r.score(a, sr))
    assert base, f"no baseline strokes matched {base_dir}/{pat} — check --baseline-mm/--baseline-dir"
    if float(np.mean(base)) < 0.05 and not allow_silent_baseline:
        raise SystemExit(f"baseline strokes in {base_dir} all score ~0 under the CURRENT reward — "
                         f"stale-domain baseline (different chain/gate era). Pass --baseline-dir "
                         f"with a sweep collected in THIS run's domain.")
    eval_path = os.path.join(run, "eval_last.json")   # PER-RUN (a global file could pair run A's
    eval_dep = None                                   # curve with run B's closed-loop bar)
    if os.path.exists(eval_path):
        _ej = json.load(open(eval_path))
        conv = _ej["scores"]; eval_dep = _ej.get("depths")
        conv_label = f"closed-loop RL\ndeterministic eval\n(n={len(conv)})"
        _es = _ej.get("bow_speed")
        _srange = None
        if os.path.exists(mani_p):
            _srange = (json.load(open(mani_p)) or {}).get("speed_range")
        if _srange:                                     # conditioned run: any eval speed inside the
            _evs = _ej.get("eval_speeds") or ([_es] if _es is not None else [])   # trained range is valid
            _out = [s for s in _evs if not (_srange[0] - 1e-6 <= s <= _srange[1] + 1e-6)]
            if _out:
                print(f"  !! eval speed(s) {_out} OUTSIDE the trained speed_range {_srange} — "
                      f"extrapolation, not the learned curve")
                conv_label += f"\n(!! speeds {_out})"
        elif _es is not None and sp_train and round(_es, 4) not in sp_train:
            print(f"  !! eval bow_speed {_es} != training bow_speed(s) {sorted(sp_train)} — "
                  f"the closed-loop bar was evaluated at a DIFFERENT speed; bars not comparable")
            conv_label += f"\n(!! speed {_es})"
    else:
        conv = rew[-n_conv:]
        conv_label = f"closed-loop RL\nlast {n_conv} train ep"

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(eps, rew, "o-", color="C0", label="tone reward")
    ax[0].set_xlabel("episode"); ax[0].set_ylabel("tone reward  P(good)", color="C0")
    ax[0].set_ylim(-0.05, 1.05)
    ax2 = ax[0].twinx()
    ax2.plot(eps, dep, "s--", color="gray", alpha=0.6, label="mean depth")
    ax2.set_ylabel("mean depth (mm)", color="gray")
    for b in boundaries:
        ax[0].axvline(b + 0.5, color="k", ls="--", lw=1, alpha=0.4)
        ax[0].text(b + 0.5, 1.03, " warm-start", fontsize=7, alpha=0.6, rotation=0)
    # learned-policy reference lines (the deterministic eval the training converges toward)
    if eval_dep is not None:
        lr, ld = float(np.mean(conv)), float(np.mean(eval_dep))
        ax[0].axhline(lr, color="C0", ls=":", lw=2)
        ax[0].text(eps[0], lr + 0.015, f"learned policy (eval) = {lr:.2f}",
                   color="C0", fontsize=9, va="bottom")
        ax2.axhline(ld, color="gray", ls=":", lw=1.5)
        ax2.text(eps[-1], ld + 0.05, f"{ld:.1f} mm", color="gray", fontsize=9, va="bottom", ha="right")
    ax[0].set_title("depth-RL learning curve")

    ax[1].boxplot([base, conv], tick_labels=[f"open-loop\nbaseline d={baseline_mm}mm\n(n={len(base)})",
                                             conv_label])
    for i, vals in enumerate([base, conv], 1):
        ax[1].scatter(np.full(len(vals), i), vals, color="C1", alpha=0.6, zorder=3)
    ax[1].set_ylabel("tone reward  P(good)"); ax[1].set_ylim(-0.05, 1.05)
    ax[1].set_title(f"open-loop {np.mean(base):.2f}  ->  closed-loop {np.mean(conv):.2f}")

    plt.tight_layout()
    out = os.path.join(run, "rl_result.png")
    plt.savefig(out, dpi=130)

    # second figure: the learned depth per stroke over training (colored by that stroke's reward)
    fig2, axd = plt.subplots(figsize=(7.5, 4))
    axd.plot(eps, dep, "-", color="gray", alpha=0.35, zorder=2)
    sc = axd.scatter(eps, dep, c=rew, cmap="viridis", vmin=0.0, vmax=1.0, s=45, zorder=3)
    # learned best-position history: TRUE (logged per update) if present, else rolling-mean approximation
    if lh:                            # stitched across the warm-start chain above
        axd.plot([r["ep"] for r in lh], [r["depth_mm"] for r in lh], "-", color="k", lw=2.5, zorder=4,
                 label="learned best depth (deterministic, per update)")
    elif len(dep) >= 5:
        w = 5
        roll = np.convolve(dep, np.ones(w) / w, mode="valid")
        axd.plot(eps[w - 1:], roll, "-", color="k", lw=2.5, zorder=4,
                 label=f"learned depth (~{w}-stroke rolling mean; true not logged this run)")
    for b in boundaries:
        axd.axvline(b + 0.5, color="k", ls="--", lw=1, alpha=0.4)
    axd.axhline(nominal_mm, color="k", ls="--", lw=1, alpha=0.6)
    axd.text(eps[-1], nominal_mm, f" nominal {nominal_mm:g}mm (start)",
             va="center", ha="right", fontsize=9, alpha=0.7)
    if eval_dep is not None:
        ld = float(np.mean(eval_dep))
        axd.axhline(ld, color="C3", ls=":", lw=2)
        axd.text(eps[0], ld + 0.06, f"learned depth = {ld:.1f}mm", color="C3", fontsize=9, va="bottom")
    axd.set_xlabel("episode (stroke)"); axd.set_ylabel("commanded depth (mm)")
    axd.set_title("learned depth over strokes")
    axd.legend(loc="upper left", fontsize=8)
    cb = fig2.colorbar(sc, ax=axd); cb.set_label("tone reward  P(good)")
    fig2.tight_layout()
    out_d = os.path.join(run, "rl_depth.png")
    fig2.savefig(out_d, dpi=130)

    print(f"  baseline (open-loop d={baseline_mm}mm): mean reward {np.mean(base):.3f}  (n={len(base)})")
    print(f"  RL converged (last {n_conv} ep):        mean reward {np.mean(conv):.3f}")
    print(f"  saved -> {out}")
    print(f"  saved -> {out_d}")


if __name__ == "__main__":
    main()
