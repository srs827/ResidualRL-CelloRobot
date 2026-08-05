"""
train_residual_robot.py — STAGED real-robot bring-up + residual-RL run.

REQUIRES a COMBINED env: ur_rtde (robot) + torch (PPO) + audiobox/sklearn/soundfile/
librosa (reward) + sounddevice (mic). The plain .venv-rtde has ur_rtde but NOT
torch/audiobox -> install those into it first. RUN AT THE ROBOT, HAND ON THE E-STOP.

Stages (--mode), increasing risk:
  check : connect, baseline.reset (moves to frog), print obs/depth/force. No bowing.
  zero  : ONE stroke, residual forced to 0 (should reproduce the baseline).
  fixed : ONE stroke, a small FIXED residual (--fixed-mm). Confirm robot/sound react.
  rl    : N episodes of residual PPO (policy residual + audio reward + PPO update).
  eval  : N DETERMINISTIC strokes from a checkpoint (no learning) -> <ckpt dir>/eval_last.json.
  ab    : N interleaved baseline-vs-policy pairs from a checkpoint -> ab_result.json.
  (zero/fixed also save the stroke's wav to the REPO root as oneshot_d..mm_s...wav + print its
   features/reward — the quickest "is the chain alive at depth X" listen.)

Author: Claude (for Zixian, 2026-06-04).
"""

import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
CLF = os.path.join(os.path.dirname(REPO), "classifier_pilot")
for _p in (HERE, REPO, CLF):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from residual_runner import (build_obs, apply_residual, OBS_FEATURES,    # noqa: E402
                             OBS_DIM, ACT_DIM, DEPTH_RESIDUAL_CLAMP)
import hardware_baseline as hb                                           # noqa: E402
from hardware_baseline import HardwareBaseline                           # noqa: E402
from audio_recorder import AudioRecorder                                 # noqa: E402
# torch (rl.ppo.ppo) is imported ONLY in --mode rl so check/zero/fixed run torch-free.

assert hb.DEPTH_RESIDUAL_CLAMP == DEPTH_RESIDUAL_CLAMP, \
    "residual clamp mismatch between hardware_baseline and residual_runner"

TIMESTEP_SEC = 0.05


def make_episode_notes(n_notes=1, dur=2.7, vel=80):   # dur 2.7s @0.109m/s => frac span ~0.60 = collect_tone 0.20-0.80
    notes, bow = [], "down"
    for _ in range(n_notes):
        notes.append({"duration": float(dur), "velocity": int(vel),
                      "dynamic": "f", "bow_dir": bow, "string": "A"})
        bow = "up" if bow == "down" else "down"
    return notes


def episode_ticks(notes):
    return sum(max(2, int(n["duration"] / TIMESTEP_SEC)) for n in notes)


def build_run_manifest(args, hw, rec, mode, bow_speed, dur):
    """One self-describing record per run (2026-07-10 review S2/provenance): everything a
    future reader needs to reproduce or trust this run's numbers. Written to the run dir
    (rl mode) and a compact subset is stored INSIDE the checkpoint (ppo save meta)."""
    import hashlib
    import subprocess
    from feature_tone_reward import DEFAULT_PATH, RMS_FLOOR, ONSET_WIN
    import residual_runner as rr
    try:
        reward_sha = hashlib.sha256(open(DEFAULT_PATH, "rb").read()).hexdigest()[:16]
        reward_mtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(DEFAULT_PATH)))
    except Exception:
        reward_sha, reward_mtime = None, None
    reward_meta = None
    _mp = os.path.splitext(DEFAULT_PATH)[0] + "_meta.json"
    if os.path.exists(_mp):
        try:
            import json as _json
            reward_meta = _json.load(open(_mp))
        except Exception:
            pass
    try:
        joints = [round(float(x), 4) for x in hw.ctrl.rtde_r.getActualQ()]
    except Exception:
        joints = None
    try:
        git_rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
                                 capture_output=True, text=True, timeout=5).stdout.strip() or None
    except Exception:
        git_rev = None
    return {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mode": mode, "episodes": args.episodes,
        "nominal_mm": hw.nominal_depth * 1000.0, "bow_speed": bow_speed, "dur_s": dur,
        "start_frac": hw.start_frac,
        "depth_min_mm": hb.DEPTH_MIN * 1000.0, "max_press_mm": hb.MAX_PRESS_DEPTH * 1000.0,
        "residual_clamp_mm": DEPTH_RESIDUAL_CLAMP * 1000.0,
        "rate_limit_mm_tick": hb.DEPTH_RATE_LIMIT * 1000.0, "force_abort_n": hb.FORCE_ABORT_N,
        "reward_joblib": os.path.basename(DEFAULT_PATH), "reward_sha256_16": reward_sha,
        "reward_mtime": reward_mtime, "reward_meta": reward_meta,
        "rms_floor": RMS_FLOOR, "onset_win_s": ONSET_WIN,
        "audio_obs": rr.AUDIO_OBS, "audio_obs_scales": [float(x) for x in rr.AUDIO_OBS_SCALES],
        "obs_dim": OBS_DIM,
        "mic_device": rec.device_name, "mic_channel": (rec.mapping or [1]),
        "joints_at_start": joints, "git_rev": git_rev,
        "per_stroke": PER_STROKE, "strokes_per_update": STROKES_PER_UPDATE,
        "attack_ramp": bool(hb.ATTACK_RAMP),
        "land_at_target": bool(hb.LAND_AT_TARGET),
        # under land-at-target the depth decision happens BEFORE reset -> the mechanical obs
        # slots describe the end of the PREVIOUS stroke (retrain-only semantic change)
        "obs_decision_point": ("pre_reset" if hb.LAND_AT_TARGET else "post_reset"),
    }


def set_speed_slider_full(ip="192.168.1.100"):
    """A leftover 20% slider from another session breaks servoL timing (hit 06-08)."""
    try:
        import rtde_io
        rtde_io.RTDEIOInterface(ip).setSpeedSlider(1.0)
        print("  speed slider -> 100%")
    except Exception as e:
        print(f"  (speed slider not set: {e}) — CHECK THE PENDANT SLIDER MANUALLY")


def ask(msg):
    if input(f"\n>>> {msg}  [Enter]=go [q]=quit: ").strip().lower() == "q":
        raise KeyboardInterrupt
    return True


def print_obs(state):
    print("  obs:", {k: round(float(state.get(k, 0.0)), 4) for k in OBS_FEATURES})


def run_check(hw):
    hw.reset()
    print(f">>> CONFIRM: bow at the contact line {hw.nominal_depth*1000:+.1f}mm (nominal), at bow "
          f"fraction {hw.start_frac:.2f} (about a third toward the tip — NOT the frog).")
    for _ in range(5):
        print_obs(hw.get_full_state())
        time.sleep(0.2)


def run_one_stroke(hw, residual_m, recorder=None):
    _ovr = None
    if hb.LAND_AT_TARGET:      # attack v2: land the guarded descent AT this stroke's target depth
        _r = float(np.clip(residual_m, -DEPTH_RESIDUAL_CLAMP, DEPTH_RESIDUAL_CLAMP))
        _ovr = float(np.clip(hw.nominal_depth + _r, hb.DEPTH_MIN, hb.MAX_PRESS_DEPTH))
    hw.reset(depth_override_m=_ovr)
    if recorder:
        stroke_s = (hb.STROKE_END_FRAC - hw.start_frac) * hb.SPAN_M / hw.bow_speed
        recorder.start(max_seconds=max(8.0, stroke_s + 2.0))   # slow strokes outlast 8s
    while not hw.is_complete():
        a_base = hw.get_baseline_action()
        if a_base is None:
            break
        a = dict(a_base)
        a["depth_residual"] = float(residual_m)   # HardwareBaseline re-clamps absolute depth
        hw.execute_timestep(a)
        time.sleep(TIMESTEP_SEC)
    if recorder:
        audio, sr = recorder.stop()
        print(f"  recorded {len(audio)/sr:.2f}s of audio")
        import soundfile as sf
        # residual_m is in METERS here (commanded_depth_mm expects the NORMALIZED action —
        # feeding meters printed d0.0mm on 2026-08-03); apply execution's clip directly:
        d_mm = float(np.clip(hw.nominal_depth + residual_m,
                             hb.DEPTH_MIN, hb.MAX_PRESS_DEPTH)) * 1000.0
        # unique per invocation (A/B needs 8+8 strokes — the old fixed name overwrote itself);
        # _ramp suffix makes the two A/B globs trivial
        fn = os.path.join(REPO, f"oneshot_{time.strftime('%Y%m%d_%H%M%S')}_d{d_mm:.1f}mm_"
                                f"s{hw.bow_speed:.3f}{'_ramp' if hb.ATTACK_RAMP else ''}"
                                f"{'_land' if hb.LAND_AT_TARGET else ''}.wav")
        sf.write(fn, audio, sr)
        from feature_tone_reward import FeatureToneReward, features
        ft = features(audio, sr)
        print(f"  d={d_mm:.0f}mm s={hw.bow_speed:.3f}: rms={ft[0]:.4f} flat={ft[1]:.3f} hf={ft[2]:.2f} "
              f"cen={ft[3]:.0f}  reward={FeatureToneReward().score(audio, sr, depth_mm=d_mm, speed_ms=hw.bow_speed):.3f}  -> {fn}")


STROKES_PER_UPDATE = 4   # R11: batch several strokes per PPO update so per-buffer
                         # advantage mean-centering compares strokes instead of
                         # cancelling the episode-level tone signal

# Depth-decision granularity. The tone reward is per-STROKE, so for now we sample ONE depth per
# stroke and hold it constant (clean tone, no per-tick jitter, clean credit assignment). The
# per-TICK ports below are kept fully intact: set PER_STROKE=False the day the reward becomes
# per-tick and per-tick streaming control resumes with no other change. (Zixian 2026-06-28)
PER_STROKE = True

GATE_N = 10           # soft reward gate: prompt if no score >0.1 in the first GATE_N episodes
                      # (10, not 5: silence-starts under a cliff reward legitimately take longer)
# (No ent_coef annealing: log_std is held ~0.3 by design — ent_coef=0.1 + LOG_STD floor, audit K10 — to
#  keep exploration for drift-tracking. The CONVERGED policy is measured via --mode eval, deterministic.)


def _decision_state(hw):
    """Pre-stroke obs state. The measured bow_speed is ~0 at reset (robot stationary), so the
    policy would NEVER see its conditioning variable — overwrite the slot with the COMMANDED
    speed of the UPCOMING stroke (2026-07-19, speed-conditioning integration). Retrain-only
    semantic change: old ckpts saw ~0 here."""
    st = hw.get_full_state()
    st["bow_speed"] = hw.bow_speed
    if hb.LAND_AT_TARGET:
        # Pre-reset obs can come from a PARKED/RETRACTED pose (ep1, post-abort): raw "depth"
        # there is -0.1..-0.3 m -> normalizes to -30..-100 and would dominate the first PPO
        # minibatch (review D3). Clamp the mechanical slots to the on-string envelope.
        st["depth"] = float(np.clip(st.get("depth", 0.0), hb.DEPTH_MIN, hb.MAX_PRESS_DEPTH))
        st["contact_force_magnitude"] = float(np.clip(st.get("contact_force_magnitude", 0.0),
                                                      0.0, hb.FORCE_ABORT_N))
    return st


def run_rl(hw, agent, reward_fn, recorder, episodes, max_rec_s, run_meta=None, speed_range=None,
           speed_seed=0):
    import json
    import soundfile as sf
    run_dir = os.path.join(REPO, "rl_runs", time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    run_meta = dict(run_meta or {})
    run_meta["run_dir"] = os.path.basename(run_dir)
    with open(os.path.join(run_dir, "manifest.json"), "w") as fh:   # provenance (review S2):
        json.dump(run_meta, fh, indent=1)                           # every run self-describes
    print(f"  persisting everything to {run_dir} (manifest.json written)")
    batch = []
    batch_scores = []     # stroke scores of the current batch (skip-uniform-zero guard)
    dead_mic = 0          # consecutive electronically-silent recordings (48V-off signature)
    mic_gate_overridden = False
    batch_has_abort = False   # abort batches are exempt from the uniform-zero skip
    gate_window = []      # collect the first GATE_N scored rewards; trip only if NONE fired (dead reward)
    learned_hist = []     # per-update DETERMINISTIC depth = the policy's current 'best position' estimate
    prev_audio = None     # the policy's EARS (v2 closed loop): last stroke's [rms,flat,hf,centroid,reward]
    speed_rng = np.random.default_rng(speed_seed) if speed_range else None   # seed recorded in the
    # manifest; default derives from wall-clock so a warm-start continuation does NOT replay the
    # parent run's exact speed sequence (review 2026-07-19: seed-0-always halved the coverage)
    for ep in range(1, episodes + 1):
        transitions = []          # defined BEFORE try: the abort handler references it
        try:
            if speed_range:       # CONDITIONED training: this stroke's speed, drawn fresh
                hw.set_bow_speed(float(speed_rng.uniform(*speed_range)))
            obs = out = None
            _ovr = None
            if hb.LAND_AT_TARGET and PER_STROKE:
                # attack v2: the depth decision moves BEFORE reset so the guarded descent can
                # land AT the stroke's target. The mechanical obs slots therefore describe the
                # PRE-reset state (end of the previous stroke) — a retrain-only semantic change,
                # recorded in the manifest as obs_decision_point=pre_reset. Ears/speed unchanged.
                obs = build_obs(_decision_state(hw), prev_audio)
                out = agent.sample(obs)
                _ovr = hb.commanded_depth_mm(float(out["action"][0]), hw.nominal_depth) / 1000.0
            hw.reset(depth_override_m=_ovr)
            recorder.start(max_seconds=max_rec_s)
            t_rec = time.time()
            t_m0 = time.time()
            if PER_STROKE and out is None:              # flag OFF: decide depth ONCE for the whole
                obs = build_obs(_decision_state(hw), prev_audio)   # stroke, HEARING the prev stroke,
                out = agent.sample(obs)                            # post-reset obs (historic order)
            while not hw.is_complete():
                if not PER_STROKE:                      # per-tick port (re-enabled when reward is per-tick)
                    obs = build_obs(hw.get_full_state(), prev_audio)
                    out = agent.sample(obs)
                a_base = hw.get_baseline_action()
                if a_base is None:
                    break
                hw.execute_timestep(apply_residual(a_base, out["clipped_action"]))
                transitions.append({"obs": obs, "action": out["action"], "value": out["value"],
                                    "log_prob": out["log_prob"], "reward": None, "done": False})
                time.sleep(TIMESTEP_SEC)
            t_m1 = time.time()
            audio, sr = recorder.stop()
        except RuntimeError as e:
            # abort (force/safety): the robot already retracted. TEACH the policy the abort was
            # bad — keep the partial episode with reward 0. (Review learning-sanity-1: silently
            # erasing aborted strokes gives over-pressing ZERO penalty -> exploration ratchets
            # toward the abort boundary.)
            print(f"  ep {ep}: ABORTED — {e}")
            try:
                _a_audio, _a_sr = recorder.stop()
            except Exception:
                _a_audio, _a_sr = np.zeros(1, dtype=np.float32), 44100
            if transitions:
                for tr in transitions:
                    tr["reward"] = 0.0
                transitions[-1]["done"] = True
                if PER_STROKE:                          # ONE decision -> ONE PPO transition
                    batch.append({"obs": transitions[-1]["obs"], "action": transitions[-1]["action"],
                                  "value": transitions[-1]["value"], "log_prob": transitions[-1]["log_prob"],
                                  "reward": 0.0, "done": True})
                else:
                    batch.extend(transitions)
                batch_scores.append(0.0)
                batch_has_abort = True     # exempt this batch from the uniform-zero skip:
                                           # the zero IS the abort penalty and must be learned
                print(f"    kept the aborted stroke with reward 0 (abort penalty)")
            elif out is not None:
                # LAND_AT_TARGET: a too-deep DECISION now aborts DURING reset (descent guard /
                # landing band) — no transitions exist yet, but the decision does, and it must
                # be punished or over-pressing gets zero penalty and exploration ratchets toward
                # the abort boundary (learning-sanity-1's pathology, relocated by v2; review D1).
                batch.append({"obs": obs, "action": out["action"], "value": out["value"],
                              "log_prob": out["log_prob"], "reward": 0.0, "done": True})
                batch_scores.append(0.0)
                batch_has_abort = True
                print(f"    kept the aborted DECISION with reward 0 (reset-abort penalty)")
            with open(os.path.join(run_dir, "episodes.jsonl"), "a") as fh:   # aborted strokes
                fh.write(json.dumps({                    # TRAIN the policy — they must appear in
                    "ep": ep, "aborted": True,           # the log too (review 2026-07-19)
                    "steps": len(transitions), "mean_reward": 0.0, "stroke_score": 0.0,
                    "bow_speed": hw.bow_speed,
                    "depth_mm": (float(np.mean([hb.commanded_depth_mm(float(t["action"][0]),
                                                                      hw.nominal_depth)
                                                for t in transitions])) if transitions
                                 else (hb.commanded_depth_mm(float(out["action"][0]),
                                                             hw.nominal_depth)
                                       if out is not None else None)),
                    "rms": 0.0, "scores": []}) + "\n")
            if len(_a_audio) > 4410:                     # ears: the policy HEARS the aborted stroke
                from feature_tone_reward import features as _afeat
                _ft = _afeat(_a_audio, _a_sr)
                prev_audio = [float(_ft[0]), float(_ft[1]), float(_ft[2]), float(_ft[3]), 0.0]
            agent.save(os.path.join(run_dir, "ppo_latest.pt"), meta=run_meta)
            if input(">>> recover the robot; [Enter]=next episode, q=quit: ").strip().lower() == "q":
                break
            continue
        if not transitions:
            print(f"  ep {ep}: no transitions, skipping")
            continue
        transitions[-1]["done"] = True

        # K6: score ONLY the actual stroke window, not pre/post silence
        i0 = max(0, int((t_m0 - t_rec) * sr))
        i1 = min(len(audio), int((t_m1 - t_rec) * sr))
        seg = audio[i0:i1]
        rms = float(np.sqrt(np.mean(seg ** 2))) if len(seg) else 0.0
        # commanded depth/speed of THIS stroke, passed to the reward: v5/v6 ignore them,
        # samantha_cnn_v1 needs them (its CNN takes physical inputs — G1 variant c)
        stroke_dmm = float(np.mean([hb.commanded_depth_mm(float(t["action"][0]), hw.nominal_depth)
                                    for t in transitions]))
        # stroke_r = FeatureToneReward.score: 0.4s ONSET window -> 4 features -> gate -> LR P(good).
        # One scalar per stroke; cross-stroke differences drive the 4-stroke-batch advantage (R11).
        # Scored ONCE and reused for the log array — with the CNN reward a second identical
        # call would run the whole torch pipeline again for the same number.
        stroke_r = (float(reward_fn.score(seg, sr, depth_mm=stroke_dmm, speed_ms=hw.bow_speed))
                    if len(seg) else 0.0)
        scores = np.array([stroke_r]) if len(seg) else np.array([])
        if len(seg):                                    # update the EARS for the next stroke's obs
            from feature_tone_reward import features as _afeat
            _ft = _afeat(seg, sr)
            prev_audio = [float(_ft[0]), float(_ft[1]), float(_ft[2]), float(_ft[3]), stroke_r]
        # Reward placement (2026-07-10, review rl-soundness-1 + rl-changes-1): PER_STROKE is
        # semantically ONE decision -> feed PPO exactly ONE transition per stroke
        # (obs, action, reward=stroke_r, done=True). GAE then degenerates to advantage = r - V:
        # correct sign, bounded value targets, no 54-duplicate dilution/mixed-sign pathology.
        # (Broadcasting inflated value targets 41x; final-tick-only placement still gave the
        # same action contradictory advantages across its duplicated ticks — both measured.)
        for tr in transitions:
            tr["reward"] = stroke_r                     # tick-level record for the .npy archive
        mean_r = stroke_r

        # R9: persist audio + transitions + summary + checkpoint every episode
        # (BEFORE the gate — a tripped gate must not destroy its own evidence)
        sf.write(os.path.join(run_dir, f"ep{ep:03d}.wav"), seg, sr)
        np.save(os.path.join(run_dir, f"ep{ep:03d}_transitions.npy"),
                np.array([(t["obs"], t["action"], t["reward"]) for t in transitions],
                         dtype=object), allow_pickle=True)
        with open(os.path.join(run_dir, "episodes.jsonl"), "a") as fh:
            fh.write(json.dumps({"ep": ep, "steps": len(transitions), "mean_reward": mean_r,
                                 "depth_mm": float(np.mean([hb.commanded_depth_mm(float(t["action"][0]), hw.nominal_depth)
                                                            for t in transitions])),
                                 "bow_speed": hw.bow_speed,
                                 "stroke_score": stroke_r, "rms": rms,
                                 "scores": [float(x) for x in scores],
                                 # per-window head scores (samantha reward only; None for
                                 # v5/v6/gated/EMPTY strokes — an empty segment skips score(),
                                 # so the attribute still holds the PREVIOUS stroke's list)
                                 "window_scores": (getattr(reward_fn, "last_window_scores", None)
                                                   if len(seg) else None)
                                 }) + "\n")

        # Two-tier go/no-go (redesigned 2026-07-10, review neg-depth-5/rl-soundness-5 — the old
        # 5-ep score gate false-fired on every legitimate silence-start under a cliff reward):
        # TIER 1 (DEAD MIC, hard): 3 consecutive recordings at electronics-silence level
        #        (rms < 1e-3 ~ the 48V-off signature, an order below even bow-off-string noise).
        # TIER 2 (dead reward, soft): no score > 0.1 in the first GATE_N eps -> prompt once;
        #        expected & overridable when the run INTENDS a silent start.
        if rms < 1e-3:
            dead_mic += 1
            print(f"  !! ep {ep}: near-silent recording (rms={rms:.5f})"
                  + ("" if mic_gate_overridden else f" — {dead_mic}/3 to mic abort."))
            if dead_mic >= 3 and not mic_gate_overridden:
                print(f"  !! MIC GATE: 3 consecutive electronically-silent recordings — 48V/cable/"
                      f"channel is dead (evidence in {run_dir}).")
                if input(">>> [Enter]=ABORT session, type 'override' to train anyway: "
                         ).strip().lower() != "override":
                    break
                mic_gate_overridden = True   # don't ask again (boolean survives resets)
        else:
            dead_mic = 0
        if len(gate_window) < GATE_N and len(scores):
            gate_window.append(stroke_r)
            if len(gate_window) == GATE_N and max(gate_window) < 0.1:
                print(f"  !! REWARD GATE: {GATE_N} episodes, best reward {max(gate_window):.3f} < 0.1. "
                      f"NORMAL if this run starts in silence (cliff reward) — override then. "
                      f"Suspicious if the start should be sounding.")
                if input(">>> [Enter]=ABORT session, type 'override' to train anyway: "
                         ).strip().lower() != "override":
                    break

        if PER_STROKE:                                  # ONE decision -> ONE PPO transition
            batch.append({"obs": transitions[-1]["obs"], "action": transitions[-1]["action"],
                          "value": transitions[-1]["value"], "log_prob": transitions[-1]["log_prob"],
                          "reward": stroke_r, "done": True})
        else:
            batch.extend(transitions)                   # per-tick port: full tick stream
        batch_scores.append(stroke_r)
        print(f"  ep {ep:2d}: steps={len(transitions)} mean_r={mean_r:.3f} rms={rms:.4f} "
              f"(batch {len(batch)} transitions)")
        if ep % STROKES_PER_UPDATE == 0 and batch:
            if max(batch_scores) <= 0.0 and not batch_has_abort:
                # Uniform-zero batch: nothing to rank, advantage normalization would only
                # amplify numerical noise (review rl-soundness-3). Keep exploring, skip update.
                print(f"  UPDATE @ep{ep}: SKIPPED (all {len(batch_scores)} strokes scored 0 — "
                      f"no signal; exploration continues unchanged)")
                batch = []
                batch_scores = []
                batch_has_abort = False
                continue
            info = agent.update(batch, build_obs(_decision_state(hw), prev_audio))
            agent.save(os.path.join(run_dir, "ppo_latest.pt"), meta=run_meta)
            # log the LEARNED depth TWICE: zero-ears reference (comparable across updates) and
            # in-context (with the current ears) — with AUDIO_OBS the output is context-dependent.
            # The reference probes at a FIXED in-band speed (range midpoint for conditioned runs):
            # measured ~0 would ask the policy below its trained band = extrapolation artifact
            # (review 2026-07-19)
            probe_speed = (0.5 * (speed_range[0] + speed_range[1])) if speed_range else hw.bow_speed
            _st_ref = hw.get_full_state(); _st_ref["bow_speed"] = probe_speed
            _ma_ref, _ = agent.predict(build_obs(_st_ref, None), deterministic=True)
            _ma_ctx, _ = agent.predict(obs, deterministic=True)
            learned_mm = hb.commanded_depth_mm(float(_ma_ref[0]), hw.nominal_depth)
            ctx_mm = hb.commanded_depth_mm(float(_ma_ctx[0]), hw.nominal_depth)
            learned_hist.append({"ep": ep, "depth_mm": learned_mm, "depth_mm_inctx": ctx_mm,
                                 "probe_speed": probe_speed})
            with open(os.path.join(run_dir, "learned_depth_history.json"), "w") as fh:
                json.dump(learned_hist, fh)
            print(f"  UPDATE @ep{ep}: pi={info['policy_loss']:+.4f} v={info['value_loss']:.3f} "
                  f"H={info['entropy']:.3f} kl={info['approx_kl']:+.4f} clip={info['clip_frac']:.2f} "
                  f"learned_depth={learned_mm:.2f}mm (in-ctx {ctx_mm:.2f})")
            batch = []
            batch_scores = []
            batch_has_abort = False
    if batch and (max(batch_scores) > 0.0 or batch_has_abort):
        # final PARTIAL batch (episodes not a multiple of STROKES_PER_UPDATE, or an abort
        # landed on the boundary): real strokes must not be silently discarded (review
        # ppo-bandit-math-1). One last update with whatever we have.
        info = agent.update(batch, build_obs(hw.get_full_state(), prev_audio))
        agent.save(os.path.join(run_dir, "ppo_latest.pt"), meta=run_meta)
        print(f"  FINAL UPDATE ({len(batch_scores)} strokes): pi={info['policy_loss']:+.4f} "
              f"v={info['value_loss']:.3f}")
    agent.save(os.path.join(run_dir, "ppo_latest.pt"), meta=run_meta)
    print(f"  done; checkpoint + data in {run_dir}")


def run_eval(hw, agent, reward_fn, recorder, episodes, max_rec_s, out_dir=REPO, eval_speeds=None):
    """DETERMINISTIC evaluation of a trained policy: act = the policy MEAN (no exploration),
    one depth per stroke, score each. Shows the converged tone without the exploration tail
    — the honest 'what the policy learned', for the open-loop-vs-closed-loop figure.
    eval_speeds: probe the learned depth(speed) CURVE — `episodes` strokes at EACH speed
    (2026-07-19 conditioning acceptance test: overlay on the grid's ear-labeled good band)."""
    scores, depths, speeds_log = [], [], []
    prev_audio = None                                   # ears: eval is sequential (hears the prev stroke)
    # INTERLEAVE speeds round-robin (not blocks): rosin warm-up / within-session drift then hits
    # every speed equally instead of confounding with the block order (review 2026-07-19)
    plan = [(s, r) for r in range(1, episodes + 1) for s in (eval_speeds or [None])]
    for stroke_i, (spd, ep) in enumerate(plan, 1):
        try:
            if spd is not None:
                hw.set_bow_speed(spd)
            if hb.LAND_AT_TARGET:
                # attack v2: deterministic decision BEFORE reset (pre-reset obs, as in training)
                obs = build_obs(_decision_state(hw), prev_audio)
                action, _ = agent.predict(obs, deterministic=True)
                hw.reset(depth_override_m=hb.commanded_depth_mm(float(action[0]),
                                                                hw.nominal_depth) / 1000.0)
            else:
                hw.reset()
                obs = build_obs(_decision_state(hw), prev_audio)
                action, _ = agent.predict(obs, deterministic=True)    # policy MEAN, no sampling
            recorder.start(max_seconds=max_rec_s); t_rec = time.time(); t_m0 = time.time()
            while not hw.is_complete():
                a_base = hw.get_baseline_action()
                if a_base is None:
                    break
                hw.execute_timestep(apply_residual(a_base, action))
                time.sleep(TIMESTEP_SEC)
            t_m1 = time.time(); audio, sr = recorder.stop()
        except RuntimeError as e:
            print(f"  eval {ep}: ABORTED — {e}")
            try:
                recorder.stop()
            except Exception:
                pass
            continue
        i0 = max(0, int((t_m0 - t_rec) * sr)); i1 = min(len(audio), int((t_m1 - t_rec) * sr))
        seg = audio[i0:i1]
        d = hb.commanded_depth_mm(float(action[0]), hw.nominal_depth)
        s = (float(reward_fn.score(seg, sr, depth_mm=d, speed_ms=hw.bow_speed))
             if len(seg) else 0.0)
        if len(seg):                                    # ears for the next eval stroke
            from feature_tone_reward import features as _afeat
            _ft = _afeat(seg, sr)
            prev_audio = [float(_ft[0]), float(_ft[1]), float(_ft[2]), float(_ft[3]), s]
        scores.append(s); depths.append(d); speeds_log.append(hw.bow_speed)
        print(f"  eval {ep:2d}" + (f" @{hw.bow_speed:.3f}m/s" if eval_speeds else "")
              + f": depth={d:.2f}mm  reward={s:.3f}")
    if scores:
        core_s = scores[1:] if len(scores) > 2 else scores   # ep1 ran on the untrained zero-ears obs
        core_d = depths[1:] if len(depths) > 2 else depths
        print(f"  EVAL deterministic policy: reward {np.mean(core_s):.3f} ± {np.std(core_s):.3f}, "
              f"depth {np.mean(core_d):.2f}mm  (n={len(core_s)}; ep1 zero-ears warmup excluded)")
        per_speed = None
        if eval_speeds:
            per_speed = {}
            for spd in eval_speeds:
                block = [i for i, v in enumerate(speeds_log) if abs(v - spd) < 1e-6]
                # warmup PER SPEED: drop each speed's own first stroke (its ears may still carry
                # another speed's context) — a global scores[1:] silently deleted the first
                # speed's whole data at --episodes 1 (review 2026-07-19)
                sel = block[1:] if len(block) > 1 else block
                if sel:                                  # the learned depth(speed) curve, one point per speed
                    per_speed[f"{spd:g}"] = {"depth_mm": float(np.mean([depths[i] for i in sel])),
                                             "reward": float(np.mean([scores[i] for i in sel])),
                                             "n": len(sel),
                                             "warmup_excluded": len(block) > 1}
            print("  learned depth(speed) curve: " + "  ".join(
                f"{k} m/s -> {v['depth_mm']:.2f}mm ({v['reward']:.2f})" for k, v in per_speed.items()))
        import json
        out = os.path.join(out_dir, "eval_last.json")   # PER-RUN (review bug: a global file let a
        with open(out, "w") as fh:                      # figure pull another run's closed-loop bar)
            json.dump({"scores": core_s, "depths": core_d, "bow_speed": hw.bow_speed,
                       "eval_speeds": eval_speeds, "per_speed": per_speed,
                       "all_speeds": speeds_log,
                       "warmup_excluded": len(scores) > 2,
                       "all_scores": scores, "all_depths": depths,
                       "attack_ramp": bool(hb.ATTACK_RAMP),
                       "land_at_target": bool(hb.LAND_AT_TARGET),
                       "run_dir": os.path.basename(os.path.normpath(out_dir))}, fh)
        print(f"  saved eval scores -> {out} (make_rl_figure reads THIS run's file)")


def run_ab(hw, agent, reward_fn, recorder, pairs, max_rec_s, out_dir=REPO, baseline_mm=None):
    """P0-1: SAME-SESSION interleaved A/B — zero-residual baseline vs deterministic policy,
    ALTERNATING stroke by stroke so drift hits both arms equally. The honest open-loop-vs-
    closed-loop comparison (the 06-29 review: the old figure mixed sessions/pipelines)."""
    import json
    import soundfile as sf
    sid = time.strftime("%Y%m%d_%H%M%S")
    wav_dir = os.path.join(out_dir, f"ab_{sid}_wavs")
    os.makedirs(wav_dir, exist_ok=True)
    log_path = os.path.join(out_dir, f"ab_{sid}_strokes.jsonl")   # per-stroke persistence (crash-safe)
    res = []
    prev_audio = None                 # shared ears: the policy hears the LAST stroke (NOTE: in AB the
    i = 0                             # policy hears a BASELINE stroke — log ears for interpretation)
    while i < 2 * pairs:
        arm = "baseline" if i % 2 == 0 else "policy"
        ears = list(prev_audio) if prev_audio is not None else None   # what THIS stroke heard
        action = None
        try:
            if hb.LAND_AT_TARGET:
                # attack v2: both arms land AT their stroke's depth (policy: pre-reset decision;
                # baseline: its fixed depth) so the A/B compares like with like
                if arm == "policy":
                    obs = build_obs(_decision_state(hw), prev_audio)
                    action, _ = agent.predict(obs, deterministic=True)
                    _ovr = hb.commanded_depth_mm(float(action[0]), hw.nominal_depth) / 1000.0
                else:
                    _ovr = (baseline_mm / 1000.0) if baseline_mm is not None else hw.nominal_depth
                hw.reset(depth_override_m=_ovr)
            else:
                hw.reset()
                if arm == "policy":
                    obs = build_obs(_decision_state(hw), prev_audio)   # commanded speed, like rl/eval
                    action, _ = agent.predict(obs, deterministic=True)  # (review 2026-07-19: measured ~0
                                                                        # ran conditioned policies OOD)
            recorder.start(max_seconds=max_rec_s)
            t_rec = time.time(); t_m0 = time.time()
            while not hw.is_complete():
                a_base = hw.get_baseline_action()
                if a_base is None:
                    break
                if arm == "policy":
                    hw.execute_timestep(apply_residual(a_base, action))
                else:
                    a = dict(a_base)                    # baseline arm = the session's BEST FIXED depth
                    a["depth_residual"] = ((baseline_mm / 1000.0 - hw.nominal_depth)
                                           if baseline_mm is not None else 0.0)
                    hw.execute_timestep(a)
                time.sleep(TIMESTEP_SEC)
            t_m1 = time.time(); audio, sr = recorder.stop()
        except RuntimeError as e:
            print(f"  ab {i + 1} ({arm}): ABORTED — {e}")
            try:
                recorder.stop()
            except Exception:
                pass
            prev_audio = None                            # honest: no usable history after an abort
            if input(">>> recover the robot; [Enter]=RETRY this stroke, q=quit: ").strip().lower() == "q":
                break
            continue                                     # retry SAME index -> pairing stays balanced
        i0 = max(0, int((t_m0 - t_rec) * sr)); i1 = min(len(audio), int((t_m1 - t_rec) * sr))
        seg = audio[i0:i1]
        d = (hb.commanded_depth_mm(float(action[0]), hw.nominal_depth) if arm == "policy"
             else (baseline_mm if baseline_mm is not None else hw.nominal_depth * 1000.0))
        s = (float(reward_fn.score(seg, sr, depth_mm=d, speed_ms=hw.bow_speed))
             if len(seg) else 0.0)
        if len(seg):
            from feature_tone_reward import features as _afeat
            _ft = _afeat(seg, sr)
            prev_audio = [float(_ft[0]), float(_ft[1]), float(_ft[2]), float(_ft[3]), s]
            sf.write(os.path.join(wav_dir, f"ab{i + 1:02d}_{arm}.wav"), seg, sr)
        row = {"i": i + 1, "arm": arm, "reward": s, "depth_mm": d, "ears": ears,
               "attack_ramp": bool(hb.ATTACK_RAMP), "land_at_target": bool(hb.LAND_AT_TARGET)}
        res.append(row)
        with open(log_path, "a") as fh:
            fh.write(json.dumps(row) + "\n")
        print(f"  ab {i + 1:2d} [{arm:8s}]: depth={d:.2f}mm  reward={s:.3f}")
        i += 1
    b = [r["reward"] for r in res if r["arm"] == "baseline"]
    p = [r["reward"] for r in res if r["arm"] == "policy"]
    if b and p:
        print(f"\n  SAME-SESSION A/B: baseline {np.mean(b):.3f}±{np.std(b):.3f} (n={len(b)})  vs  "
              f"policy {np.mean(p):.3f}±{np.std(p):.3f} (n={len(p)})")
        out = os.path.join(out_dir, f"ab_results_{sid}.json")     # timestamped: reruns never clobber
        with open(out, "w") as fh:
            json.dump({"strokes": res, "baseline_mean": float(np.mean(b)),
                       "policy_mean": float(np.mean(p)), "bow_speed": hw.bow_speed,
                       "nominal_mm": hw.nominal_depth * 1000.0,
                       "attack_ramp": bool(hb.ATTACK_RAMP),
                       "land_at_target": bool(hb.LAND_AT_TARGET),
                       "note": "policy's ears in AB = the preceding BASELINE stroke (differs from "
                               "training where it heard its own previous stroke)"}, fh)
        print(f"  saved -> {out}  (+ {log_path}, wavs in {wav_dir})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["check", "zero", "fixed", "rl", "eval", "ab"], required=True)
    ap.add_argument("--fixed-mm", type=float, default=0.3, help="fixed residual (mode=fixed)")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--notes", type=int, default=1)
    ap.add_argument("--dur", type=float, default=None,
                    help="note duration s (default: auto = frac-span travel / bow speed + margin)")
    ap.add_argument("--bow-speed", type=float, default=None,
                    help=f"bow speed m/s (default NOMINAL_SPEED={hb.NOMINAL_SPEED:.4f} = note vel 80)")
    ap.add_argument("--speed-range", default=None, metavar="LO,HI",
                    help="mode rl: CONDITIONED training — each episode's speed drawn uniformly "
                         "from [LO,HI] m/s (e.g. 0.05,0.15); speed enters the obs, depth stays "
                         "the only action. Mutually exclusive with --bow-speed.")
    ap.add_argument("--eval-speeds", default=None, metavar="S1,S2,..",
                    help="mode eval: probe the learned depth(speed) curve — --episodes strokes "
                         "at EACH listed speed (e.g. 0.05,0.09,0.15)")
    ap.add_argument("--speed-seed", type=int, default=None,
                    help="rng seed for --speed-range sampling (default: derived from wall clock "
                         "and recorded in the manifest — pass the recorded value to reproduce)")
    ap.add_argument("--checkpoint", default="", help="ppo_latest.pt: eval/ab source (default: latest "
                                                     "rl_runs), or WARM-START weights for --mode rl")
    ap.add_argument("--nominal-mm", type=float, default=None,
                    help="baseline nominal depth mm (default hardware_baseline.NOMINAL_DEPTH; a policy "
                         "is GLUED to the nominal it trained with — use the same value in eval/ab)")
    ap.add_argument("--ab-baseline-mm", type=float, default=None,
                    help="mode ab: the baseline arm's FIXED depth mm (default = nominal). Set this to the "
                         "session's best fixed depth — a detuned-nominal baseline would rig the A/B")
    ap.add_argument("--attack-ramp", action="store_true",
                    help="attack primitive v1' (2026-08-02): staircase speed ramp over the first "
                         "150ms + 90/95%% depth for the first ~100ms of every stroke; honored by "
                         "all motion modes (zero/fixed/rl/eval/ab). SUPERSEDED by --land-at-target "
                         "(2026-08-03 blind A/B: v1' flat on both instruments and the ear)")
    ap.add_argument("--land-at-target", action="store_true",
                    help="attack primitive v2 (2026-08-03): the guarded descent lands AT each "
                         "stroke's target depth (policy/fixed/baseline), so the stroke starts "
                         "with force established instead of pressing in at full bow speed "
                         "(blind test: press-in 14/16 scratchy vs landed 3/3 clean at 4mm). "
                         "zero/fixed/eval/ab + rl(PER_STROKE) honor it; mode=check and "
                         "PER_STROKE=False rl descend to nominal as before. Landing sanity "
                         "band scales with depth; landing ceiling 4.5mm (deeper stays "
                         "in-stroke territory).")
    ap.add_argument("--force-nominal", action="store_true",
                    help="eval/ab: run even if --nominal-mm mismatches the checkpoint's stored training nominal")
    args = ap.parse_args()

    if args.notes != 1:
        raise SystemExit("--notes must be 1: the frac-window stroke is a single down-bow "
                         "(the own frac counter has no note boundaries / up-bow yet)")
    speed_range = None
    if args.speed_range:
        if args.bow_speed:
            raise SystemExit("--speed-range and --bow-speed are mutually exclusive")
        if args.mode != "rl":
            raise SystemExit("--speed-range is only for --mode rl (eval probes with --eval-speeds)")
        try:
            lo, hi = (float(x) for x in args.speed_range.split(","))
        except ValueError:
            raise SystemExit(f"--speed-range wants LO,HI — got {args.speed_range!r}")
        if not (0.02 <= lo < hi <= 0.30):
            raise SystemExit(f"--speed-range [{lo},{hi}] outside sane band 0.02-0.30 or LO>=HI")
        speed_range = (lo, hi)
    eval_speeds = None
    if args.eval_speeds:
        if args.mode != "eval":
            raise SystemExit("--eval-speeds is only for --mode eval")
        eval_speeds = [float(x) for x in args.eval_speeds.split(",")]
        if any(not (0.02 <= s <= 0.30) for s in eval_speeds):
            raise SystemExit(f"--eval-speeds {eval_speeds}: outside sane band 0.02-0.30")
    bow_speed = args.bow_speed if args.bow_speed else hb.NOMINAL_SPEED
    # duration/tick budget must fit the SLOWEST stroke: the frac counter ends strokes early at
    # faster speeds, but a note/recording sized for the fast case would TRUNCATE slow strokes
    slowest = min([bow_speed] + ([speed_range[0]] if speed_range else [])
                  + (eval_speeds if eval_speeds else []))
    travel = (hb.STROKE_END_FRAC - 0.20) * hb.SPAN_M          # stroke length in metres (frac 0.20-0.80)
    dur = args.dur if args.dur else travel / slowest + 0.15    # note must outlast the frac window
    need = travel / slowest + 0.10
    if dur < need:   # an explicit short --dur would silently truncate every stroke (note runs out first)
        print(f"  !! --dur {dur:.2f}s < frac window needs {need:.2f}s at {slowest:.3f} m/s — clamping up")
        dur = need
    notes = make_episode_notes(args.notes, dur)
    n_steps = episode_ticks(notes)
    print(f"mode={args.mode}  notes={len(notes)}  episode_ticks={n_steps}  "
          f"bow_speed={bow_speed:.4f} m/s  (stroke ~{travel / bow_speed:.2f}s)")
    print(">>> HAND ON THE E-STOP. <<<")

    if args.attack_ramp:
        hb.ATTACK_RAMP = True          # module flag: every stroke in every mode honors it
    if hb.ATTACK_RAMP:                 # banner regardless of source (flag OR env var)
        print("  ATTACK RAMP v1' ON: speed x(1/3,2/3) ticks 0-1, depth x(0.90,0.95) ticks 0-1")
    if args.land_at_target:
        hb.LAND_AT_TARGET = True       # module flag: every stroke in every mode honors it
    if hb.LAND_AT_TARGET:              # banner regardless of source (flag OR env var)
        print("  LAND-AT-TARGET v2 ON: guarded descent lands at each stroke's target depth "
              "(attack starts with force established; landing band scales with depth)")
    if hb.ATTACK_RAMP and hb.LAND_AT_TARGET:
        raise SystemExit("--attack-ramp and --land-at-target are mutually exclusive: the ramp's "
                         "depth scale would LIFT ~10% off the landed target on tick 0 — the "
                         "opposite of v2's design (review D6).")
    nominal = args.nominal_mm / 1000.0 if args.nominal_mm is not None else hb.NOMINAL_DEPTH
    hw = HardwareBaseline(notes, nominal_depth=nominal, start_frac=0.20, bow_speed=bow_speed)
    print(f"  nominal depth = {nominal * 1000:.1f}mm" +
          ("  (POLICY IS GLUED TO ITS TRAINING NOMINAL — keep it consistent across rl/eval/ab)"
           if args.nominal_mm is not None else ""))
    set_speed_slider_full()
    try:
        if args.mode == "check":
            if ask("place bow at light contact (frac 0.20) and read obs (NO bowing)?"):
                run_check(hw)
        elif args.mode == "zero":
            if ask("ONE stroke, ZERO residual (should == baseline)?"):
                run_one_stroke(hw, 0.0, AudioRecorder())
        elif args.mode == "fixed":
            d = float(np.clip(args.fixed_mm / 1000.0, -DEPTH_RESIDUAL_CLAMP, DEPTH_RESIDUAL_CLAMP))
            if ask(f"ONE stroke, FIXED residual {d*1000:+.2f} mm. Start SMALL & POSITIVE; "
                   f"CONFIRM it presses INTO the string (if the bow LIFTS instead, the "
                   f"E_PRESS sign is wrong -> E-STOP and re-run verify_contact_line.py). "
                   f"Proceed?"):
                run_one_stroke(hw, d, AudioRecorder())
        elif args.mode == "rl":
            from rl.ppo.ppo import PPOAgent                 # lazy: torch only for RL
            from feature_tone_reward import FeatureToneReward   # interim feature reward (interior optimum 2026-06-27)
            agent = PPOAgent(env=None, obs_dim=OBS_DIM, act_dim=ACT_DIM, n_steps=n_steps,
                             n_epochs=4, batch_size=16, ent_coef=0.1, target_kl=0.03,
                             device="cpu", seed=0)          # ent 0.1 = locked decision
            if args.checkpoint:                             # WARM-START: continue a previous run
                _ck = os.path.abspath(args.checkpoint)
                ckpt_meta = agent.load(_ck)
                _cn = ckpt_meta.get("nominal_mm")
                if _cn is not None and abs(_cn - hw.nominal_depth * 1000.0) > 0.01 and not args.force_nominal:
                    raise SystemExit(f"warm-start ckpt was trained at nominal {_cn:.1f}mm but this run "
                                     f"uses {hw.nominal_depth * 1000:.1f}mm — the learned residuals would "
                                     f"be re-based. Pass --nominal-mm {_cn:g} or --force-nominal.")
                if ckpt_meta.get("obs_bow_speed_source", "measured_legacy") != "commanded_per_stroke":
                    print("  !! warm-start ckpt trained with MEASURED (~0) bow_speed in the obs; this "
                          "code now feeds the COMMANDED speed — that obs dim shifts ~0 -> 0.3-1.0 "
                          "(distribution shift). Fine if you are RETRAINING it into the new "
                          "semantics; do not expect its old behavior on the first strokes.")
                _dp = ckpt_meta.get("obs_decision_point", "post_reset")
                _now = "pre_reset" if hb.LAND_AT_TARGET else "post_reset"
                if _dp != _now:
                    print(f"  !! warm-start ckpt obs_decision_point={_dp}, this run is {_now} — "
                          f"obs/anatomy semantics shift; fine if RETRAINING into the new "
                          f"semantics (review D2, warn-only for warm-start).")
                print(f"  WARM-START from {_ck} (ckpt nominal {_cn})")
            max_rec_s = n_steps * TIMESTEP_SEC + 4.0        # K15: never truncate the stroke
            rec = AudioRecorder()                           # prints the resolved input device
            if not rec.is_focusrite and input(
                    ">>> Input is NOT the Focusrite — reward domain shift! "
                    "[Enter]=abort, 'yes'=continue anyway: ").strip().lower() != "yes":
                raise SystemExit("aborted: wrong audio input device")
            manifest = build_run_manifest(args, hw, rec, "rl", bow_speed, dur)
            if args.checkpoint:
                manifest["warm_start_from"] = os.path.abspath(args.checkpoint)
            # UNCONDITIONAL (review 2026-07-19): every new run's obs carries the COMMANDED speed
            # in the bow_speed slot (_decision_state) — mark it so eval/warm-start can detect
            # legacy ckpts that trained on measured (~0)
            manifest["obs_bow_speed_source"] = "commanded_per_stroke"
            speed_seed = None
            if speed_range:
                manifest["speed_range"] = list(speed_range)
                speed_seed = args.speed_seed if args.speed_seed is not None else int(time.time()) & 0xFFFFFFFF
                manifest["speed_seed"] = speed_seed
                print(f"  speed sampling: U{speed_range}, seed {speed_seed}")
            if ask(f"RL: {args.episodes} episodes, PPO update every {STROKES_PER_UPDATE} "
                   + (f"strokes, speeds ~U{speed_range} (CONDITIONED)" if speed_range else "strokes")
                   + " (motion + contact)?"):
                run_rl(hw, agent, FeatureToneReward(), rec, args.episodes, max_rec_s,
                       run_meta=manifest, speed_range=speed_range,
                       speed_seed=(speed_seed if speed_seed is not None else 0))
        elif args.mode in ("eval", "ab"):
            import glob as _glob
            from rl.ppo.ppo import PPOAgent
            from feature_tone_reward import FeatureToneReward
            cands = sorted(_glob.glob(os.path.join(REPO, "rl_runs", "*", "ppo_latest.pt")))
            if not (args.checkpoint or cands):
                raise SystemExit("no rl_runs/*/ppo_latest.pt found — pass --checkpoint")
            ckpt = os.path.abspath(args.checkpoint or cands[-1])   # abspath: else eval_last.json lands in cwd
            import torch as _torch
            _w = _torch.load(ckpt, map_location="cpu")["actor"]
            _in = next(w for k, w in _w.items() if k.endswith("weight")).shape[-1]
            if _in != OBS_DIM:
                raise SystemExit(f"checkpoint {ckpt} is {_in}-dim but OBS_DIM={OBS_DIM} (AUDIO_OBS obs "
                                 f"change) — TRAIN a fresh policy first, or pass a matching --checkpoint")
            agent = PPOAgent(env=None, obs_dim=OBS_DIM, act_dim=ACT_DIM, n_steps=n_steps, device="cpu", seed=0)
            ckpt_meta = agent.load(ckpt)
            _cn = ckpt_meta.get("nominal_mm")
            if _cn is not None and abs(_cn - hw.nominal_depth * 1000.0) > 0.01:
                msg = (f"checkpoint was TRAINED at nominal {_cn:.1f}mm but this run uses "
                       f"{hw.nominal_depth * 1000:.1f}mm — the policy's residuals would be re-based "
                       f"by the difference (review provenance-4). Pass --nominal-mm {_cn:g} "
                       f"or --force-nominal to override.")
                if not args.force_nominal:
                    raise SystemExit(msg)
                print(f"  !! {msg}  (FORCED anyway)")
            elif _cn is None:
                print("  (pre-manifest checkpoint: no stored nominal — VERIFY --nominal-mm matches "
                      "the training run yourself)")
            import residual_runner as _rr
            _cs = ckpt_meta.get("audio_obs_scales")
            if _cs is not None and not np.allclose(np.asarray(_cs, dtype=np.float32),
                                                   _rr.AUDIO_OBS_SCALES, rtol=1e-4):
                print(f"  !! checkpoint was trained with AUDIO_OBS_SCALES={_cs} but the code now uses "
                      f"{list(map(float, _rr.AUDIO_OBS_SCALES))} — the policy's audio obs are RESCALED "
                      f"vs training (distribution shift). Eval is unreliable; retrain under the new scales.")
            if ckpt_meta.get("obs_bow_speed_source", "measured_legacy") != "commanded_per_stroke":
                print("  !! checkpoint trained with MEASURED (~0) bow_speed in the obs; eval/ab now "
                      "feed the COMMANDED speed (~0.3-1.0 normalized) — obs distribution shift, "
                      "behavior may differ from training. Retrain for trustworthy numbers.")
            _dp = ckpt_meta.get("obs_decision_point", "post_reset")
            _now = "pre_reset" if hb.LAND_AT_TARGET else "post_reset"
            if _dp != _now:
                raise SystemExit(f"checkpoint obs_decision_point={_dp} but this run is {_now} "
                                 f"(--land-at-target mismatch): both the obs semantics AND the "
                                 f"attack anatomy differ from training — add/drop --land-at-target "
                                 f"to match the checkpoint (review D2).")
            max_rec_s = n_steps * TIMESTEP_SEC + 4.0
            rec = AudioRecorder()
            if not rec.is_focusrite and input(          # same hard gate as --mode rl (review ab-mic-1)
                    ">>> Input is NOT the Focusrite — reward domain shift! "
                    "[Enter]=abort, 'yes'=continue anyway: ").strip().lower() != "yes":
                raise SystemExit("aborted: wrong audio input device")
            if args.mode == "eval":
                _n = args.episodes * (len(eval_speeds) if eval_speeds else 1)
                if ask(f"EVAL: {_n} DETERMINISTIC strokes from {ckpt}"
                       + (f" across speeds {eval_speeds}" if eval_speeds else "") + "?"):
                    run_eval(hw, agent, FeatureToneReward(), rec, args.episodes, max_rec_s,
                             out_dir=os.path.dirname(ckpt), eval_speeds=eval_speeds)
            else:
                bmm = args.ab_baseline_mm
                if bmm is not None and abs(bmm / 1000.0 - hw.nominal_depth) > DEPTH_RESIDUAL_CLAMP + 1e-9:
                    raise SystemExit(f"--ab-baseline-mm {bmm} is {abs(bmm - hw.nominal_depth*1000):.1f}mm "
                                     f"from nominal {hw.nominal_depth*1000:.1f} — beyond the ±"
                                     f"{DEPTH_RESIDUAL_CLAMP*1000:.0f}mm residual authority; the robot "
                                     f"would silently clamp it and the log would lie. Choose a baseline "
                                     f"within reach or change --nominal-mm.")
                if bmm is None:
                    print("  !! --ab-baseline-mm not given: baseline arm = nominal "
                          f"({hw.nominal_depth * 1000:.1f}mm). If nominal is detuned, the A/B is RIGGED.")
                if ask(f"A/B: {args.episodes} pairs = {2 * args.episodes} strokes, baseline arm at "
                       f"{(bmm if bmm is not None else hw.nominal_depth * 1000):.1f}mm vs policy, from {ckpt}?"):
                    run_ab(hw, agent, FeatureToneReward(), rec, args.episodes, max_rec_s,
                           out_dir=os.path.dirname(ckpt), baseline_mm=bmm)
    finally:
        hw.close()


if __name__ == "__main__":
    main()
