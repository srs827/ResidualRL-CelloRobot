# Update — Aug 13-14 (Zixian)

All architectural changes are done (a day ahead of the Friday deadline), and
last night we ran a full dress rehearsal of Saturday's pipeline on the robot
— calibration → training → crash-recovery → checkpoint selection → playback
— on string_crossings and t1, end to end. Saturday is safe to run.
Per-file details: PATCHES.md in robot-cello-rl-summer2026/melody_driver (our archive repo); this branch contains everything runnable.

## Tasks done

1. Merged the 8/12 version; every item on the to-do list is closed:
   best-model retention, onset-penalty short-note bug, `--perform` mode,
   envelope-dip check (ran on real data: 0% mid-note stalls — the
   continuous-renderer fix holds).
2. **Per-session gain calibration** (`rl/calibrate_gain.py`): zero-residual
   pass on a long-note piece, median residual vs the loudness model,
   `--write` into loudness_model.json. Measured **+6.77 dB** last night
   (sd 0.21, twice). This also resolved why earlier measurements disagreed
   (8.2 vs 4.2): short notes read 3–6 dB below the steady-state model —
   transient physics, not gain — so **calibration must use long-note pieces
   only**. Rosin changes the offset, hence per-session.
3. **Duration-aware dynamics grading**: notes whose physical level cap
   (loudness model at accel_max·T/4) sits below the written zone are graded
   against the reachable band (floor lowered, written ceiling kept). On t1,
   8/25 strokes were physically capped by up to 5.8 dB; zero-residual best
   effort now grades 1.0 instead of being unfixably penalized.
4. **Best-model retention**: `--ckpt-every 10` numbered checkpoints +
   `rl/select_best.py` (deterministic sweep, selects on return, top-2
   confirmation on fresh seeds).
5. **`--perform`** per the §5 spec (absolute-onset scheduling, async
   scoring), plus a **compiled mode**: the whole piece is precomputed and
   dispatched as single blended moveL paths per contiguous run. Measured:
   stroke-wise dispatch costs a constant ~124 ms/note → tempo 1.163×;
   compiled runs at **1.012–1.033×**. Use `--compile` for Monday's
   recording; the driver arm stays stroke-wise (its sensing needs it).
6. **Driver (drift-adaptation arm) ported and validated in mock** on the
   new architecture: level channel −0.074/dB, depth channel −0.257/mm,
   masked controls ~0. The 6-dim action space needs ~25k steps from
   scratch, so a mock-pretrained warm-start checkpoint is shipped
   (`melody_driver/checkpoints/driver_mock25k_obs27.zip`); Saturday's
   driver run just resumes from it and fine-tunes.

## Rehearsal results (deterministic numbers)

- **The train→eval gap is fixed.** Deterministic return beat the
  training-time average on both pieces (t1: **7.27 vs 5.45**;
  string_crossings: 5.24 vs 4.15) — compare the challengepiece case
  (train 0.468 → deterministic 0.015). With the measurement and reward
  fixes in place, the deterministic policy is now the better one, as it
  should be.
- **Retention caught its first real case**: on t1 the best checkpoint was
  ep60 (7.27 confirmed) while `final` scored 6.81 — "keep the last one"
  would have shipped a 6% worse policy.
- **First policy-vs-baseline win** (t1, compiled playback, per-zone signed
  error from the recordings): p notes **−0.79 dB quieter**, f notes
  **+0.59 dB fuller**, mf/mp unchanged — the dynamic contrast widens in
  both directions the score asks for, and it is audible. Caveat: one
  render per arm so far; worth 2–3 renders each on Saturday for error bars.
- string_crossings was a wash vs baseline (0.459 vs 0.457): after honest
  calibration the baseline already sits near the dynamics ceiling on an
  all-mf long-note etude — the pieces with dynamic contrast and short
  notes are where the policy has room to work.

## Decisions worth flagging (and why)

1. **obs 18 → 22** (full previous action). With the 6-dim envelope action,
   the two obs slots for "previous action" held only two speed residuals —
   the policy could not see its own previous depth choice, which makes
   r_smooth's depth component unpredictable and rewards a constant depth
   action ("depth shaping went unused" is exactly that signature).
   Checkpoints were already invalidated by the 2→6 action change, so this
   was the free moment. RL_METHOD §3's "18 dimensions" is superseded.
2. **Frame consistency ride-along**: `r_dynamic` corrects for
   gain_offset_db internally but the logged `err_db` stayed in the raw mic
   frame; the logging layer now writes `err_db_model` alongside. One-line
   fix upstream if you want it in piece_env itself.
3. **Classifier OOD guardrail**: fresh rosin roughly doubles contact
   torque, which pushes the physical features outside the training range —
   tone_quality/attack_quality rail to 0.0 on strokes that sound
   unremarkable, while `overall` barely moves. Added a ±2.5σ clip on the
   standardized physical features (bounds come from each checkpoint's own
   stored normalizer). Within range nothing changes; beyond it the score
   freezes at the worst in-distribution judgement instead of free-falling.
   Probably relevant for classifier work generally.

## For Saturday (template in PATCHES.md)

- Prefix long robot commands with **`caffeinate -dims`** — two RTDE
  disconnects traced to Mac sleep/USB suspend; zero in 28 min once
  prefixed. Resume protocol works if anything still drops
  (`--resume <latest> --episode-offset <episodes done>`).
- 1-minute gain calibration first (long-note piece, `--write`).
- Budgets: **timesteps = strokes × ≥100 passes**.
- After training run `select_best.py` — don't use `final`.
- Recording/playback via `perform.py --compile`.

## Open items

- Fresh-rosin sessions compress tone_quality scores even in-range; if tone
  sits floored all session, a per-session head-weight rebalance is one
  flag away (`--reward-weights`, recorded in the run header).
- The policy-vs-baseline dynamics result needs replication (n=1/arm).
- Driver's real-hardware fine-tune is Saturday's remaining first.
