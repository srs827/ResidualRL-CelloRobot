# writeup-aug19

Four bug fixes from the aug18 Part 6/7 lists, hardware-verified, sitting on
branch
`fixes-aug19` (PR #4). Then the Part 8 item 2 experiment: the full loop —
calibrate, gate, train, select — run end to end on `twinkle_twinkle-open`.
The loop works. The one question still open is the paired baseline-vs-policy
comparison, deferred for a physical reason described in Part 2.4.

---

## Part 1: Fixes (PR #4, one commit each)

1. **`train_piece_logged` honours `--save-dir`** as the parent of the
   timestamped run directory, and rewrites the forwarded value so
   checkpoints, stroke audio and the jsonl log land in the same run dir.
   Mock runs no longer pile into `rl/checkpoints_piece/` next to real ones.

2. **The audio input stream is now shared across `HardwareExecutor`
   instances.** This closes the regression my 8/18 crash fix introduced:
   with `close()` no longer closing the stream, driver_eval's
   one-executor-per-cell loop leaked one open stream per cell. The stream
   lives in a module-level holder keyed by (device, channel, samplerate);
   a module-level callback routes chunks to whichever executor is currently
   recording. Verified on the real Scarlett: two executors built in
   sequence share one stream object, each receives chunks only while it is
   the owner, and the never-close invariant that stops the CoreAudio
   teardown segfault is kept.

3. **`perform.py` refuses compile-only flags without `--compile`**
   (`--render`, `--flatten-envelope`, `--max-run-s`, `--fixed-action`,
   `--only`) instead of silently falling through to the live path — the
   trap that cost two A/B takes on the 18th.

4. **`ab_compare` speaks both render dialects.** The blank-score bug was
   not stdout parsing: the summary was hardcoded to render_compiled's
   schema (a wav literally named `compiled_full.wav`, `tempo_ratio`,
   `mean_quality_posthoc`), and the default render is now baseline, which
   names the wav itself and reports wall/written plus per-note drift. The
   wav is discovered by glob, tempo falls back to wall/written, and a
   drift-ms column was added. Verified in a real yunpiece run: both wavs
   listed, tempo 1.011/1.010, drift +6.7 ms (baseline) vs +32.9 ms
   (policy) — that +32.9 is §7 item 6 of writeup-aug18 showing up live,
   the moveL(path) cost on shaped notes. Post-hoc q is still not computed
   on the baseline-render path; the summary now says so instead of
   printing blanks. That gap belongs in perform.py and should be a
   hardware-verified change, so I did not patch it blind.

---

## Part 2: The full loop on twinkle (Part 8 item 2, executed)

Goal as stated in writeup-aug18: separate "the pipeline is sound" from
"this piece is hard". 42 strokes, every note 0.5–1.0 s, single dynamic,
zero retakes — fully inside the judge's training domain, fill = 1
everywhere.

### 2.1 Calibration, and the duration bias confirming itself

Measured on the training piece per the aug18 lesson:

```
gain offset = +6.55 dB   (robust sd 0.32, n 42)     yunpiece 8/18: +5.06
```

+1.49 dB above yunpiece — the sign the duration bias predicts (median
0.5 s vs 0.111 s). Better: the bias is visible **inside** the piece.
Every 7th stroke — the phrase-end 1.0 s notes, the only duration change
in the piece — reads +8.1 to +8.3 dB against +6.5 for the 0.5 s notes.
One doubling of duration, ~+1.7 dB, right in the direction and rough
size of §3.2's +7.73 dB/decade. The regression's strongest evidence so
far cost zero extra robot time; it was sitting in the calibration
printout.

### 2.2 The gate, on in-distribution repertoire

```
stroke  8:  total 3.04   stroke 20:  total 7.03   stroke 33:  total 1.35
mean 3.81 -> GO
```

Two readings matter more than the GO itself.

**The noise floor collapsed.** Total repeat sd 0.013–0.020 against
0.118–0.128 on yunpiece — 8x lower. r_dynamic repeat sd, the term that
owned 81–90% of yunpiece's noise floor, is 0.000/0.030/0.000 here. On
0.5 s windows the measurement chain is simply quiet. The pipeline is
sound; yunpiece is hard. That is the separation the experiment was
designed to produce, and the gate produced it before the training run.

**The judge may be weak in-domain too.** quality SNR came out at
1.03/1.79/0.81 — on full-window, in-distribution notes, against ±35%
speed and ±1 mm depth probes; n is small, but the direction is the same
at all three strokes. The CNN's short-note inversion (aug18 Part 2) and its per-stroke
resolution are two separate problems, and the second one does not go
away on friendly repertoire. The total here was carried by r_defect
(2.18/12.60) and r_dynamic (7.42 where not saturated). This hardens the
case for the short-note annotation + retrain plan (Part 5).

(Strokes 8 and 33 show r_dynamic at sd 0.000 on both columns — every
trial inside the zone, the deadband saturating as designed.)

### 2.3 Training: 42 notes x 100 passes

`--timesteps 4200 --ckpt-every 10`, 0.84 s/step, ~35 s/episode.

The ethernet link dropped at episode 21 — the 4th drop of the session,
each cured by reseating the AX88179B dongle; swap the dongle or cable
before the next long run. First live use of `--resume-run`: it found the
750-step checkpoint plus replay buffer, derived episode offset 22 from
the stroke log, and the first resumed episode printed as 23 with scores
continuous with the run. Episodes 18–22 were played after the last
checkpoint and before the drop, so their ~200 steps were replayed —
that is the by-design cost of checkpoint spacing, not a resume bug.

The curve: mean tone 0.572 at the start, last-10 peak **0.641** around
episode 62, 0.600 at the end; ent_coef annealed 0.85 → 0.011. No
floor-bowing at any point — per-stroke speeds stayed 0.112–0.137, dyn
0.87–0.96 — the seg0 + regime fixes doing on twinkle what they were
built to do. dyn slid in the last thirty episodes (in-zone dropping to
18–25/42 on bad episodes) while tone stayed flat; see 2.4 before
reading that as reward trade-off.

One physical series logged at every set-down through the session:

```
contact at u=0.5:   3.11 N  ->  2.72  ->  2.49  ->  2.29 N
                    (gate)     (calib)   (train)    (eval)
```

Monotonic, -0.8 N over ~2.5 h of continuous playing.

### 2.4 Evaluation, and why the final comparison is postponed

`select_best --episodes 2`, deterministic policy:

```
ckpt_ep0060   return 14.06   tone 0.421      <- return winner
ckpt_ep0070          13.90        0.404
ckpt_ep0080          12.19        0.401
ckpt_ep0090          11.95        0.404
ckpt_ep0100          12.47        0.424
final                13.83        0.504      <- tone winner
confirm ep0060 (n=4) 13.83        0.410
```

Two findings and one stop decision.

**§7.5 reproduced, with a twist.** Return ranks ep0060 first — matching
the training curve's peak almost exactly — while tone puts `final`
ahead by a margin (+0.08) that n=2 can neither dismiss nor confirm. On
the 17th the measurement and the docstring heuristic disagreed; this time
the two measurements disagree with each other. select_best should print
a confidence interval and decline to pick inside it.

**Deterministic eval scored far BELOW stochastic training** — tone
0.40–0.50 against 0.60–0.64 — the wrong direction, and it hit every
checkpoint equally. By ear, every eval take was audibly scratchy. Put
together with the contact series above, the rig itself degraded over
~5000 strokes. Two hypotheses fit all three observations (falling
contact, global tone drop, scratch):

- **rosin depletion** — the hair stops gripping, stick-slip breaks up;
- **the cello shifted on its stand** — it is not rigidly fixed, and a
  string line that crept away from the bow gives shallower contact at
  the same commanded depth.

Discriminating test, next session, ~2 minutes: inspect the stand marks
and the rosin dust on hair and string; re-probe contact, reposition the
cello, re-probe again. Reposition recovers the force → it slipped; only
rosin recovers it → it's the hair.

Either way the lesson stands: **evaluation must sit adjacent to
training or behind a rig refresh.** A late eval on a drifting rig
punishes every checkpoint equally and silently. This is also a
candidate mechanism for §7.2's repeat-sd growth — the trials that
replay longest are the trials furthest apart in rig state.

### 2.5 What is still open

The paired baseline-vs-policy `ab_compare` on twinkle — the actual
"did training win" question — deferred until the rig is back in a stable state. It is a 10-minute
run: rosin/reposition, recalibrate (rosin moves the gain), then the
paired comparison with ep0060. Note: I interrupted select_best during
its confirm pass, so check whether `best.json` was written before
reusing it.

---

## Part 3: Three physical observations

**Sympathetic resonance is loud on this rig.** A metallic halo over
every open-A note; damping the C/G/D strings by hand kills it. I did
not damp them mid-session — the calibration and the gate were measured
undamped, and the judge's 500 training recordings presumably are too,
so muting mid-protocol would have changed the acoustic domain under
everything downstream. Proposal: a short blinded A/B (damped vs
undamped baseline take) and let ears decide what the recording
configuration should be. Testable side-hypothesis: the ringing adds
energy off the 220 Hz series and may be depressing hnr and period_corr
a little; the same A/B answers it.

**twinkle knocks.** A repeated mechanical knocking during play that
yunpiece does not produce. Suspicion: every twinkle note is ≥0.4 s, so
every note is split into 3 segments and every note dispatches as
moveL(path) — the segment-junction accelerations may simply be audible.
yunpiece is 86% single-segment moveL and never exposes this. Test:
`--flatten-envelope` A/B; if the knocking disappears, §7.6's "price of
the swell" has an audible component on top of the 14–20 ms.

**The contact series** (Part 2.3): the strongest single piece of
within-session drift evidence we have logged, and it was free.

---

## Part 4: The gain knob is excluded (§7.1 partial answer)

A few days ago I put thin witness tape across the Scarlett's gain
knobs. Checked this session: unmoved. Combined with the offset stability —
+5.22 measured on t1 on the 17th, +5.06 on yunpiece on the 18th,
0.16 dB apart once piece-matched — the electrical chain has been
steady all week. The 3.9 dB take-level jump has to come from what was
played and how it was measured (piece duration mix, metric, contact
state), not from the preamp. That narrows §7.1 to things we can test
from logs.

---

## Part 5: A suggestion on the plan (for discussion)

The gate quantified what we suspected: 92% of yunpiece's notes are
shorter than the judge's 0.5 s window (70% under 0.15 s). t1 is 84%
under; twinkle and string_crossings are 0%; batman is 22%. Much of this
month's reward engineering has been compensating for that mismatch, and
the compensating terms are the ones that keep needing repair.

With the draft due Friday (8/22) and ICRA on 9/15, here is what I would
suggest — open to discussion:

1. **Let the paper advance on twinkle first.** This session provides the
   full-loop demonstration on in-distribution repertoire (gate GO at
   3.81, an 8x-lower noise floor, a real learning curve,
   resume-after-crash) — pending only the paired comparison under stable conditions.
   The paper could then be framed as: closed-loop RL demonstrated where
   the judge is validated, plus a quantified reward-integrity analysis
   of what happens outside that domain (the gate, the regime split,
   the 92% number). The OOD analysis stands as a finding in its own
   right.
2. **Short-note annotation + judge retrain, near term.** The pipeline
   (label_studio_bridge) already exists. If the retrain lands inside the
   paper window, a trained yunpiece goes into this paper as well — the
   OOD analysis then reads as a before/after story rather than as a
   boundary.
3. **yunpiece stays the target** — challenge piece and drift-evidence
   source now, a training result in the paper once the judge can see
   it.
4. The repertoire itself gives the evaluation section a ready-made
   difficulty axis: 0% → 22% → 84% → 92% out-of-window. If a step
   between twinkle and the hard pieces is useful, batman sits in the
   middle (278 notes, 78% in-window).

If a different ordering makes more sense, happy to talk it through —
the constraint driving all of this is the Friday draft.

---

## Part 6: Measured wrong, or nearly, this session

- **"Eval below training" is a drift artifact until proven otherwise.**
  I nearly read the select_best table as "the policy got worse" before
  the contact series and a listen said the rig had degraded under every
  checkpoint equally.
- **The split ranking** (return → ep0060, tone → final) is undecidable at
  n=2/n=4. Nothing should be concluded from `final`'s 0.504 until a
  fresh-rig eval repeats it.
- **Four link drops in one session**, all cured by reseating the USB
  dongle, one of them killing a training run mid-episode. The resume
  path held, but the dongle is now the least reliable component on the
  rig. Swap it before the next long run.
- I stopped the final training step by hand at 4236/4200 steps; the
  interrupt-save path worked and saved latest + buffer + final. For the record: the stop condition is
  timesteps, not the episode number printed on screen, and the two
  drift apart by exactly the episodes lost to a crash.

---

## Next session

Ready to hand over: after landing whatever fixes go in (PR #4
included), this is either the 25-minute final comparison on this session's checkpoints
or a fresh twinkle retrain end to end (~1.5 h — the recipe is proven).
Otherwise it waits for my next lab slot.

1. Stand marks + rosin inspection; contact probe, reposition, probe
   again (discriminates Part 2.4's hypotheses). Rosin if needed.
2. Recalibrate on twinkle (rosin moves the gain).
3. `ab_compare` baseline vs ckpt_ep0060 — the deciding comparison. (Check
   `best.json` first; I interrupted the confirm pass.)
4. If time: damped-vs-undamped A/B; `--flatten-envelope` knock test.

## Files changed

`rl/train_piece_logged.py`, `rl/piece_hardware.py`, `rl/perform.py`,
`rl/ab_compare.py` (PR #4, branch `fixes-aug19`). Training artifacts in
`rl/checkpoints_piece/run_20260819_015423` (pre-crash) and
`run_20260819_021409` (resumed, ckpt_ep0030–0100 + final).
