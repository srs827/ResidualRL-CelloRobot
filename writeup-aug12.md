# Tasks Done on August 11
1) widened speed + depth residuals
2) switched to servoL briefly -> tested with poor results because the trajectory is not smooth (there is a better solution for this of breaking each note into 3 continuous/smooth note segments for mid-stroke control)

    - BaselineControls/servo_player.py implements full servoL streaming.
      We found that no lookahead (since the trajectory must be prepared)
      was satisfactory when compared against moveL's automatically smooth calculated trajectories. (attempted 0.03, 0.10, 0.20 lookaheads). 
      High lookahead is smoother but with a noticable lag, whereas low lookahead sounds jagged. MoveL, however, generates the smooth trajectory internally.
    - to allow multiple speeds and depths per note, we use 3 segments per stroke that get blended together. then we can start the note slower with a soft attack, press slightly more after onset, and add swells/tapers inside long notes. 
    - essentially, streaming positions through servoL sounds worse than moveL

3) multi-head mechanism, not just overall sound quality (emphasis on tone_quality)
4) scalar-feature defect penalties such as hnr flag for scratchiness in training
5) attack/release for short notes via length-dependent reweighting
6) variable-length windows
7) onset acceleration penalty, implemenented but need to sort out bug for short notes: they need more accelation than longer notes so there should be less penalty for short notes for "harsh" attack 
 
# Tasks Left To-Do

1) Best-model retention (choose best policy to keep instead of last one)
2) Rebalance dynamics vs. scratchiness: currently the reward is not correctly tuned so when it is supposed to play quietly it is giving a really scractchy sound. We need to modify the reward to fix this issue.
    - Options include: raise HNR's weight, adjust bow speed tolerance and windows esp. for long notes 
3) Harsh bow-direction changes
4) bow-region sweeping to figure out bowing regions 
    - maybe some experiments with alternate bowings, angle if time permits 
5) training with far more episodes 
6) fix delay issues and note onset/rhythmic issues in the RL loop through better scheduling
    - maybe play around with asynchronous scoring given to worker thread where main loop can proceed immediately (avoid delays, esp. for short notes during the learning process)
7) check for volume dips in the middle of the note due to segmentation (envelope_outlier_rate parameter) 

# Notes and Results from Yesterday's Training

## Runs completed

1) **string_crossings** (14 notes, all 1.0 s, 100% envelope-eligible) — 1500 steps / 107 episodes / 30 min
2) **challengepiecedynamics** (182 notes, median 0.111 s, 14% envelope-eligible) — 4000 steps / 22 episodes / 22 min

## 1) Mic gain calibration 

The measured level came in 4.20 dB hotter than the loudness model predicted (−17.6 dBFS vs −21.8), which put every stroke above even the `f` ceiling. `r_dynamic` was returning a constant zero, 0/14 strokes in zone, for 25% of the reward weight.

After adding `gain_offset_db = 4.20`: 14/14 in zone, return 1.23 → 4.89.

Every dynamics result before this point was measuring nothing. The zones are absolute dBFS, so they are tied to the mic and gain that recorded `dataset_a_final` — re-measure whenever the audio path changes.

## 2) Segmentation of notes

| | tone quality | return |
|---|---|---|
| baseline (flat) | 0.679 ± 0.007 | 5.00 |
| policy, rest-to-rest segments | 0.671 ± 0.009 | 5.08 |
| **policy, continuous segments** | **0.750 ± 0.015** | 5.52 |

The renderer bug: each of the 3 segments was solved as an independent rest-to-rest trapezoid, so the bow decelerated to near zero twice inside every note. Audible as the note breaking into three pieces. Fixed by giving each segment a cruise speed (scaled by the whole-stroke peak/mean ratio) and raising the blend cap 5 mm -> 25 mm.

Rendering properly added +0.079 tone. The policy learned a consistent soft onset (segment-1 speed residual −0.13 on all 14 strokes); depth shaping went unused.

## 3) challengepiece — failed

| | tone | dynamics | in-zone | return |
|---|---|---|---|---|
| baseline | **0.496** | 0.339 | 18.7% | **27.05** |
| policy | **0.015** | 0.386 | 14.1% | 18.55 |

Tone collapsed to near zero. The policy is worse on its own reward too (18.55 vs 27.05), so this is an undertrained policy.

Cause is probably low episode count:

| | strokes/episode | steps | complete pieces seen |
|---|---|---|---|
| string_crossings (worked) | 14 | 1500 | **107** |
| challengepiece (failed) | 182 | 4000 | **22** |

Training-time average (0.468) masked a bad deterministic mean action; exploration rarely visited it, so the critic never evaluated it properly. Best-model retention would have caught this automatically, so we shoiuld implement this.

## 4) Scratchiness 

HNR measured across training (12 dB = clean, 3 dB = scratchy):

| | HNR | median duration |
|---|---|---|
| p notes (91% of piece) | **5.2 dB** | 0.111 s |
| f notes (9%) | **9.4 dB** | 0.500 s |

Flat from episode 1 to episode 20: the RL doesn't fix it. The forte notes are 4.2 dB cleaner because they are longer and faster-bowed.

The reward blocks the fix. Cleaning a scratchy note needs more bow speed, but more speed means more volume, which leaves zone `p`:

```
dynamics anchor (keeps bow slow)  0.25
HNR      (clean/non-scratchy)         0.15 x 1/4 of r_defect  =~ 0.04
```

Roughly 6:1 in favour of staying quiet and scratchy. This is a reward-design problem that we need to fix.

## 5) Bugs/unresolved issues

- Saved stroke audio is peak-normalized, so absolute level cannot be measured from it. 
- `--perform` does not exist and must be implemented:
    - play the piece at true tempo with the learned policy instead of training-loop timing
    - needs to implement:
        1) absolute-onset scheduling: before each stroke, sleep until episode_t0 + stroke.onset rather than sleeping
                                      until gap_before. this allows delays to stop compounding and have short notes
                                      stop absorbing a disproportionate share.
        2) async scoring: hand (audio, physical) to a worker thread where the main loop can proceed immediately
                          goal to remove ~50ms per note delay between strokes 
        3) report timing accuracy at the end: total wall time vs written duration + worst per-note slip
        4) behavior when late: don't sleep, play immediately, and log the slip. no catch-up by shortening notes
        5) don't change the training itself, just playback afterwards. (playback + eval only)
## 6) Timing measurements

From the actual baseline run: 52.9 s wall vs 52.5 s written = 1.01×. The baseline holds tempo, and the arm delivers the acceleration asked of it (p90 4.33, p99 6.87 m/s², well above the 1.2 nominal). So the hardware and planner are fine.

The rhythm distortion is entirely in the RL loop: ~50 ms of dead time per note (31 ms scoring + ~20 ms window-tail wait). Because it is **constant per note**, it distorts the *ratios*:

| written note | stretched to |
|---|---|
| 0.111 s | **1.45×** |
| 0.500 s | 1.10× |

Short notes inflate four times more than long ones, so 4.5:1 rhythmic proportions collapse to 3.4:1. That is why it reads as wrong rhythm rather than slow tempo.

## Takeaways

1) **The classifier missed defects that were obvious by ear — three times in one day** (the segmented envelope, servoL, and the challengepiece ranking). It caught the fourth (tone 0.015). Human listening remains the ground truth, and the paper should say so.

2) **Episode count matters more than step count.** 4000 steps sounds like more than 1500, but across a 182-stroke episode it is 22 passes versus 107.

3) **Absolute dBFS is fragile.** A gain change silently zeroed a quarter of the reward with no error and no obviously wrong number — just a score of 0.001 that looked like poor performance.

4) **Some problems are structural.** pp + 0.111 s notes forces a slow bow, which is the corner where the string stops speaking. No amount of training fixes that under the current reward; the reward has to change.

5) **A reward with no sensor for a defect will reward the defect.** Both the mid-note stall and the scratchiness went unpunished because nothing measured them.
