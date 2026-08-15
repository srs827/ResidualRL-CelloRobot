"""
bowing_search.py

Find a bowing that survives a given bow speed, by asking the real planner.

WHY THIS EXISTS

Bow position is a running sum, so whether note 140 has room depends on
directions chosen a hundred notes earlier. The planner assigns direction
greedily -- it flips only once the intended direction has already run out of
room -- which is one note of lookahead against a whole-piece constraint. When
it loses it repositions during a rest (RESET), and if the rest is too short it
does what it can (RESET~) and the piece runs late from there. Raising
--speed-scale makes every stroke eat more hair, so the margin that made the
greedy choice survivable disappears.

WHY NOT SOLVE IT ANALYTICALLY

The obvious approach -- dynamic programming over (stroke, bow position) -- was
tried and does not work, for a reason worth recording: **stroke length depends
on direction**. The planner shortens strokes for the acceleration limit, trims
them to a comfort band, and clamps to whatever room the chosen direction
leaves. A DP needs lengths known before direction is chosen, and they are not.
The first version of this file assumed they were; its "zero retake" bowings
produced MORE retakes in the real planner (1 -> 4), because forcing a direction
also removes the planner's ability to flip out of trouble.

So the objective here is the planner's own output. Slower per evaluation, but
it is measuring the thing that will actually happen.

METHOD

  1. Plan once with the score's own bowing to get a starting assignment.
  2. Local search: flip one bow group at a time, re-plan, keep the flip if the
     cost drops. Repeat until a full pass yields no improvement.

Slurred notes are one group and always share a direction, so the search can
never split a slur.

Usage:
    python BaselineControls/bowing_search.py <score.mxl> --speed-scale 1.3
    python BaselineControls/bowing_search.py <score.mxl> --sweep
    python BaselineControls/bowing_search.py <score.mxl> --speed-scale 1.3 \
        --write BaselineControls/annotations/challenge_bowing.json
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import play_midi_pieces as PMP  # noqa: E402

# What each planner event costs the performance.
#
# RETAKE and RESET~ are the only ones that break rhythm: a retake lifts and
# repositions the bow, and RESET~ is the planner saying outright that the rest
# was too short for the move it needed. A plain RESET fits inside its rest and
# is inaudible. SHRINK/CAP/DISTRIB only make a stroke slightly shorter and
# quieter, which costs tone, not time.
EVENT_COST = {
    "RETAKE": 10.0,
    "RESET~": 10.0,
    "RESET": 1.0,
    "FLIP": 0.5,
    "SPLIT": 0.5,
    "SHRINK": 0.1,
    "CAP": 0.05,
    "DISTRIB": 0.05,
}


def bow_groups(notes):
    """Collapse into units that get ONE direction each. A slur is one stroke."""
    groups, i = [], 0
    while i < len(notes):
        n = notes[i]
        if n.slur_id is not None:
            j = i
            while j < len(notes) and notes[j].slur_id == n.slur_id:
                j += 1
            groups.append(list(range(i, j)))
            i = j
        else:
            groups.append([i])
            i += 1
    return groups


def make_planner(args):
    return PMP.BowPlanner(
        start_u=args.start_u, enable_swell=True, accel_max=args.accel_max,
        retake_speed=args.retake_speed, retake_accel=args.retake_accel,
        lookahead=True, articulation_ref=args.articulation_ref,
        depth_offset=args.depth_offset / 1000.0, speed_scale=args.speed_scale)


def evaluate(notes, groups, dirs, args):
    """
    Cost of a bowing, as the planner actually renders it.

    `dirs[i] is None` leaves that group to the planner, which matters: pinning
    every note removes its ability to flip out of trouble, and a search that
    could only pin would be strictly weaker than the greedy planner it is
    trying to beat.
    """
    for gi, g in enumerate(groups):
        for ni in g:
            notes[ni].bow_dir = dirs[gi]
            notes[ni].bow_dir_source = "annotation" if dirs[gi] else "auto"
    strokes = make_planner(args).plan(notes)
    counts = {}
    for s in strokes:
        for e in (s.events or []):
            key = e.strip()
            counts[key] = counts.get(key, 0) + 1
    cost = sum(EVENT_COST.get(k, 0.0) * v for k, v in counts.items())
    return cost, counts, strokes


def search(notes, args, max_passes=6, verbose=True):
    groups = bow_groups(notes)
    original = [notes[g[0]].bow_dir for g in groups]

    # Seed: let the planner choose everything, then read back what it chose so
    # the search starts from its answer rather than from an arbitrary one.
    dirs = list(original)
    cost, counts, strokes = evaluate(notes, groups, dirs, args)
    if verbose:
        print(f"  start   cost {cost:7.2f}   {fmt(counts)}")

    seeded = []
    si = 0
    for g in groups:
        seeded.append(strokes[min(si, len(strokes) - 1)].direction)
        si += len(g)
    dirs = seeded
    cost, counts, _ = evaluate(notes, groups, dirs, args)
    if verbose:
        print(f"  pinned  cost {cost:7.2f}   {fmt(counts)}")

    for p in range(max_passes):
        improved = False
        for gi in range(len(groups)):
            if original[gi] is not None:
                continue          # the score asked for this bowing explicitly
            for cand in ("down", "up", None):
                if cand == dirs[gi]:
                    continue
                trial = list(dirs)
                trial[gi] = cand
                c, cnt, _ = evaluate(notes, groups, trial, args)
                if c < cost - 1e-9:
                    dirs, cost, counts, improved = trial, c, cnt, True
        if verbose:
            print(f"  pass {p+1}  cost {cost:7.2f}   {fmt(counts)}")
        if not improved:
            break

    for gi, g in enumerate(groups):          # leave the notes as we found them
        for ni in g:
            notes[ni].bow_dir = original[gi]
    return dirs, cost, counts, groups


def fmt(counts):
    order = ["RETAKE", "RESET~", "RESET", "FLIP", "SPLIT", "SHRINK", "CAP", "DISTRIB"]
    return "  ".join(f"{k} {counts[k]}" for k in order if counts.get(k))or "clean"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("score")
    ap.add_argument("--tempo-scale", type=float, default=1.0)
    ap.add_argument("--speed-scale", type=float, default=1.0)
    ap.add_argument("--articulation-ref", type=float, default=0.5)
    ap.add_argument("--depth-offset", type=float, default=0.0)
    ap.add_argument("--start-u", type=float, default=0.5)
    # Take these from the module, not from literals: a default that drifts
    # from play_midi_pieces.py silently searches a plan the robot will never
    # play. Hardcoding expand_dynamics=True here once produced a search whose
    # "improvement" was worse in the real run, because rescaling the dynamics
    # changes every speed and therefore every stroke length.
    ap.add_argument("--accel-max", type=float, default=PMP.ACCEL_MAX)
    ap.add_argument("--retake-speed", type=float, default=PMP.RETAKE_SPEED)
    ap.add_argument("--retake-accel", type=float, default=PMP.MOVE_ACCEL)
    # build_score reads these; defaults match play_midi_pieces.py so the plan
    # searched here is the plan that gets played.
    ap.add_argument("--midi", default=None)
    ap.add_argument("--annotations", default=None)
    ap.add_argument("--bowing", choices=["alternate","rule-of-downbow"], default="alternate")
    ap.add_argument("--no-repeats", action="store_true")
    ap.add_argument("--expand-dynamics", action="store_true")   # OFF by default, as in the CLI
    ap.add_argument("--no-merge-slurs", action="store_true")
    ap.add_argument("--sweep", action="store_true",
                    help="search at a range of bow speeds and report how fast "
                         "the piece can be bowed before the plan degrades")
    ap.add_argument("--write", metavar="JSON")
    a = ap.parse_args()

    # build_score, not parse_musicxml: the real run merges slurs, applies any
    # existing annotations and drops grace notes, which is the difference
    # between 185 parsed notes and the 182 strokes the robot plays. Searching
    # on the parsed list optimises a piece that is never performed.
    a.midi = a.score          # build_score reads the path from args.midi
    notes, _ = PMP.build_score(a)
    groups = bow_groups(notes)
    print(f"{Path(a.score).name}: {len(notes)} notes, {len(groups)} bow groups\n")

    if a.sweep:
        print("searching at each bow speed (higher speed-scale = faster bow):\n")
        for s in (1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6):
            a.speed_scale = s
            fresh = copy.deepcopy(notes)
            _, cost, counts, _ = search(fresh, a, verbose=False)
            print(f"  speed-scale {s:<4}  cost {cost:7.2f}   {fmt(counts)}")
        return 0

    dirs, cost, counts, groups = search(notes, a)
    print(f"\nbest: cost {cost:.2f}   {fmt(counts)}")

    if a.write:
        entries = []
        for g, d in zip(groups, dirs):
            if d is None:
                continue          # left to the planner on purpose
            for ni in g:
                entries.append({"index": notes[ni].index, "bow": d})
        out = {"_comment": [
            f"Bowing found by bowing_search.py at --speed-scale {a.speed_scale} "
            f"--articulation-ref {a.articulation_ref}, scored on the real "
            f"planner's output.",
            f"result: {fmt(counts)} (cost {cost:.2f})",
            "Groups the search left to the planner are omitted, so it keeps "
            "the freedom to flip out of trouble where that is the better move.",
            "Key is 'bow' (down|up); indices are pre-merge score order.",
        ], "notes": entries}
        Path(a.write).write_text(json.dumps(out, indent=2))
        print(f"wrote {len(entries)} bow overrides -> {a.write}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
