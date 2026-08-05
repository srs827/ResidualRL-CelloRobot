#!/usr/bin/env python3
"""
annotate.py — Robot Cello Dataset Annotation Tool

Walks through your 640 recordings (5 configs × 5 depths × 5 speeds × 5 repeats
+ 3 bad conditions × 5 repeats) and collects quality scores, saving progress
after every annotation so you can stop and resume any time.

Usage:
    python annotate.py --audio-dir dataset_a_configs/audio --meta dataset_a_configs/metadata.jsonl --annotator SK

Controls (shown on screen during annotation):
    1 2 3 4  = quality score
    ENTER    = replay current recording
    s        = skip (come back later)
    q        = quit and save progress

Quality scale:
    1 = Poor:  no Helmholtz motion — raucous/scratchy, flautando, or no sound
    2 = Fair:  some tone but noticeable problems (grainy, thin, uneven)
    3 = Good:  mostly resonant, minor inconsistencies, acceptable
    4 = Great: clear, resonant, controlled for this dataset

Annotation load: ~640 recordings × 90sec = ~16 hours total.
At 25 recordings/session (~45 min), that's 26 sessions.
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from datetime import datetime

try:
    import sounddevice as sd
    import soundfile as sf
    import numpy as np
except ImportError:
    sd = None
    sf = None
    np = None


# ── Playback ──────────────────────────────────────────────────────

def play_audio(fpath, skip_silence_s=0.2, gain=3.0):
    """Play an audio file, skipping the pre-buffer silence. Boosts gain
    slightly since robot recordings tend to be quiet."""
    if sd is None or sf is None or np is None:
        print("Missing dependencies. Run: pip install sounddevice soundfile numpy")
        return
    try:
        audio, sr = sf.read(str(fpath))
        audio = audio.flatten().astype('float32')
        start = int(skip_silence_s * sr)
        audio = audio[start:]
        audio = audio * gain
        # clip to prevent distortion after gain
        audio = np.clip(audio, -1.0, 1.0)
        sd.play(audio, sr)
        sd.wait()
    except Exception as e:
        print(f"  [playback error: {e}]")


# ── Metadata I/O ──────────────────────────────────────────────────

def load_records(meta_file):
    records = []
    with open(meta_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get('record_type', 'stroke') != 'stroke':
                continue
            records.append(record)
    return records


def load_task_keys(task_file):
    tasks = json.loads(Path(task_file).read_text())
    keys = []
    for task in tasks:
        data = task.get("data", {})
        if "stroke_id" not in data or "repeat" not in data:
            continue
        keys.append((str(data["stroke_id"]), int(data["repeat"])))
    return keys


def save_records(records, meta_file):
    """Rewrite metadata with updated stroke annotations, preserving sample rows."""
    updated = {
        (r.get('stroke_id'), int(r.get('repeat', -1))): r
        for r in records
        if r.get('record_type', 'stroke') == 'stroke'
    }
    tmp = str(meta_file) + ".tmp"
    with open(tmp, 'w') as f:
        with open(meta_file) as src:
            for line in src:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if record.get('record_type', 'stroke') == 'stroke':
                    key = (record.get('stroke_id'), int(record.get('repeat', -1)))
                    record = updated.get(key, record)
                f.write(json.dumps(record) + '\n')
    os.replace(tmp, str(meta_file))   # atomic on most filesystems


def already_annotated_by(record, annotator_id):
    return any(a['annotator'] == annotator_id for a in record.get('annotations', []))


def unannotated_count(records, annotator_id):
    return sum(1 for r in records if not already_annotated_by(r, annotator_id))


# ── Stats helpers ─────────────────────────────────────────────────

def print_progress(records, annotator_id):
    total   = len(records)
    done    = total - unannotated_count(records, annotator_id)
    scores  = []
    for r in records:
        for a in r.get('annotations', []):
            if a['annotator'] == annotator_id:
                scores.append(a['overall'])
    dist = {s: scores.count(s) for s in [1, 2, 3, 4]}
    print(f"\n  Progress: {done}/{total} annotated ({100*done/total:.0f}%)")
    if scores:
        print(f"  Score distribution: " +
              " | ".join(f"{s}★={dist[s]}" for s in [1,2,3,4]))
        print(f"  Mean score: {sum(scores)/len(scores):.2f}")
    print()


# ── Main annotation loop ──────────────────────────────────────────

def annotate(meta_file, audio_dir, annotator_id, session_size=25,
             randomize=True, show_condition=True, task_file=None):
    meta_file = Path(meta_file)
    audio_dir = Path(audio_dir)

    print(f"\n{'='*55}")
    print(f"  Robot Cello Annotation Tool")
    print(f"  Annotator: {annotator_id}")
    print(f"  Metadata:  {meta_file}")
    print(f"  Audio:     {audio_dir}")
    if task_file:
        print(f"  Task file: {task_file}")
    print(f"{'='*55}")
    print("""
  Quality scale:
    1 = Poor:  raucous/scratchy, flautando, or basically no sound
    2 = Fair:  some tone but clearly problematic (grainy, thin, uneven)
    3 = Good:  mostly resonant, acceptable, minor issues only
    4 = Great: clear, resonant, controlled for this dataset

  Controls:
    1 2 3 4  rate the recording
    ENTER    replay
    s        skip (come back later)
    q        quit and save
""")

    records = load_records(meta_file)
    if task_file:
        ordered_keys = load_task_keys(task_file)
        allowed = set(ordered_keys)
        by_key = {(str(r["stroke_id"]), int(r["repeat"])): r for r in records}
        records = [by_key[key] for key in ordered_keys if key in by_key]
        missing = len(allowed) - len({(str(r["stroke_id"]), int(r["repeat"])) for r in records})
        if missing:
            print(f"  Warning: {missing} task records were not found in metadata.")

    print_progress(records, annotator_id)

    # Build the queue for this session: unannotated records in randomized order
    queue = [r for r in records if not already_annotated_by(r, annotator_id)]
    if not queue:
        print("All recordings already annotated by you. Done!")
        return

    if randomize:
        random.shuffle(queue)
    queue = queue[:session_size]

    print(f"  This session: {len(queue)} recordings")
    print(f"  ({unannotated_count(records, annotator_id)} total remaining)\n")
    input("  Press ENTER to start...")
    print()

    annotated_this_session = 0
    for i, record in enumerate(queue):
        fpath = audio_dir / record['audio_file']

        if not fpath.exists():
            print(f"  [MISSING: {fpath}] skipping")
            continue

        # Header
        print(f"─ [{i+1}/{len(queue)}] ", end="")
        if show_condition:
            ctype = record.get('condition_type', '')
            label = record.get('condition_label', '')
            depth = record.get('commanded', {}).get('depth_m', 0)
            speed = record.get('commanded', {}).get('speed', 0)
            bow_dir = record.get('commanded', {}).get('bow_dir', '')
            print(f"{label}  [{ctype}]")
            print(f"  depth={depth*1000:.1f}mm  speed={speed:.2f}m/s  "
                  f"dir={bow_dir}  peak={record['audio_peak']:.3f}")
        else:
            print(f"Recording {record['audio_file']}")

        # Auto-play once
        play_audio(fpath)

        # Get score
        score = None
        while score is None:
            try:
                inp = input("  Score [1-4] / ENTER=replay / s=skip / q=quit: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\n  Interrupted — saving progress.")
                save_records(records, meta_file)
                return

            if inp == 'q':
                print(f"\n  Saved. Annotated {annotated_this_session} this session.")
                print_progress(records, annotator_id)
                save_records(records, meta_file)
                return
            elif inp == 's':
                print("  Skipped.\n")
                break
            elif inp == '':
                play_audio(fpath)
            elif inp in ['1', '2', '3', '4']:
                score = int(inp)
            else:
                print("  Type 1, 2, 3, or 4.")

        if score is None:
            continue   # was skipped

        # Save annotation directly into the record
        annotation = {
            'annotator':   annotator_id,
            'timestamp':   datetime.now().isoformat(),
            'overall':     score,
        }

        # Find the matching record in the full list and update it
        for r in records:
            if (r['stroke_id'] == record['stroke_id'] and
                    r['repeat'] == record['repeat']):
                r.setdefault('annotations', []).append(annotation)
                break

        # Write after every annotation — never lose progress
        save_records(records, meta_file)
        annotated_this_session += 1
        print(f"  → {score}★  saved.\n")

    print(f"\n{'='*55}")
    print(f"  Session complete: {annotated_this_session} recordings annotated.")
    print_progress(records, annotator_id)
    remaining = unannotated_count(records, annotator_id)
    if remaining:
        print(f"  {remaining} remaining — re-run to continue.\n")
    else:
        print("  All done! Ready for classifier training.\n")


# ── Agreement check ───────────────────────────────────────────────

def check_agreement(meta_file):
    """Print inter-annotator agreement stats. Run after multiple annotators."""
    try:
        from sklearn.metrics import cohen_kappa_score
    except ImportError:
        print("Run: pip install scikit-learn")
        return

    records = load_records(meta_file)
    multi   = [r for r in records if len(r.get('annotations', [])) >= 2]
    print(f"\nRecordings with >=2 annotations: {len(multi)}")

    if not multi:
        print("Need overlap recordings to compute agreement.")
        return

    # Build per-annotator score dicts
    ann_scores = {}
    for r in multi:
        key = f"{r['stroke_id']}_{r['repeat']}"
        for a in r['annotations']:
            aid = a['annotator']
            ann_scores.setdefault(aid, {})[key] = a['overall']

    names = list(ann_scores.keys())
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            a1, a2  = names[i], names[j]
            shared  = sorted(set(ann_scores[a1]) & set(ann_scores[a2]))
            if len(shared) < 5:
                print(f"  {a1} vs {a2}: only {len(shared)} shared — need more")
                continue
            s1 = [ann_scores[a1][k] for k in shared]
            s2 = [ann_scores[a2][k] for k in shared]
            kappa = cohen_kappa_score(s1, s2)
            status = "✓ good" if kappa >= 0.6 else ("⚠ marginal" if kappa >= 0.4 else "✗ poor")
            print(f"  {a1} vs {a2}: κ={kappa:.3f}  {status}  (n={len(shared)})")


# ── Entry point ───────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Annotate robot cello recordings")
    parser.add_argument('--meta',       required=True, help="Path to metadata.jsonl")
    parser.add_argument('--audio-dir',  required=True, help="Directory containing .wav files")
    parser.add_argument('--annotator',  required=True, help="Your initials or unique ID")
    parser.add_argument('--session-size', type=int, default=25,
                        help="Max recordings per session (default 25)")
    parser.add_argument('--no-shuffle', action='store_true',
                        help="Don't shuffle; annotate in metadata order")
    parser.add_argument('--check-agreement', action='store_true',
                        help="Print inter-annotator kappa and exit")
    parser.add_argument('--hide-condition', action='store_true',
                        help="Hide condition labels while annotating (blind)")
    parser.add_argument('--task-file',
                        help="Optional Label Studio tasks_ANNOTATOR.json file limiting this annotator to a subset")
    args = parser.parse_args()

    if args.check_agreement:
        check_agreement(args.meta)
    else:
        annotate(
            meta_file    = args.meta,
            audio_dir    = args.audio_dir,
            annotator_id = args.annotator,
            session_size = args.session_size,
            randomize    = not args.no_shuffle,
            show_condition = not args.hide_condition,
            task_file = args.task_file,
        )
