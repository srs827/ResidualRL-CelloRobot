#!/usr/bin/env python3
"""
Prepare a recorded dataset for human annotation.

The recording script can append duplicate metadata if a run is resumed after
overwriting the same audio/state filenames. This utility keeps the latest stroke
metadata for each (stroke_id, repeat), keeps the matching latest state_sample
rows, and optionally removes unreferenced audio/state files.
"""

import argparse
import json
import os
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path


def load_rows(meta_path):
    rows = []
    with open(meta_path) as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            rows.append((line_no, json.loads(line)))
    return rows


def record_type(record):
    return record.get("record_type", "stroke")


def stroke_key(record):
    return str(record["stroke_id"]), int(record["repeat"])


def state_key(record):
    return (
        str(record["stroke_id"]),
        int(record["repeat"]),
        int(record["sample_index"]),
    )


def choose_rows(rows):
    latest_stroke_line = {}
    kept_strokes = {}
    for line_no, record in rows:
        if record_type(record) != "stroke":
            continue
        key = stroke_key(record)
        latest_stroke_line[key] = line_no
        kept_strokes[key] = record

    timeline_lengths = {
        key: len(record.get("state_timeline", []))
        for key, record in kept_strokes.items()
    }

    latest_state_line = {}
    kept_states = {}
    for line_no, record in rows:
        if record_type(record) != "state_sample":
            continue
        key = stroke_key(record)
        sample_index = int(record.get("sample_index", -1))
        if key not in kept_strokes:
            continue
        if sample_index < 0 or sample_index >= timeline_lengths.get(key, 0):
            continue
        sample_key = state_key(record)
        latest_state_line[sample_key] = line_no
        kept_states[sample_key] = record

    kept_rows = []
    removed_rows = []
    for line_no, record in rows:
        rtype = record_type(record)
        keep = True
        if rtype == "stroke":
            keep = latest_stroke_line.get(stroke_key(record)) == line_no
        elif rtype == "state_sample":
            sample_key = state_key(record)
            keep = latest_state_line.get(sample_key) == line_no
        if keep:
            kept_rows.append(record)
        else:
            removed_rows.append((line_no, record))

    return kept_rows, removed_rows, kept_strokes


def referenced_files(strokes):
    audio = set()
    states = set()
    for record in strokes.values():
        if record.get("audio_file"):
            audio.add(record["audio_file"])
        if record.get("state_file"):
            states.add(record["state_file"])
    return audio, states


def file_health(dataset_dir, strokes):
    audio_refs, state_refs = referenced_files(strokes)
    audio_files = {p.name for p in (dataset_dir / "audio").glob("*.wav")}
    state_files = {p.name for p in (dataset_dir / "states").glob("*.npy")}
    return {
        "missing_audio": sorted(audio_refs - audio_files),
        "missing_states": sorted(state_refs - state_files),
        "unreferenced_audio": sorted(audio_files - audio_refs),
        "unreferenced_states": sorted(state_files - state_refs),
    }


def write_jsonl(path, records):
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    os.replace(tmp, path)


def backup_file(path):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = path.with_name(f"{path.name}.bak_{stamp}")
    shutil.copy2(path, backup)
    return backup


def remove_unreferenced(dataset_dir, health):
    removed = []
    for subdir, names in (
        ("audio", health["unreferenced_audio"]),
        ("states", health["unreferenced_states"]),
    ):
        for name in names:
            path = dataset_dir / subdir / name
            path.unlink()
            removed.append(path)
    return removed


def summarize(rows, kept_rows, removed_rows, kept_strokes, health):
    before_types = Counter(record_type(record) for _, record in rows)
    after_types = Counter(record_type(record) for record in kept_rows)
    print("Metadata rows:")
    print(f"  before: {len(rows)} {dict(before_types)}")
    print(f"  after:  {len(kept_rows)} {dict(after_types)}")
    print(f"  removed duplicate/orphan rows: {len(removed_rows)}")
    print(f"  unique strokes kept: {len(kept_strokes)}")
    print("File references:")
    print(f"  missing audio files:       {len(health['missing_audio'])}")
    print(f"  missing state files:       {len(health['missing_states'])}")
    print(f"  unreferenced audio files:  {len(health['unreferenced_audio'])}")
    print(f"  unreferenced state files:  {len(health['unreferenced_states'])}")


def main():
    parser = argparse.ArgumentParser(description="Deduplicate annotation dataset metadata")
    parser.add_argument("dataset_dir", help="Dataset directory containing metadata.jsonl, audio/, and states/")
    parser.add_argument("--dry-run", action="store_true", help="Report changes without modifying files")
    parser.add_argument("--no-backup", action="store_true", help="Do not create metadata.jsonl backup")
    parser.add_argument(
        "--remove-unreferenced-files",
        action="store_true",
        help="Delete audio/state files that are not referenced by kept stroke rows",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    meta = dataset_dir / "metadata.jsonl"
    rows = load_rows(meta)
    kept_rows, removed_rows, kept_strokes = choose_rows(rows)
    health = file_health(dataset_dir, kept_strokes)
    summarize(rows, kept_rows, removed_rows, kept_strokes, health)

    if args.dry_run:
        print("Dry run only; no files changed.")
        return

    if health["missing_audio"] or health["missing_states"]:
        raise SystemExit("Refusing to rewrite: kept metadata references missing files.")

    if not args.no_backup:
        backup = backup_file(meta)
        print(f"Backup written: {backup}")

    write_jsonl(meta, kept_rows)
    print(f"Rewrote metadata: {meta}")

    if args.remove_unreferenced_files:
        removed_files = remove_unreferenced(dataset_dir, health)
        print(f"Removed unreferenced files: {len(removed_files)}")


if __name__ == "__main__":
    main()
