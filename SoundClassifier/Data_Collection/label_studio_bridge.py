#!/usr/bin/env python3
"""
Label Studio bridge for robot cello audio annotations.

Exports per-annotator task files from metadata.jsonl and imports completed
Label Studio JSON exports back into the existing record["annotations"] list.
"""

import argparse
import json
import random
import re
from collections import Counter
from datetime import datetime
from itertools import combinations
from pathlib import Path


LABEL_CONFIG = """<View>
  <Style>
    .info { background:#f5f5f5; padding:8px; border-radius:4px;
            font-family:monospace; font-size:12px; margin-bottom:10px; }
  </Style>

  <View className="info">
    <Text name="info" value="Intended dynamic: $dynamic_est | type: $condition_type"/>
  </View>

  <Audio name="audio" value="$audio"/>

  <Header value="LAYER 1 — Technical Quality (all required, 1=poor … 4=great)"/>

  <Header value="Overall quality (holistic — trust your ear, not the average of the others)"/>
  <Choices name="overall" toName="audio" choice="single-radio" showInline="true" required="true">
    <Choice value="1" hint="No Helmholtz: raucous/scratchy, flautando, or no sound"/>
    <Choice value="2" hint="Tone present but clearly flawed: grainy, thin, uneven"/>
    <Choice value="3" hint="Mostly resonant; minor issues; performance-acceptable"/>
    <Choice value="4" hint="Great for this dataset: clear, resonant, and controlled"/>
  </Choices>

  <Header value="Tone quality (resonance, clarity, warmth of the sustained tone)"/>
  <Choices name="tone_quality" toName="audio" choice="single-radio" showInline="true" required="true">
    <Choice value="1" hint="Noise-dominated, no clear pitch"/>
    <Choice value="2" hint="Pitch present but thin/inconsistent"/>
    <Choice value="3" hint="Good tone, minor blemishes"/>
    <Choice value="4" hint="Great tone for this dataset: rich, clear fundamental"/>
  </Choices>

  <Header value="Bow control (mid-stroke evenness and stability)"/>
  <Choices name="bow_control" toName="audio" choice="single-radio" showInline="true" required="true">
    <Choice value="1" hint="Jerky, bouncing, strongly variable"/>
    <Choice value="2" hint="Noticeable speed/pressure variation"/>
    <Choice value="3" hint="Mostly smooth, minor inconsistencies"/>
    <Choice value="4" hint="Great control for this dataset: smooth and controlled"/>
  </Choices>

  <Header value="Attack quality (the first ~300ms — replay and focus on the onset)"/>
  <Choices name="attack_quality" toName="audio" choice="single-radio" showInline="true" required="true">
    <Choice value="1" hint="Scratch, thump, or failed engagement"/>
    <Choice value="2" hint="Delayed or noisy onset"/>
    <Choice value="3" hint="Clean with minor roughness"/>
    <Choice value="4" hint="Immediate, clean, controlled onset"/>
  </Choices>

  <Header value="Release quality (the last ~300ms — replay and focus on the ending)"/>
  <Choices name="release_quality" toName="audio" choice="single-radio" showInline="true" required="true">
    <Choice value="1" hint="Scratch/abrupt noisy stop"/>
    <Choice value="2" hint="Uneven or slightly noisy ending"/>
    <Choice value="3" hint="Clean ending, minor issues"/>
    <Choice value="4" hint="Smooth, controlled release"/>
  </Choices>

  <Header value="Dynamic accuracy (did the stroke achieve the INTENDED dynamic shown above?)"/>
  <Choices name="dynamic_accuracy" toName="audio" choice="single-radio" showInline="true" required="true">
    <Choice value="1" hint="Completely different dynamic than intended"/>
    <Choice value="2" hint="Noticeably off (e.g. mf when f intended)"/>
    <Choice value="3" hint="Close; slight deviation"/>
    <Choice value="4" hint="Matches the intended dynamic"/>
  </Choices>

  <Header value="LAYER 2 — Consistency Through The Stroke (required)"/>

  <Header value="Dynamic consistency: how does loudness change during the sustained stroke?"/>
  <Choices name="dynamic_consistency" toName="audio" choice="single-radio" showInline="true" required="true">
    <Choice value="much_softer" hint="Clearly fades a lot from start to finish"/>
    <Choice value="softer" hint="Gets somewhat softer"/>
    <Choice value="equal" hint="Stays about the same level"/>
    <Choice value="louder" hint="Gets somewhat louder"/>
    <Choice value="much_louder" hint="Clearly grows a lot from start to finish"/>
  </Choices>

  <Header value="Tone quality consistency: where is the tone clearest?"/>
  <Choices name="tone_consistency" toName="audio" choice="single-radio" showInline="true" required="true">
    <Choice value="much_clearer_start" hint="Much clearer at the start than the finish"/>
    <Choice value="clearer_start" hint="Somewhat clearer at the start"/>
    <Choice value="consistent" hint="Similar clarity throughout"/>
    <Choice value="clearer_finish" hint="Somewhat clearer at the finish"/>
    <Choice value="much_clearer_finish" hint="Much clearer at the finish than the start"/>
  </Choices>

  <Header value="Surface-noise consistency: where is scratch/noise most noticeable?"/>
  <Choices name="noise_consistency" toName="audio" choice="single-radio" showInline="true" required="true">
    <Choice value="much_noisier_start" hint="Much noisier at the start"/>
    <Choice value="noisier_start" hint="Somewhat noisier at the start"/>
    <Choice value="even" hint="Noise level is about even"/>
    <Choice value="noisier_finish" hint="Somewhat noisier at the finish"/>
    <Choice value="much_noisier_finish" hint="Much noisier at the finish"/>
  </Choices>

  <Header value="LAYER 3 — Tonal Character (optional; skip when Layer 1 is ambiguous)"/>

  <Header value="Grainy — 1 2 3 4 5 — Smooth"/>
  <Choices name="grainy_smooth" toName="audio" choice="single-radio" showInline="true">
    <Choice value="1" hint="Rough texture, audible grit"/>
    <Choice value="2"/>
    <Choice value="3" hint="Balanced / neutral"/>
    <Choice value="4"/>
    <Choice value="5" hint="Smooth texture, little grit"/>
  </Choices>

  <Header value="Harsh — 1 2 3 4 5 — Sweet"/>
  <Choices name="harsh_sweet" toName="audio" choice="single-radio" showInline="true">
    <Choice value="1" hint="Biting or unpleasant edge"/>
    <Choice value="2"/>
    <Choice value="3" hint="Balanced / neutral"/>
    <Choice value="4"/>
    <Choice value="5" hint="Pleasant, rounded tone"/>
  </Choices>

  <Header value="Thin — 1 2 3 4 5 — Rich"/>
  <Choices name="thin_rich" toName="audio" choice="single-radio" showInline="true">
    <Choice value="1" hint="Small, weak, or narrow tone"/>
    <Choice value="2"/>
    <Choice value="3" hint="Balanced / neutral"/>
    <Choice value="4"/>
    <Choice value="5" hint="Fuller, richer tone"/>
  </Choices>

  <Header value="Airy — 1 2 3 4 5 — Clear"/>
  <Choices name="airy_clear" toName="audio" choice="single-radio" showInline="true">
    <Choice value="1" hint="Breathy/whispery bow noise in the tone"/>
    <Choice value="2"/>
    <Choice value="3" hint="Balanced / neutral"/>
    <Choice value="4"/>
    <Choice value="5" hint="Clear pitch with little airy noise"/>
  </Choices>

  <Header value="Dry — 1 2 3 4 5 — Resonant"/>
  <Choices name="dry_resonant" toName="audio" choice="single-radio" showInline="true">
    <Choice value="1" hint="Short, non-ringing, little bloom"/>
    <Choice value="2"/>
    <Choice value="3" hint="Balanced / neutral"/>
    <Choice value="4"/>
    <Choice value="5" hint="Ringing, resonant, sustained bloom"/>
  </Choices>

  <Header value="Flags / notes"/>
  <Choices name="skip_flag" toName="audio" choice="single-radio" showInline="true">
    <Choice value="normal"/>
    <Choice value="audio_problem" hint="Mic issue, silence, clipping, artifact"/>
    <Choice value="robot_problem" hint="Mechanical noise, bow slip, fault audible"/>
  </Choices>
  <TextArea name="notes" toName="audio" placeholder="Optional comments…" rows="2" editable="true"/>
</View>
"""

ANNOTATOR_INSTRUCTIONS = """\
- Rate Layer 1 on EVERY recording. Replay before attack and release ratings.
- Rate Layer 2 only when Layer 1 was unambiguous for you (roughly every 3rd
  recording is a good pace); leave blank otherwise.
- `overall` is a holistic judgment, not the mean of the other five.
"""


def load_jsonl(path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("record_type", "stroke") != "stroke":
                continue
            records.append(record)
    return records


def save_jsonl(records, path):
    updated = {
        (record.get("stroke_id"), int(record.get("repeat", -1))): record
        for record in records
        if record.get("record_type", "stroke") == "stroke"
    }
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w") as f:
        with open(path) as src:
            for line in src:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if record.get("record_type", "stroke") == "stroke":
                    key = (record.get("stroke_id"), int(record.get("repeat", -1)))
                    record = updated.get(key, record)
                f.write(json.dumps(record) + "\n")
    tmp.replace(path)


def record_key(record):
    return str(record["stroke_id"]), int(record["repeat"])


def audio_value(record, audio_dir, document_root):
    audio_path = (audio_dir / record["audio_file"]).resolve()
    if document_root:
        rel = audio_path.relative_to(document_root.resolve())
        return f"/data/local-files/?d={rel.as_posix()}"
    return str(audio_path)


def _derive_dynamic_levels(condition_label):
    """Pull the 'Dxxx'/'Sxxx' depth/speed level tags back out of a systematic
    condition_label like 'standard_Dfirm_Svery_fast' -> ('firm', 'very_fast').
    Returns (None, None) for labels that don't follow this pattern (bad_*, TEST_*)."""
    if not condition_label:
        return None, None
    match = re.match(r"^[A-Za-z]+_D([A-Za-z_]+)_S([A-Za-z_]+)$", condition_label)
    if not match:
        return None, None
    return match.group(1), match.group(2)


def task_for_record(record, audio_dir, document_root, hide_condition):
    commanded = record.get("commanded", {})
    depth_level, speed_level = _derive_dynamic_levels(record.get("condition_label"))
    data = {
        "audio": audio_value(record, audio_dir, document_root),
        "stroke_id": record["stroke_id"],
        "repeat": record["repeat"],
        "audio_file": record["audio_file"],
        "string": record.get("string"),
        "condition_type": record.get("condition_type"),
        "audio_peak": record.get("audio_peak"),
        # Needed even in blind mode: you can't judge dynamic_accuracy without
        # knowing the intended dynamic. That's the question, not a bias leak.
        "dynamic_est": commanded.get("dynamic_est"),
    }
    if not hide_condition:
        data.update(
            {
                "condition_label": record.get("condition_label"),
                "config": record.get("config") or commanded.get("config"),
                "depth_m": commanded.get("depth_m"),
                "speed": commanded.get("speed"),
                "bow_dir": commanded.get("bow_dir"),
                "depth_level": depth_level,
                "speed_level": speed_level,
            }
        )
    return {"data": data}


def balanced_assignments(records, annotators, annotations_per_sample, seed):
    if annotations_per_sample > len(annotators):
        raise ValueError("annotations-per-sample cannot exceed annotator count")

    rng = random.Random(seed)
    counts = Counter()
    assignments = {annotator: [] for annotator in annotators}
    shuffled = list(records)
    rng.shuffle(shuffled)

    if annotations_per_sample == 2:
        pair_counts = Counter()
        pairs = [tuple(pair) for pair in combinations(annotators, 2)]
        total_assignments = len(records) * 2
        target_per_annotator = total_assignments // len(annotators)
        exact_annotator_balance = total_assignments % len(annotators) == 0

        for record in shuffled:
            eligible = []
            for pair in pairs:
                if exact_annotator_balance and any(counts[name] >= target_per_annotator for name in pair):
                    continue
                eligible.append(pair)
            if not eligible:
                eligible = pairs

            pair = min(
                eligible,
                key=lambda p: (
                    pair_counts[p],
                    max(counts[p[0]], counts[p[1]]),
                    counts[p[0]] + counts[p[1]],
                    rng.random(),
                ),
            )
            pair_counts[pair] += 1
            for annotator in pair:
                assignments[annotator].append(record)
                counts[annotator] += 1

        for annotator in assignments:
            rng.shuffle(assignments[annotator])
        return assignments, counts, pair_counts

    for record in shuffled:
        chosen = sorted(annotators, key=lambda name: (counts[name], rng.random()))
        chosen = chosen[:annotations_per_sample]
        for annotator in chosen:
            assignments[annotator].append(record)
            counts[annotator] += 1

    for annotator in assignments:
        rng.shuffle(assignments[annotator])
    return assignments, counts, Counter()


def export_tasks(args):
    meta = Path(args.meta)
    audio_dir = Path(args.audio_dir)
    out_dir = Path(args.out_dir)
    document_root = Path(args.document_root) if args.document_root else None

    records = load_jsonl(meta)
    if args.exclude_already_annotated:
        records = [r for r in records if len(r.get("annotations", [])) < args.annotations_per_sample]

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "label_config.xml").write_text(LABEL_CONFIG)
    (out_dir / "annotator_instructions.txt").write_text(ANNOTATOR_INSTRUCTIONS)

    assignments, counts, pair_counts = balanced_assignments(
        records=records,
        annotators=args.annotators,
        annotations_per_sample=args.annotations_per_sample,
        seed=args.seed,
    )

    manifest = {
        "created_at": datetime.now().isoformat(),
        "meta": str(meta),
        "audio_dir": str(audio_dir),
        "document_root": str(document_root) if document_root else None,
        "annotations_per_sample": args.annotations_per_sample,
        "annotators": args.annotators,
        "n_records": len(records),
        "task_files": {},
        "pair_counts": {" + ".join(pair): count for pair, count in sorted(pair_counts.items())},
    }

    for annotator, assigned_records in assignments.items():
        tasks = [
            task_for_record(r, audio_dir, document_root, args.hide_condition)
            for r in assigned_records
        ]
        out_path = out_dir / f"tasks_{annotator}.json"
        out_path.write_text(json.dumps(tasks, indent=2))
        manifest["task_files"][annotator] = {
            "path": str(out_path),
            "n_tasks": len(tasks),
        }

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Exported {len(records)} records into {len(args.annotators)} task files:")
    for annotator in args.annotators:
        print(f"  {annotator}: {counts[annotator]} tasks")
    if pair_counts:
        print("\nShared pair counts:")
        for pair, count in sorted(pair_counts.items()):
            print(f"  {' + '.join(pair)}: {count}")
    print(f"\nLabel config:  {out_dir / 'label_config.xml'}")
    print(f"Instructions:  {out_dir / 'annotator_instructions.txt'} (paste into the project's instructions field)")
    print(f"Manifest:     {out_dir / 'manifest.json'}")


def _parse_choice_value(raw):
    """"1" -> 1; "3_neutral" -> 3 (Layer-2-style value+label encodings); anything
    else (e.g. skip_flag's "audio_problem") passes through as a plain string."""
    if raw.isdigit():
        return int(raw)
    prefix = raw.split("_", 1)[0]
    if prefix.isdigit():
        return int(prefix)
    return raw


def parse_fields(task):
    """
    Extract every annotated field from a task's (first) annotation, keyed by
    its `from_name`: Choices -> int for numeric values, else the raw string;
    TextArea -> string. Returns ({}, None) if the task has no annotation yet.
    """
    annotations = task.get("annotations", [])
    if not annotations:
        return {}, None

    annotation = annotations[0]
    fields = {}
    for result in annotation.get("result", []):
        from_name = result.get("from_name")
        if not from_name:
            continue
        value = result.get("value", {})
        if "choices" in value:
            choices = value.get("choices") or []
            if choices:
                fields[from_name] = _parse_choice_value(str(choices[0]))
        elif "text" in value:
            texts = value.get("text") or []
            fields[from_name] = texts[0] if texts else ""
    return fields, annotation


def parse_export_spec(spec):
    if "=" in spec:
        annotator, path = spec.split("=", 1)
        return annotator.strip(), Path(path)

    path = Path(spec)
    match = re.search(r"(?:export|annotations|tasks)_?([A-Za-z0-9_-]+)", path.stem)
    if match:
        return match.group(1), path
    return path.stem, path


SKIP_FLAG_VALUES = {"audio_problem", "robot_problem"}


def import_annotations(args):
    meta = Path(args.meta)
    records = load_jsonl(meta)
    by_key = {record_key(r): r for r in records}

    imported = 0
    flagged = 0
    skipped = 0
    for spec in args.exports:
        annotator, export_path = parse_export_spec(spec)
        tasks = json.loads(export_path.read_text())
        for task in tasks:
            data = task.get("data", {})
            key = (str(data.get("stroke_id")), int(data.get("repeat")))
            record = by_key.get(key)
            fields, raw_annotation = parse_fields(task)

            if record is None or not fields:
                skipped += 1
                continue

            existing = record.setdefault("annotations", [])
            if any(a.get("annotator") == annotator for a in existing):
                skipped += 1
                continue

            completed_at = raw_annotation.get("completed_at") if raw_annotation else None
            entry = {
                "annotator": annotator,
                "timestamp": completed_at or datetime.now().isoformat(),
                "source": "label_studio",
            }

            skip_flag = fields.get("skip_flag")
            if skip_flag in SKIP_FLAG_VALUES:
                # Mic/robot problems make the 1-4 ratings meaningless -- record
                # the flag (and any notes) but don't import quality scores.
                entry["skip_flag"] = skip_flag
                if "notes" in fields:
                    entry["notes"] = fields["notes"]
                existing.append(entry)
                flagged += 1
                continue

            if "overall" not in fields:
                skipped += 1
                continue

            entry.update(fields)
            existing.append(entry)
            imported += 1

    if not args.dry_run:
        save_jsonl(records, meta)

    print(f"Imported annotations: {imported}")
    print(f"Flagged (audio/robot problem): {flagged}")
    print(f"Skipped annotations:  {skipped}")
    if args.dry_run:
        print("Dry run only; metadata was not changed.")


def main():
    parser = argparse.ArgumentParser(description="Bridge metadata.jsonl and Label Studio")
    subparsers = parser.add_subparsers(dest="command", required=True)

    export = subparsers.add_parser("export", help="Create Label Studio task files")
    export.add_argument("--meta", required=True, help="Path to metadata.jsonl")
    export.add_argument("--audio-dir", required=True, help="Directory containing WAV files")
    export.add_argument("--out-dir", default="label_studio_tasks", help="Output directory")
    export.add_argument("--annotators", nargs="+", required=True, help="Annotator IDs")
    export.add_argument("--annotations-per-sample", type=int, default=3)
    export.add_argument("--seed", type=int, default=42)
    export.add_argument("--document-root", help="Label Studio local-files document root")
    export.add_argument("--hide-condition", action="store_true", help="Blind annotation")
    export.add_argument(
        "--exclude-already-annotated",
        action="store_true",
        help="Only export records with fewer than the requested annotation count",
    )
    export.set_defaults(func=export_tasks)

    imp = subparsers.add_parser("import", help="Merge Label Studio exports into metadata")
    imp.add_argument("--meta", required=True, help="Path to metadata.jsonl")
    imp.add_argument(
        "--exports",
        nargs="+",
        required=True,
        help="Export JSON paths, preferably as ANNOTATOR=path/to/export.json",
    )
    imp.add_argument("--dry-run", action="store_true")
    imp.set_defaults(func=import_annotations)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
