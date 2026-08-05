"""
validate_dataset.py

Quick sanity check on the dataset after each session.

Usage:
    python validate_dataset.py
"""

import json
import numpy as np
from collections import Counter, defaultdict
from pathlib import Path
import soundfile as sf


AUDIO_DIR = Path("dataset/audio")
STATE_DIR = Path("dataset/states")
META_FILE = Path("dataset/metadata.jsonl")


def load_records():
    if not META_FILE.exists():
        print("No metadata.jsonl found.")
        return []
    records = []
    with open(META_FILE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get('record_type', 'stroke') != 'stroke':
                continue
            records.append(record)
    return records


def validate():
    records = load_records()
    if not records:
        return

    annotated   = [r for r in records if r.get('annotations')]
    unannotated = [r for r in records if not r.get('annotations')]

    print(f"\n-- Dataset Summary --------------------------------")
    print(f"Total recordings:    {len(records)}")
    print(f"Annotated:           {len(annotated)}")
    print(f"Unannotated:         {len(unannotated)}")

    # Session breakdown
    sessions = Counter(r['session_id'] for r in records)
    print(f"\nSessions ({len(sessions)}):")
    for s, count in sorted(sessions.items()):
        print(f"  {s}: {count} recordings")

    # Condition type breakdown
    types = Counter(r.get('condition_type', 'unknown') for r in records)
    print(f"\nCondition types:")
    for t, count in types.items():
        print(f"  {t}: {count}")

    # Score distribution (annotated only)
    if annotated:
        scores = []
        for r in annotated:
            for ann in r['annotations']:
                if 'overall' in ann:
                    scores.append(ann['overall'])
        if scores:
            dist = Counter(scores)
            print(f"\nScore distribution ({len(scores)} annotations):")
            for s in [1, 2, 3, 4]:
                count = dist.get(s, 0)
                bar   = '█' * (count // 2)
                print(f"  {s}: {count:4d}  {bar}")

            if dist.get(3, 0) + dist.get(4, 0) > 0.75 * len(scores):
                print("  Distribution skewed toward good: add more bad conditions")

    # Audio file integrity
    print(f"\nAudio file check...")
    missing = 0
    corrupt = 0
    clipped = 0

    for r in records:
        fpath = AUDIO_DIR / r['audio_file']
        if not fpath.exists():
            missing += 1
        else:
            try:
                audio, sr = sf.read(str(fpath))
                if len(audio) < sr * 0.5:
                    corrupt += 1
                if np.max(np.abs(audio)) > 0.95:
                    clipped += 1
            except Exception:
                corrupt += 1

    print(f"  Missing: {missing}")
    print(f"  Corrupt/short: {corrupt}")
    print(f"  Clipped (peak > 0.95): {clipped}")

    # Force measurement quality
    print(f"\nForce measurement quality...")
    high_deviation = 0
    high_variance  = 0

    for r in records:
        m = r.get('measured', {})
        c = r.get('commanded', {})

        if m and c:
            cmd_f  = c.get('force', 0)
            meas_f = m.get('force_mean', 0)
            std_f  = m.get('force_std', 0)

            if abs(meas_f - cmd_f) > 0.8 and cmd_f > 0:
                high_deviation += 1
            if std_f > 0.4:
                high_variance += 1

    print(f"  High deviation (|cmd-meas| > 0.8N): {high_deviation}")
    print(f"  High variance (std > 0.4N): {high_variance}")

    if high_deviation > len(records) * 0.1:
        print("  Many high deviation recordings: check force control calibration")

    # Readiness checklist
    print(f"\n-- Readiness for classifier training ---------------------")
    checks = {
        'Minimum recordings (200+)':    len(records) >= 200,
        'Bad conditions present':       types.get('bad', 0) >= 20,
        'Multiple sessions (2+)':       len(sessions) >= 2,
        'Some annotated (50+)':         len(annotated) >= 50,
        'No missing audio':             missing == 0,
        'Low clipping rate (<5%)':      clipped < len(records) * 0.05,
    }

    all_pass = True
    for check, passed in checks.items():
        status = 'yes' if passed else 'no'
        print(f"  {status} {check}")
        if not passed:
            all_pass = False

    print()
    if all_pass:
        print("Dataset ready for classifier training")
    else:
        print("Address issues above before training")


if __name__ == '__main__':
    validate()
