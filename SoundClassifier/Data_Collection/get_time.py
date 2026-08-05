import json
from pathlib import Path

meta_file = Path("dataset_a_configs/metadata.jsonl")

total_seconds = 0.0
recordings = 0
with open(meta_file) as f:
    for line in f:
        r = json.loads(line)
        if r.get('record_type', 'stroke') != 'stroke':
            continue
        total_seconds += r['commanded']['duration']
        recordings += 1

hours   = int(total_seconds // 3600)
minutes = int((total_seconds % 3600) // 60)
seconds = total_seconds % 60

print(f"Total commanded stroke duration: {hours}h {minutes}m {seconds:.1f}s")
print(f"  ({total_seconds:.1f} seconds across {recordings} recordings)")
