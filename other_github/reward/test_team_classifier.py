"""
test_team_classifier.py — does the TEAM DeepMLP classifier (deep_mlp.pt + scaler.pkl,
6 Essentia-via-librosa features + pitch) give a usable reward on OUR robot audio?
Mirrors the A2 check we did for the Audiobox probe. Replicates RealSoundClassifier's
inference directly (no cello_env dependency).

Run:  classifier_pilot/.venv/bin/python reward/test_team_classifier.py
Author: Claude (for Zixian, 2026-06-07).
"""
import glob
import json
import os
import sys
from collections import defaultdict

import joblib
import numpy as np
import pandas as pd
import soundfile as sf
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from reward.classifier_models.deep_mlp import DeepMLP                # noqa: E402
from reward.classifier_models.feature_extractor import extract_features  # noqa: E402

MDIR = os.path.join(REPO, "reward", "classifier_models")
ROBOT = os.path.join(REPO, "pre_experiment", "audio")
STUDIO = os.path.join(REPO, "..", "classifier_pilot", "data")
PITCH_A3 = 57.0   # cello open A string
PRE = 0.2
SWEEPS = {"225203": "depth", "231210": "speed", "231815": "beta"}

names = json.load(open(os.path.join(MDIR, "selected_features.json")))["selected_features"]
if "pitch" not in names:
    names.append("pitch")
scaler = joblib.load(os.path.join(MDIR, "scaler.pkl"))
model = DeepMLP(input_dim=len(names))
model.load_state_dict(torch.load(os.path.join(MDIR, "deep_mlp.pt"), map_location="cpu"))
model.eval()


def p_good(seg, sr, pitch=PITCH_A3):
    feat = extract_features(seg, sr, pitch_midi=pitch)
    X = pd.DataFrame([feat])[names].values
    with torch.no_grad():
        probs = torch.softmax(model(torch.tensor(scaler.transform(X), dtype=torch.float32)), 1)[0]
    return float(probs[1])   # P(Good)


def load(path, stroke=False, dur=None):
    a, sr = sf.read(path)
    if a.ndim > 1:
        a = a.mean(1)
    a = a.astype(np.float32)
    if stroke and dur:
        a = a[int(PRE * sr):int((PRE + dur) * sr)]
    return a, sr


# --- robot audio ---
meta = [json.loads(l) for l in open(os.path.join(ROBOT, "..", "metadata.jsonl")) if l.strip()]
g = defaultdict(list)
for m in meta:
    sid = next((s for s in SWEEPS if s in m["audio_file"]), None)
    if not sid:
        continue
    a, sr = load(os.path.join(ROBOT, m["audio_file"]), stroke=True, dur=float(m["commanded"]["duration"]))
    g[(SWEEPS[sid], m["condition_label"])].append(p_good(a, sr))
    print(".", end="", flush=True)
print(" robot done")

print("\n=== TEAM classifier P(Good) on ROBOT audio ===")
for (sweep, lbl), v in sorted(g.items()):
    print(f"  {sweep:6s} {lbl:24s} P(Good)={np.mean(v):.3f}")
allrobot = [x for v in g.values() for x in v]

# --- studio sanity ---
print("\n=== studio sanity (should: good > bad) ===")
for tag in ("good", "bad"):
    fs = sorted(glob.glob(os.path.join(STUDIO, tag, "*.wav")))[:10]
    vals = []
    for f in fs:
        a, sr = load(f)
        vals.append(p_good(a[: int(2 * sr)], sr))
    print(f"  STUDIO {tag:4s} P(Good) mean={np.mean(vals):.3f}  range=[{min(vals):.3f},{max(vals):.3f}]")

print(f"\n=== verdict ===")
ar = np.array(allrobot)
print(f"robot P(Good): mean={ar.mean():.3f} std={ar.std():.3f} range=[{ar.min():.3f},{ar.max():.3f}]")
print("std<0.05 or pinned at one class -> degenerate on robot audio (same problem).")
