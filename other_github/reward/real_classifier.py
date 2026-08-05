"""
real_classifier.py

Wraps the team-provided DeepMLP into our SoundClassifierBase interface so
the rest of the RL pipeline can use it interchangeably with MockSoundClassifier.

Pipeline:
    audio (np.ndarray, mono, sr=44100)
        │
        ├─ feature_extractor   (librosa replacement for Essentia)
        │       ↓ 7 raw features
        ├─ scaler.pkl          (sklearn StandardScaler, trained by them)
        │       ↓ 7 scaled features
        ├─ DeepMLP             (deep_mlp.pt weights)
        │       ↓ 2 logits
        └─ softmax             → (label, confidence)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import torch

from rl.sac.cello_env import SoundClassifierBase
from reward.classifier_models.deep_mlp import DeepMLP
from reward.classifier_models.feature_extractor import (
    extract_features,
    estimate_pitch_midi,
    DEFAULT_SR,
)


_HERE = Path(__file__).parent
_MODEL_DIR = _HERE / "classifier_models"


class RealSoundClassifier(SoundClassifierBase):
    """
    Real sound-quality classifier wrapping the team-provided DeepMLP.

    Args:
        pitch_midi : if known (e.g. A3 = 57.0), pass it to skip pitch estimation
                     and exactly match the original notebook's behaviour.
                     If None, pitch is estimated from audio with librosa.pyin.
        device     : torch device, default "cpu".

    Notes:
        - Returns (label, confidence) per SoundClassifierBase.
        - current_force kwarg is accepted but ignored (real model only looks
          at audio).
        - Internal state captured in `last_details` for the demo script to
          display intermediate values.
    """

    CLASS_MAP = {0: "Bad", 1: "Good"}

    def __init__(
        self,
        pitch_midi: Optional[float] = None,
        device: str = "cpu",
    ):
        self.pitch_midi = pitch_midi
        self.device = torch.device(device)

        # Load feature list and append pitch to match the training-time order
        with (_MODEL_DIR / "selected_features.json").open() as f:
            self.feature_names = json.load(f)["selected_features"]
        if "pitch" not in self.feature_names:
            self.feature_names.append("pitch")

        # Load scaler (sklearn StandardScaler) and model weights
        self.scaler = joblib.load(_MODEL_DIR / "scaler.pkl")

        self.model = DeepMLP(input_dim=len(self.feature_names)).to(self.device)
        state = torch.load(_MODEL_DIR / "deep_mlp.pt", map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()

        # Diagnostic info populated each call — handy for the demo.
        self.last_details: dict = {}

    # ------------------------------------------------------------------
    def predict(
        self,
        audio: np.ndarray,
        current_force: Optional[float] = None,  # accepted for API compat, ignored
        sr: int = DEFAULT_SR,
    ) -> tuple[str, float]:
        """
        Run inference on a mono float32 audio buffer.

        Args:
            audio        : 1-D numpy array, mono float32
            current_force: ignored (kept for SoundClassifierBase compatibility)
            sr           : sample rate (default 44100)

        Returns:
            (label, confidence) — label in {"Good", "Bad"}, confidence in [0, 1].
        """
        # 1. Decide pitch
        if self.pitch_midi is not None:
            pitch = float(self.pitch_midi)
        else:
            pitch = estimate_pitch_midi(audio, sr)

        # 2. Extract 7 features
        feat_dict = extract_features(audio, sr, pitch_midi=pitch)

        # 3. Order them as the scaler expects, then scale
        feat_df  = pd.DataFrame([feat_dict])
        X        = feat_df[self.feature_names].values  # (1, 7)
        X_scaled = self.scaler.transform(X)            # (1, 7)

        # 4. Forward through DeepMLP
        with torch.no_grad():
            x = torch.tensor(X_scaled, dtype=torch.float32, device=self.device)
            logits = self.model(x)
            probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred   = int(np.argmax(probs))

        label      = self.CLASS_MAP[pred]
        confidence = float(probs[pred])

        # Stash intermediates so demo.py can render them nicely
        self.last_details = {
            "features_raw":    feat_dict,
            "features_scaled": X_scaled[0].tolist(),
            "feature_names":   self.feature_names,
            "logits":          logits.cpu().numpy()[0].tolist(),
            "probs":           probs.tolist(),
            "label":           label,
            "confidence":      confidence,
            "pitch_midi":      pitch,
        }

        return label, confidence
