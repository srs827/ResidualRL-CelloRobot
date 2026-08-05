"""
classifier.py

Real-time inference wrapper around MelQualityCNN, matching the same
predict() contract as example_sound_classifier.py's ExSoundClassifier so it
drops into reward/sliding_window.py / reward/reward_funct.py unchanged:

    classifier = BowingQualityClassifier()
    score = classifier.predict(audio_chunk, physical_params, string='A')

`audio_chunk` arrives already peak-normalized, one window (sliding_window.py
uses 500ms @ 44.1kHz = 22050 samples; window_sec/sr are configurable there,
but must match WINDOW_SEC/SR here for the model's input shape to mean what
it was trained on).

`physical_params` is the existing 6-dim vector from rl/env.py's
_get_physical_params(): [measured_force, abs(force_deviation), bow_speed,
bow_position, ft_lateral, ft_torque]. dataset.py's PHYSICAL_FEATURE_NAMES
was deliberately defined in this same order/semantics so no remapping is
needed here -- see dataset.py's docstring for the offline/online unit
caveat.

If no trained checkpoint exists yet (see train_classifier.py), falls back to
a hand-tuned heuristic over the audio features alone, in the same spirit as
ExSoundClassifier's placeholder -- usable immediately, not a substitute for
the trained model.
"""

from pathlib import Path

import numpy as np
import torch

import audio_features as af
import quality_classifier as qc
from dataset import N_PHYSICAL_FEATURES, TIER1_FIELDS

DEFAULT_CHECKPOINT = Path(__file__).parent / 'checkpoints' / 'quality_cnn.pt'
WINDOW_SEC = 0.5
SR = af.SR
WINDOW_SAMPLES = int(round(WINDOW_SEC * SR))


def _fit_window_length(audio, target_len=WINDOW_SAMPLES):
    audio = np.asarray(audio, dtype=np.float32).flatten()
    if len(audio) == target_len:
        return audio
    if len(audio) > target_len:
        start = (len(audio) - target_len) // 2
        return audio[start:start + target_len]
    if len(audio) <= 1:
        return np.zeros(target_len, dtype=np.float32)
    return np.pad(audio, (0, target_len - len(audio)), mode='reflect')


def heuristic_quality_score(scalar_features: np.ndarray) -> float:
    """
    No-checkpoint fallback. Rewards high HNR, low spectral flatness, low
    envelope instability, and a clean (low-overshoot) attack -- the same
    "what makes a stroke sound good" intuition as ExSoundClassifier's force
    model, just over audio features instead of force.
    """
    names = af.SCALAR_FEATURE_NAMES
    f = dict(zip(names, scalar_features))

    hnr_term = np.clip(f['hnr_db_mean'] / 20.0, 0.0, 1.0)          # ~0dB poor, ~20dB clean
    flatness_term = 1.0 - np.clip(f['flatness_mean'] * 4.0, 0.0, 1.0)
    envelope_term = 1.0 - np.clip(f['envelope_cv'] * 2.0, 0.0, 1.0)
    attack_term = 1.0 - np.clip((f['attack_overshoot'] - 1.0), 0.0, 1.0)

    score = 0.35 * hnr_term + 0.30 * flatness_term + 0.20 * envelope_term + 0.15 * attack_term
    return float(np.clip(score, 0.0, 1.0))


class BowingQualityClassifier:
    """Loads a trained checkpoint if present; otherwise predict() uses the heuristic fallback."""

    def __init__(self, checkpoint_path=DEFAULT_CHECKPOINT, device='cpu'):
        self.device = torch.device(device)
        self.model = None
        self.scalar_norm = None
        self.physical_norm = None
        self.schema_version = None

        checkpoint_path = Path(checkpoint_path)
        if checkpoint_path.exists():
            self._load_checkpoint(checkpoint_path)
        else:
            print(f"[BowingQualityClassifier] No checkpoint at {checkpoint_path} -- "
                  f"using heuristic fallback until train_classifier.py is run.")

    def _load_checkpoint(self, checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.schema_version = ckpt.get('schema_version', 1)
        self.model = qc.MelQualityCNN(multitask=self.schema_version >= 2).to(self.device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.eval()
        self.scalar_norm = qc.FeatureNormalizer.from_state_dict(ckpt['scalar_norm'])
        self.physical_norm = qc.FeatureNormalizer.from_state_dict(ckpt['physical_norm'])
        print(f"[BowingQualityClassifier] Loaded schema_version={self.schema_version} checkpoint "
              f"trained on {ckpt.get('n_annotated_recordings', '?')} recordings "
              f"(val_mse={ckpt.get('best_val_mse', float('nan')):.4f}).")

    def _prepare_tensors(self, audio_chunk, physical_params, window_pos):
        if audio_chunk is None:
            audio = np.zeros(WINDOW_SAMPLES, dtype=np.float32)
        else:
            audio = _fit_window_length(audio_chunk)

        physical = (np.asarray(physical_params, dtype=np.float32)
                    if physical_params is not None else np.zeros(N_PHYSICAL_FEATURES, dtype=np.float32))
        scalar = af.extract_scalar_features(audio, sr=SR)

        mel = qc.audio_to_mel_tensor(audio, sr=SR, target_frames=qc.TARGET_FRAMES)
        mel_t = torch.from_numpy(mel[np.newaxis].astype(np.float32)).to(self.device)
        scalar_t = torch.from_numpy(self.scalar_norm(scalar)[np.newaxis].astype(np.float32)).to(self.device)
        physical_t = torch.from_numpy(self.physical_norm(physical)[np.newaxis].astype(np.float32)).to(self.device)
        window_pos_t = None
        if window_pos is not None:
            window_pos_t = torch.tensor([[float(window_pos)]], dtype=torch.float32, device=self.device)
        return scalar, mel_t, scalar_t, physical_t, window_pos_t

    def predict(self, audio_chunk, physical_params, string='A', window_pos=None) -> float:
        """
        audio_chunk:     np.ndarray, ~WINDOW_SAMPLES of peak-normalized audio, or None
        physical_params: np.ndarray (6,) -- see module docstring
        string:          unused for now (single-string dataset); accepted for interface
                          compatibility and future multi-string conditioning
        window_pos:      optional float in [0,1], the window's position within the note
                          (0=onset, 1=release); defaults to mid-stroke (0.5) when omitted,
                          so existing callers are unaffected.
        """
        scalar, mel_t, scalar_t, physical_t, window_pos_t = self._prepare_tensors(
            audio_chunk, physical_params, window_pos)

        if self.model is None:
            return heuristic_quality_score(scalar)

        with torch.no_grad():
            score = self.model(mel_t, scalar_t, physical_t, window_pos_t).item()
        return float(np.clip(score, 0.0, 1.0))

    def predict_detailed(self, audio_chunk, physical_params, string='A', window_pos=None) -> dict:
        """
        Same inputs as predict(), but returns every Tier-1 dimension score:
            {'overall': ..., 'tone_quality': ..., 'attack_quality': ...,
             'release_quality': ..., 'bow_control': ..., 'dynamic_accuracy': ...}
        Falls back to {'overall': heuristic} with no checkpoint loaded, or when
        a loaded checkpoint predates the multi-task heads (schema_version 1).
        """
        scalar, mel_t, scalar_t, physical_t, window_pos_t = self._prepare_tensors(
            audio_chunk, physical_params, window_pos)

        if self.model is None:
            return {'overall': heuristic_quality_score(scalar)}

        with torch.no_grad():
            out = self.model.forward_multitask(mel_t, scalar_t, physical_t, window_pos_t)
        result = {field: float(np.clip(out[field].item(), 0.0, 1.0))
                  for field in TIER1_FIELDS if field in out}
        return result
