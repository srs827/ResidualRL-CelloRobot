"""
quality_classifier.py

The CNN itself: a small conv stack over the log-mel spectrogram of a 500ms
bowing window, fused with the hand-engineered audio features (HNR, spectral
flatness, F0 stability, envelope shape, attack quality, MFCCs -- see
audio_features.py) and the robot's physical state (dataset.py /
PHYSICAL_FEATURE_NAMES), producing one quality score in [0,1].

This module only defines the model + a normalization helper. Training lives
in train_classifier.py; real-time inference (matching the predict() contract
used by reward/sliding_window.py) lives in classifier.py.
"""

import numpy as np
import torch
import torch.nn as nn

import audio_features as af
from dataset import N_PHYSICAL_FEATURES, TIER1_FIELDS

TARGET_FRAMES = af.mel_spectrogram_frames_for_duration(0.5)   # 500ms window -> fixed frame count
AUX_FIELDS = [f for f in TIER1_FIELDS if f != 'overall']


class FeatureNormalizer:
    """Z-score normalization fit on the training split, applied identically at inference."""

    def __init__(self, mean, std):
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)

    def __call__(self, x):
        return (x - self.mean) / self.std

    def state_dict(self):
        return {'mean': self.mean, 'std': self.std}

    @classmethod
    def from_state_dict(cls, state):
        return cls(state['mean'], state['std'])


class MelQualityCNN(nn.Module):
    """
    Input:
        mel        (B, 1, n_mels, T)         log-mel spectrogram of the window
        scalar     (B, N_SCALAR_FEATURES)     normalized hand-engineered audio features
        physical   (B, N_PHYSICAL_FEATURES)   normalized robot physical-state features
        window_pos (B,) or (B, 1)             window's position in [0,1] within the
                                               recording's active region (0=onset, 1=release);
                                               defaults to 0.5 (mid-stroke) when not given.
                                               Ignored when multitask=False.
    Output:
        forward()           -> (B,) overall quality score in [0,1] (sigmoid).
        forward_multitask()  -> {field: (B,) tensor} for every TIER1_FIELDS entry
                                 (just {'overall': ...} when multitask=False).

    multitask=True (default, schema_version 2) trunk pools frequency fully but
    keeps time, then summarizes time with both mean and std pooling (128-dim)
    -- plain global-average pooling throws away exactly the temporal structure
    ("clean attack, rough finish") that attack/release/consistency judgments
    need. multitask=False reproduces the original schema_version-1 single-head
    architecture (conv -> AdaptiveAvgPool2d(1) -> single head, no window_pos)
    verbatim, key-for-key, so classifier.py can load old checkpoints exactly.
    """

    def __init__(self, n_mels=af.N_MELS, n_scalar=af.N_SCALAR_FEATURES,
                 n_physical=N_PHYSICAL_FEATURES, dropout=0.3, multitask=True):
        super().__init__()
        self.multitask = multitask

        if multitask:
            self.conv = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            )
            self.temporal_pool = nn.AdaptiveAvgPool2d((1, None))
            conv_out_dim = 64 * 2   # mean + std over time

            fusion_dim = conv_out_dim + n_scalar + n_physical + 1   # +1 for window_pos
            self.head_overall = nn.Sequential(
                nn.Linear(fusion_dim, 32), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(32, 1),
            )
            self.heads_aux = nn.ModuleDict({
                field: nn.Sequential(
                    nn.Linear(fusion_dim, 16), nn.ReLU(),
                    nn.Linear(16, 1),
                )
                for field in AUX_FIELDS
            })
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
            )
            fusion_dim = 64 + n_scalar + n_physical
            self.head = nn.Sequential(
                nn.Linear(fusion_dim, 32), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(32, 1),
            )

    def _fuse(self, mel, scalar, physical, window_pos=None):
        if not self.multitask:
            x = self.conv(mel).flatten(1)
            return torch.cat([x, scalar, physical], dim=1)

        x = self.conv(mel)                         # (B, 64, 1, T')
        x = self.temporal_pool(x).squeeze(2)        # (B, 64, T')
        x = torch.cat([x.mean(dim=-1), x.std(dim=-1)], dim=1)   # (B, 128)

        if window_pos is None:
            window_pos = torch.full((mel.shape[0], 1), 0.5, dtype=mel.dtype, device=mel.device)
        window_pos = window_pos.reshape(mel.shape[0], 1).to(dtype=mel.dtype, device=mel.device)

        return torch.cat([x, scalar, physical, window_pos], dim=1)

    def forward_multitask(self, mel, scalar, physical, window_pos=None):
        fused = self._fuse(mel, scalar, physical, window_pos)
        if not self.multitask:
            return {'overall': torch.sigmoid(self.head(fused)).squeeze(-1)}
        out = {'overall': torch.sigmoid(self.head_overall(fused)).squeeze(-1)}
        for field, head in self.heads_aux.items():
            out[field] = torch.sigmoid(head(fused)).squeeze(-1)
        return out

    def forward(self, mel, scalar, physical, window_pos=None):
        fused = self._fuse(mel, scalar, physical, window_pos)
        head = self.head if not self.multitask else self.head_overall
        return torch.sigmoid(head(fused)).squeeze(-1)


def audio_to_mel_tensor(audio, sr=af.SR, target_frames=TARGET_FRAMES):
    """Raw audio window -> (1, n_mels, target_frames) float32 array, ready to batch+tensor."""
    log_mel = af.compute_log_mel_spectrogram(audio, sr=sr, target_frames=target_frames)
    return log_mel[np.newaxis, :, :]
