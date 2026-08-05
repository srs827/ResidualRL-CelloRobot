"""
reward/sliding_window.py

Wraps the quality classifier in a rolling audio buffer.
Receives 50ms audio chunks, scores on 500ms windows.
Returns a score every timestep once buffer is full.

Works with both MockSoundClassifier and real BowingQualityClassifier —
same interface, just pass the model in.
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from constants import TIMESTEP_SEC


class SlidingWindowClassifier:
    """
    Maintains a rolling audio buffer.
    Receives 50ms chunks, scores on 500ms windows.

    Usage:
        clf = SlidingWindowClassifier(model, sr=44100)
        clf.reset()

        # each RL timestep:
        score = clf.push(audio_chunk_50ms, physical_params, string)
        # score is None for first 10 steps (buffer filling)
        # score is float [0,1] after that
    """

    def __init__(
        self,
        model,
        sr           = 44100,
        window_sec   = 0.5,     # 500ms — what classifier was trained on
        hop_sec      = 0.05,    # 50ms  — one per RL timestep
    ):
        self.model        = model
        self.sr           = sr
        self.window_size  = int(window_sec * sr)   # 22050 samples
        self.hop_size     = int(hop_sec * sr)      # 2205 samples

        self.buffer       = np.zeros(self.window_size, dtype=np.float32)
        self.samples_seen = 0
        self.last_score   = None

    def push(
        self,
        audio_chunk,        # np.ndarray (hop_size,) — 50ms of audio
                            # OR None if using mock (no real mic)
        physical_params,    # np.ndarray (6,) — measured values
        string = 'A',
    ):
        """
        Push one 50ms chunk into the buffer.
        Returns quality score [0,1], or None while buffer is filling.
        """
        # Mock path: no real audio — generate silence-length chunk
        if audio_chunk is None:
            audio_chunk = np.zeros(self.hop_size, dtype=np.float32)

        # Validate chunk size — pad or truncate if needed
        if len(audio_chunk) < self.hop_size:
            audio_chunk = np.pad(audio_chunk, (0, self.hop_size - len(audio_chunk)))
        elif len(audio_chunk) > self.hop_size:
            audio_chunk = audio_chunk[:self.hop_size]

        # Roll buffer and insert new audio
        self.buffer       = np.roll(self.buffer, -self.hop_size)
        self.buffer[-self.hop_size:] = audio_chunk.astype(np.float32)
        self.samples_seen += self.hop_size

        # Don't score until buffer has filled once
        if self.samples_seen < self.window_size:
            return None

        return self._score(physical_params, string)

    def _score(self, physical_params, string):
        """Run classifier on current buffer."""
        audio = self.buffer.copy()

        # Normalize amplitude — same as classifier training preprocessing
        peak = np.max(np.abs(audio))
        if peak > 1e-6:
            audio = audio / peak
        else:
            # Silence — return low score
            self.last_score = 0.1
            return self.last_score

        score = self.model.predict(
            audio_chunk     = audio,
            physical_params = physical_params,
            string          = string,
        )

        self.last_score = float(score)
        return self.last_score

    def reset(self):
        """Call at start of each episode."""
        self.buffer       = np.zeros(self.window_size, dtype=np.float32)
        self.samples_seen = 0
        self.last_score   = None