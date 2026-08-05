"""
audio_recorder.py — minimal mic recorder for an RL episode, mirroring the capture in
Samantha's recording_script.py (sounddevice, 44.1 kHz mono, Focusrite if present).

Usage:
    rec = AudioRecorder()
    rec.start(max_seconds=5.0)     # at episode reset
    ... robot bows the stroke ...
    audio, sr = rec.stop()         # trims to elapsed; feed to sliding_reward.batch_score

Needs `sounddevice` (in .venv-rtde). Author: Claude (for Zixian, 2026-06-04).
"""

import time

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 44100
CHANNELS = 1


def find_focusrite_device():
    try:
        for i, d in enumerate(sd.query_devices()):
            name = d["name"].lower()
            if d["max_input_channels"] >= 1 and ("focusrite" in name or "scarlett" in name):
                return i   # Focusrite interfaces enumerate as 'Scarlett 2i2 USB' on macOS
    except Exception:
        pass
    return None


class AudioRecorder:
    def __init__(self, device=None, sample_rate=SAMPLE_RATE, channels=CHANNELS):
        self.device = device if device is not None else find_focusrite_device()
        self.sr = sample_rate
        self.channels = channels
        self._buf = None
        self._t0 = None
        try:
            d = sd.query_devices(self.device, "input") if self.device is not None \
                else sd.query_devices(kind="input")
            self.device_name = d["name"]
        except Exception:
            self.device_name = "UNKNOWN"
        self.is_focusrite = "focusrite" in self.device_name.lower() or \
                            "scarlett" in self.device_name.lower()
        # Scarlett SOLO (4th gen): the XLR mic input is ANALOGUE 2 (input 1 = the front Inst jack).
        # Recording channel 1 there captures the silent jack. Map to the mic channel.
        self.mapping = [2] if "solo" in self.device_name.lower() else None
        print(f"  AudioRecorder input: '{self.device_name}' (recording ch {self.mapping or [1]})"
              + ("" if self.is_focusrite else "  !! NOT the Focusrite — system default mic"))

    def start(self, max_seconds: float = 8.0):
        """Begin recording into a ZEROED buffer (an uninitialized tail would poison
        rms / final-tick rewards with stale memory garbage)."""
        n = int(max_seconds * self.sr)
        self._buf = np.zeros((n, self.channels), dtype="float32")
        if self.mapping:
            sd.rec(n, samplerate=self.sr, mapping=self.mapping,
                   dtype="float32", device=self.device, out=self._buf)
        else:
            sd.rec(n, samplerate=self.sr, channels=self.channels,
                   dtype="float32", device=self.device, out=self._buf)
        self._t0 = time.time()

    def stop(self):
        """Stop, trim to elapsed time, return (mono_float32, sr)."""
        elapsed = time.time() - self._t0 if self._t0 else 0.0
        sd.stop()
        audio = self._buf
        self._buf, self._t0 = None, None
        if audio is None:
            return np.zeros(1, dtype=np.float32), self.sr
        n = min(int(elapsed * self.sr), len(audio))
        audio = audio[:max(n, 1)]
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        return audio.astype(np.float32), self.sr
