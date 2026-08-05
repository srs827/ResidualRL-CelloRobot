# MicRecorder — Documentation

## Overview

`mic_recorder.py` is a reusable microphone recording module for the Robot Cello project. It continuously records fixed-length WAV audio chunks from a wireless microphone and saves them to a specified folder. The sound classification team can read these files to score audio quality during RL training.

---

## Installation

```bash
pip install sounddevice numpy
```

On macOS, if `sounddevice` fails to install:

```bash
brew install portaudio
pip install sounddevice numpy
```

**macOS permission:** The first time you run the recorder, macOS will ask for microphone access. Click **Allow**. If you accidentally denied it, go to **System Settings → Privacy & Security → Microphone** and enable it for your terminal app (Terminal / iTerm / VS Code).

---

## Quick Start

### Record a single 1-second chunk

```python
from mic_recorder import MicRecorder

recorder = MicRecorder(output_dir="recordings", chunk_duration=1.0)
path = recorder.record_one()
# Returns: "recordings/chunk_0000.wav"
```

### Continuous recording in background

```python
recorder = MicRecorder(output_dir="cello_audio", chunk_duration=1.0)
recorder.start()       # Starts recording in a background thread

# ... robot plays cello, main program continues ...

recorder.stop()        # Stops recording
# cello_audio/ now contains chunk_0000.wav, chunk_0001.wav, ...
```

### Feed audio to the sound classifier

```python
recorder.start()

# Whenever you need a score:
latest_path = recorder.get_latest()     # filepath of the most recent chunk
audio, sr = soundfile.read(latest_path)
label, confidence = classifier.predict(audio)
```

---

## API Reference

### `MicRecorder(output_dir, chunk_duration, sample_rate, device)`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `output_dir` | str | `"recordings"` | Folder to save WAV files (created automatically) |
| `chunk_duration` | float | `1.0` | Length of each WAV file in seconds |
| `sample_rate` | int | `44100` | Audio sample rate (44100 = CD quality) |
| `device` | int or None | `None` | Audio device ID. `None` = auto-detect wireless mic |

### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `record_one()` | `str` (filepath) | Record a single chunk. Blocks until done. |
| `start()` | None | Start continuous recording in background thread. |
| `stop()` | `int` (chunk count) | Stop continuous recording. |
| `get_latest()` | `str` or `None` | Filepath of the most recently saved chunk. |
| `reset_count()` | None | Reset chunk counter to 0 for a new session. |
| `list_devices()` | None | Print all available audio input devices (static method). |

---

## Auto-Detection Logic

When `device=None`, the recorder automatically finds the microphone using this priority:

1. **USB audio device** — The wireless microphone's USB receiver typically shows up as `USBAudio1.0` or similar. Any device with "usb" in its name is selected first.

2. **External input device** — Any input device that is not the built-in MacBook mic and not a virtual device (Teams, Zoom, etc.).

3. **Default input (fallback)** — If no external device is found, the system default input (usually the built-in MacBook microphone) is used.

To override auto-detection:

```python
# Force a specific device
recorder = MicRecorder(device=1)
```

To see all available devices:

```python
MicRecorder.list_devices()
```

---

## Output Format

Each chunk is saved as a **16-bit mono WAV file** at the specified sample rate.

```
recordings/
├── chunk_0000.wav    ← seconds 0–1
├── chunk_0001.wav    ← seconds 1–2
├── chunk_0002.wav    ← seconds 2–3
└── ...
```

File naming: `chunk_XXXX.wav` where `XXXX` is a zero-padded counter starting from 0.

---

## Command-Line Usage

The module can also be run standalone for testing:

```bash
# List available audio devices
python3 mic_recorder.py --list

# Record a single chunk
python3 mic_recorder.py --mode one

# Record continuously for 10 seconds
python3 mic_recorder.py --mode continuous --seconds 10

# Specify output folder and chunk length
python3 mic_recorder.py --mode continuous --output cello_audio --duration 0.5 --seconds 10

# Force a specific device
python3 mic_recorder.py --mode one --device 1

# Play the recorded file (macOS)
afplay recordings/chunk_0000.wav
```

---

## Integration with CelloEnv

To connect with `cello_env.py`, override the `_get_audio()` stub in a hardware subclass:

```python
import soundfile as sf
from mic_recorder import MicRecorder

class HardwareCelloEnv(CelloEnv):
    def __init__(self, classifier, **kwargs):
        super().__init__(classifier, **kwargs)
        self.recorder = MicRecorder(
            output_dir="cello_audio",
            chunk_duration=self.audio_samples / 44100,
        )
        self.recorder.start()

    def _get_audio(self) -> np.ndarray:
        latest = self.recorder.get_latest()
        if latest is None:
            return np.zeros(self.audio_samples, dtype=np.float32)
        audio, sr = sf.read(latest)
        return audio.astype(np.float32)

    def close(self):
        self.recorder.stop()
        super().close()
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: sounddevice` | Run `pip3 install sounddevice`. On Mac, may need `brew install portaudio` first. |
| Microphone not detected | Check that the USB receiver is plugged in. Run `list_devices()` to verify. |
| macOS permission denied | Go to System Settings → Privacy & Security → Microphone → enable your terminal app. |
| RMS ≈ 0.000 (silence) | Mic is connected but not picking up sound. Check if the mic is turned on and not muted. |
| `pip` installs to wrong Python | Use `pip3` instead of `pip` to match `python3`. |
