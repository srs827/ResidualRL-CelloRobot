"""
mic_recorder.py

Reusable microphone recording module for Robot Cello.
Auto-detects the wireless microphone. Other scripts just call:

    from mic_recorder import MicRecorder

    recorder = MicRecorder(output_dir="recordings", chunk_duration=1.0)
    recorder.start()   # starts recording in background
    # ... do other things, WAV files appear in recordings/ ...
    recorder.stop()    # stops recording

Or for one-shot use:

    recorder.record_one()   # record a single chunk, return filepath
"""

import sounddevice as sd
import numpy as np
import wave
import os
import time
import threading


class MicRecorder:
    """
    Continuous microphone recorder that saves fixed-length WAV chunks.
    
    Args:
        output_dir:     folder to save WAV files (created if not exists)
        chunk_duration: length of each WAV file in seconds (default 1.0)
        sample_rate:    audio sample rate (default 44100)
        device:         audio device ID (None = auto-detect wireless mic)
    """

    def __init__(
        self,
        output_dir: str = "recordings",
        chunk_duration: float = 1.0,
        sample_rate: int = 44100,
        device: int = None,
    ):
        self.output_dir = output_dir
        self.chunk_duration = chunk_duration
        self.sample_rate = sample_rate
        self.chunk_count = 0

        # Auto-detect mic if not specified
        if device is not None:
            self.device = device
        else:
            self.device = self._auto_detect_mic()

        self._running = False
        self._thread = None
        self._buffer = np.zeros(0, dtype='float32')
        self._lock = threading.Lock()

        os.makedirs(self.output_dir, exist_ok=True)

        dev_info = sd.query_devices(self.device)
        print(f"[MicRecorder] Using device [{self.device}]: {dev_info['name']}")
        print(f"[MicRecorder] Chunk duration: {chunk_duration}s, "
              f"Sample rate: {sample_rate}Hz")
        print(f"[MicRecorder] Output dir: {output_dir}/")

    # ------------------------------------------------------------------
    # Auto-detect microphone
    # ------------------------------------------------------------------

    @staticmethod
    def _auto_detect_mic() -> int:
        """
        Auto-detect the wireless microphone.
        
        Priority:
        1. USB audio device (wireless mic's USB receiver)
        2. Any non-built-in input device
        3. Default input device (built-in mic as fallback)
        """
        devices = sd.query_devices()

        # Priority 1: look for USB audio (wireless mic receiver)
        for i, d in enumerate(devices):
            if d['max_input_channels'] > 0:
                name = d['name'].lower()
                if 'usb' in name:
                    print(f"[MicRecorder] Auto-detected USB mic: [{i}] {d['name']}")
                    return i

        # Priority 2: look for any external mic (not built-in, not virtual)
        skip_keywords = ['macbook', 'built-in', 'teams', 'zoom', 'virtual']
        for i, d in enumerate(devices):
            if d['max_input_channels'] > 0:
                name = d['name'].lower()
                if not any(kw in name for kw in skip_keywords):
                    print(f"[MicRecorder] Auto-detected external mic: [{i}] {d['name']}")
                    return i

        # Priority 3: fallback to default input
        default = sd.default.device[0]  # default input device index
        if default is not None and default >= 0:
            dev_info = sd.query_devices(default)
            print(f"[MicRecorder] Using default input: [{default}] {dev_info['name']}")
            return default

        raise RuntimeError(
            "No microphone found. Run MicRecorder.list_devices() to see available devices."
        )

    @staticmethod
    def list_devices():
        """Print all available audio devices."""
        print("\nAvailable audio devices:")
        print("=" * 60)
        devices = sd.query_devices()
        for i, d in enumerate(devices):
            if d['max_input_channels'] > 0:
                print(f"  [{i}] {d['name']}  (INPUT, channels={d['max_input_channels']})")
        print()

    # ------------------------------------------------------------------
    # Save WAV
    # ------------------------------------------------------------------

    def _save_wav(self, filepath: str, data: np.ndarray):
        """Save numpy float32 array as 16-bit mono WAV."""
        audio_int16 = np.int16(np.clip(data, -1.0, 1.0) * 32767)
        with wave.open(filepath, 'w') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio_int16.tobytes())

    # ------------------------------------------------------------------
    # Record single chunk
    # ------------------------------------------------------------------

    def record_one(self) -> str:
        """
        Record a single chunk and return the filepath.
        Blocks until recording is done.
        
        Returns:
            filepath to the saved WAV file
            
        Usage:
            recorder = MicRecorder(output_dir="recordings", chunk_duration=1.0)
            path = recorder.record_one()
            # path = "recordings/chunk_0000.wav"
        """
        samples = int(self.chunk_duration * self.sample_rate)

        audio = sd.rec(
            frames=samples,
            samplerate=self.sample_rate,
            channels=1,
            dtype='float32',
            device=self.device,
        )
        sd.wait()

        filepath = os.path.join(self.output_dir, f"chunk_{self.chunk_count:04d}.wav")
        self._save_wav(filepath, audio.flatten())

        rms = np.sqrt(np.mean(audio ** 2))
        print(f"[MicRecorder] Saved {filepath}  |  RMS={rms:.4f}")

        self.chunk_count += 1
        return filepath

    # ------------------------------------------------------------------
    # Continuous recording (background thread)
    # ------------------------------------------------------------------

    def start(self):
        """
        Start continuous recording in background.
        WAV files appear in output_dir as they are recorded.
        Call stop() to finish.
        
        Usage:
            recorder = MicRecorder(output_dir="recordings")
            recorder.start()
            # ... robot plays cello ...
            recorder.stop()
        """
        if self._running:
            print("[MicRecorder] Already recording.")
            return

        self._running = True
        self._buffer = np.zeros(0, dtype='float32')
        self._thread = threading.Thread(target=self._record_loop, daemon=True)
        self._thread.start()
        print("[MicRecorder] Recording started.")

    def stop(self) -> int:
        """
        Stop continuous recording.
        
        Returns:
            number of chunks recorded
        """
        if not self._running:
            print("[MicRecorder] Not recording.")
            return self.chunk_count

        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None

        print(f"[MicRecorder] Stopped. Total chunks: {self.chunk_count}")
        return self.chunk_count

    def _record_loop(self):
        """Background thread: continuous audio stream, cut into chunks."""
        samples_per_chunk = int(self.chunk_duration * self.sample_rate)

        def callback(indata, frames, time_info, status):
            if status:
                print(f"[MicRecorder] ⚠ {status}")
            with self._lock:
                self._buffer = np.append(self._buffer, indata[:, 0])

        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype='float32',
                device=self.device,
                callback=callback,
                blocksize=1024,
            ):
                while self._running:
                    should_save = False
                    chunk = None

                    with self._lock:
                        if len(self._buffer) >= samples_per_chunk:
                            chunk = self._buffer[:samples_per_chunk].copy()
                            self._buffer = self._buffer[samples_per_chunk:]
                            should_save = True

                    if should_save and chunk is not None:
                        filepath = os.path.join(
                            self.output_dir,
                            f"chunk_{self.chunk_count:04d}.wav"
                        )
                        self._save_wav(filepath, chunk)
                        self.chunk_count += 1
                    else:
                        time.sleep(0.01)

        except Exception as e:
            print(f"[MicRecorder] Error: {e}")
            self._running = False

    # ------------------------------------------------------------------
    # Get latest chunk
    # ------------------------------------------------------------------

    def get_latest(self) -> str:
        """
        Return the filepath of the most recently saved chunk.
        Useful for feeding to the sound classifier.
        
        Returns:
            filepath, or None if no chunks recorded yet
        """
        if self.chunk_count == 0:
            return None
        return os.path.join(self.output_dir, f"chunk_{self.chunk_count - 1:04d}.wav")

    def reset_count(self):
        """Reset chunk counter to 0 (for a new recording session)."""
        self.chunk_count = 0


# ============================================================
# Standalone test
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test MicRecorder")
    parser.add_argument("--list", action="store_true", help="List devices")
    parser.add_argument("--output", type=str, default="recordings")
    parser.add_argument("--duration", type=float, default=1.0)
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument("--mode", choices=["one", "continuous"], default="one",
                        help="'one' = single chunk, 'continuous' = keep recording")
    parser.add_argument("--seconds", type=int, default=5,
                        help="How many seconds to record in continuous mode")
    args = parser.parse_args()

    if args.list:
        MicRecorder.list_devices()
    else:
        recorder = MicRecorder(
            output_dir=args.output,
            chunk_duration=args.duration,
            device=args.device,
        )

        if args.mode == "one":
            path = recorder.record_one()
            print(f"\nDone! Play it: afplay {path}")

        elif args.mode == "continuous":
            recorder.start()
            print(f"\nRecording for {args.seconds} seconds...")
            time.sleep(args.seconds)
            recorder.stop()
            latest = recorder.get_latest()
            if latest:
                print(f"Play latest: afplay {latest}")
