"""
Rolling system-audio capture -> on-demand 30s snapshot -> Whisper transcription.

- Captures at 16 kHz mono directly, so we avoid resampling and stereo downmix later.
- Keeps a 30s ring buffer in RAM.
- On trigger, freezes the last 30s, writes WAV bytes to a temp file, runs faster-whisper, returns text.

Dependencies:
  pip install sounddevice numpy faster-whisper
"""

import io
import wave
import threading
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
import tempfile
from typing import Optional

# ------------------ Config ------------------

TARGET_SR = 16_000     # Whisper-friendly sample rate
CHANNELS  = 1          # mono is fine for speech, and lighter
BUFFER_SEC = 30        # rolling window duration
BLOCKSIZE = 1024       # audio callback frames

# If you need a specific input device (e.g., BlackHole), set it here:
#   sd.default.device = (input_device_index, None)
# Print devices with: print(sd.query_devices())
# sd.default.device = (None, None)

# ------------------ Ring buffer ------------------

class AudioRing:
    """Fixed-size ring buffer storing float32 mono audio at TARGET_SR."""
    def __init__(self, seconds: int, sr: int):
        self.sr = sr
        self.samples = seconds * sr
        self.buf = np.zeros(self.samples, dtype=np.float32)
        self.idx = 0
        self.lock = threading.Lock()

    def write(self, mono_f32: np.ndarray):
        """Append a 1-D float32 array (mono) into the ring."""
        n = mono_f32.shape[0]
        with self.lock:
            end = self.idx + n
            if end <= self.samples:
                self.buf[self.idx:end] = mono_f32
            else:
                k = self.samples - self.idx
                self.buf[self.idx:] = mono_f32[:k]
                self.buf[:n-k] = mono_f32[k:]
            self.idx = (self.idx + n) % self.samples

    def snapshot(self) -> np.ndarray:
        """Return a copy of the last `seconds` worth of samples in time order."""
        with self.lock:
            i = self.idx
            return np.concatenate([self.buf[i:], self.buf[:i]]).copy()

# ------------------ Capture ------------------

class SystemAudioCapture:
    """
    Starts a sounddevice InputStream capturing at TARGET_SR mono.
    Writes into a ring buffer.

    macOS: set the input device to BlackHole (or equivalent).
    Windows: select WASAPI loopback device (sd.query_devices()).
    """

    def __init__(self, ring: AudioRing, sr: int = TARGET_SR, blocksize: int = BLOCKSIZE, channels: int = CHANNELS):
        self.ring = ring
        self.sr = sr
        self.channels = channels
        self.blocksize = blocksize
        self.stream: Optional[sd.InputStream] = None

    def _cb(self, indata, frames, time_info, status):
        if status:
            # Non-fatal, but good to see dropouts etc.
            print(status)
        # indata shape: (frames, channels) float32
        if self.channels == 1:
            mono = indata[:, 0].astype(np.float32)
        else:
            mono = np.mean(indata, axis=1).astype(np.float32)
        self.ring.write(mono)

    def start(self):
        self.stream = sd.InputStream(
            samplerate=self.sr,
            channels=self.channels,
            dtype="float32",
            blocksize=self.blocksize,
            callback=self._cb,
        )
        self.stream.start()

    def stop(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

# ------------------ Snapshot -> WAV bytes ------------------

def float32_to_wav_bytes(mono_f32: np.ndarray, sr: int = TARGET_SR) -> bytes:
    """
    Convert float32 mono [-1, 1] to 16-bit PCM WAV bytes (in-memory).
    """
    pcm = np.clip(mono_f32, -1.0, 1.0)
    pcm16 = (pcm * 32767.0).astype(np.int16)
    buff = io.BytesIO()
    with wave.open(buff, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # int16
        wf.setframerate(sr)
        wf.writeframes(pcm16.tobytes())
    return buff.getvalue()

# ------------------ Whisper transcription ------------------

class LocalWhisper:
    """
    Tiny, fast Whisper wrapper for <=30s clips.
    Loads once; reuses model for all triggers.
    """
    def __init__(self, model_size: str = "tiny", compute_type: str = "int8"):
        # CPU-friendly defaults
        self.model = WhisperModel(model_size, device="cpu", compute_type=compute_type)

    def transcribe_wav_bytes(self, wav_bytes: bytes) -> str:
        # faster-whisper wants a file path; write to a temp file briefly.
        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
            tmp.write(wav_bytes)
            tmp.flush()
            segments, _ = self.model.transcribe(tmp.name, beam_size=1)
        return " ".join(seg.text.strip() for seg in segments)

# ------------------ Wiring it together ------------------

# 1) Create ring + capture and start it
ring = AudioRing(seconds=BUFFER_SEC, sr=TARGET_SR)
cap = SystemAudioCapture(ring, sr=TARGET_SR, channels=CHANNELS)
cap.start()
print(f"Capturing system audio → {BUFFER_SEC}s ring @ {TARGET_SR} Hz mono. (Ctrl+C to stop)")

# 2) Create a Whisper engine once (reuse across triggers)
whisper_engine = LocalWhisper(model_size="tiny", compute_type="int8")

def on_trigger_transcribe_last_30s() -> str:
    """
    Freeze the last 30s, encode to WAV (in memory), run Whisper, return text.
    Call this from your hotkey handler or button click.
    """
    mono = ring.snapshot()
    wav_bytes = float32_to_wav_bytes(mono, sr=TARGET_SR)
    text = whisper_engine.transcribe_wav_bytes(wav_bytes)
    return text

# Example: simple REPL trigger (press Enter to transcribe the last 30s)
if __name__ == "__main__":
    try:
        while True:
            input("Press Enter to transcribe the last 30 seconds...")
            transcript = on_trigger_transcribe_last_30s()
            print("\n--- Transcript ---")
            print(transcript or "(no speech detected)")
            print("------------------\n")
    except KeyboardInterrupt:
        print("Stopping capture...")
        cap.stop()
