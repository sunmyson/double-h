"""

python -m pip install -U pip setuptools wheel
python -m pip install "ctranslate2<5" faster-whisper



System OUTPUT audio -> 30s rolling buffer -> on-demand Whisper transcription.

This version explicitly captures the computer's *output* audio by auto-selecting a
loopback/virtual device per OS (BlackHole on macOS, WASAPI loopback on Windows,
'monitor' sources on Linux).

Install:
  pip install sounddevice numpy faster-whisper
Also:
  macOS: install BlackHole (existential.audio), set Multi-Output (speakers+BlackHole)
  Windows: none (WASAPI loopback exposed automatically)
  Linux: use PulseAudio/PipeWire 'monitor' sources (pavucontrol helps)
"""

import io
import sys
import wave
import threading
import tempfile
from typing import Optional, Tuple

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

# ------------------ Config ------------------

TARGET_SR  = 16_000   # Whisper-friendly
CHANNELS   = 1        # capture mono (downmix in callback if needed)
BUFFER_SEC = 30
BLOCKSIZE  = 1024

# ------------------ Device selection ------------------

def find_system_output_input_device(preferred_name: Optional[str] = None) -> Tuple[int, int]:
    """
    Return (input_device_index, max_input_channels) for a loopback/virtual device
    that captures system OUTPUT audio.

    - macOS: searches for 'BlackHole'/'Loopback'/'Soundflower'
    - Windows: searches for '(loopback)' devices (WASAPI)
    - Linux:  searches for 'monitor' devices (PulseAudio/PipeWire)
    - If preferred_name is given, match that first (case-insensitive substring)
    """
    devs = sd.query_devices()
    name_key = "name"

    # Helper for case-insensitive substring match on device name
    def matches(i: int, needle: str) -> bool:
        try:
            return needle.lower() in devs[i][name_key].lower()
        except Exception:
            return False

    # 1) Preferred name (if provided)
    if preferred_name:
        for i, d in enumerate(devs):
            if d["max_input_channels"] > 0 and matches(i, preferred_name):
                return i, d["max_input_channels"]

    # 2) OS-specific heuristics
    plat = sys.platform
    candidates = []

    if plat == "darwin":  # macOS
        for i, d in enumerate(devs):
            if d["max_input_channels"] > 0 and (
                matches(i, "BlackHole") or matches(i, "Loopback") or matches(i, "Soundflower")
            ):
                candidates.append((i, d["max_input_channels"]))

    elif plat.startswith("win"):  # Windows (WASAPI loopback exposes "(loopback)")
        for i, d in enumerate(devs):
            if d["max_input_channels"] > 0 and matches(i, "(loopback)"):
                candidates.append((i, d["max_input_channels"]))

    else:  # Linux (PulseAudio/PipeWire monitor sources)
        for i, d in enumerate(devs):
            if d["max_input_channels"] > 0 and matches(i, "monitor"):
                candidates.append((i, d["max_input_channels"]))

    if candidates:
        return candidates[0]

    # If we got here, we didn't find a loopback device.
    msg = [
        "Could not find a system-output capture (loopback) device.",
        "Fixes:",
    ]
    if plat == "darwin":
        msg += [
            "- Install BlackHole: https://existential.audio/blackhole/",
            "- In System Settings → Sound:",
            "    • Output: Multi-Output (Your speakers + BlackHole)",
            "    • Input: BlackHole",
            "- Then re-run. (Or pass preferred_name='BlackHole' here.)",
        ]
    elif plat.startswith("win"):
        msg += [
            "- Ensure WASAPI loopback devices are enabled.",
            "- Try selecting a device with '(loopback)' in its name from sd.query_devices().",
        ]
    else:
        msg += [
            "- Use PulseAudio/PipeWire 'monitor' sources (install pavucontrol).",
            "- Select the output's monitor as input.",
        ]
    raise RuntimeError("\n".join(msg))

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
        with self.lock:
            i = self.idx
            return np.concatenate([self.buf[i:], self.buf[:i]]).copy()

# ------------------ Capture ------------------

class SystemOutputCapture:
    """
    Captures *system output* audio by opening an InputStream on a loopback/virtual device.
    Captures at TARGET_SR; downmixes to mono in the callback.
    """
    def __init__(self, ring: AudioRing, device_name: Optional[str] = None):
        self.ring = ring
        self.stream: Optional[sd.InputStream] = None
        self.device_index, self.max_in = find_system_output_input_device(device_name)

    def _cb(self, indata, frames, time_info, status):
        if status:
            print(status)
        # indata: (frames, nch)
        if indata.shape[1] == 1:
            mono = indata[:, 0].astype(np.float32)
        else:
            mono = np.mean(indata, axis=1).astype(np.float32)
        self.ring.write(mono)

    def start(self):
        self.stream = sd.InputStream(
            samplerate=TARGET_SR,
            channels=min(self.max_in, 2),   # handle mono/2ch devices
            dtype="float32",
            blocksize=BLOCKSIZE,
            device=self.device_index,
            callback=self._cb,
        )
        self.stream.start()

    def stop(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

# ------------------ WAV bytes helper ------------------

def float32_to_wav_bytes(mono_f32: np.ndarray, sr: int = TARGET_SR) -> bytes:
    pcm = np.clip(mono_f32, -1.0, 1.0)
    pcm16 = (pcm * 32767.0).astype(np.int16)
    buff = io.BytesIO()
    with wave.open(buff, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # int16
        wf.setframerate(sr)
        wf.writeframes(pcm16.tobytes())
    return buff.getvalue()

# ------------------ Whisper wrapper ------------------

class LocalWhisper:
    def __init__(self, model_size: str = "tiny", compute_type: str = "int8"):
        self.model = WhisperModel(model_size, device="cpu", compute_type=compute_type)

    def transcribe_wav_bytes(self, wav_bytes: bytes) -> str:
        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
            tmp.write(wav_bytes)
            tmp.flush()
            segments, _ = self.model.transcribe(tmp.name, beam_size=1)
        return " ".join(seg.text.strip() for seg in segments)

# ------------------ Wire it up ------------------

ring = AudioRing(seconds=BUFFER_SEC, sr=TARGET_SR)
cap  = SystemOutputCapture(ring, device_name=None)  # or e.g., 'BlackHole'
cap.start()
print(f"[OK] Capturing *system output* → {BUFFER_SEC}s ring @ {TARGET_SR} Hz mono. (Ctrl+C to stop)")

whisper_engine = LocalWhisper(model_size="tiny", compute_type="int8")

def transcribe_last_30s() -> str:
    mono = ring.snapshot()
    wav_bytes = float32_to_wav_bytes(mono, sr=TARGET_SR)
    return whisper_engine.transcribe_wav_bytes(wav_bytes)

# Simple demo loop
if __name__ == "__main__":
    try:
        while True:
            input("Press Enter to transcribe the last 30 seconds of SYSTEM audio...")
            print("\n--- Transcript ---")
            print(transcribe_last_30s() or "(no speech detected)")
            print("------------------\n")
    except KeyboardInterrupt:
        print("Stopping capture...")
        cap.stop()
