"""Utilities for capturing the last 30 seconds of system audio and transcribing it.

The capture path favours built-in, OS-provided loopback devices so that Windows
users do not need to install extra audio routing tools. On Windows 10/11 the
code tries WASAPI loopback (available by default). On macOS and Linux a
loopback/monitor device is still required; provide its name via the
``DOUBLE_H_AUDIO_DEVICE`` environment variable if auto-detection fails.
"""

from __future__ import annotations

import atexit
import io
import os
import sys
import tempfile
import threading
import wave
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

try:
    import sounddevice as sd
except OSError as exc:  # pragma: no cover - environment-specific
    raise RuntimeError(
        "The sounddevice package could not load its PortAudio backend.\n"
        "Install PortAudio and reinstall the Python dependency, e.g.:\n"
        "  • Ubuntu/Debian: sudo apt install libportaudio2\n"
        "                 sudo apt install portaudio19-dev  # for building\n"
        "  • macOS (Homebrew): brew install portaudio\n"
        "  • Windows: reinstall sounddevice (pip install --force-reinstall sounddevice)\n"
        "Then reinstall sounddevice inside your virtualenv."
    ) from exc

from faster_whisper import WhisperModel

TARGET_SR = 16_000
BUFFER_SEC = 30
BLOCKSIZE = 1024
DEVICE_ENV_VAR = "DOUBLE_H_AUDIO_DEVICE"


class CaptureSetupError(RuntimeError):
    """Generic audio-capture setup failure."""


class DeviceNotFoundError(CaptureSetupError):
    """Raised when no suitable loopback device can be located."""


@dataclass
class AudioRing:
    seconds: int
    sr: int

    def __post_init__(self) -> None:
        self.samples = self.seconds * self.sr
        self.buf = np.zeros(self.samples, dtype=np.float32)
        self.idx = 0
        self.lock = threading.Lock()

    def write(self, mono_f32: np.ndarray) -> None:
        n = mono_f32.shape[0]
        with self.lock:
            end = self.idx + n
            if end <= self.samples:
                self.buf[self.idx:end] = mono_f32
            else:
                k = self.samples - self.idx
                self.buf[self.idx:] = mono_f32[:k]
                self.buf[:n - k] = mono_f32[k:]
            self.idx = (self.idx + n) % self.samples

    def snapshot(self) -> np.ndarray:
        with self.lock:
            i = self.idx
            return np.concatenate([self.buf[i:], self.buf[:i]]).copy()


class SystemOutputCapture:
    def __init__(self, ring: AudioRing, device_name: Optional[str] = None):
        self.ring = ring
        self._preferred_name = device_name or os.getenv(DEVICE_ENV_VAR)
        self.stream: Optional[sd.InputStream] = None
        self.source_description = "(not started)"
        self._fallback_to_microphone = False

    def start(self) -> None:
        if self.stream is not None:
            return
        stream = self._build_stream()
        stream.start()
        self.stream = stream
        suffix = " (microphone fallback)" if self._fallback_to_microphone else ""
        print(
            f"[OK] Capturing audio from {self.source_description}{suffix} → {BUFFER_SEC}s ring @ {TARGET_SR} Hz."
        )

    def stop(self) -> None:
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

    # -------------------------- stream construction --------------------------

    def _build_stream(self) -> sd.InputStream:
        try:
            return self._build_loopback_stream()
        except DeviceNotFoundError as exc:
            print(f"[WARN] {exc} Falling back to the default microphone.")
            self._fallback_to_microphone = True
            return self._build_microphone_stream()

    def _build_loopback_stream(self) -> sd.InputStream:
        if sys.platform.startswith("win"):
            stream = self._try_windows_loopback()
            if stream:
                return stream
            raise DeviceNotFoundError("No WASAPI loopback playback device was found.")

        device_index, max_in = find_system_output_input_device(self._preferred_name)
        info = sd.query_devices(device_index)
        channels = self._safe_channel_count(max_in)
        self.source_description = f"loopback input '{info['name']}'"
        return sd.InputStream(
            samplerate=TARGET_SR,
            channels=channels,
            dtype="float32",
            blocksize=BLOCKSIZE,
            device=device_index,
            callback=self._callback,
        )

    def _try_windows_loopback(self) -> Optional[sd.InputStream]:
        if not hasattr(sd, "WasapiSettings"):
            raise DeviceNotFoundError(
                "sounddevice was built without WASAPI support; upgrade to 0.4.5+"
            )
        device_index = resolve_output_device_index(self._preferred_name)
        if device_index is None:
            return None
        info = sd.query_devices(device_index)
        channels = self._safe_channel_count(info.get("max_output_channels", 2))
        self.source_description = f"WASAPI loopback '{info['name']}'"
        return sd.InputStream(
            samplerate=TARGET_SR,
            channels=channels,
            dtype="float32",
            blocksize=BLOCKSIZE,
            device=device_index,
            callback=self._callback,
            extra_settings=sd.WasapiSettings(loopback=True),
        )

    def _build_microphone_stream(self) -> sd.InputStream:
        device_index = resolve_input_device_index(self._preferred_name)
        info = sd.query_devices(device_index)
        channels = self._safe_channel_count(info.get("max_input_channels", 1))
        self.source_description = f"microphone '{info['name']}'"
        return sd.InputStream(
            samplerate=TARGET_SR,
            channels=channels,
            dtype="float32",
            blocksize=BLOCKSIZE,
            device=device_index,
            callback=self._callback,
        )

    @staticmethod
    def _safe_channel_count(raw: Optional[int]) -> int:
        try:
            value = int(raw)
        except (TypeError, ValueError):
            value = 1
        return max(1, min(value, 2))

    def _callback(self, indata, frames, time_info, status):  # noqa: D401
        if status:
            print(status)
        if indata.shape[1] == 1:
            mono = indata[:, 0].astype(np.float32)
        else:
            mono = np.mean(indata, axis=1).astype(np.float32)
        self.ring.write(mono)


def resolve_output_device_index(preferred_name: Optional[str]) -> Optional[int]:
    devices = sd.query_devices()
    candidates = []
    for idx, dev in enumerate(devices):
        if dev.get("max_output_channels", 0) <= 0:
            continue
        if preferred_name and preferred_name.lower() not in dev.get("name", "").lower():
            continue
        candidates.append(idx)
    if candidates:
        return candidates[0]

    output_index = _extract_default_device_index("output")
    if output_index is not None:
        return output_index

    for idx, dev in enumerate(devices):
        if dev.get("max_output_channels", 0) > 0:
            return idx
    return None


def resolve_input_device_index(preferred_name: Optional[str]) -> int:
    devices = sd.query_devices()
    candidates = []
    for idx, dev in enumerate(devices):
        if dev.get("max_input_channels", 0) <= 0:
            continue
        if preferred_name and preferred_name.lower() not in dev.get("name", "").lower():
            continue
        candidates.append(idx)
    if candidates:
        return candidates[0]

    input_index = _extract_default_device_index("input")
    if input_index is not None:
        return input_index

    for idx, dev in enumerate(devices):
        if dev.get("max_input_channels", 0) > 0:
            return idx

    hints = ["sounddevice reports no audio input devices."]
    if sys.platform.startswith("linux"):
        hints.extend(
            [
                "Ensure PulseAudio/PipeWire is running and exposes a monitor or microphone input.",
                "Consider installing pavucontrol to enable the output's monitor stream.",
            ]
        )
        if os.getenv("WSL_DISTRO_NAME"):
            hints.append(
                "Audio capture is unavailable inside WSL by default; run the app on the host OS or set up a PulseAudio bridge."
            )
    else:
        hints.append("Connect or enable a microphone/loopback device and try again.")

    raise CaptureSetupError(" ".join(hints))


def find_system_output_input_device(preferred_name: Optional[str]) -> Tuple[int, int]:
    devices = sd.query_devices()

    def matches(name: str) -> bool:
        return preferred_name and preferred_name.lower() in name.lower()

    if preferred_name:
        for idx, dev in enumerate(devices):
            if dev.get("max_input_channels", 0) > 0 and matches(dev.get("name", "")):
                return idx, int(dev["max_input_channels"])

    plat = sys.platform
    def loopback_candidate(name: str) -> bool:
        lname = name.lower()
        if plat == "darwin":
            return any(key in lname for key in ("blackhole", "loopback", "soundflower"))
        if plat.startswith("win"):
            return "(loopback)" in lname
        return "monitor" in lname

    for idx, dev in enumerate(devices):
        if dev.get("max_input_channels", 0) > 0 and loopback_candidate(dev.get("name", "")):
            return idx, int(dev["max_input_channels"])

    raise DeviceNotFoundError(
        "No dedicated loopback input device detected. "
        "Specify one via DOUBLE_H_AUDIO_DEVICE or install an OS loopback driver."
    )


def _extract_default_device_index(kind: str) -> Optional[int]:
    default_device = sd.default.device
    if default_device is None:
        return None

    index: Optional[int]
    if isinstance(default_device, int):
        index = default_device
    elif isinstance(default_device, (tuple, list)):
        position = 0 if kind == "input" else 1
        try:
            value = default_device[position]
        except IndexError:
            value = None
        index = value if isinstance(value, int) else None
    else:
        value = getattr(default_device, kind, None)
        index = value if isinstance(value, int) else None

    if isinstance(index, int) and index >= 0:
        return index
    return None


class LocalWhisper:
    def __init__(self, model_size: str = "tiny", compute_type: str = "int8"):
        self.model = WhisperModel(model_size, device="cpu", compute_type=compute_type)

    def transcribe_wav_bytes(self, wav_bytes: bytes) -> str:
        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
            tmp.write(wav_bytes)
            tmp.flush()
            segments, _ = self.model.transcribe(tmp.name, beam_size=1)
        return " ".join(seg.text.strip() for seg in segments)


_ring: Optional[AudioRing] = None
_capture: Optional[SystemOutputCapture] = None
_whisper: Optional[LocalWhisper] = None
_init_lock = threading.Lock()


def ensure_audio_capture() -> None:
    global _ring, _capture, _whisper
    if _capture and _capture.stream:
        return
    with _init_lock:
        if _capture and _capture.stream:
            return
        if _ring is None:
            _ring = AudioRing(seconds=BUFFER_SEC, sr=TARGET_SR)
        if _capture is None:
            _capture = SystemOutputCapture(_ring)
        _capture.start()
        if _whisper is None:
            _whisper = LocalWhisper(model_size="tiny", compute_type="int8")


def float32_to_wav_bytes(mono_f32: np.ndarray, sr: int = TARGET_SR) -> bytes:
    pcm = np.clip(mono_f32, -1.0, 1.0)
    pcm16 = (pcm * 32767.0).astype(np.int16)
    buff = io.BytesIO()
    with wave.open(buff, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm16.tobytes())
    return buff.getvalue()


def transcribe_last_30s() -> str:
    ensure_audio_capture()
    assert _ring is not None and _whisper is not None
    mono = _ring.snapshot()
    wav_bytes = float32_to_wav_bytes(mono, sr=TARGET_SR)
    return _whisper.transcribe_wav_bytes(wav_bytes)


def _cleanup() -> None:
    if _capture:
        _capture.stop()


atexit.register(_cleanup)


if __name__ == "__main__":
    ensure_audio_capture()
    try:
        while True:
            input("Press Enter to transcribe the last 30 seconds of SYSTEM audio...")
            print("\n--- Transcript ---")
            print(transcribe_last_30s() or "(no speech detected)")
            print("------------------\n")
    except KeyboardInterrupt:  # pragma: no cover - manual exit
        print("Stopping capture...")
        _cleanup()
