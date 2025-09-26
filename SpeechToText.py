"""
Fast local speech-to-text for short (<~30s) MP3 clips using faster-whisper.

Dependencies:
    pip install faster-whisper
    # (Optional) ffmpeg must be installed on your system to decode MP3:
    # macOS: brew install ffmpeg
    # Ubuntu: sudo apt-get install ffmpeg
"""

from faster_whisper import WhisperModel

def transcribe_mp3_fast_local(file_path: str,
                              model_size: str = "tiny",
                              compute_type: str = "int8") -> str:
    """
    Transcribe an MP3 (up to ~30s) to text quickly on CPU.

    Parameters
    ----------
    file_path : str
        Path to the MP3 file (speech audio).
    model_size : str, optional
        Whisper checkpoint size. Use "tiny" or "base" for speed (default "tiny").
    compute_type : str, optional
        CTranslate2 compute type. For CPU-only machines, prefer "int8" or "int8_float32".
        Avoid "int8_float16" on CPUs that lack fast float16 support.

    Returns
    -------
    transcript : str
        Concatenated transcription text.
    """
    # device="cpu" ensures we don’t try to use unavailable FP16 paths on CPU-only systems.
    model = WhisperModel(model_size, device="cpu", compute_type=compute_type)

    # Low-latency settings: beam_size=1; keep defaults otherwise.
    segments, _ = model.transcribe(file_path, beam_size=1)

    # Join all segment texts
    return " ".join(seg.text.strip() for seg in segments)


if __name__ == "__main__":
    print(transcribe_mp3_fast_local("ExampleAudio.mp3"))
