"""Adapter voice offline tùy chọn; không tự cài hoặc tải model âm thanh."""
from __future__ import annotations

import json
import wave
from pathlib import Path


class VoiceUnavailable(RuntimeError):
    pass


def transcribe_wav(wav_path: str | Path, model_path: str | Path) -> str:
    """Nhận dạng WAV bằng Vosk local; model phải được người dùng tải/cài sẵn."""
    try:
        from vosk import Model, KaldiRecognizer
    except ImportError as exc:
        raise VoiceUnavailable("Cài extra voice: pip install '.[voice]'") from exc
    model_dir = Path(model_path)
    if not model_dir.is_dir():
        raise VoiceUnavailable(f"Không tìm thấy Vosk model local: {model_dir}")
    with wave.open(str(wav_path), "rb") as audio:
        if audio.getnchannels() != 1 or audio.getsampwidth() != 2:
            raise VoiceUnavailable("WAV cần mono PCM 16-bit")
        recognizer = KaldiRecognizer(Model(str(model_dir)), audio.getframerate())
        parts = []
        while True:
            chunk = audio.readframes(4000)
            if not chunk:
                break
            if recognizer.AcceptWaveform(chunk):
                parts.append(json.loads(recognizer.Result()).get("text", ""))
        parts.append(json.loads(recognizer.FinalResult()).get("text", ""))
    return " ".join(x for x in parts if x).strip()


def speak_local(text: str, rate: int = 175) -> None:
    """Đọc văn bản qua TTS hệ thống; pyttsx3 không cần gọi cloud."""
    try:
        import pyttsx3
    except ImportError as exc:
        raise VoiceUnavailable("Cài extra voice: pip install '.[voice]'") from exc
    engine = pyttsx3.init()
    engine.setProperty("rate", max(100, min(int(rate), 240)))
    engine.say(text)
    engine.runAndWait()
    engine.stop()
