"""Local Whisper transcription wrapper (faster-whisper backend)."""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
from faster_whisper import WhisperModel


@dataclass
class TranscriptResult:
    text: str
    language: str
    language_probability: float


class Transcriber:
    def __init__(
        self,
        model_size: str | None = None,
        device: str | None = None,
        compute_type: str | None = None,
        language: str | None = None,
    ) -> None:
        self.language = language or os.getenv("WHISPER_LANGUAGE") or None
        self.model = WhisperModel(
            model_size or os.getenv("WHISPER_MODEL", "small"),
            device=device or os.getenv("WHISPER_DEVICE", "cpu"),
            compute_type=compute_type or os.getenv("WHISPER_COMPUTE_TYPE", "int8"),
        )

    def transcribe(self, audio: np.ndarray) -> TranscriptResult | None:
        """Transcribe mono float32 PCM in [-1, 1] sampled at 16 kHz."""
        if audio.size == 0:
            return None
        segments, info = self.model.transcribe(
            audio,
            language=self.language,
            vad_filter=True,
            beam_size=1,
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()
        if not text:
            return None
        return TranscriptResult(
            text=text,
            language=info.language,
            language_probability=info.language_probability,
        )
