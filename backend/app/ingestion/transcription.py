import asyncio
from collections.abc import Iterable, Sequence
from importlib import import_module
from pathlib import Path
from typing import Protocol, cast

from app.ingestion.errors import IngestionError
from app.ingestion.schemas import TimedWord


class WhisperWordResult(Protocol):
    start: float | None
    end: float | None
    word: str


class WhisperSegmentResult(Protocol):
    words: Sequence[WhisperWordResult] | None


class WhisperModel(Protocol):
    def transcribe(
        self,
        audio_path: str,
        word_timestamps: bool,
        vad_filter: bool,
    ) -> tuple[Iterable[WhisperSegmentResult], object]: ...


class WhisperModelFactory(Protocol):
    def __call__(self, model_name: str, device: str, compute_type: str) -> WhisperModel: ...


class FasterWhisperTranscriber:
    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        compute_type: str = "int8",
        model_factory: WhisperModelFactory | None = None,
    ) -> None:
        if not model_name.strip():
            raise ValueError("A faster-whisper model name is required")
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.model_factory = model_factory
        self.model: WhisperModel | None = None

    async def transcribe(self, audio_path: Path) -> list[TimedWord]:
        return await asyncio.to_thread(self._transcribe, audio_path)

    def _transcribe(self, audio_path: Path) -> list[TimedWord]:
        model = self._get_model()
        try:
            segments, _ = model.transcribe(
                str(audio_path),
                word_timestamps=True,
                vad_filter=True,
            )
            words: list[TimedWord] = []
            for segment in segments:
                if segment.words is None:
                    raise IngestionError("Transcription returned no word timestamps")
                for word in segment.words:
                    if not word.word.strip():
                        continue
                    if word.start is None or word.end is None:
                        raise IngestionError("Transcription returned a word without timestamps")
                    words.append(TimedWord(float(word.start), float(word.end), word.word))
            return words
        except (OSError, RuntimeError, ValueError) as error:
            raise IngestionError(f"Transcription failed for {audio_path}: {error}") from error

    def _get_model(self) -> WhisperModel:
        if self.model is not None:
            return self.model
        if self.model_factory is None:
            try:
                module = import_module("faster_whisper")
            except ImportError as error:
                raise IngestionError(
                    "faster-whisper is not installed; install the ingestion dependency group"
                ) from error
            self.model_factory = cast(WhisperModelFactory, module.__dict__["WhisperModel"])
        try:
            self.model = self.model_factory(
                self.model_name,
                device=self.device,
                compute_type=self.compute_type,
            )
        except (OSError, RuntimeError, ValueError) as error:
            raise IngestionError(
                f"Could not initialize faster-whisper model {self.model_name}: {error}"
            ) from error
        return self.model
