from dataclasses import dataclass
from pathlib import Path

import pytest

from app.ingestion.errors import IngestionError
from app.ingestion.schemas import TimedWord
from app.ingestion.transcription import FasterWhisperTranscriber, WhisperModelFactory


@dataclass
class FakeWord:
    start: float | None
    end: float | None
    word: str


@dataclass
class FakeSegment:
    words: list[FakeWord] | None


class FakeModel:
    def __init__(self, segments: list[FakeSegment]) -> None:
        self.segments = segments
        self.calls = 0

    def transcribe(
        self,
        audio_path: str,
        word_timestamps: bool,
        vad_filter: bool,
    ) -> tuple[list[FakeSegment], object]:
        assert word_timestamps
        assert vad_filter
        self.calls += 1
        return self.segments, object()


def _factory(model: FakeModel, calls: list[tuple[str, str, str]]) -> WhisperModelFactory:
    def create(model_name: str, device: str, compute_type: str) -> FakeModel:
        calls.append((model_name, device, compute_type))
        return model

    return create


async def test_transcriber_returns_word_timestamps_and_reuses_model(tmp_path: Path) -> None:
    model = FakeModel([FakeSegment([FakeWord(0.0, 0.4, " Hello"), FakeWord(0.4, 0.8, " world.")])])
    model_initializations: list[tuple[str, str, str]] = []
    transcriber = FasterWhisperTranscriber(
        "small",
        model_factory=_factory(model, model_initializations),
    )

    first_result = await transcriber.transcribe(tmp_path / "first.wav")
    second_result = await transcriber.transcribe(tmp_path / "second.wav")

    assert first_result == [TimedWord(0.0, 0.4, " Hello"), TimedWord(0.4, 0.8, " world.")]
    assert second_result == first_result
    assert model_initializations == [("small", "cpu", "int8")]
    assert model.calls == 2


async def test_transcriber_accepts_speechless_audio(tmp_path: Path) -> None:
    transcriber = FasterWhisperTranscriber(
        "small",
        model_factory=_factory(FakeModel([]), []),
    )

    assert await transcriber.transcribe(tmp_path / "silence.wav") == []


async def test_transcriber_rejects_missing_word_timestamps(tmp_path: Path) -> None:
    transcriber = FasterWhisperTranscriber(
        "small",
        model_factory=_factory(FakeModel([FakeSegment(None)]), []),
    )

    with pytest.raises(IngestionError, match="no word timestamps"):
        await transcriber.transcribe(tmp_path / "audio.wav")
