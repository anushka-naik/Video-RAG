from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TimedWord:
    start_s: float
    end_s: float
    text: str


@dataclass(frozen=True, slots=True)
class Sentence:
    start_s: float
    end_s: float
    text: str


@dataclass(frozen=True, slots=True)
class TranscriptSegment:
    start_s: float
    end_s: float
    transcript_text: str
    slide_text: str | None = None

    @property
    def text(self) -> str:
        return self.transcript_text
