from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from uuid import UUID


@dataclass(frozen=True, slots=True)
class RetrievedEvidence:
    id: UUID
    video_id: UUID
    video_title: str
    start_s: float
    end_s: float
    source_type: Literal["transcript", "keyframe"]
    transcript_text: str
    slide_text: str | None
    ocr_text: str | None
    image_path: Path | None
    rank_score: float

    @property
    def quote(self) -> str | None:
        text = " ".join(
            part.strip()
            for part in (self.transcript_text, self.slide_text or "", self.ocr_text or "")
            if part.strip()
        )
        return text or None
