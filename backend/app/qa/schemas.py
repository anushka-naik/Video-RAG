from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field, model_validator


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=10_000)
    collection_id: UUID | None = None
    video_id: UUID | None = None
    thread_id: UUID | None = None
    top_k: int = Field(default=5, ge=1, le=10)

    @model_validator(mode="after")
    def validate_scope(self) -> ChatRequest:
        if (self.collection_id is None) == (self.video_id is None):
            raise ValueError("Provide exactly one of collection_id or video_id")
        if not self.question.strip():
            raise ValueError("Question cannot be empty")
        return self


class Citation(BaseModel):
    evidence_id: UUID
    video_id: UUID
    video_title: str
    start_s: float
    end_s: float
    source_type: Literal["transcript", "keyframe"]
    quote: str | None


class ChatResponse(BaseModel):
    thread_id: UUID
    assistant_message_id: UUID
    answer: str
    citations: list[Citation]
    usage: dict[str, int]
