from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class CollectionCreate(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    description: str | None = None


class VideoCreate(BaseModel):
    collection_id: UUID
    title: str = Field(min_length=1, max_length=500)
    source_url: str = Field(min_length=1)


class CollectionDetails(BaseModel):
    id: UUID
    name: str
    description: str | None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class IngestionAccepted(BaseModel):
    video_id: UUID
    job_id: UUID
    status: Literal["queued", "processing", "ready", "failed"]


class JobDetails(BaseModel):
    id: UUID
    video_id: UUID
    stage: str
    progress: int
    status: Literal["queued", "processing", "completed", "failed"]
    error: str | None

    model_config = ConfigDict(from_attributes=True)


class VideoDetails(BaseModel):
    id: UUID
    collection_id: UUID
    title: str
    source_url: str | None
    duration_s: float | None
    summary: str | None
    status: Literal["queued", "processing", "ready", "failed"]
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
