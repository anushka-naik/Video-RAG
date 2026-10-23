from collections.abc import Awaitable, Callable
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.db.base import Base
from app.db.models import Chapter, Collection, Job, Keyframe, Segment, Video
from app.ingestion.errors import IngestionError
from app.ingestion.pipeline import ChapterArtifact, IngestionResult, KeyframeArtifact
from app.ingestion.schemas import TranscriptSegment
from app.workers.ingestion import WorkerContext, WorkerSettings, process_video


class SuccessfulPipeline:
    def __init__(self) -> None:
        self.calls = 0

    async def process(
        self,
        video_id: UUID,
        collection_id: UUID,
        title: str,
        source_url: str | None,
        source_path: Path | None,
        video_directory: Path,
        report_progress: Callable[[str, int], Awaitable[None]],
    ) -> IngestionResult:
        self.calls += 1
        await report_progress("transcribe", 30)
        return IngestionResult(
            media_path=video_directory / "media.mp4",
            duration_s=60,
            segments=[TranscriptSegment(0, 30, "First segment.", "slide one")],
            keyframes=[
                KeyframeArtifact(uuid4(), 5, video_directory / "frame.png", "hash", "slide one")
            ],
            chapters=[ChapterArtifact(0, 0, 30, "Opening", "Opening summary.")],
            summary="Video summary.",
        )


class FailedPipeline:
    async def process(
        self,
        video_id: UUID,
        collection_id: UUID,
        title: str,
        source_url: str | None,
        source_path: Path | None,
        video_directory: Path,
        report_progress: Callable[[str, int], Awaitable[None]],
    ) -> IngestionResult:
        raise IngestionError("Transcription failed")


class UnexpectedFailurePipeline:
    async def process(
        self,
        video_id: UUID,
        collection_id: UUID,
        title: str,
        source_url: str | None,
        source_path: Path | None,
        video_directory: Path,
        report_progress: Callable[[str, int], Awaitable[None]],
    ) -> IngestionResult:
        raise RuntimeError("Unexpected provider failure")


def test_arq_worker_registers_ingestion_job() -> None:
    assert WorkerSettings.functions == [process_video]
    assert WorkerSettings.max_jobs == 1


async def _worker_context(
    tmp_path: Path,
    pipeline: SuccessfulPipeline | FailedPipeline,
) -> tuple[
    WorkerContext,
    async_sessionmaker[AsyncSession],
    AsyncEngine,
    UUID,
    UUID,
]:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    collection_id = uuid4()
    video_id = uuid4()
    job_id = uuid4()
    async with session_factory() as session:
        session.add_all(
            [
                Collection(id=collection_id, name="Worker test"),
                Video(
                    id=video_id,
                    collection_id=collection_id,
                    title="Worker video",
                    status="queued",
                ),
                Job(id=job_id, video_id=video_id, stage="queued", status="queued"),
            ]
        )
        await session.commit()
    context: WorkerContext = {
        "session_factory": session_factory,
        "pipeline": pipeline,
        "media_storage_path": tmp_path,
    }
    return context, session_factory, engine, job_id, video_id


async def test_worker_persists_results_and_skips_completed_jobs(tmp_path: Path) -> None:
    pipeline = SuccessfulPipeline()
    context, session_factory, engine, job_id, video_id = await _worker_context(tmp_path, pipeline)

    await process_video(context, str(job_id))
    await process_video(context, str(job_id))

    async with session_factory() as session:
        video = await session.get(Video, video_id)
        job = await session.get(Job, job_id)
        segment_count = await session.scalar(
            select(func.count()).select_from(Segment).where(Segment.video_id == video_id)
        )
        chapter_count = await session.scalar(
            select(func.count()).select_from(Chapter).where(Chapter.video_id == video_id)
        )
        keyframe_count = await session.scalar(
            select(func.count()).select_from(Keyframe).where(Keyframe.video_id == video_id)
        )

    assert pipeline.calls == 1
    assert video is not None and video.status == "ready"
    assert video.summary == "Video summary."
    assert job is not None and job.status == "completed" and job.progress == 100
    assert segment_count == chapter_count == keyframe_count == 1
    await engine.dispose()


async def test_worker_records_ingestion_errors(tmp_path: Path) -> None:
    context, session_factory, engine, job_id, video_id = await _worker_context(
        tmp_path, FailedPipeline()
    )

    with pytest.raises(IngestionError, match="Transcription failed"):
        await process_video(context, str(job_id))

    async with session_factory() as session:
        video = await session.get(Video, video_id)
        job = await session.get(Job, job_id)

    assert video is not None and video.status == "failed"
    assert job is not None and job.status == "failed"
    assert job.error == "Transcription failed"
    await engine.dispose()


async def test_worker_records_unexpected_provider_errors(tmp_path: Path) -> None:
    context, session_factory, engine, job_id, video_id = await _worker_context(
        tmp_path, UnexpectedFailurePipeline()
    )

    with pytest.raises(RuntimeError, match="Unexpected provider failure"):
        await process_video(context, str(job_id))

    async with session_factory() as session:
        video = await session.get(Video, video_id)
        job = await session.get(Job, job_id)

    assert video is not None and video.status == "failed"
    assert job is not None and job.status == "failed"
    assert job.error == "Unexpected provider failure"
    await engine.dispose()
