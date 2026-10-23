import asyncio
import logging
from pathlib import Path
from typing import NotRequired, TypedDict
from uuid import UUID, uuid5

from arq.connections import RedisSettings
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.config import settings
from app.db.models import Chapter, Job, Keyframe, Segment, Video
from app.ingestion.errors import IngestionError
from app.ingestion.factory import create_ingestion_runtime
from app.ingestion.pipeline import IngestionPipeline, IngestionResult
from app.ingestion.providers import QdrantVectorStore

logger = logging.getLogger(__name__)


class WorkerContext(TypedDict):
    session_factory: async_sessionmaker[AsyncSession]
    pipeline: IngestionPipeline
    media_storage_path: Path
    engine: NotRequired[AsyncEngine]
    vector_store: NotRequired[QdrantVectorStore]


async def on_startup(context: WorkerContext) -> None:
    engine = create_async_engine(settings.database_url, pool_pre_ping=True)
    try:
        pipeline, vector_store = create_ingestion_runtime(settings)
    except Exception:
        await engine.dispose()
        raise
    context["engine"] = engine
    context["session_factory"] = async_sessionmaker(engine, expire_on_commit=False)
    context["pipeline"] = pipeline
    context["media_storage_path"] = settings.media_storage_path
    context["vector_store"] = vector_store


async def on_shutdown(context: WorkerContext) -> None:
    try:
        if "vector_store" in context:
            await asyncio.to_thread(context["vector_store"].close)
    finally:
        if "engine" in context:
            await context["engine"].dispose()


async def process_video(context: WorkerContext, job_id: str) -> None:
    try:
        parsed_job_id = UUID(job_id)
    except ValueError as error:
        raise IngestionError(f"Invalid ingestion job ID: {job_id}") from error

    async with context["session_factory"]() as session:
        job = await session.get(Job, parsed_job_id)
        if job is None:
            raise IngestionError(f"Ingestion job {parsed_job_id} does not exist")
        if job.status == "completed":
            return
        video = await session.get(Video, job.video_id)
        if video is None:
            raise IngestionError(f"Video {job.video_id} for job {parsed_job_id} does not exist")

        job.status = "processing"
        job.error = None
        video.status = "processing"
        await session.commit()

        async def report_progress(stage: str, progress: int) -> None:
            job.stage = stage
            job.progress = max(job.progress, progress)
            await session.commit()

        try:
            result = await context["pipeline"].process(
                video_id=video.id,
                collection_id=video.collection_id,
                title=video.title,
                source_url=video.source_url,
                source_path=Path(video.media_path) if video.media_path is not None else None,
                video_directory=context["media_storage_path"] / str(video.id),
                report_progress=report_progress,
            )
            await _persist_result(session, video, job, result)
        except Exception as error:
            await session.rollback()
            failed_job = await session.get(Job, parsed_job_id)
            failed_video = (
                await session.get(Video, failed_job.video_id) if failed_job is not None else None
            )
            if failed_job is None or failed_video is None:
                raise IngestionError(
                    f"Could not reload job and video after ingestion failure for {parsed_job_id}"
                ) from error
            logger.exception("Video ingestion failed for job %s", parsed_job_id)
            failed_job.status = "failed"
            failed_job.error = str(error)
            failed_video.status = "failed"
            await session.commit()
            raise


async def _persist_result(
    session: AsyncSession,
    video: Video,
    job: Job,
    result: IngestionResult,
) -> None:
    await session.execute(delete(Keyframe).where(Keyframe.video_id == video.id))
    await session.execute(delete(Segment).where(Segment.video_id == video.id))
    await session.execute(delete(Chapter).where(Chapter.video_id == video.id))

    chapter_rows = [
        Chapter(
            id=uuid5(video.id, f"chapter:{chapter.ordinal}"),
            video_id=video.id,
            ordinal=chapter.ordinal,
            start_s=chapter.start_s,
            end_s=chapter.end_s,
            title=chapter.title,
            summary=chapter.summary,
        )
        for chapter in result.chapters
    ]
    session.add_all(chapter_rows)

    segment_rows = []
    for index, segment in enumerate(result.segments):
        midpoint_s = (segment.start_s + segment.end_s) / 2
        chapter_id = next(
            (
                chapter.id
                for chapter in chapter_rows
                if chapter.start_s <= midpoint_s <= chapter.end_s
            ),
            None,
        )
        segment_rows.append(
            Segment(
                id=uuid5(video.id, f"segment:{index}:{segment.start_s:.3f}:{segment.end_s:.3f}"),
                video_id=video.id,
                chapter_id=chapter_id,
                start_s=segment.start_s,
                end_s=segment.end_s,
                transcript_text=segment.transcript_text,
                slide_text=segment.slide_text,
            )
        )
    session.add_all(segment_rows)
    session.add_all(
        [
            Keyframe(
                id=keyframe.id,
                video_id=video.id,
                timestamp_s=keyframe.timestamp_s,
                image_path=str(keyframe.image_path),
                phash=keyframe.phash,
                ocr_text=keyframe.ocr_text,
            )
            for keyframe in result.keyframes
        ]
    )

    video.media_path = str(result.media_path)
    video.duration_s = result.duration_s
    video.summary = result.summary
    video.status = "ready"
    job.stage = "complete"
    job.progress = 100
    job.status = "completed"
    job.error = None
    await session.commit()


class WorkerSettings:
    functions = [process_video]
    on_startup = on_startup
    on_shutdown = on_shutdown
    redis_settings = RedisSettings.from_dsn(settings.redis_url)
    max_jobs = 1
