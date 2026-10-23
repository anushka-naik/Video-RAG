from collections.abc import AsyncIterator
from typing import cast

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db.session import SessionFactory
from app.ingestion.queue import ArqJobQueue, JobQueue
from app.ingestion.service import (
    CollectionService,
    JobService,
    VideoIngestionService,
    VideoService,
)
from app.qa.runtime import QARuntime, create_qa_runtime
from app.qa.service import VideoQuestionService


async def get_session() -> AsyncIterator[AsyncSession]:
    async with SessionFactory() as session:
        yield session


def get_job_queue() -> JobQueue:
    return ArqJobQueue(settings.redis_url)


async def get_qa_runtime(request: Request) -> QARuntime:
    async with request.app.state.qa_runtime_lock:
        runtime = cast(QARuntime | None, request.app.state.qa_runtime)
        if runtime is None:
            runtime = create_qa_runtime(settings)
            request.app.state.qa_runtime = runtime
    return runtime


def get_video_question_service(
    session: AsyncSession = Depends(get_session),
    runtime: QARuntime = Depends(get_qa_runtime),
) -> VideoQuestionService:
    return VideoQuestionService(
        session=session,
        vector_store=runtime.vector_store,
        text_encoder=runtime.text_encoder,
        visual_encoder=runtime.visual_encoder,
        planner=runtime.planner,
        answerer=runtime.answerer,
        media_storage_path=settings.media_storage_path,
    )


def get_collection_service(
    session: AsyncSession = Depends(get_session),
) -> CollectionService:
    return CollectionService(session)


def get_video_ingestion_service(
    session: AsyncSession = Depends(get_session),
    queue: JobQueue = Depends(get_job_queue),
) -> VideoIngestionService:
    return VideoIngestionService(
        session=session,
        queue=queue,
        media_storage_path=settings.media_storage_path,
        maximum_upload_size_bytes=settings.max_video_upload_bytes,
    )


def get_video_service(
    session: AsyncSession = Depends(get_session),
) -> VideoService:
    return VideoService(session)


def get_job_service(
    session: AsyncSession = Depends(get_session),
) -> JobService:
    return JobService(session)
