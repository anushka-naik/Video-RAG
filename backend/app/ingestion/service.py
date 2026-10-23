import logging
from collections.abc import AsyncIterable
from pathlib import Path
from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Collection, Job, Video
from app.ingestion.media import save_upload, validate_public_http_url
from app.ingestion.queue import JobQueue, QueueUnavailable

logger = logging.getLogger(__name__)
SUPPORTED_UPLOAD_SUFFIXES = {
    ".avi",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp4",
    ".mpeg",
    ".mpg",
    ".ts",
    ".webm",
}


class CollectionNotFound(LookupError):
    pass


class VideoIngestionService:
    def __init__(
        self,
        session: AsyncSession,
        queue: JobQueue,
        media_storage_path: Path,
        maximum_upload_size_bytes: int,
    ) -> None:
        self.session = session
        self.queue = queue
        self.media_storage_path = media_storage_path
        self.maximum_upload_size_bytes = maximum_upload_size_bytes

    async def create_video(
        self,
        collection_id: UUID,
        title: str,
        source_url: str | None,
        upload_chunks: AsyncIterable[bytes] | None,
        upload_filename: str | None,
    ) -> tuple[Video, Job]:
        has_url = source_url is not None and bool(source_url.strip())
        has_upload = upload_chunks is not None
        if has_url == has_upload:
            raise ValueError("Provide exactly one of source_url or an uploaded video file")
        normalized_url = source_url.strip() if source_url is not None else None
        if normalized_url is not None:
            await validate_public_http_url(normalized_url)

        collection = await self.session.get(Collection, collection_id)
        if collection is None:
            raise CollectionNotFound(f"Collection {collection_id} does not exist")
        normalized_title = title.strip()
        if not normalized_title:
            raise ValueError("Video title cannot be empty")

        video_id = uuid4()
        job_id = uuid4()
        upload_path: Path | None = None
        if upload_chunks is not None:
            suffix = Path(upload_filename or "").suffix.lower()
            if suffix and suffix not in SUPPORTED_UPLOAD_SUFFIXES:
                raise ValueError(f"Unsupported uploaded video extension: {suffix}")
            suffix = suffix or ".upload"
            upload_path = self.media_storage_path / str(video_id) / f"source{suffix}"
            await save_upload(upload_chunks, upload_path, self.maximum_upload_size_bytes)

        video = Video(
            id=video_id,
            collection_id=collection_id,
            title=normalized_title,
            source_url=normalized_url,
            media_path=str(upload_path) if upload_path is not None else None,
            status="queued",
        )
        job = Job(id=job_id, video_id=video_id, stage="queued", progress=0, status="queued")
        self.session.add_all([video, job])
        persisted = False
        try:
            await self.session.commit()
            persisted = True
        finally:
            if not persisted and upload_path is not None:
                try:
                    upload_path.unlink(missing_ok=True)
                except OSError:
                    logger.exception("Failed to remove an unpersisted video upload")

        try:
            await self.queue.enqueue(job_id)
        except QueueUnavailable as error:
            job.status = "failed"
            job.error = str(error)
            video.status = "failed"
            await self.session.commit()
            raise

        return video, job


class CollectionService:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create_collection(self, name: str, description: str | None) -> Collection:
        normalized_name = name.strip()
        if not normalized_name:
            raise ValueError("Collection name cannot be empty")
        collection = Collection(name=normalized_name, description=description)
        self.session.add(collection)
        await self.session.commit()
        await self.session.refresh(collection)
        return collection


class JobService:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get_job(self, job_id: UUID) -> Job | None:
        result = await self.session.execute(select(Job).where(Job.id == job_id))
        return result.scalar_one_or_none()


class VideoService:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get_video(self, video_id: UUID) -> Video | None:
        return await self.session.get(Video, video_id)
