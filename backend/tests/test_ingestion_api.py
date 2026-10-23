from collections.abc import AsyncIterator
from pathlib import Path
from uuid import UUID

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.api.dependencies import get_job_queue, get_session
from app.db.base import Base
from app.db.models import Job, Video
from app.ingestion.queue import QueueUnavailable
from app.main import app


class FakeQueue:
    def __init__(self) -> None:
        self.enqueued_ids: list[UUID] = []
        self.fail = False

    async def enqueue(self, job_id: UUID) -> None:
        self.enqueued_ids.append(job_id)
        if self.fail:
            raise QueueUnavailable("Queue is unavailable")


@pytest.fixture
async def ingestion_client(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[tuple[AsyncClient, FakeQueue, async_sessionmaker[AsyncSession]]]:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    queue = FakeQueue()

    async def override_session() -> AsyncIterator[AsyncSession]:
        async with session_factory() as session:
            yield session

    async def validate_url(url: str) -> None:
        assert url.startswith("https://")

    app.dependency_overrides[get_session] = override_session
    app.dependency_overrides[get_job_queue] = lambda: queue
    monkeypatch.setattr("app.api.dependencies.settings.media_storage_path", tmp_path)
    monkeypatch.setattr("app.api.dependencies.settings.max_video_upload_bytes", 100)
    monkeypatch.setattr("app.ingestion.service.validate_public_http_url", validate_url)
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        yield client, queue, session_factory
    app.dependency_overrides.clear()
    await engine.dispose()


async def test_url_ingestion_creates_video_and_queued_job(
    ingestion_client: tuple[AsyncClient, FakeQueue, async_sessionmaker[AsyncSession]],
) -> None:
    client, queue, _ = ingestion_client
    collection_response = await client.post("/collections", json={"name": "Lectures"})
    collection_id = collection_response.json()["id"]

    response = await client.post(
        "/videos",
        json={
            "collection_id": collection_id,
            "title": "Sample lecture",
            "source_url": "https://example.com/lecture.mp4",
        },
    )

    assert response.status_code == 202
    accepted = response.json()
    assert accepted["status"] == "queued"
    assert UUID(accepted["job_id"]) in queue.enqueued_ids

    job_response = await client.get(f"/jobs/{accepted['job_id']}")
    video_response = await client.get(f"/videos/{accepted['video_id']}")

    assert job_response.status_code == 200
    assert job_response.json()["status"] == "queued"
    assert video_response.status_code == 200
    assert video_response.json()["title"] == "Sample lecture"
    assert video_response.json()["summary"] is None


async def test_upload_ingestion_saves_file_before_enqueue(
    ingestion_client: tuple[AsyncClient, FakeQueue, async_sessionmaker[AsyncSession]],
    tmp_path: Path,
) -> None:
    client, queue, _ = ingestion_client
    collection_response = await client.post("/collections", json={"name": "Uploads"})

    response = await client.post(
        "/videos/upload",
        data={
            "collection_id": collection_response.json()["id"],
            "title": "Uploaded video",
        },
        files={"file": ("lecture.mp4", b"fixture-bytes", "video/mp4")},
    )

    assert response.status_code == 202
    assert list(tmp_path.glob("*/source.mp4"))[0].read_bytes() == b"fixture-bytes"
    assert UUID(response.json()["job_id"]) in queue.enqueued_ids


async def test_queue_failure_marks_video_and_job_failed(
    ingestion_client: tuple[AsyncClient, FakeQueue, async_sessionmaker[AsyncSession]],
) -> None:
    client, queue, session_factory = ingestion_client
    collection_response = await client.post("/collections", json={"name": "Failure case"})
    queue.fail = True

    response = await client.post(
        "/videos",
        json={
            "collection_id": collection_response.json()["id"],
            "title": "Queue failure",
            "source_url": "https://example.com/lecture.mp4",
        },
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "Queue is unavailable"
    async with session_factory() as session:
        job = await session.scalar(select(Job).where(Job.id == queue.enqueued_ids[-1]))
        assert job is not None
        video = await session.get(Video, job.video_id)
        assert video is not None
        assert job.status == "failed"
        assert video.status == "failed"
