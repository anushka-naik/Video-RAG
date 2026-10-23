from collections.abc import Sequence
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.db.base import Base
from app.db.models import Collection, Keyframe, Message, Segment, Thread, Video
from app.ingestion.errors import IngestionError
from app.ingestion.providers import (
    IMAGE_VECTOR_COLLECTION,
    TEXT_VECTOR_COLLECTION,
    VectorSearchResult,
)
from app.qa.answering import ABSTENTION_ANSWER, GeneratedAnswer
from app.qa.planning import PlannedQueries
from app.qa.service import VideoNotReady, VideoQuestionService


class FakeVectorStore:
    def __init__(
        self,
        text_hits: list[VectorSearchResult] | None = None,
        visual_hits: list[VectorSearchResult] | None = None,
    ) -> None:
        self.hits = {
            TEXT_VECTOR_COLLECTION: text_hits or [],
            IMAGE_VECTOR_COLLECTION: visual_hits or [],
        }
        self.calls: list[tuple[str, UUID, list[UUID], int]] = []

    def search(
        self,
        collection_name: str,
        vector: list[float],
        collection_id: UUID,
        ready_video_ids: Sequence[UUID],
        limit: int,
    ) -> list[VectorSearchResult]:
        self.calls.append((collection_name, collection_id, list(ready_video_ids), limit))
        return self.hits[collection_name]


class FakeTextEncoder:
    def encode(self, documents: list[str]) -> list[list[float]]:
        assert documents
        return [[0.1, 0.2] for _ in documents]


class FakeVisualEncoder:
    def encode_texts(self, texts: list[str]) -> list[list[float]]:
        assert texts
        return [[0.3, 0.4] for _ in texts]


class FakePlanner:
    def __init__(
        self,
        additional_queries: list[str] | None = None,
        usage: dict[str, int] | None = None,
    ) -> None:
        self.additional_queries = additional_queries or []
        self.usage = usage or {}

    async def plan(self, question: str) -> PlannedQueries:
        return PlannedQueries(
            queries=[question, *self.additional_queries],
            usage=self.usage,
        )


class FakeAnswerer:
    def __init__(self, answer: GeneratedAnswer | None = None) -> None:
        self.generated = answer or GeneratedAnswer(
            answer="The lecturer defines a vector as a list of values.",
            citation_ids=[],
            abstain=False,
            usage={"input_tokens": 10, "output_tokens": 5},
        )
        self.calls = 0

    async def answer(self, question: str, evidence: Sequence[object]) -> GeneratedAnswer:
        self.calls += 1
        return self.generated


async def _database(
    tmp_path: Path,
) -> tuple[
    async_sessionmaker[AsyncSession],
    AsyncEngine,
    UUID,
    UUID,
    UUID,
    UUID,
    Path,
]:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    collection_id, video_id, segment_id, keyframe_id = uuid4(), uuid4(), uuid4(), uuid4()
    image_path = tmp_path / "frame.jpg"
    image_path.write_bytes(b"frame")
    async with factory() as session:
        session.add_all(
            [
                Collection(id=collection_id, name="QA test"),
                Video(
                    id=video_id,
                    collection_id=collection_id,
                    title="Vector lecture",
                    status="ready",
                    duration_s=120,
                ),
                Segment(
                    id=segment_id,
                    video_id=video_id,
                    start_s=12,
                    end_s=20,
                    transcript_text="A vector is a list of values.",
                    slide_text="Vectors",
                ),
                Keyframe(
                    id=keyframe_id,
                    video_id=video_id,
                    timestamp_s=15,
                    image_path=str(image_path),
                    phash="abc",
                    ocr_text="Vectors are ordered values.",
                ),
            ]
        )
        await session.commit()
    return factory, engine, collection_id, video_id, segment_id, keyframe_id, image_path


async def test_question_service_fuses_sources_and_persists_citations(tmp_path: Path) -> None:
    (
        factory,
        engine,
        collection_id,
        video_id,
        segment_id,
        keyframe_id,
        _,
    ) = await _database(tmp_path)
    store = FakeVectorStore(
        text_hits=[VectorSearchResult(segment_id, 0.8, {})],
        visual_hits=[VectorSearchResult(keyframe_id, 0.7, {})],
    )
    answerer = FakeAnswerer(
        GeneratedAnswer(
            answer="The lecturer defines a vector as a list of values.",
            citation_ids=[str(segment_id), str(keyframe_id)],
            abstain=False,
            usage={"input_tokens": 20, "output_tokens": 8},
        )
    )
    planner = FakePlanner(
        ["vector definition"],
        usage={"input_tokens": 3, "output_tokens": 2},
    )
    async with factory() as session:
        service = VideoQuestionService(
            session,
            store,
            FakeTextEncoder(),
            FakeVisualEncoder(),
            planner,
            answerer,
            tmp_path,
        )
        response = await service.ask(
            "What is a vector?",
            collection_id,
            None,
            None,
            5,
        )

    assert response.answer == "The lecturer defines a vector as a list of values."
    assert [citation.evidence_id for citation in response.citations] == [
        segment_id,
        keyframe_id,
    ]
    assert response.citations[0].start_s == 12
    assert response.citations[1].source_type == "keyframe"
    assert len(store.calls) == 4
    assert all(call[1] == collection_id and call[2] == [video_id] for call in store.calls)
    assert answerer.calls == 1
    assert response.usage == {"input_tokens": 23, "output_tokens": 10}
    async with factory() as session:
        thread = await session.get(Thread, response.thread_id)
        messages = list(
            await session.scalars(select(Message).where(Message.thread_id == response.thread_id))
        )
    assert thread is not None and thread.scope == f"collection:{collection_id}"
    assert len(messages) == 2
    assistant_message = next(message for message in messages if message.role == "assistant")
    assert assistant_message.content == response.answer
    assert len(assistant_message.citations) == 2
    await engine.dispose()


async def test_question_service_abstains_when_no_ready_evidence_exists(tmp_path: Path) -> None:
    factory, engine, collection_id, _, _, _, _ = await _database(tmp_path)
    answerer = FakeAnswerer()
    async with factory() as session:
        service = VideoQuestionService(
            session,
            FakeVectorStore(),
            FakeTextEncoder(),
            FakeVisualEncoder(),
            FakePlanner(),
            answerer,
            tmp_path,
        )
        response = await service.ask("What is discussed?", collection_id, None, None, 5)

    assert response.answer == ABSTENTION_ANSWER
    assert response.citations == []
    assert response.usage == {}
    assert answerer.calls == 0
    await engine.dispose()


async def test_question_service_rejects_non_ready_video(tmp_path: Path) -> None:
    factory, engine, collection_id, video_id, _, _, _ = await _database(tmp_path)
    async with factory() as session:
        video = await session.get(Video, video_id)
        assert video is not None
        video.status = "processing"
        await session.commit()
        service = VideoQuestionService(
            session,
            FakeVectorStore(),
            FakeTextEncoder(),
            FakeVisualEncoder(),
            FakePlanner(),
            FakeAnswerer(),
            tmp_path,
        )
        with pytest.raises(VideoNotReady):
            await service.ask("What is discussed?", None, video_id, None, 5)
    await engine.dispose()


async def test_question_service_rejects_citations_outside_retrieved_evidence(
    tmp_path: Path,
) -> None:
    factory, engine, collection_id, _, segment_id, _, _ = await _database(tmp_path)
    answerer = FakeAnswerer(
        GeneratedAnswer(
            answer="Unsupported answer.",
            citation_ids=[str(uuid4())],
            abstain=False,
            usage={},
        )
    )
    store = FakeVectorStore(text_hits=[VectorSearchResult(segment_id, 0.8, {})])
    async with factory() as session:
        service = VideoQuestionService(
            session,
            store,
            FakeTextEncoder(),
            FakeVisualEncoder(),
            FakePlanner(),
            answerer,
            tmp_path,
        )
        with pytest.raises(IngestionError, match="not retrieved"):
            await service.ask("What is a vector?", collection_id, None, None, 5)
        await session.rollback()
    await engine.dispose()


async def test_question_service_uses_canonical_abstention_for_model_abstain(
    tmp_path: Path,
) -> None:
    factory, engine, collection_id, _, segment_id, _, _ = await _database(tmp_path)
    answerer = FakeAnswerer(
        GeneratedAnswer(
            answer="I am not sure.",
            citation_ids=[],
            abstain=True,
            usage={"input_tokens": 2, "output_tokens": 3},
        )
    )
    store = FakeVectorStore(text_hits=[VectorSearchResult(segment_id, 0.8, {})])
    async with factory() as session:
        service = VideoQuestionService(
            session,
            store,
            FakeTextEncoder(),
            FakeVisualEncoder(),
            FakePlanner(),
            answerer,
            tmp_path,
        )
        response = await service.ask("What is a vector?", collection_id, None, None, 5)

    assert response.answer == ABSTENTION_ANSWER
    assert response.citations == []
    assert response.usage == {"input_tokens": 2, "output_tokens": 3}
    await engine.dispose()


async def test_question_service_can_capture_evaluation_trace_without_persisting(
    tmp_path: Path,
) -> None:
    factory, engine, _, video_id, segment_id, _, _ = await _database(tmp_path)
    store = FakeVectorStore(text_hits=[VectorSearchResult(segment_id, 0.8, {})])
    answerer = FakeAnswerer(
        GeneratedAnswer(
            answer="The lecturer defines a vector as a list of values.",
            citation_ids=[str(segment_id)],
            abstain=False,
            usage={},
        )
    )
    async with factory() as session:
        service = VideoQuestionService(
            session,
            store,
            FakeTextEncoder(),
            FakeVisualEncoder(),
            FakePlanner(),
            answerer,
            tmp_path,
        )
        response, evidence = await service.ask_with_trace(
            "What is a vector?",
            None,
            video_id,
            None,
            5,
            persist=False,
        )

    assert len(evidence) == 1
    async with factory() as session:
        assert await session.get(Thread, response.thread_id) is None
        messages = list(
            await session.scalars(select(Message).where(Message.thread_id == response.thread_id))
        )
    assert messages == []
    await engine.dispose()
