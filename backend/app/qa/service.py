import asyncio
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Literal, Protocol
from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Collection, Keyframe, Message, Segment, Thread, Video
from app.ingestion.errors import IngestionError
from app.ingestion.providers import (
    IMAGE_VECTOR_COLLECTION,
    TEXT_VECTOR_COLLECTION,
    VectorSearchResult,
)
from app.qa.answering import ABSTENTION_ANSWER, Answerer, GeneratedAnswer
from app.qa.evidence import RetrievedEvidence
from app.qa.planning import PlannedQueries, QueryPlanner
from app.qa.schemas import ChatResponse, Citation

logger = logging.getLogger(__name__)
RRF_CONSTANT = 60


class CollectionNotFound(LookupError):
    pass


class VideoNotFound(LookupError):
    pass


class VideoNotReady(RuntimeError):
    pass


class ThreadNotFound(LookupError):
    pass


class ThreadScopeMismatch(ValueError):
    pass


class SearchVectorStore(Protocol):
    def search(
        self,
        collection_name: str,
        vector: list[float],
        collection_id: UUID,
        ready_video_ids: Sequence[UUID],
        limit: int,
    ) -> list[VectorSearchResult]: ...


class TextQueryEncoder(Protocol):
    def encode(self, documents: list[str]) -> list[list[float]]: ...


class VisualQueryEncoder(Protocol):
    def encode_texts(self, texts: list[str]) -> list[list[float]]: ...


@dataclass(slots=True)
class _Candidate:
    point_id: UUID
    source_type: Literal["transcript", "keyframe"]
    rank_score: float


class VideoQuestionService:
    def __init__(
        self,
        session: AsyncSession,
        vector_store: SearchVectorStore,
        text_encoder: TextQueryEncoder,
        visual_encoder: VisualQueryEncoder,
        planner: QueryPlanner,
        answerer: Answerer,
        media_storage_path: Path,
    ) -> None:
        self.session = session
        self.vector_store = vector_store
        self.text_encoder = text_encoder
        self.visual_encoder = visual_encoder
        self.planner = planner
        self.answerer = answerer
        self.media_storage_path = media_storage_path

    async def ask(
        self,
        question: str,
        collection_id: UUID | None,
        video_id: UUID | None,
        thread_id: UUID | None,
        top_k: int,
    ) -> ChatResponse:
        response, _ = await self.ask_with_trace(
            question,
            collection_id,
            video_id,
            thread_id,
            top_k,
        )
        return response

    async def ask_with_trace(
        self,
        question: str,
        collection_id: UUID | None,
        video_id: UUID | None,
        thread_id: UUID | None,
        top_k: int,
        persist: bool = True,
    ) -> tuple[ChatResponse, list[RetrievedEvidence]]:
        normalized_question = question.strip()
        if not normalized_question:
            raise ValueError("Question cannot be empty")
        if (collection_id is None) == (video_id is None):
            raise ValueError("Provide exactly one of collection_id or video_id")
        if not 1 <= top_k <= 10:
            raise ValueError("top_k must be between 1 and 10")

        collection_id, ready_videos, scope = await self._resolve_scope(collection_id, video_id)
        thread = await self._resolve_thread(thread_id, scope)
        query_plan = (
            await self.planner.plan(normalized_question)
            if ready_videos
            else PlannedQueries(
                queries=[],
                usage={},
            )
        )
        evidence = (
            await self._retrieve(
                query_plan.queries,
                collection_id,
                ready_videos,
                top_k,
            )
            if ready_videos
            else []
        )
        generated = (
            await self.answerer.answer(normalized_question, evidence)
            if evidence
            else GeneratedAnswer(
                answer=ABSTENTION_ANSWER,
                citation_ids=[],
                abstain=True,
                usage={},
            )
        )
        if generated.abstain:
            generated = GeneratedAnswer(
                answer=ABSTENTION_ANSWER,
                citation_ids=[],
                abstain=True,
                usage=generated.usage,
            )
        usage = _merge_usage(query_plan.usage, generated.usage)
        generated = GeneratedAnswer(
            answer=generated.answer,
            citation_ids=generated.citation_ids,
            abstain=generated.abstain,
            usage=usage,
        )
        citations = _validate_answer(generated, evidence)

        thread_id_value = thread.id
        assistant_message_id = uuid4()
        if persist:
            self.session.add(
                Message(
                    id=uuid4(),
                    thread_id=thread_id_value,
                    role="user",
                    content=normalized_question,
                    citations=[],
                    usage={},
                )
            )
            self.session.add(
                Message(
                    id=assistant_message_id,
                    thread_id=thread_id_value,
                    role="assistant",
                    content=generated.answer,
                    citations=[citation.model_dump(mode="json") for citation in citations],
                    usage=generated.usage,
                )
            )
            await self.session.commit()
        return (
            ChatResponse(
                thread_id=thread_id_value,
                assistant_message_id=assistant_message_id,
                answer=generated.answer,
                citations=citations,
                usage=generated.usage,
            ),
            evidence,
        )

    async def _resolve_scope(
        self,
        collection_id: UUID | None,
        video_id: UUID | None,
    ) -> tuple[UUID, list[Video], str]:
        if video_id is not None:
            video = await self.session.get(Video, video_id)
            if video is None:
                raise VideoNotFound(f"Video {video_id} does not exist")
            if video.status != "ready":
                raise VideoNotReady(f"Video {video_id} is not ready for questions")
            return video.collection_id, [video], f"video:{video.id}"
        if collection_id is None:
            raise ValueError("A collection or video scope is required")
        collection = await self.session.get(Collection, collection_id)
        if collection is None:
            raise CollectionNotFound(f"Collection {collection_id} does not exist")
        result = await self.session.scalars(
            select(Video)
            .where(Video.collection_id == collection_id, Video.status == "ready")
            .order_by(Video.created_at, Video.id)
        )
        ready_videos = list(result)
        return collection_id, ready_videos, f"collection:{collection_id}"

    async def _resolve_thread(self, thread_id: UUID | None, scope: str) -> Thread:
        if thread_id is None:
            thread = Thread(id=uuid4(), scope=scope)
            self.session.add(thread)
            await self.session.flush()
            return thread
        existing_thread = await self.session.get(Thread, thread_id)
        if existing_thread is None:
            raise ThreadNotFound(f"Thread {thread_id} does not exist")
        if existing_thread.scope != scope:
            raise ThreadScopeMismatch("Thread scope does not match the requested video scope")
        return existing_thread

    async def _retrieve(
        self,
        queries: Sequence[str],
        collection_id: UUID,
        ready_videos: Sequence[Video],
        top_k: int,
    ) -> list[RetrievedEvidence]:
        if not ready_videos:
            return []
        ready_video_ids = [video.id for video in ready_videos]
        titles = {video.id: video.title for video in ready_videos}
        text_vectors, visual_vectors = await asyncio.gather(
            asyncio.to_thread(self.text_encoder.encode, list(queries)),
            asyncio.to_thread(self.visual_encoder.encode_texts, list(queries)),
        )
        _validate_embedding_count(text_vectors, len(queries), "text")
        _validate_embedding_count(visual_vectors, len(queries), "visual")
        text_rankings, visual_rankings = await asyncio.gather(
            asyncio.gather(
                *(
                    asyncio.to_thread(
                        self.vector_store.search,
                        TEXT_VECTOR_COLLECTION,
                        vector,
                        collection_id,
                        ready_video_ids,
                        top_k,
                    )
                    for vector in text_vectors
                )
            ),
            asyncio.gather(
                *(
                    asyncio.to_thread(
                        self.vector_store.search,
                        IMAGE_VECTOR_COLLECTION,
                        vector,
                        collection_id,
                        ready_video_ids,
                        top_k,
                    )
                    for vector in visual_vectors
                )
            ),
        )
        candidates = _fuse_rankings(text_rankings, visual_rankings)
        segment_ids = [item.point_id for item in candidates if item.source_type == "transcript"]
        keyframe_ids = [item.point_id for item in candidates if item.source_type == "keyframe"]
        segments = {
            segment.id: segment
            for segment in await self.session.scalars(
                select(Segment).where(
                    Segment.id.in_(segment_ids),
                    Segment.video_id.in_(ready_video_ids),
                )
            )
        }
        keyframes = {
            frame.id: frame
            for frame in await self.session.scalars(
                select(Keyframe).where(
                    Keyframe.id.in_(keyframe_ids),
                    Keyframe.video_id.in_(ready_video_ids),
                )
            )
        }
        evidence: list[RetrievedEvidence] = []
        for candidate in candidates:
            if len(evidence) >= top_k:
                break
            if candidate.source_type == "transcript":
                segment = segments.get(candidate.point_id)
                if segment is None:
                    logger.warning("Ignoring stale transcript vector %s", candidate.point_id)
                    continue
                title = titles.get(segment.video_id)
                if title is None:
                    logger.warning("Ignoring transcript from non-ready video %s", segment.video_id)
                    continue
                item = RetrievedEvidence(
                    id=segment.id,
                    video_id=segment.video_id,
                    video_title=title,
                    start_s=segment.start_s,
                    end_s=segment.end_s,
                    source_type="transcript",
                    transcript_text=segment.transcript_text,
                    slide_text=segment.slide_text,
                    ocr_text=None,
                    image_path=None,
                    rank_score=candidate.rank_score,
                )
                if item.quote:
                    evidence.append(item)
                continue

            keyframe = keyframes.get(candidate.point_id)
            if keyframe is None:
                logger.warning("Ignoring stale keyframe vector %s", candidate.point_id)
                continue
            title = titles.get(keyframe.video_id)
            if title is None:
                logger.warning("Ignoring keyframe from non-ready video %s", keyframe.video_id)
                continue
            image_path = self._resolve_image_path(Path(keyframe.image_path))
            if image_path is None and not keyframe.ocr_text:
                logger.warning(
                    "Ignoring keyframe without an available image or OCR: %s",
                    keyframe.id,
                )
                continue
            if image_path is None:
                logger.warning("Using OCR-only evidence for keyframe %s", keyframe.id)
            evidence.append(
                RetrievedEvidence(
                    id=keyframe.id,
                    video_id=keyframe.video_id,
                    video_title=title,
                    start_s=keyframe.timestamp_s,
                    end_s=keyframe.timestamp_s,
                    source_type="keyframe",
                    transcript_text="",
                    slide_text=None,
                    ocr_text=keyframe.ocr_text,
                    image_path=image_path,
                    rank_score=candidate.rank_score,
                )
            )
            if len(evidence) == top_k:
                break
        return evidence

    def _resolve_image_path(self, stored_path: Path) -> Path | None:
        storage_root = self.media_storage_path.resolve()
        image_path = stored_path if stored_path.is_absolute() else storage_root / stored_path
        resolved_path = image_path.resolve()
        if not resolved_path.is_relative_to(storage_root):
            logger.warning("Ignoring keyframe path outside media storage: %s", resolved_path)
            return None
        return resolved_path if resolved_path.is_file() else None


def _validate_embedding_count(
    vectors: Sequence[Sequence[float]],
    expected_count: int,
    modality: str,
) -> None:
    if len(vectors) != expected_count or any(not vector for vector in vectors):
        raise IngestionError(
            f"{modality.capitalize()} query encoder returned {len(vectors)} vectors "
            f"for {expected_count} search queries"
        )
    if not all(isfinite(float(value)) for vector in vectors for value in vector):
        raise IngestionError(f"{modality.capitalize()} query encoder returned non-finite values")


def _fuse_rankings(
    text_rankings: Sequence[Sequence[VectorSearchResult]],
    visual_rankings: Sequence[Sequence[VectorSearchResult]],
) -> list[_Candidate]:
    candidates: dict[UUID, _Candidate] = {}
    _add_rankings(candidates, "transcript", text_rankings)
    _add_rankings(candidates, "keyframe", visual_rankings)
    return sorted(candidates.values(), key=lambda item: (-item.rank_score, str(item.point_id)))


def _add_rankings(
    candidates: dict[UUID, _Candidate],
    source_type: Literal["transcript", "keyframe"],
    rankings: Sequence[Sequence[VectorSearchResult]],
) -> None:
    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            candidate = candidates.get(hit.point_id)
            if candidate is None:
                candidate = _Candidate(hit.point_id, source_type, 0.0)
                candidates[hit.point_id] = candidate
            candidate.rank_score += 1 / (RRF_CONSTANT + rank)


def _merge_usage(*usages: dict[str, int]) -> dict[str, int]:
    totals: dict[str, int] = {}
    for usage in usages:
        for name, count in usage.items():
            totals[name] = totals.get(name, 0) + count
    return totals


def _validate_answer(
    answer: GeneratedAnswer,
    evidence: Sequence[RetrievedEvidence],
) -> list[Citation]:
    if answer.abstain:
        if answer.citation_ids:
            raise IngestionError("An abstaining answer must not include citations")
        return []
    if not answer.answer.strip():
        raise IngestionError("Question answering returned an empty answer")
    evidence_by_id = {str(item.id): item for item in evidence}
    if not answer.citation_ids:
        raise IngestionError("An answer based on video evidence must include citations")
    if len(answer.citation_ids) != len(set(answer.citation_ids)):
        raise IngestionError("Question answering returned duplicate citation IDs")
    citations: list[Citation] = []
    for evidence_id in answer.citation_ids:
        item = evidence_by_id.get(evidence_id)
        if item is None:
            raise IngestionError(
                f"Question answering cited evidence that was not retrieved: {evidence_id}"
            )
        citations.append(
            Citation(
                evidence_id=item.id,
                video_id=item.video_id,
                video_title=item.video_title,
                start_s=item.start_s,
                end_s=item.end_s,
                source_type=item.source_type,
                quote=item.quote,
            )
        )
    return citations
