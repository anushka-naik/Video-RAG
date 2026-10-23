from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import pytest

from app.ingestion.errors import IngestionError
from app.ingestion.pipeline import KeyframeArtifact
from app.ingestion.providers import (
    IMAGE_VECTOR_COLLECTION,
    TEXT_VECTOR_COLLECTION,
    QdrantVideoIndexer,
    VectorPoint,
)
from app.ingestion.schemas import TranscriptSegment
from app.ingestion.summarization import (
    AnthropicChapterSummarizer,
    AnthropicClientFactory,
)


class FakeTextEncoder:
    def encode(self, documents: list[str]) -> list[list[float]]:
        return [[float(len(document)), 1.0] for document in documents]


class FakeImageEncoder:
    def encode(self, image_paths: list[Path]) -> list[list[float]]:
        return [[float(index), 1.0] for index, _ in enumerate(image_paths)]


class FakeVectorStore:
    def __init__(self) -> None:
        self.deleted: list[tuple[str, object]] = []
        self.collections: dict[str, int] = {}
        self.points: dict[str, list[VectorPoint]] = {}

    def ensure_collection(self, collection_name: str, vector_size: int) -> None:
        self.collections[collection_name] = vector_size

    def delete_video(self, collection_name: str, video_id: object) -> None:
        self.deleted.append((collection_name, video_id))
        self.points[collection_name] = []

    def upsert(self, collection_name: str, points: list[VectorPoint]) -> None:
        self.points.setdefault(collection_name, []).extend(points)

    def close(self) -> None:
        return None


async def test_indexer_updates_text_and_image_points_with_video_metadata(
    tmp_path: Path,
) -> None:
    video_id = uuid4()
    collection_id = uuid4()
    frame_id = uuid4()
    frame_path = tmp_path / "frame.jpg"
    store = FakeVectorStore()
    indexer = QdrantVideoIndexer(
        store,
        FakeTextEncoder(),
        FakeImageEncoder(),
        batch_size=1,
    )

    await indexer.index(
        video_id,
        collection_id,
        [TranscriptSegment(1, 3, "A verified statement.", "A slide heading.")],
        [KeyframeArtifact(frame_id, 2, frame_path, "abc", "A slide heading.")],
    )

    text_point = store.points[TEXT_VECTOR_COLLECTION][0]
    image_point = store.points[IMAGE_VECTOR_COLLECTION][0]
    assert store.deleted == [
        (TEXT_VECTOR_COLLECTION, video_id),
        (IMAGE_VECTOR_COLLECTION, video_id),
    ]
    assert store.collections == {TEXT_VECTOR_COLLECTION: 2, IMAGE_VECTOR_COLLECTION: 2}
    assert text_point.payload == {
        "video_id": str(video_id),
        "collection_id": str(collection_id),
        "segment_id": str(text_point.id),
        "start_s": 1,
        "end_s": 3,
        "transcript_text": "A verified statement.",
        "slide_text": "A slide heading.",
        "modality": "text",
    }
    assert image_point.id == frame_id
    assert image_point.payload["timestamp_s"] == 2
    assert image_point.payload["modality"] == "image"


@dataclass
class FakeTextBlock:
    text: str


@dataclass
class FakeMessage:
    content: list[FakeTextBlock]


class FakeAnthropicMessages:
    def __init__(self, response_text: str) -> None:
        self.response_text = response_text
        self.responses = [response_text]
        self.request: dict[str, object] | None = None
        self.requests: list[dict[str, object]] = []

    def create(
        self,
        *,
        model: str,
        max_tokens: int,
        system: str,
        messages: list[dict[str, str]],
    ) -> FakeMessage:
        self.request = {
            "model": model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": messages,
        }
        self.requests.append(self.request)
        response_index = min(len(self.requests) - 1, len(self.responses) - 1)
        return FakeMessage([FakeTextBlock(self.responses[response_index])])


class FakeAnthropicClient:
    def __init__(self, messages: FakeAnthropicMessages) -> None:
        self.messages = messages


def _client_factory(
    messages: FakeAnthropicMessages,
    expected_timeout: int = 120,
) -> AnthropicClientFactory:
    def factory(*, api_key: str, timeout: int) -> FakeAnthropicClient:
        assert api_key == "test-key"
        assert timeout == expected_timeout
        return FakeAnthropicClient(messages)

    return factory


async def test_summarizer_parses_evidence_based_chapters_and_checks_duration() -> None:
    messages = FakeAnthropicMessages(
        '{"summary":"A lesson on verified facts.",'
        '"chapters":[{"start_s":0,"end_s":15,"title":"Introduction",'
        '"summary":"The speaker introduces the subject."}]}'
    )
    summarizer = AnthropicChapterSummarizer(
        api_key="test-key",
        model_name="configured-model",
        timeout_seconds=10,
        client_factory=_client_factory(messages, expected_timeout=10),
    )

    chapters, summary = await summarizer.summarize(
        "Lecture",
        30,
        [TranscriptSegment(0, 15, "Welcome to the lecture.", None)],
        [],
    )

    assert chapters[0].ordinal == 0
    assert chapters[0].title == "Introduction"
    assert summary == "0-30s: A lesson on verified facts."
    assert messages.request is not None
    assert "only the supplied transcript and on-screen text" in str(messages.request["system"])


async def test_summarizer_rejects_chapters_beyond_video_duration() -> None:
    messages = FakeAnthropicMessages(
        '{"summary":"Summary.","chapters":[{"start_s":0,"end_s":31,'
        '"title":"Ending","summary":"The video ends."}]}'
    )
    summarizer = AnthropicChapterSummarizer(
        api_key="test-key",
        model_name="configured-model",
        client_factory=_client_factory(messages),
    )

    with pytest.raises(IngestionError, match="outside its evidence window"):
        await summarizer.summarize(
            "Lecture",
            30,
            [TranscriptSegment(0, 30, "Evidence.", None)],
            [],
        )


async def test_summarizer_splits_long_video_evidence_into_time_windows() -> None:
    messages = FakeAnthropicMessages(
        '{"summary":"First part.","chapters":[{"start_s":0,"end_s":10,'
        '"title":"First","summary":"First part."}]}'
    )
    messages.responses.append(
        '{"summary":"Second part.","chapters":[{"start_s":10,"end_s":20,'
        '"title":"Second","summary":"Second part."}]}'
    )
    summarizer = AnthropicChapterSummarizer(
        api_key="test-key",
        model_name="configured-model",
        window_seconds=10,
        client_factory=_client_factory(messages),
    )

    chapters, summary = await summarizer.summarize(
        "Lecture",
        20,
        [
            TranscriptSegment(0, 10, "First evidence.", None),
            TranscriptSegment(10, 20, "Second evidence.", None),
        ],
        [],
    )

    assert [chapter.title for chapter in chapters] == ["First", "Second"]
    assert len(messages.requests) == 2
    assert summary == "0-10s: First part.\n10-20s: Second part."


async def test_summarizer_requires_explicit_credentials_and_model_id() -> None:
    summarizer = AnthropicChapterSummarizer(api_key=None, model_name=None)

    with pytest.raises(IngestionError, match="ANTHROPIC_API_KEY"):
        await summarizer.summarize("Lecture", 30, [], [])
