from types import SimpleNamespace
from typing import cast
from uuid import uuid4

import pytest

from app.ingestion.errors import IngestionError
from app.ingestion.providers import QdrantClientPort, QdrantModels, QdrantVectorStore, VectorPoint

qdrant_models = pytest.importorskip("qdrant_client.models")


class FakeQdrantClient:
    def __init__(self) -> None:
        self.collections: set[str] = set()
        self.created_config: object | None = None
        self.deleted_filter: object | None = None
        self.upserted: list[object] = []
        self.queried_filter: object | None = None
        self.search_points: list[object] = [
            SimpleNamespace(
                id=str(uuid4()),
                score=0.75,
                payload={"video_id": str(uuid4()), "modality": "text"},
            )
        ]

    def collection_exists(self, collection_name: str) -> bool:
        return collection_name in self.collections

    def create_collection(self, *, collection_name: str, vectors_config: object) -> object:
        self.collections.add(collection_name)
        self.created_config = vectors_config
        return True

    def delete(
        self,
        *,
        collection_name: str,
        points_selector: object,
        wait: bool = True,
    ) -> object:
        self.deleted_filter = points_selector
        return True

    def upsert(
        self,
        *,
        collection_name: str,
        points: list[object],
        wait: bool = True,
    ) -> object:
        self.upserted.extend(points)
        return True

    def query_points(
        self,
        *,
        collection_name: str,
        query: list[float],
        query_filter: object,
        limit: int,
        with_payload: bool,
    ) -> object:
        self.queried_filter = query_filter
        return SimpleNamespace(points=self.search_points)

    def close(self) -> None:
        return None


def test_qdrant_adapter_builds_sdk_models_and_search_filters() -> None:
    client = FakeQdrantClient()
    models = cast(QdrantModels, qdrant_models)
    store = QdrantVectorStore(cast(QdrantClientPort, client), models)
    collection_id = uuid4()
    video_id = uuid4()
    point_id = uuid4()

    store.ensure_collection("video_segments", 2)
    store.upsert(
        "video_segments",
        [VectorPoint(point_id, [0.1, 0.2], {"video_id": str(video_id)})],
    )
    store.delete_video("video_segments", video_id)
    results = store.search(
        "video_segments",
        [0.3, 0.4],
        collection_id,
        [video_id],
        5,
    )

    assert client.created_config == qdrant_models.VectorParams(
        size=2,
        distance=qdrant_models.Distance.COSINE,
    )
    assert len(client.upserted) == 1
    assert client.deleted_filter is not None
    assert isinstance(client.queried_filter, qdrant_models.Filter)
    assert client.queried_filter.must[0].key == "collection_id"
    assert client.queried_filter.must[0].match.value == str(collection_id)
    assert client.queried_filter.must[1].key == "video_id"
    assert client.queried_filter.must[1].match.any == [str(video_id)]
    assert len(results) == 1
    assert results[0].score == 0.75


@pytest.mark.parametrize(
    ("point_id", "payload", "message"),
    [
        ("not-a-uuid", {}, "invalid point ID"),
        (str(uuid4()), None, "without a payload"),
    ],
)
def test_qdrant_search_rejects_malformed_results(
    point_id: str,
    payload: dict[str, object] | None,
    message: str,
) -> None:
    client = FakeQdrantClient()
    client.search_points = [SimpleNamespace(id=point_id, score=0.5, payload=payload)]
    store = QdrantVectorStore(
        cast(QdrantClientPort, client),
        cast(QdrantModels, qdrant_models),
    )
    client.collections.add("video_segments")

    with pytest.raises(IngestionError, match=message):
        store.search("video_segments", [0.1], uuid4(), [uuid4()], 2)


def test_qdrant_search_skips_collection_when_no_points_exist() -> None:
    client = FakeQdrantClient()
    store = QdrantVectorStore(
        cast(QdrantClientPort, client),
        cast(QdrantModels, qdrant_models),
    )

    assert store.search("empty", [0.1], uuid4(), [uuid4()], 2) == []
