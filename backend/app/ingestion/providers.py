import asyncio
from collections.abc import Iterable, Mapping, Sequence
from contextlib import AbstractContextManager, ExitStack
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Protocol, cast
from uuid import UUID, uuid5

from app.ingestion.errors import IngestionError
from app.ingestion.pipeline import KeyframeArtifact, VideoIndexer
from app.ingestion.schemas import TranscriptSegment

TEXT_VECTOR_COLLECTION = "video_segments"
IMAGE_VECTOR_COLLECTION = "video_keyframes"


class TextEmbeddingModel(Protocol):
    def embed(self, documents: list[str]) -> Iterable[Sequence[float]]: ...


class TextEmbeddingFactory(Protocol):
    def __call__(self, model_name: str) -> TextEmbeddingModel: ...


class TextEncoder(Protocol):
    def encode(self, documents: list[str]) -> list[list[float]]: ...


class FastEmbedTextEncoder:
    def __init__(
        self,
        model_name: str,
        model_factory: TextEmbeddingFactory | None = None,
    ) -> None:
        if not model_name.strip():
            raise ValueError("A text embedding model name is required")
        self.model_name = model_name
        self.model_factory = model_factory
        self.model: TextEmbeddingModel | None = None

    def encode(self, documents: list[str]) -> list[list[float]]:
        if not documents:
            return []
        model = self._get_model()
        try:
            return [[float(value) for value in vector] for vector in model.embed(documents)]
        except (OSError, RuntimeError, ValueError) as error:
            raise IngestionError(f"Text embedding failed: {error}") from error

    def _get_model(self) -> TextEmbeddingModel:
        if self.model is not None:
            return self.model
        if self.model_factory is None:
            try:
                module = import_module("fastembed")
            except ImportError as error:
                raise IngestionError(
                    "fastembed is not installed; install the ingestion dependency group"
                ) from error
            self.model_factory = cast(TextEmbeddingFactory, module.__dict__["TextEmbedding"])
        try:
            self.model = self.model_factory(self.model_name)
        except (OSError, RuntimeError, ValueError) as error:
            raise IngestionError(
                f"Could not initialize text embedding model {self.model_name}: {error}"
            ) from error
        return self.model


class ImageObject(Protocol):
    def __enter__(self) -> object: ...

    def __exit__(self, error_type: object, error: object, traceback: object) -> bool | None: ...


class ImageModule(Protocol):
    def open(self, path: str) -> ImageObject: ...


class ImageEncoder(Protocol):
    def encode(self, image_paths: list[Path]) -> list[list[float]]: ...


class ImageProcessor(Protocol):
    def __call__(
        self,
        *,
        return_tensors: str,
        images: list[object] | None = None,
        text: list[str] | None = None,
        padding: str | bool | None = None,
    ) -> Mapping[str, object]: ...


class ImageTensor(Protocol):
    def tolist(self) -> object: ...


class ImageModel(Protocol):
    def eval(self) -> object: ...

    def get_image_features(self, **inputs: object) -> ImageTensor: ...

    def get_text_features(self, **inputs: object) -> ImageTensor: ...


class PretrainedFactory(Protocol):
    def from_pretrained(self, model_name: str) -> object: ...


class TransformersModule(Protocol):
    AutoModel: PretrainedFactory
    AutoProcessor: PretrainedFactory


class InferenceModule(Protocol):
    def inference_mode(self) -> AbstractContextManager[object]: ...


class SiglipImageEncoder:
    def __init__(
        self,
        model_name: str,
        processor_factory: PretrainedFactory | None = None,
        model_factory: PretrainedFactory | None = None,
        image_module: ImageModule | None = None,
        inference_module: InferenceModule | None = None,
    ) -> None:
        if not model_name.strip():
            raise ValueError("A visual embedding model name is required")
        self.model_name = model_name
        self.processor_factory = processor_factory
        self.model_factory = model_factory
        self.image_module = image_module
        self.inference_module = inference_module
        self.processor: ImageProcessor | None = None
        self.model: ImageModel | None = None

    def encode(self, image_paths: list[Path]) -> list[list[float]]:
        if not image_paths:
            return []
        processor, model, image_module, inference_module = self._get_components()
        try:
            with ExitStack() as stack:
                images = [
                    stack.enter_context(image_module.open(str(image_path)))
                    for image_path in image_paths
                ]
                inputs = processor(images=images, return_tensors="pt")
                with inference_module.inference_mode():
                    features = model.get_image_features(**inputs).tolist()
            vectors = cast(Sequence[Sequence[float]], features)
            return [[float(value) for value in vector] for vector in vectors]
        except (OSError, RuntimeError, ValueError) as error:
            raise IngestionError(f"Visual embedding failed: {error}") from error

    def encode_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        processor, model, _, inference_module = self._get_components()
        try:
            inputs = processor(text=texts, return_tensors="pt", padding="max_length")
            with inference_module.inference_mode():
                features = model.get_text_features(**inputs).tolist()
            vectors = cast(Sequence[Sequence[float]], features)
            return [[float(value) for value in vector] for vector in vectors]
        except (OSError, RuntimeError, ValueError) as error:
            raise IngestionError(f"Visual text embedding failed: {error}") from error

    def _get_components(
        self,
    ) -> tuple[ImageProcessor, ImageModel, ImageModule, InferenceModule]:
        if (
            self.processor is not None
            and self.model is not None
            and self.image_module is not None
            and self.inference_module is not None
        ):
            return self.processor, self.model, self.image_module, self.inference_module
        try:
            transformers = cast(TransformersModule, import_module("transformers"))
            image_module = cast(ImageModule, import_module("PIL.Image"))
            inference_module = cast(InferenceModule, import_module("torch"))
        except ImportError as error:
            raise IngestionError(
                "Transformers, Pillow, and PyTorch are required for visual embeddings; "
                "install the ingestion dependency group"
            ) from error
        processor_factory = self.processor_factory or transformers.AutoProcessor
        model_factory = self.model_factory or transformers.AutoModel
        try:
            self.processor = cast(
                ImageProcessor,
                processor_factory.from_pretrained(self.model_name),
            )
            self.model = cast(ImageModel, model_factory.from_pretrained(self.model_name))
            self.model.eval()
        except (OSError, RuntimeError, ValueError) as error:
            raise IngestionError(
                f"Could not initialize visual embedding model {self.model_name}: {error}"
            ) from error
        self.image_module = self.image_module or image_module
        self.inference_module = self.inference_module or inference_module
        return self.processor, self.model, self.image_module, self.inference_module


@dataclass(frozen=True, slots=True)
class VectorPoint:
    id: UUID
    vector: list[float]
    payload: dict[str, object]


@dataclass(frozen=True, slots=True)
class VectorSearchResult:
    point_id: UUID
    score: float
    payload: dict[str, object]


class VectorStore(Protocol):
    def ensure_collection(self, collection_name: str, vector_size: int) -> None: ...

    def delete_video(self, collection_name: str, video_id: UUID) -> None: ...

    def upsert(self, collection_name: str, points: list[VectorPoint]) -> None: ...

    def close(self) -> None: ...


class QdrantScoredPoint(Protocol):
    id: object
    score: float
    payload: dict[str, object] | None


class QdrantQueryResponse(Protocol):
    points: Sequence[QdrantScoredPoint]


class QdrantClientPort(Protocol):
    def collection_exists(self, collection_name: str) -> bool: ...

    def create_collection(self, *, collection_name: str, vectors_config: object) -> object: ...

    def delete(
        self,
        *,
        collection_name: str,
        points_selector: object,
        wait: bool = True,
    ) -> object: ...

    def upsert(
        self,
        *,
        collection_name: str,
        points: list[object],
        wait: bool = True,
    ) -> object: ...

    def query_points(
        self,
        *,
        collection_name: str,
        query: list[float],
        query_filter: object,
        limit: int,
        with_payload: bool,
    ) -> QdrantQueryResponse: ...

    def close(self) -> None: ...


class QdrantClientFactory(Protocol):
    def __call__(self, *, url: str, timeout: int) -> QdrantClientPort: ...


class DistanceValues(Protocol):
    COSINE: object


class QdrantModelFactory(Protocol):
    def __call__(self, **fields: object) -> object: ...


class QdrantModels(Protocol):
    Distance: DistanceValues
    VectorParams: QdrantModelFactory
    PointStruct: QdrantModelFactory
    MatchValue: QdrantModelFactory
    MatchAny: QdrantModelFactory
    FieldCondition: QdrantModelFactory
    Filter: QdrantModelFactory
    FilterSelector: QdrantModelFactory


class QdrantVectorStore:
    def __init__(self, client: QdrantClientPort, models: QdrantModels) -> None:
        self.client = client
        self.models = models

    def ensure_collection(self, collection_name: str, vector_size: int) -> None:
        if self.client.collection_exists(collection_name=collection_name):
            return
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=self.models.VectorParams(
                size=vector_size,
                distance=self.models.Distance.COSINE,
            ),
        )

    def delete_video(self, collection_name: str, video_id: UUID) -> None:
        if not self.client.collection_exists(collection_name=collection_name):
            return
        video_filter = self.models.Filter(
            must=[
                self.models.FieldCondition(
                    key="video_id",
                    match=self.models.MatchValue(value=str(video_id)),
                )
            ]
        )
        self.client.delete(
            collection_name=collection_name,
            points_selector=self.models.FilterSelector(filter=video_filter),
            wait=True,
        )

    def upsert(self, collection_name: str, points: list[VectorPoint]) -> None:
        if not points:
            return
        qdrant_points = [
            self.models.PointStruct(
                id=str(point.id),
                vector=point.vector,
                payload=point.payload,
            )
            for point in points
        ]
        self.client.upsert(
            collection_name=collection_name,
            points=qdrant_points,
            wait=True,
        )

    def search(
        self,
        collection_name: str,
        vector: list[float],
        collection_id: UUID,
        ready_video_ids: Sequence[UUID],
        limit: int,
    ) -> list[VectorSearchResult]:
        if not ready_video_ids:
            return []
        if limit <= 0:
            raise ValueError("Qdrant search limit must be positive")
        if not self.client.collection_exists(collection_name=collection_name):
            return []
        query_filter = self.models.Filter(
            must=[
                self.models.FieldCondition(
                    key="collection_id",
                    match=self.models.MatchValue(value=str(collection_id)),
                ),
                self.models.FieldCondition(
                    key="video_id",
                    match=self.models.MatchAny(any=[str(video_id) for video_id in ready_video_ids]),
                ),
            ]
        )
        response = self.client.query_points(
            collection_name=collection_name,
            query=vector,
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
        )
        results: list[VectorSearchResult] = []
        for point in response.points:
            if point.payload is None:
                raise IngestionError("Qdrant returned a search result without a payload")
            try:
                point_id = UUID(str(point.id))
            except ValueError as error:
                raise IngestionError(f"Qdrant returned an invalid point ID: {point.id}") from error
            results.append(
                VectorSearchResult(
                    point_id=point_id,
                    score=point.score,
                    payload=point.payload,
                )
            )
        return results

    def close(self) -> None:
        self.client.close()


def create_qdrant_vector_store(url: str, timeout_seconds: int) -> QdrantVectorStore:
    try:
        client_module = import_module("qdrant_client")
        models_module = cast(QdrantModels, import_module("qdrant_client.models"))
    except ImportError as error:
        raise IngestionError(
            "qdrant-client is not installed; install the ingestion dependency group"
        ) from error
    factory = cast(QdrantClientFactory, client_module.__dict__["QdrantClient"])
    try:
        client = factory(url=url, timeout=timeout_seconds)
    except (OSError, RuntimeError, ValueError) as error:
        raise IngestionError(f"Could not initialize Qdrant at {url}: {error}") from error
    return QdrantVectorStore(client, models_module)


class QdrantVideoIndexer(VideoIndexer):
    def __init__(
        self,
        vector_store: VectorStore,
        text_encoder: TextEncoder,
        image_encoder: ImageEncoder,
        batch_size: int = 32,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("Embedding batch size must be positive")
        self.vector_store = vector_store
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder
        self.batch_size = batch_size

    async def index(
        self,
        video_id: UUID,
        collection_id: UUID,
        segments: Sequence[TranscriptSegment],
        keyframes: Sequence[KeyframeArtifact],
    ) -> None:
        text_records = [
            (
                uuid5(video_id, f"segment:{index}:{segment.start_s:.3f}:{segment.end_s:.3f}"),
                segment,
                _segment_document(segment),
            )
            for index, segment in enumerate(segments)
            if segment.transcript_text.strip() or (segment.slide_text or "").strip()
        ]
        image_frames = list(keyframes)
        text_vectors = await self._encode_texts([record[2] for record in text_records])
        image_vectors = await self._encode_images([frame.image_path for frame in image_frames])
        _validate_embedding_count(text_vectors, len(text_records), "text")
        _validate_embedding_count(image_vectors, len(image_frames), "visual")

        await asyncio.to_thread(
            self.vector_store.delete_video,
            TEXT_VECTOR_COLLECTION,
            video_id,
        )
        await asyncio.to_thread(
            self.vector_store.delete_video,
            IMAGE_VECTOR_COLLECTION,
            video_id,
        )
        text_points = [
            VectorPoint(
                id=record[0],
                vector=vector,
                payload={
                    "video_id": str(video_id),
                    "collection_id": str(collection_id),
                    "segment_id": str(record[0]),
                    "start_s": record[1].start_s,
                    "end_s": record[1].end_s,
                    "transcript_text": record[1].transcript_text,
                    "slide_text": record[1].slide_text or "",
                    "modality": "text",
                },
            )
            for record, vector in zip(text_records, text_vectors, strict=True)
        ]
        image_points = [
            VectorPoint(
                id=frame.id,
                vector=vector,
                payload={
                    "video_id": str(video_id),
                    "collection_id": str(collection_id),
                    "keyframe_id": str(frame.id),
                    "timestamp_s": frame.timestamp_s,
                    "ocr_text": frame.ocr_text or "",
                    "phash": frame.phash or "",
                    "modality": "image",
                },
            )
            for frame, vector in zip(image_frames, image_vectors, strict=True)
        ]
        await self._upsert(TEXT_VECTOR_COLLECTION, text_points)
        await self._upsert(IMAGE_VECTOR_COLLECTION, image_points)

    async def _encode_texts(self, documents: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for offset in range(0, len(documents), self.batch_size):
            batch = documents[offset : offset + self.batch_size]
            vectors.extend(await asyncio.to_thread(self.text_encoder.encode, batch))
        return vectors

    async def _encode_images(self, image_paths: list[Path]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for offset in range(0, len(image_paths), self.batch_size):
            batch = image_paths[offset : offset + self.batch_size]
            vectors.extend(await asyncio.to_thread(self.image_encoder.encode, batch))
        return vectors

    async def _upsert(self, collection_name: str, points: list[VectorPoint]) -> None:
        if not points:
            return
        vector_size = len(points[0].vector)
        if vector_size == 0:
            raise IngestionError("Embedding provider returned an empty vector")
        if any(len(point.vector) != vector_size for point in points):
            raise IngestionError("Embedding provider returned inconsistent vector dimensions")
        await asyncio.to_thread(
            self.vector_store.ensure_collection,
            collection_name,
            vector_size,
        )
        for offset in range(0, len(points), self.batch_size):
            batch = points[offset : offset + self.batch_size]
            await asyncio.to_thread(self.vector_store.upsert, collection_name, batch)


def _segment_document(segment: TranscriptSegment) -> str:
    parts = []
    if segment.transcript_text.strip():
        parts.append(f"Transcript: {segment.transcript_text.strip()}")
    if segment.slide_text and segment.slide_text.strip():
        parts.append(f"On-screen text: {segment.slide_text.strip()}")
    return "\n".join(parts)


def _validate_embedding_count(
    vectors: Sequence[Sequence[float]],
    expected: int,
    modality: str,
) -> None:
    if len(vectors) != expected:
        raise IngestionError(
            f"{modality.capitalize()} embedding provider returned {len(vectors)} vectors "
            f"for {expected} inputs"
        )
