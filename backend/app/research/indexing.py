from __future__ import annotations

import asyncio
from collections.abc import Iterable, Sequence
from importlib import import_module
from pathlib import Path
from typing import Protocol, cast
from uuid import NAMESPACE_URL, uuid5

from app.config import Settings
from app.ingestion.media import MediaProcessor
from app.ingestion.pipeline import KeyframeArtifact
from app.ingestion.schemas import TimedWord
from app.ingestion.segmentation import segment_sentences, sentences_from_words
from app.ingestion.transcription import FasterWhisperTranscriber
from app.research.store import ResearchItem, ResearchStore


class _VectorModel(Protocol):
    def passage_embed(self, texts: Iterable[str]) -> Iterable[Sequence[float]]: ...

    def query_embed(self, text: str) -> Iterable[Sequence[float]]: ...


class _VectorModelFactory(Protocol):
    def __call__(self, model_name: str, **kwargs: object) -> _VectorModel: ...


class _ImageModel(Protocol):
    def embed(self, images: Iterable[str]) -> Iterable[Sequence[float]]: ...


class _ImageModelFactory(Protocol):
    def __call__(self, model_name: str, **kwargs: object) -> _ImageModel: ...


class _FastEmbedModule(Protocol):
    TextEmbedding: _VectorModelFactory
    ImageEmbedding: _ImageModelFactory


class ResearchIndex:
    def __init__(
        self,
        store: ResearchStore,
        settings: Settings,
        data_directory: Path,
        text_model_name: str = "BAAI/bge-base-en-v1.5",
        cross_modal_model_name: str = "jinaai/jina-clip-v1",
        whisper_model_name: str = "base.en",
        sample_interval_s: float = 2.0,
        transcribe: bool = True,
        model_factory: _VectorModelFactory | None = None,
        image_model_factory: _ImageModelFactory | None = None,
    ) -> None:
        if sample_interval_s <= 0:
            raise ValueError("Frame sampling interval must be positive")
        self.store = store
        self.settings = settings
        self.data_directory = data_directory.resolve()
        self.text_model_name = text_model_name
        self.cross_modal_model_name = cross_modal_model_name
        self.whisper_model_name = whisper_model_name
        self.sample_interval_s = sample_interval_s
        self.transcribe = transcribe
        self.model_factory = model_factory
        self.image_model_factory = image_model_factory
        self.text_model: _VectorModel | None = None
        self.cross_modal_text_model: _VectorModel | None = None
        self.image_model: _ImageModel | None = None
        self.media_processor = MediaProcessor(
            ffmpeg_executable=settings.ffmpeg_executable,
            ffprobe_executable=settings.ffprobe_executable,
            timeout_seconds=settings.media_timeout_seconds,
        )
        self.transcriber = FasterWhisperTranscriber(
            model_name=whisper_model_name,
            device=settings.whisper_device,
            compute_type=settings.whisper_compute_type,
        )

    async def index_video(self, external_id: str, source_path: Path) -> dict[str, object]:
        if not source_path.is_file():
            raise FileNotFoundError(f"Video does not exist: {source_path}")
        video_uuid = uuid5(NAMESPACE_URL, f"perception-test:{external_id}")
        video_directory = self.data_directory / "videos" / external_id
        video_directory.mkdir(parents=True, exist_ok=True)

        artifacts = await self.media_processor.prepare_video(source_path, video_directory)
        words: list[TimedWord] = []
        if self.transcribe and artifacts.audio_path is not None:
            words = await self.transcriber.transcribe(artifacts.audio_path)
        sentences = sentences_from_words(words)
        segments = segment_sentences(
            sentences,
            target_duration_s=12.0,
            maximum_duration_s=16.0,
            overlap_ratio=0.15,
        )
        frames = await _extract_sampled_frames(
            artifacts.media_path,
            video_directory / "frames",
            artifacts.metadata.duration_s,
            self.sample_interval_s,
            self.settings.ffmpeg_executable,
            self.settings.media_timeout_seconds,
        )

        dense_vectors = await asyncio.to_thread(
            self._encode_passages,
            [segment.transcript_text for segment in segments],
        )
        clip_vectors = await asyncio.to_thread(
            self._encode_images,
            [frame.image_path for frame in frames],
        )
        if len(dense_vectors) != len(segments) or len(clip_vectors) != len(frames):
            raise RuntimeError("Embedding model returned a different number of vectors than inputs")
        items = [
            ResearchItem(
                id=uuid5(video_uuid, f"segment:{index}:{segment.start_s:.3f}"),
                video_id=external_id,
                kind="transcript",
                start_s=segment.start_s,
                end_s=segment.end_s,
                text=segment.transcript_text,
                image_path=None,
                dense_vector=list(vector),
                clip_vector=None,
            )
            for index, (segment, vector) in enumerate(zip(segments, dense_vectors, strict=True))
        ]
        items.extend(
            ResearchItem(
                id=frame.id,
                video_id=external_id,
                kind="keyframe",
                start_s=frame.timestamp_s,
                end_s=frame.timestamp_s,
                text=frame.ocr_text or "",
                image_path=frame.image_path,
                dense_vector=None,
                clip_vector=list(vector),
            )
            for frame, vector in zip(frames, clip_vectors, strict=True)
        )
        self.store.replace_video(
            external_id=external_id,
            title=external_id,
            source_path=source_path,
            processed_path=artifacts.media_path,
            duration_s=artifacts.metadata.duration_s,
            items=items,
        )
        return {
            "video_id": external_id,
            "duration_s": artifacts.metadata.duration_s,
            "transcript_chunks": len(segments),
            "sampled_frames": len(frames),
            "transcription_enabled": self.transcribe and artifacts.audio_path is not None,
        }

    def embed_text_query(self, text: str, model: str = "dense") -> list[float]:
        if model == "dense":
            encoded = self._get_text_model().query_embed(text)
        elif model == "clip":
            encoded = self._get_cross_modal_text_model().query_embed(text)
        else:
            raise ValueError(f"Unknown query embedding model: {model}")
        vectors = [list(map(float, vector)) for vector in encoded]
        if len(vectors) != 1:
            raise RuntimeError(f"Query encoder returned {len(vectors)} vectors for one query")
        return vectors[0]

    def _encode_passages(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return [list(map(float, vector)) for vector in self._get_text_model().passage_embed(texts)]

    def _encode_images(self, image_paths: list[Path]) -> list[list[float]]:
        if not image_paths:
            return []
        return [
            list(map(float, vector))
            for vector in self._get_image_model().embed(str(path) for path in image_paths)
        ]

    def _get_text_model(self) -> _VectorModel:
        if self.text_model is None:
            factory = self.model_factory or self._fastembed().TextEmbedding
            self.text_model = factory(self.text_model_name)
        return self.text_model

    def _get_cross_modal_text_model(self) -> _VectorModel:
        if self.cross_modal_text_model is None:
            factory = self.model_factory or self._fastembed().TextEmbedding
            self.cross_modal_text_model = factory(self.cross_modal_model_name)
        return self.cross_modal_text_model

    def _get_image_model(self) -> _ImageModel:
        if self.image_model is None:
            factory = self.image_model_factory or self._fastembed().ImageEmbedding
            self.image_model = factory(self.cross_modal_model_name)
        return self.image_model

    @staticmethod
    def _fastembed() -> _FastEmbedModule:
        try:
            return cast(_FastEmbedModule, import_module("fastembed"))
        except ImportError as error:
            raise RuntimeError(
                "FastEmbed is required for the local research index; install the research extra"
            ) from error


async def _extract_sampled_frames(
    media_path: Path,
    frame_directory: Path,
    duration_s: float,
    interval_s: float,
    ffmpeg_executable: str,
    timeout_seconds: int,
) -> list[KeyframeArtifact]:
    from app.ingestion.media import run_media_command

    frame_directory.mkdir(parents=True, exist_ok=True)
    pattern = frame_directory / "frame-%06d.jpg"
    expected_count = max(1, int(duration_s / interval_s))
    existing = sorted(frame_directory.glob("frame-*.jpg"))
    if len(existing) != expected_count:
        for image in existing:
            image.unlink()
        await run_media_command(
            [
                ffmpeg_executable,
                "-nostdin",
                "-y",
                "-i",
                str(media_path),
                "-vf",
                f"fps=1/{interval_s:g},scale='min(640,iw)':-2",
                "-q:v",
                "3",
                str(pattern),
            ],
            timeout_seconds,
        )
        existing = sorted(frame_directory.glob("frame-*.jpg"))
    if not existing:
        raise RuntimeError(f"FFmpeg did not extract any frames from {media_path}")
    video_id = uuid5(NAMESPACE_URL, f"local-video:{media_path}")
    return [
        KeyframeArtifact(
            id=uuid5(video_id, f"frame:{index}:{interval_s}"),
            timestamp_s=min(index * interval_s, duration_s),
            image_path=image_path,
            phash=None,
        )
        for index, image_path in enumerate(existing)
    ]
