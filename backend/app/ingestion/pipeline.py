import asyncio
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from uuid import UUID, uuid5

from app.ingestion.errors import IngestionError
from app.ingestion.media import MediaProcessor
from app.ingestion.schemas import TimedWord, TranscriptSegment
from app.ingestion.segmentation import segment_sentences, sentences_from_words


@dataclass(frozen=True, slots=True)
class KeyframeArtifact:
    id: UUID
    timestamp_s: float
    image_path: Path
    phash: str | None
    ocr_text: str | None = None


@dataclass(frozen=True, slots=True)
class ChapterArtifact:
    ordinal: int
    start_s: float
    end_s: float
    title: str
    summary: str


@dataclass(frozen=True, slots=True)
class IngestionResult:
    media_path: Path
    duration_s: float
    segments: list[TranscriptSegment]
    keyframes: list[KeyframeArtifact]
    chapters: list[ChapterArtifact]
    summary: str | None


ProgressCallback = Callable[[str, int], Awaitable[None]]


class Transcriber(Protocol):
    async def transcribe(self, audio_path: Path) -> Sequence[TimedWord]: ...


class KeyframeExtractor(Protocol):
    async def extract(
        self,
        video_id: UUID,
        media_path: Path,
        duration_s: float,
    ) -> Sequence[KeyframeArtifact]: ...


class OCRReader(Protocol):
    async def read(self, image_path: Path) -> str: ...


class VideoIndexer(Protocol):
    async def index(
        self,
        video_id: UUID,
        collection_id: UUID,
        segments: Sequence[TranscriptSegment],
        keyframes: Sequence[KeyframeArtifact],
    ) -> None: ...


class ChapterSummarizer(Protocol):
    async def summarize(
        self,
        title: str,
        duration_s: float,
        segments: Sequence[TranscriptSegment],
        keyframes: Sequence[KeyframeArtifact],
    ) -> tuple[Sequence[ChapterArtifact], str | None]: ...


class IngestionPipeline:
    def __init__(
        self,
        media_processor: MediaProcessor,
        transcriber: Transcriber,
        keyframe_extractor: KeyframeExtractor,
        ocr_reader: OCRReader,
        indexer: VideoIndexer,
        chapter_summarizer: ChapterSummarizer,
        ocr_concurrency: int = 4,
    ) -> None:
        if ocr_concurrency <= 0:
            raise ValueError("OCR concurrency must be positive")
        self.media_processor = media_processor
        self.transcriber = transcriber
        self.keyframe_extractor = keyframe_extractor
        self.ocr_reader = ocr_reader
        self.indexer = indexer
        self.chapter_summarizer = chapter_summarizer
        self.ocr_concurrency = ocr_concurrency

    async def process(
        self,
        video_id: UUID,
        collection_id: UUID,
        title: str,
        source_url: str | None,
        source_path: Path | None,
        video_directory: Path,
        report_progress: ProgressCallback,
    ) -> IngestionResult:
        if source_url is None and source_path is None:
            raise IngestionError("Video has neither a source URL nor an uploaded media file")

        await report_progress("acquire", 5)
        acquired_path = (
            await self.media_processor.download_url(source_url, video_directory)
            if source_url is not None
            else source_path
        )
        if acquired_path is None:
            raise IngestionError("Video acquisition did not produce a media file")

        await report_progress("normalize", 15)
        artifacts = await self.media_processor.prepare_video(acquired_path, video_directory)

        await report_progress("transcribe", 30)
        words = (
            list(await self.transcriber.transcribe(artifacts.audio_path))
            if artifacts.audio_path is not None
            else []
        )
        segments = segment_sentences(sentences_from_words(words))

        await report_progress("keyframes", 50)
        extracted_frames = await self.keyframe_extractor.extract(
            video_id,
            artifacts.media_path,
            artifacts.metadata.duration_s,
        )

        await report_progress("ocr", 65)
        ocr_semaphore = asyncio.Semaphore(self.ocr_concurrency)
        keyframes = await asyncio.gather(
            *(self._read_keyframe(video_id, frame, ocr_semaphore) for frame in extracted_frames)
        )
        segments = _attach_slide_text(segments, keyframes)

        await report_progress("embed", 80)
        await self.indexer.index(video_id, collection_id, segments, keyframes)

        await report_progress("chapters", 90)
        if any(
            segment.transcript_text.strip() or segment.slide_text for segment in segments
        ) or any(keyframe.ocr_text for keyframe in keyframes):
            chapters, summary = await self.chapter_summarizer.summarize(
                title,
                artifacts.metadata.duration_s,
                segments,
                keyframes,
            )
        else:
            chapters, summary = (), None

        return IngestionResult(
            media_path=artifacts.media_path,
            duration_s=artifacts.metadata.duration_s,
            segments=segments,
            keyframes=list(keyframes),
            chapters=list(chapters),
            summary=summary,
        )

    async def _read_keyframe(
        self,
        video_id: UUID,
        frame: KeyframeArtifact,
        semaphore: asyncio.Semaphore,
    ) -> KeyframeArtifact:
        async with semaphore:
            text = await self.ocr_reader.read(frame.image_path)
        return KeyframeArtifact(
            id=uuid5(video_id, f"keyframe:{frame.timestamp_s:.3f}"),
            timestamp_s=frame.timestamp_s,
            image_path=frame.image_path,
            phash=frame.phash,
            ocr_text=text.strip() or None,
        )


def _attach_slide_text(
    segments: Sequence[TranscriptSegment],
    keyframes: Sequence[KeyframeArtifact],
) -> list[TranscriptSegment]:
    attached: list[TranscriptSegment] = []
    for segment in segments:
        slide_text = " ".join(
            keyframe.ocr_text
            for keyframe in keyframes
            if keyframe.ocr_text is not None
            and segment.start_s <= keyframe.timestamp_s <= segment.end_s
        )
        attached.append(
            TranscriptSegment(
                start_s=segment.start_s,
                end_s=segment.end_s,
                transcript_text=segment.transcript_text,
                slide_text=slide_text or None,
            )
        )
    return attached
