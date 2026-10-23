from collections.abc import Sequence
from pathlib import Path
from uuid import UUID, uuid4

from app.ingestion.media import MediaArtifacts, MediaMetadata
from app.ingestion.pipeline import (
    ChapterArtifact,
    IngestionPipeline,
    KeyframeArtifact,
)
from app.ingestion.schemas import TimedWord, TranscriptSegment


class FakeMediaProcessor:
    def __init__(self, audio_path: Path | None) -> None:
        self.audio_path = audio_path
        self.downloaded_url: str | None = None

    async def download_url(self, url: str, video_directory: Path) -> Path:
        self.downloaded_url = url
        return video_directory / "source.mp4"

    async def prepare_video(self, source_path: Path, video_directory: Path) -> MediaArtifacts:
        return MediaArtifacts(
            media_path=video_directory / "media.mp4",
            audio_path=self.audio_path,
            metadata=MediaMetadata(duration_s=120, has_audio=self.audio_path is not None),
        )


class FakeTranscriber:
    async def transcribe(self, audio_path: Path) -> Sequence[TimedWord]:
        return [TimedWord(10, 11, "A fact.")]


class FakeKeyframeExtractor:
    async def extract(
        self,
        video_id: UUID,
        media_path: Path,
        duration_s: float,
    ) -> Sequence[KeyframeArtifact]:
        return [KeyframeArtifact(uuid4(), 10.5, media_path, "abc123")]


class FakeOCRReader:
    async def read(self, image_path: Path) -> str:
        return "  slide text  "


class FakeIndexer:
    def __init__(self) -> None:
        self.segments: Sequence[TranscriptSegment] = ()
        self.keyframes: Sequence[KeyframeArtifact] = ()

    async def index(
        self,
        video_id: UUID,
        collection_id: UUID,
        segments: Sequence[TranscriptSegment],
        keyframes: Sequence[KeyframeArtifact],
    ) -> None:
        self.segments = segments
        self.keyframes = keyframes


class FakeChapterSummarizer:
    def __init__(self) -> None:
        self.called = False

    async def summarize(
        self,
        title: str,
        duration_s: float,
        segments: Sequence[TranscriptSegment],
        keyframes: Sequence[KeyframeArtifact],
    ) -> tuple[Sequence[ChapterArtifact], str | None]:
        self.called = True
        return [ChapterArtifact(0, 10, 11, "Topic", "Supported summary.")], "Video summary."


async def test_pipeline_orders_stages_and_keeps_slide_text_separate(tmp_path: Path) -> None:
    audio_path = tmp_path / "audio.wav"
    media_processor = FakeMediaProcessor(audio_path)
    indexer = FakeIndexer()
    summarizer = FakeChapterSummarizer()
    pipeline = IngestionPipeline(
        media_processor,
        FakeTranscriber(),
        FakeKeyframeExtractor(),
        FakeOCRReader(),
        indexer,
        summarizer,
    )
    progress: list[tuple[str, int]] = []

    async def report_progress(stage: str, value: int) -> None:
        progress.append((stage, value))

    result = await pipeline.process(
        video_id=uuid4(),
        collection_id=uuid4(),
        title="Test lecture",
        source_url="https://example.com/lecture.mp4",
        source_path=None,
        video_directory=tmp_path,
        report_progress=report_progress,
    )

    assert media_processor.downloaded_url == "https://example.com/lecture.mp4"
    assert result.segments == [TranscriptSegment(10, 11, "A fact.", "slide text")]
    assert indexer.segments == result.segments
    assert result.keyframes[0].ocr_text == "slide text"
    assert result.chapters[0].title == "Topic"
    assert result.summary == "Video summary."
    assert summarizer.called
    assert progress == [
        ("acquire", 5),
        ("normalize", 15),
        ("transcribe", 30),
        ("keyframes", 50),
        ("ocr", 65),
        ("embed", 80),
        ("chapters", 90),
    ]


async def test_pipeline_finishes_video_without_speech_or_slides(tmp_path: Path) -> None:
    media_processor = FakeMediaProcessor(audio_path=None)
    indexer = FakeIndexer()
    summarizer = FakeChapterSummarizer()
    pipeline = IngestionPipeline(
        media_processor,
        FakeTranscriber(),
        EmptyKeyframeExtractor(),
        FakeOCRReader(),
        indexer,
        summarizer,
    )

    async def report_progress(stage: str, value: int) -> None:
        assert 0 <= value <= 100

    result = await pipeline.process(
        video_id=uuid4(),
        collection_id=uuid4(),
        title="Silent video",
        source_url=None,
        source_path=tmp_path / "source.mp4",
        video_directory=tmp_path,
        report_progress=report_progress,
    )

    assert result.segments == []
    assert result.keyframes == []
    assert result.chapters == []
    assert result.summary is None
    assert indexer.segments == []
    assert not summarizer.called


class EmptyKeyframeExtractor:
    async def extract(
        self,
        video_id: UUID,
        media_path: Path,
        duration_s: float,
    ) -> Sequence[KeyframeArtifact]:
        return []
