import asyncio
import shutil
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from app.ingestion.media import MediaProcessor, save_upload


async def _chunks(*items: bytes) -> AsyncIterator[bytes]:
    for item in items:
        yield item


async def test_save_upload_streams_chunks_atomically(tmp_path: Path) -> None:
    destination = tmp_path / "video" / "source.mp4"

    size_bytes = await save_upload(_chunks(b"video", b"data"), destination, 9)

    assert size_bytes == 9
    assert destination.read_bytes() == b"videodata"
    assert not destination.with_name("source.mp4.partial").exists()


async def test_save_upload_removes_partial_file_when_size_limit_is_exceeded(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "source.mp4"

    with pytest.raises(ValueError, match="exceeds"):
        await save_upload(_chunks(b"larger", b" than the limit"), destination, 10)

    assert not destination.exists()
    assert not destination.with_name("source.mp4.partial").exists()


async def test_save_upload_rejects_empty_content(tmp_path: Path) -> None:
    destination = tmp_path / "source.mp4"

    with pytest.raises(ValueError, match="empty"):
        await save_upload(_chunks(), destination, 10)

    assert not destination.exists()


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "http://127.0.0.1:8080/video",
        "http://user:password@example.com/video",
    ],
)
async def test_download_rejects_invalid_or_non_public_urls(url: str, tmp_path: Path) -> None:
    processor = MediaProcessor()

    with pytest.raises(ValueError):
        await processor.download_url(url, tmp_path / "video")

    assert not (tmp_path / "video").exists()


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="FFmpeg and ffprobe are required for the media integration test",
)
async def test_prepare_video_normalizes_video_and_extracts_mono_audio(tmp_path: Path) -> None:
    source_path = tmp_path / "fixture.mp4"
    process = await asyncio.create_subprocess_exec(
        "ffmpeg",
        "-nostdin",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "testsrc=duration=2:size=160x120:rate=24",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=1000:duration=2",
        "-shortest",
        "-c:v",
        "libx264",
        "-c:a",
        "aac",
        str(source_path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await process.communicate()
    assert process.returncode == 0, stderr.decode(errors="replace")

    artifacts = await MediaProcessor().prepare_video(source_path, tmp_path / "prepared")

    assert artifacts.media_path.is_file()
    assert artifacts.audio_path is not None
    assert artifacts.audio_path.is_file()
    assert artifacts.metadata.duration_s > 0
    assert artifacts.metadata.has_audio


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="FFmpeg and ffprobe are required for the media integration test",
)
async def test_prepare_video_supports_videos_without_audio(tmp_path: Path) -> None:
    source_path = tmp_path / "silent.mp4"
    process = await asyncio.create_subprocess_exec(
        "ffmpeg",
        "-nostdin",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "testsrc=duration=1:size=160x120:rate=24",
        "-c:v",
        "libx264",
        str(source_path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await process.communicate()
    assert process.returncode == 0, stderr.decode(errors="replace")

    artifacts = await MediaProcessor().prepare_video(source_path, tmp_path / "prepared")

    assert artifacts.media_path.is_file()
    assert artifacts.audio_path is None
    assert artifacts.metadata.duration_s > 0
    assert not artifacts.metadata.has_audio
