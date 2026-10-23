import asyncio
import importlib.util
import shutil
from pathlib import Path
from uuid import uuid4

import pytest

from app.ingestion.keyframes import PySceneDetectKeyframeExtractor, _hash_distance
from app.ingestion.ocr import normalize_ocr_text


def test_hash_distance_counts_different_bits() -> None:
    assert _hash_distance("0000", "0003") == 2


def test_ocr_normalization_unifies_whitespace_and_unicode() -> None:
    assert normalize_ocr_text("  A\u00a0slide\n with   text  ") == "A slide with text"


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None
    or shutil.which("ffprobe") is None
    or importlib.util.find_spec("scenedetect") is None
    or importlib.util.find_spec("PIL") is None
    or importlib.util.find_spec("imagehash") is None,
    reason="FFmpeg, PySceneDetect, Pillow, and ImageHash are required",
)
async def test_keyframe_extraction_uses_scene_boundaries_and_perceptual_hash(
    tmp_path: Path,
) -> None:
    media_path = tmp_path / "fixture.mp4"
    process = await asyncio.create_subprocess_exec(
        "ffmpeg",
        "-nostdin",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "testsrc=duration=2:size=160x120:rate=24",
        "-c:v",
        "libx264",
        str(media_path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await process.communicate()
    assert process.returncode == 0, stderr.decode(errors="replace")

    keyframes = await PySceneDetectKeyframeExtractor(interval_seconds=1).extract(
        uuid4(),
        media_path,
        2,
    )

    assert keyframes
    assert keyframes[0].timestamp_s == 0
    assert all(frame.image_path.is_file() for frame in keyframes)
    assert all(frame.phash is not None for frame in keyframes)
