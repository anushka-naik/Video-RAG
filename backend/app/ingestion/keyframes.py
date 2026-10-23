import asyncio
import logging
from collections.abc import Sequence
from importlib import import_module
from pathlib import Path
from typing import Protocol, cast
from uuid import UUID, uuid5

from app.ingestion.errors import IngestionError
from app.ingestion.media import run_media_command
from app.ingestion.pipeline import KeyframeArtifact

logger = logging.getLogger(__name__)


class FrameTimecode(Protocol):
    def get_seconds(self) -> float: ...


class VideoCapture(Protocol):
    def release(self) -> None: ...


class VideoStream(Protocol):
    @property
    def capture(self) -> VideoCapture: ...


class SceneManager(Protocol):
    def add_detector(self, detector: object) -> None: ...

    def detect_scenes(self, video: VideoStream) -> None: ...

    def get_scene_list(self) -> Sequence[tuple[FrameTimecode, FrameTimecode]]: ...


class SceneDetectionModule(Protocol):
    def open_video(self, path: str) -> VideoStream: ...

    def SceneManager(self) -> SceneManager: ...


class ContentDetectorModule(Protocol):
    def ContentDetector(self) -> object: ...


class ImageObject(Protocol):
    def __enter__(self) -> object: ...

    def __exit__(self, error_type: object, error: object, traceback: object) -> bool | None: ...


class ImageModule(Protocol):
    def open(self, path: str) -> ImageObject: ...


class ImageHashModule(Protocol):
    def phash(self, image: object) -> object: ...


class PySceneDetectKeyframeExtractor:
    def __init__(
        self,
        ffmpeg_executable: str = "ffmpeg",
        interval_seconds: int = 10,
        phash_distance: int = 5,
        timeout_seconds: int = 14400,
    ) -> None:
        if interval_seconds <= 0 or not 0 <= phash_distance <= 64 or timeout_seconds <= 0:
            raise ValueError("Keyframe interval, perceptual distance, and timeout are invalid")
        self.ffmpeg_executable = ffmpeg_executable
        self.interval_seconds = interval_seconds
        self.phash_distance = phash_distance
        self.timeout_seconds = timeout_seconds

    async def extract(
        self,
        video_id: UUID,
        media_path: Path,
        duration_s: float,
    ) -> list[KeyframeArtifact]:
        if duration_s <= 0:
            raise ValueError("Video duration must be greater than zero")
        timestamps = await asyncio.to_thread(self._detect_scene_times, media_path)
        if not timestamps:
            timestamps = [
                float(timestamp) for timestamp in range(0, int(duration_s), self.interval_seconds)
            ]
        timestamps = sorted({timestamp for timestamp in timestamps if 0 <= timestamp < duration_s})
        if not timestamps:
            timestamps = [0.0]

        keyframe_directory = media_path.parent / "keyframes"
        keyframe_directory.mkdir(parents=True, exist_ok=True)
        keyframes: list[KeyframeArtifact] = []
        seen_hashes: list[str] = []
        for timestamp_s in timestamps:
            timestamp_ms = round(timestamp_s * 1000)
            image_path = keyframe_directory / f"frame-{timestamp_ms:012d}.jpg"
            if not image_path.is_file():
                await run_media_command(
                    [
                        self.ffmpeg_executable,
                        "-nostdin",
                        "-y",
                        "-ss",
                        f"{timestamp_s:.3f}",
                        "-i",
                        str(media_path),
                        "-frames:v",
                        "1",
                        "-q:v",
                        "2",
                        str(image_path),
                    ],
                    self.timeout_seconds,
                )
            phash = await asyncio.to_thread(_calculate_phash, image_path)
            if any(
                _hash_distance(phash, existing) <= self.phash_distance for existing in seen_hashes
            ):
                image_path.unlink(missing_ok=True)
                continue
            seen_hashes.append(phash)
            keyframes.append(
                KeyframeArtifact(
                    id=uuid5(video_id, f"keyframe:{timestamp_s:.3f}"),
                    timestamp_s=timestamp_s,
                    image_path=image_path,
                    phash=phash,
                )
            )
        return keyframes

    def _detect_scene_times(self, media_path: Path) -> list[float]:
        try:
            scene_module = cast(
                SceneDetectionModule,
                import_module("scenedetect"),
            )
            detector_module = cast(
                ContentDetectorModule,
                import_module("scenedetect.detectors"),
            )
        except ImportError:
            logger.warning("PySceneDetect is not installed; using fixed-interval keyframes")
            return []

        video = scene_module.open_video(str(media_path))
        try:
            manager = scene_module.SceneManager()
            manager.add_detector(detector_module.ContentDetector())
            manager.detect_scenes(video)
            return [start.get_seconds() for start, _ in manager.get_scene_list()]
        except (OSError, RuntimeError, ValueError):
            logger.warning(
                "PySceneDetect failed for %s; using fixed-interval keyframes",
                media_path,
                exc_info=True,
            )
            return []
        finally:
            video.capture.release()


def _calculate_phash(image_path: Path) -> str:
    try:
        image_module = cast(ImageModule, import_module("PIL.Image"))
        image_hash_module = cast(ImageHashModule, import_module("imagehash"))
    except ImportError as error:
        raise IngestionError(
            "Pillow and ImageHash are required to deduplicate keyframes; "
            "install the ingestion dependency group"
        ) from error
    try:
        with image_module.open(str(image_path)) as image:
            return str(image_hash_module.phash(image))
    except OSError as error:
        raise IngestionError(f"Could not calculate a perceptual hash for {image_path}") from error


def _hash_distance(first: str, second: str) -> int:
    return (int(first, 16) ^ int(second, 16)).bit_count()
