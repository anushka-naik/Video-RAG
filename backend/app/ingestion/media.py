import asyncio
import ipaddress
import json
import math
import socket
from collections.abc import AsyncIterable
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from pydantic import BaseModel, ValidationError

from app.ingestion.errors import IngestionError


class MediaProcessingError(IngestionError):
    pass


@dataclass(frozen=True, slots=True)
class MediaMetadata:
    duration_s: float
    has_audio: bool


@dataclass(frozen=True, slots=True)
class MediaArtifacts:
    media_path: Path
    audio_path: Path | None
    metadata: MediaMetadata


class _ProbeFormat(BaseModel):
    duration: str


class _ProbeStream(BaseModel):
    codec_type: str


class _ProbeOutput(BaseModel):
    format: _ProbeFormat
    streams: list[_ProbeStream]


class MediaProcessor:
    def __init__(
        self,
        ffmpeg_executable: str = "ffmpeg",
        ffprobe_executable: str = "ffprobe",
        timeout_seconds: int = 14400,
        max_download_bytes: int = 10_737_418_240,
    ) -> None:
        if timeout_seconds <= 0 or max_download_bytes <= 0:
            raise ValueError("Media time and size limits must be positive")
        self.ffmpeg_executable = ffmpeg_executable
        self.ffprobe_executable = ffprobe_executable
        self.timeout_seconds = timeout_seconds
        self.max_download_bytes = max_download_bytes

    async def download_url(self, url: str, video_directory: Path) -> Path:
        await validate_public_http_url(url)
        download_directory = video_directory / "source"
        download_directory.mkdir(parents=True, exist_ok=True)
        manifest_path = download_directory / "source.url"
        if manifest_path.is_file() and manifest_path.read_text() == url:
            source_path = await self._find_download(download_directory)
            if source_path is not None:
                return source_path

        output_template = str(download_directory / "source.%(ext)s")
        command = [
            "yt-dlp",
            "--no-playlist",
            "--no-progress",
            "--quiet",
            "--retries",
            "3",
            "--fragment-retries",
            "3",
            "--max-filesize",
            str(self.max_download_bytes),
            "--merge-output-format",
            "mp4",
            "--output",
            output_template,
            "--print",
            "after_move:filepath",
            url,
        ]
        stdout = await run_media_command(command, self.timeout_seconds)
        output_lines = [line for line in stdout.splitlines() if line.strip()]
        if not output_lines:
            raise MediaProcessingError("yt-dlp completed without reporting the downloaded file")
        source_path = Path(output_lines[-1]).resolve()
        if source_path.parent != download_directory.resolve() or not source_path.is_file():
            raise MediaProcessingError("yt-dlp reported a missing or unexpected downloaded file")
        await asyncio.to_thread(manifest_path.write_text, url)
        return source_path

    async def prepare_video(self, source_path: Path, video_directory: Path) -> MediaArtifacts:
        source_metadata = await self.probe(source_path)
        normalized_path = video_directory / "media.mp4"
        video_directory.mkdir(parents=True, exist_ok=True)
        if normalized_path.is_file():
            metadata = await self.probe(normalized_path)
        else:
            partial_path = video_directory / "media.partial.mp4"
            partial_path.unlink(missing_ok=True)
            await run_media_command(
                [
                    self.ffmpeg_executable,
                    "-nostdin",
                    "-y",
                    "-i",
                    str(source_path),
                    "-map",
                    "0:v:0",
                    "-map",
                    "0:a?",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "fast",
                    "-crf",
                    "23",
                    "-c:a",
                    "aac",
                    "-movflags",
                    "+faststart",
                    str(partial_path),
                ],
                self.timeout_seconds,
            )
            metadata = await self.probe(partial_path)
            partial_path.replace(normalized_path)

        if source_metadata.duration_s <= 0 or metadata.duration_s <= 0:
            raise MediaProcessingError("Video duration must be greater than zero")

        audio_path: Path | None = None
        expected_audio_path = video_directory / "audio.wav"
        if metadata.has_audio:
            if not expected_audio_path.is_file():
                partial_audio_path = video_directory / "audio.partial.wav"
                partial_audio_path.unlink(missing_ok=True)
                await run_media_command(
                    [
                        self.ffmpeg_executable,
                        "-nostdin",
                        "-y",
                        "-i",
                        str(normalized_path),
                        "-map",
                        "0:a:0",
                        "-vn",
                        "-ac",
                        "1",
                        "-ar",
                        "16000",
                        "-c:a",
                        "pcm_s16le",
                        str(partial_audio_path),
                    ],
                    self.timeout_seconds,
                )
                partial_audio_path.replace(expected_audio_path)
            audio_path = expected_audio_path
        else:
            expected_audio_path.unlink(missing_ok=True)

        return MediaArtifacts(normalized_path, audio_path, metadata)

    async def probe(self, media_path: Path) -> MediaMetadata:
        stdout = await run_media_command(
            [
                self.ffprobe_executable,
                "-v",
                "error",
                "-show_entries",
                "format=duration:stream=codec_type",
                "-of",
                "json",
                str(media_path),
            ],
            self.timeout_seconds,
        )
        try:
            output = _ProbeOutput.model_validate_json(stdout)
            duration_s = float(output.format.duration)
        except (ValidationError, ValueError, json.JSONDecodeError) as error:
            raise MediaProcessingError(
                f"ffprobe returned invalid media metadata for {media_path}"
            ) from error
        if not math.isfinite(duration_s):
            raise MediaProcessingError(f"ffprobe returned a non-finite duration for {media_path}")
        if not any(stream.codec_type == "video" for stream in output.streams):
            raise MediaProcessingError(f"Media file contains no video stream: {media_path}")
        return MediaMetadata(
            duration_s=duration_s,
            has_audio=any(stream.codec_type == "audio" for stream in output.streams),
        )

    async def _find_download(self, download_directory: Path) -> Path | None:
        valid_suffixes = {
            ".avi",
            ".flv",
            ".m4v",
            ".mkv",
            ".mov",
            ".mp4",
            ".mpeg",
            ".mpg",
            ".ts",
            ".webm",
        }
        for candidate in sorted(download_directory.glob("source.*")):
            if candidate.suffix not in valid_suffixes:
                continue
            await self.probe(candidate)
            return candidate
        return None


async def save_upload(
    chunks: AsyncIterable[bytes],
    destination: Path,
    maximum_size_bytes: int,
) -> int:
    if maximum_size_bytes <= 0:
        raise ValueError("Maximum upload size must be positive")
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial_path = destination.with_name(f"{destination.name}.partial")
    partial_path.unlink(missing_ok=True)
    size_bytes = 0
    completed = False
    try:
        with partial_path.open("wb") as output:
            async for chunk in chunks:
                size_bytes += len(chunk)
                if size_bytes > maximum_size_bytes:
                    raise ValueError(f"Upload exceeds the {maximum_size_bytes}-byte limit")
                await asyncio.to_thread(output.write, chunk)
        if size_bytes == 0:
            raise ValueError("Uploaded video is empty")
        partial_path.replace(destination)
        completed = True
        return size_bytes
    finally:
        if not completed:
            partial_path.unlink(missing_ok=True)


async def validate_public_http_url(url: str) -> None:
    parsed_url = urlsplit(url)
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.hostname:
        raise ValueError("Video source URL must use HTTP or HTTPS and include a hostname")
    if parsed_url.username is not None or parsed_url.password is not None:
        raise ValueError("Video source URL cannot include embedded credentials")
    try:
        addresses = await asyncio.get_running_loop().getaddrinfo(
            parsed_url.hostname,
            parsed_url.port or (443 if parsed_url.scheme == "https" else 80),
            type=socket.SOCK_STREAM,
        )
    except OSError as error:
        raise MediaProcessingError(
            f"Could not resolve video source host {parsed_url.hostname}"
        ) from error
    resolved_addresses = {ipaddress.ip_address(address[4][0]) for address in addresses}
    if not resolved_addresses or any(not address.is_global for address in resolved_addresses):
        raise ValueError("Video source URL must resolve only to public IP addresses")


async def run_media_command(command: list[str], timeout_seconds: int) -> str:
    try:
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError as error:
        raise MediaProcessingError(
            f"Required media executable was not found: {command[0]}"
        ) from error

    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout_seconds)
    except asyncio.CancelledError:
        process.kill()
        await process.communicate()
        raise
    except TimeoutError as error:
        process.kill()
        await process.communicate()
        raise MediaProcessingError(
            f"Media command exceeded its {timeout_seconds}-second timeout"
        ) from error

    if process.returncode != 0:
        detail = stderr.decode(errors="replace").strip()[-2000:]
        raise MediaProcessingError(
            f"{Path(command[0]).name} failed with exit code {process.returncode}: {detail}"
        )
    return stdout.decode(errors="replace")
