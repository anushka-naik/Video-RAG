import unicodedata
from pathlib import Path

from app.ingestion.media import run_media_command


class TesseractOCRReader:
    def __init__(
        self,
        executable: str = "tesseract",
        language: str = "eng",
        timeout_seconds: int = 120,
    ) -> None:
        if not executable.strip() or not language.strip() or timeout_seconds <= 0:
            raise ValueError("OCR executable, language, and timeout must be configured")
        self.executable = executable
        self.language = language
        self.timeout_seconds = timeout_seconds

    async def read(self, image_path: Path) -> str:
        text = await run_media_command(
            [
                self.executable,
                str(image_path),
                "stdout",
                "-l",
                self.language,
                "--psm",
                "6",
            ],
            self.timeout_seconds,
        )
        return normalize_ocr_text(text)


def normalize_ocr_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).split())
