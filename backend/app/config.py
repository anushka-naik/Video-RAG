from pathlib import Path

from pydantic import AliasChoices, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    app_env: str = "development"
    database_url: str = "postgresql+asyncpg://video_rag:video_rag@localhost:5432/video_rag"
    redis_url: str = "redis://:video_rag@localhost:6379/0"
    qdrant_url: str = "http://localhost:6333"
    anthropic_api_key: SecretStr | None = None
    anthropic_planning_model: str | None = None
    anthropic_fast_model: str | None = None
    openai_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("VIDEO_RAG_OPENAI_API_KEY", "OPENAI_API_KEY"),
    )
    openai_model: str | None = Field(
        default=None,
        validation_alias=AliasChoices("VIDEO_RAG_OPENAI_MODEL", "openai_model"),
    )
    answer_provider: str = Field(
        default="ollama",
        validation_alias=AliasChoices("VIDEO_RAG_ANSWER_PROVIDER", "answer_provider"),
    )
    ollama_base_url: str = Field(
        default="http://localhost:11434/v1",
        validation_alias=AliasChoices("VIDEO_RAG_OLLAMA_BASE_URL", "ollama_base_url"),
    )
    ollama_model: str | None = Field(
        default="qwen3-vl:8b",
        validation_alias=AliasChoices("VIDEO_RAG_OLLAMA_MODEL", "ollama_model"),
    )
    ollama_context_length: int = Field(
        default=16384,
        validation_alias=AliasChoices(
            "VIDEO_RAG_OLLAMA_CONTEXT_LENGTH",
            "ollama_context_length",
        ),
    )
    ollama_timeout_seconds: int = Field(
        default=300,
        validation_alias=AliasChoices(
            "VIDEO_RAG_OLLAMA_TIMEOUT_SECONDS",
            "ollama_timeout_seconds",
        ),
    )
    request_timeout_seconds: int = 30
    media_storage_path: Path = PROJECT_ROOT / "media"
    ffmpeg_executable: str = "ffmpeg"
    ffprobe_executable: str = "ffprobe"
    media_timeout_seconds: int = 14400
    max_video_upload_bytes: int = 10_737_418_240
    whisper_model: str = "small"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"
    keyframe_interval_seconds: int = 10
    ocr_concurrency: int = 4
    summary_window_seconds: int = 900
    tesseract_executable: str = "tesseract"
    ocr_language: str = "eng"
    text_embedding_model: str = "BAAI/bge-base-en-v1.5"
    visual_embedding_model: str = "google/siglip-base-patch16-224"

    @field_validator("media_storage_path", mode="before")
    @classmethod
    def resolve_media_storage_path(cls, value: Path | str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else PROJECT_ROOT / path

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_prefix="VIDEO_RAG_",
        env_ignore_empty=True,
        extra="ignore",
    )


settings = Settings()
