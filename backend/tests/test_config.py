from pathlib import Path

from app.config import PROJECT_ROOT, Settings


def test_relative_media_storage_path_is_resolved_from_project_root() -> None:
    settings = Settings(_env_file=None, media_storage_path=Path("media"))

    assert settings.media_storage_path == PROJECT_ROOT / "media"


def test_openai_answer_configuration_accepts_standard_environment_names(
    monkeypatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("VIDEO_RAG_ANSWER_PROVIDER", "ollama")
    monkeypatch.setenv("VIDEO_RAG_OLLAMA_MODEL", "qwen3-vl:8b")
    monkeypatch.setenv("VIDEO_RAG_OLLAMA_CONTEXT_LENGTH", "16384")
    monkeypatch.setenv("VIDEO_RAG_OLLAMA_TIMEOUT_SECONDS", "300")
    monkeypatch.setenv("VIDEO_RAG_OPENAI_MODEL", "vision-model")
    settings = Settings(_env_file=None)

    assert settings.openai_api_key is not None
    assert settings.openai_api_key.get_secret_value() == "test-key"
    assert settings.answer_provider == "ollama"
    assert settings.openai_model == "vision-model"
    assert settings.ollama_model == "qwen3-vl:8b"
    assert settings.ollama_context_length == 16384
    assert settings.ollama_timeout_seconds == 300
    assert settings.ollama_base_url == "http://localhost:11434/v1"
