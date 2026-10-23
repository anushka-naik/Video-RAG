import base64
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

import pytest

from app.ingestion.errors import IngestionError
from app.qa.answering import (
    ABSTENTION_ANSWER,
    AnthropicGroundedAnswerer,
    OllamaGroundedAnswerer,
    OpenAIGroundedAnswerer,
    create_grounded_answerer,
)
from app.qa.evidence import RetrievedEvidence


@dataclass
class FakeTextBlock:
    text: str


@dataclass
class FakeUsage:
    input_tokens: int = 12
    output_tokens: int = 7


@dataclass
class FakeResponse:
    content: list[FakeTextBlock]
    usage: FakeUsage


class FakeMessages:
    def __init__(self, response_text: str) -> None:
        self.response_text = response_text
        self.request: dict[str, object] | None = None

    def create(
        self,
        *,
        model: str,
        max_tokens: int,
        system: str,
        messages: list[dict[str, object]],
    ) -> FakeResponse:
        self.request = {
            "model": model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": messages,
        }
        return FakeResponse([FakeTextBlock(self.response_text)], FakeUsage())


class FakeClient:
    def __init__(self, messages: FakeMessages) -> None:
        self.messages = messages


@dataclass
class FakeOpenAIUsage:
    prompt_tokens: int = 14
    completion_tokens: int = 8


@dataclass
class FakeOpenAIMessage:
    content: str | None


@dataclass
class FakeOpenAIChoice:
    message: FakeOpenAIMessage


@dataclass
class FakeOpenAIResponse:
    choices: list[FakeOpenAIChoice]
    usage: FakeOpenAIUsage | None = field(default_factory=FakeOpenAIUsage)


class FakeOpenAICompletions:
    def __init__(self, response_text: str | None) -> None:
        self.response_text = response_text
        self.request: dict[str, object] | None = None

    def create(
        self,
        *,
        model: str,
        max_tokens: int,
        messages: list[dict[str, object]],
        response_format: dict[str, str] | None = None,
        extra_body: dict[str, object] | None = None,
    ) -> FakeOpenAIResponse:
        self.request = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        if response_format is not None:
            self.request["response_format"] = response_format
        if extra_body is not None:
            self.request["extra_body"] = extra_body
        return FakeOpenAIResponse([FakeOpenAIChoice(FakeOpenAIMessage(self.response_text))])


class FakeOpenAIChat:
    def __init__(self, completions: FakeOpenAICompletions) -> None:
        self.completions = completions


class FakeOpenAIClient:
    def __init__(self, completions: FakeOpenAICompletions) -> None:
        self.chat = FakeOpenAIChat(completions)


def _openai_client_factory(
    completions: FakeOpenAICompletions,
    expected_timeout: int = 120,
    expected_base_url: str | None = None,
):
    def factory(
        *,
        api_key: str,
        timeout: int,
        base_url: str | None = None,
    ) -> FakeOpenAIClient:
        assert api_key in {"test-openai-key", "ollama"}
        assert timeout == expected_timeout
        assert base_url == expected_base_url
        return FakeOpenAIClient(completions)

    return factory


def _client_factory(messages: FakeMessages):
    def factory(*, api_key: str, timeout: int) -> FakeClient:
        assert api_key == "test-key"
        assert timeout == 15
        return FakeClient(messages)

    return factory


def _evidence(image_path: Path | None = None) -> RetrievedEvidence:
    return RetrievedEvidence(
        id=uuid4(),
        video_id=uuid4(),
        video_title="Test lecture",
        start_s=2,
        end_s=3,
        source_type="keyframe" if image_path is not None else "transcript",
        transcript_text="" if image_path is not None else "A vector is ordered data.",
        slide_text=None,
        ocr_text="Vectors" if image_path is not None else None,
        image_path=image_path,
        rank_score=0.01,
    )


async def test_grounded_answerer_validates_ids_and_sends_visual_evidence(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "frame.jpg"
    image_path.write_bytes(b"jpeg fixture")
    evidence = _evidence(image_path)
    messages = FakeMessages(
        f'{{"answer":"The slide shows vectors.","citation_ids":["{evidence.id}"],"abstain":false}}'
    )
    answerer = AnthropicGroundedAnswerer(
        api_key="test-key",
        model_name="configured-vision-model",
        timeout_seconds=15,
        client_factory=_client_factory(messages),
    )

    generated = await answerer.answer("What is on the slide?", [evidence])

    assert generated.answer == "The slide shows vectors."
    assert generated.citation_ids == [str(evidence.id)]
    assert generated.usage == {"input_tokens": 12, "output_tokens": 7}
    assert messages.request is not None
    request_messages = messages.request["messages"]
    assert isinstance(request_messages, list)
    content = request_messages[0]["content"]
    assert isinstance(content, list)
    image_block = next(block for block in content if block["type"] == "image")
    assert image_block["source"]["data"] == base64.b64encode(b"jpeg fixture").decode("ascii")


async def test_grounded_answerer_rejects_unknown_citation_ids() -> None:
    evidence = _evidence()
    messages = FakeMessages(
        '{"answer":"Unsupported.","citation_ids":["not-retrieved"],"abstain":false}'
    )
    answerer = AnthropicGroundedAnswerer(
        api_key="test-key",
        model_name="configured-model",
        client_factory=_client_factory_with_default_timeout(messages),
    )

    with pytest.raises(IngestionError, match="not retrieved"):
        await answerer.answer("What is a vector?", [evidence])


async def test_grounded_answerer_abstains_without_evidence() -> None:
    answerer = AnthropicGroundedAnswerer(api_key=None, model_name=None)

    generated = await answerer.answer("Question?", [])

    assert generated.answer == ABSTENTION_ANSWER
    assert generated.abstain
    assert generated.citation_ids == []
    assert generated.usage == {}


async def test_openai_answerer_validates_citations_and_sends_vision_frames(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "frame.jpg"
    image_path.write_bytes(b"jpeg fixture")
    evidence = _evidence(image_path)
    completions = FakeOpenAICompletions(
        f'{{"answer":"The slide shows vectors.","citation_ids":["{evidence.id}"],"abstain":false}}'
    )
    answerer = OpenAIGroundedAnswerer(
        api_key="test-openai-key",
        model_name="configured-vision-model",
        timeout_seconds=15,
        client_factory=_openai_client_factory(completions, expected_timeout=15),
    )

    generated = await answerer.answer("What is on the slide?", [evidence])

    assert generated.answer == "The slide shows vectors."
    assert generated.citation_ids == [str(evidence.id)]
    assert generated.usage == {"input_tokens": 14, "output_tokens": 8}
    assert completions.request is not None
    request_messages = completions.request["messages"]
    assert isinstance(request_messages, list)
    content = request_messages[1]["content"]
    assert isinstance(content, list)
    image_block = next(block for block in content if block["type"] == "image_url")
    assert image_block["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert completions.request["response_format"] == {"type": "json_object"}


async def test_ollama_answerer_uses_native_api_and_vision_payload(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "frame.jpg"
    image_path.write_bytes(b"jpeg fixture")
    evidence = _evidence(image_path)
    captured: dict[str, object] = {}

    def fake_request(
        endpoint: str,
        payload: dict[str, object],
        timeout_seconds: int,
    ) -> dict[str, object]:
        captured.update({"endpoint": endpoint, "payload": payload, "timeout": timeout_seconds})
        return {
            "message": {
                "content": (
                    f'{{"answer":"The slide shows vectors.","citation_ids":'
                    f'["{evidence.id}"],"abstain":false}}'
                )
            },
            "prompt_eval_count": 100,
            "eval_count": 20,
        }

    answerer = OllamaGroundedAnswerer(
        model_name="qwen3-vl:8b",
        base_url="http://localhost:11434",
        context_length=16384,
        request_json=fake_request,
    )

    generated = await answerer.answer("What is on the slide?", [evidence])

    assert generated.answer == "The slide shows vectors."
    assert generated.usage == {"input_tokens": 100, "output_tokens": 20}
    assert captured["endpoint"] == "http://localhost:11434/api/chat"
    payload = captured["payload"]
    assert isinstance(payload, dict)
    assert payload["format"] == "json"
    assert payload["options"] == {
        "num_ctx": 16384,
        "num_predict": 2048,
        "temperature": 0,
    }
    request_messages = payload["messages"]
    assert isinstance(request_messages, list)
    content = request_messages[1]["content"]
    assert isinstance(content, str)
    assert isinstance(request_messages[1]["images"], list)
    assert request_messages[1]["images"][0] == base64.b64encode(b"jpeg fixture").decode("ascii")


async def test_openai_answerer_rejects_unknown_citation_ids() -> None:
    evidence = _evidence()
    completions = FakeOpenAICompletions(
        '{"answer":"Unsupported.","citation_ids":["not-retrieved"],"abstain":false}'
    )
    answerer = OpenAIGroundedAnswerer(
        api_key="test-openai-key",
        model_name="configured-model",
        client_factory=_openai_client_factory(completions),
    )

    with pytest.raises(IngestionError, match="not retrieved"):
        await answerer.answer("What is a vector?", [evidence])


async def test_openai_answerer_abstains_without_evidence() -> None:
    answerer = OpenAIGroundedAnswerer(api_key=None, model_name=None)

    generated = await answerer.answer("Question?", [])

    assert generated.answer == ABSTENTION_ANSWER
    assert generated.abstain
    assert generated.citation_ids == []


def test_answerer_factory_supports_local_and_hosted_providers() -> None:
    assert isinstance(
        create_grounded_answerer("openai", "key", "model", 30),
        OpenAIGroundedAnswerer,
    )
    ollama_answerer = create_grounded_answerer(
        "ollama",
        None,
        "qwen3-vl:8b",
        30,
        "http://localhost:11434/v1",
    )
    assert isinstance(ollama_answerer, OllamaGroundedAnswerer)
    assert ollama_answerer.base_url == "http://localhost:11434"
    assert ollama_answerer.context_length == 16384
    assert isinstance(
        create_grounded_answerer("anthropic", "key", "model", 30),
        AnthropicGroundedAnswerer,
    )
    with pytest.raises(ValueError, match="Unsupported answer provider"):
        create_grounded_answerer("unknown", "key", "model", 30)


def _client_factory_with_default_timeout(messages: FakeMessages):
    def factory(*, api_key: str, timeout: int) -> FakeClient:
        assert api_key == "test-key"
        assert timeout == 120
        return FakeClient(messages)

    return factory
