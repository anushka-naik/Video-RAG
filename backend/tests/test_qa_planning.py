from dataclasses import dataclass

import pytest

from app.ingestion.errors import IngestionError
from app.qa.planning import AnthropicQueryPlanner


@dataclass
class FakeTextBlock:
    text: str


@dataclass
class FakeUsage:
    input_tokens: int = 4
    output_tokens: int = 3


@dataclass
class FakeResponse:
    content: list[FakeTextBlock]
    usage: FakeUsage


class FakeMessages:
    def __init__(self, response_text: str) -> None:
        self.response_text = response_text
        self.system_prompt: str | None = None

    def create(
        self,
        *,
        model: str,
        max_tokens: int,
        system: str,
        messages: list[dict[str, str]],
    ) -> FakeResponse:
        self.system_prompt = system
        return FakeResponse([FakeTextBlock(self.response_text)], FakeUsage())


class FakeClient:
    def __init__(self, messages: FakeMessages) -> None:
        self.messages = messages


def _client_factory(messages: FakeMessages):
    def factory(*, api_key: str, timeout: int) -> FakeClient:
        assert api_key == "test-key"
        assert timeout == 10
        return FakeClient(messages)

    return factory


async def test_planner_keeps_original_and_deduplicates_rewrites() -> None:
    messages = FakeMessages('{"queries":["vector definition","What is a vector?"]}')
    planner = AnthropicQueryPlanner(
        api_key="test-key",
        model_name="configured-model",
        timeout_seconds=10,
        client_factory=_client_factory(messages),
    )

    plan = await planner.plan("What is a vector?")

    assert plan.queries == ["What is a vector?", "vector definition"]
    assert plan.usage == {"input_tokens": 4, "output_tokens": 3}
    assert messages.system_prompt is not None
    assert "do not answer it" in messages.system_prompt


async def test_planner_requires_configured_api_key_and_model() -> None:
    planner = AnthropicQueryPlanner(api_key=None, model_name=None)

    with pytest.raises(IngestionError, match="ANTHROPIC_API_KEY"):
        await planner.plan("Question?")
