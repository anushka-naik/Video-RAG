import asyncio
from collections.abc import Sequence
from importlib import import_module
from typing import Protocol, cast

from pydantic import BaseModel, ConfigDict, Field

from app.ingestion.errors import IngestionError


class PlannedQueries(BaseModel):
    queries: list[str]
    usage: dict[str, int]


class QueryPlanner(Protocol):
    async def plan(self, question: str) -> PlannedQueries: ...


class _PlannedQueryOutput(BaseModel):
    queries: list[str] = Field(min_length=1, max_length=2)

    model_config = ConfigDict(extra="forbid")


class _TextBlock(Protocol):
    text: str


class _Usage(Protocol):
    input_tokens: int
    output_tokens: int


class _Response(Protocol):
    content: Sequence[_TextBlock]
    usage: _Usage


class _Messages(Protocol):
    def create(
        self,
        *,
        model: str,
        max_tokens: int,
        system: str,
        messages: list[dict[str, str]],
    ) -> _Response: ...


class _Client(Protocol):
    messages: _Messages


class _ClientFactory(Protocol):
    def __call__(self, *, api_key: str, timeout: int) -> _Client: ...


class _AnthropicModule(Protocol):
    Anthropic: _ClientFactory


class AnthropicQueryPlanner:
    def __init__(
        self,
        api_key: str | None,
        model_name: str | None,
        timeout_seconds: int = 30,
        max_output_tokens: int = 512,
        client_factory: _ClientFactory | None = None,
    ) -> None:
        if timeout_seconds <= 0 or max_output_tokens <= 0:
            raise ValueError("Planner timeout and output limit must be positive")
        self.api_key = api_key
        self.model_name = model_name
        self.timeout_seconds = timeout_seconds
        self.max_output_tokens = max_output_tokens
        self.client_factory = client_factory
        self.client: _Client | None = None

    async def plan(self, question: str) -> PlannedQueries:
        return await asyncio.to_thread(self._plan, question)

    def _plan(self, question: str) -> PlannedQueries:
        client = self._get_client()
        try:
            response = client.messages.create(
                model=self.model_name or "",
                max_tokens=self.max_output_tokens,
                system=(
                    "Create up to two concise semantic search rewrites of the user question. "
                    "Preserve its intent and named entities; do not answer it, add assumed facts, "
                    "or introduce entities not present in the question. Return only JSON with a "
                    "single key 'queries' containing one or two strings."
                ),
                messages=[{"role": "user", "content": question}],
            )
        except (OSError, RuntimeError, ValueError) as error:
            raise IngestionError(f"Retrieval query planning failed: {error}") from error
        if not response.content:
            raise IngestionError("Retrieval query planning returned an empty response")
        try:
            output = _PlannedQueryOutput.model_validate_json(response.content[0].text)
        except ValueError as error:
            raise IngestionError(
                f"Retrieval query planning returned invalid JSON or a schema: {error}"
            ) from error
        queries = [question.strip()]
        for query in output.queries:
            normalized_query = query.strip()
            if not normalized_query:
                raise IngestionError("Retrieval query planner returned an empty query")
            if normalized_query.casefold() not in {item.casefold() for item in queries}:
                queries.append(normalized_query)
        return PlannedQueries(
            queries=queries[:3],
            usage={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
        )

    def _get_client(self) -> _Client:
        if self.client is not None:
            return self.client
        if not self.api_key:
            raise IngestionError("VIDEO_RAG_ANTHROPIC_API_KEY is required for retrieval planning")
        if not self.model_name:
            raise IngestionError(
                "VIDEO_RAG_ANTHROPIC_PLANNING_MODEL is required for retrieval planning"
            )
        if self.client_factory is None:
            try:
                module = cast(_AnthropicModule, import_module("anthropic"))
            except ImportError as error:
                raise IngestionError(
                    "anthropic is not installed; install the ingestion dependency group"
                ) from error
            self.client_factory = module.Anthropic
        try:
            self.client = self.client_factory(
                api_key=self.api_key,
                timeout=self.timeout_seconds,
            )
        except (OSError, RuntimeError, ValueError) as error:
            raise IngestionError(f"Could not initialize Anthropic client: {error}") from error
        return self.client
