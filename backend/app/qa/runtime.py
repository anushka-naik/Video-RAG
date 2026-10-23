from dataclasses import dataclass

from app.config import Settings
from app.ingestion.providers import (
    FastEmbedTextEncoder,
    QdrantVectorStore,
    SiglipImageEncoder,
    create_qdrant_vector_store,
)
from app.qa.answering import AnthropicGroundedAnswerer
from app.qa.planning import AnthropicQueryPlanner


@dataclass(slots=True)
class QARuntime:
    vector_store: QdrantVectorStore
    text_encoder: FastEmbedTextEncoder
    visual_encoder: SiglipImageEncoder
    planner: AnthropicQueryPlanner
    answerer: AnthropicGroundedAnswerer

    def close(self) -> None:
        self.vector_store.close()


def create_qa_runtime(settings: Settings) -> QARuntime:
    return QARuntime(
        vector_store=create_qdrant_vector_store(
            settings.qdrant_url,
            settings.request_timeout_seconds,
        ),
        text_encoder=FastEmbedTextEncoder(settings.text_embedding_model),
        visual_encoder=SiglipImageEncoder(settings.visual_embedding_model),
        planner=AnthropicQueryPlanner(
            api_key=(
                settings.anthropic_api_key.get_secret_value()
                if settings.anthropic_api_key is not None
                else None
            ),
            model_name=settings.anthropic_planning_model,
            timeout_seconds=settings.request_timeout_seconds,
        ),
        answerer=AnthropicGroundedAnswerer(
            api_key=(
                settings.anthropic_api_key.get_secret_value()
                if settings.anthropic_api_key is not None
                else None
            ),
            model_name=settings.anthropic_fast_model,
            timeout_seconds=settings.request_timeout_seconds,
        ),
    )
