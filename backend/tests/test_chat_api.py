from uuid import uuid4

from httpx import ASGITransport, AsyncClient

from app.api.dependencies import get_video_question_service
from app.main import app
from app.qa.schemas import ChatResponse


class FakeQuestionService:
    def __init__(self) -> None:
        self.question: str | None = None

    async def ask(
        self,
        question: str,
        collection_id: object,
        video_id: object,
        thread_id: object,
        top_k: int,
    ) -> ChatResponse:
        self.question = question
        return ChatResponse(
            thread_id=uuid4(),
            assistant_message_id=uuid4(),
            answer="Supported answer.",
            citations=[],
            usage={},
        )


async def test_chat_endpoint_returns_answer_and_thread_identifiers() -> None:
    service = FakeQuestionService()
    app.dependency_overrides[get_video_question_service] = lambda: service
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/chat",
                json={
                    "question": "What happened?",
                    "collection_id": str(uuid4()),
                },
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["answer"] == "Supported answer."
    assert service.question == "What happened?"
    assert response.json()["thread_id"]
    assert response.json()["assistant_message_id"]


async def test_chat_endpoint_rejects_ambiguous_scope() -> None:
    app.dependency_overrides[get_video_question_service] = lambda: FakeQuestionService()
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/chat",
                json={
                    "question": "What happened?",
                    "collection_id": str(uuid4()),
                    "video_id": str(uuid4()),
                },
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 422
