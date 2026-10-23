import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel

from app.api import chat, collections, jobs, videos


class HealthResponse(BaseModel):
    status: str


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            runtime = application.state.qa_runtime
            if runtime is not None:
                await asyncio.to_thread(runtime.close)

    application = FastAPI(title="Video RAG API", version="0.1.0", lifespan=lifespan)
    application.state.qa_runtime = None
    application.state.qa_runtime_lock = asyncio.Lock()
    application.include_router(collections.router)
    application.include_router(videos.router)
    application.include_router(jobs.router)
    application.include_router(chat.router)

    @application.get("/health", response_model=HealthResponse, tags=["health"])
    async def health() -> HealthResponse:
        return HealthResponse(status="ok")

    return application


app = create_app()
