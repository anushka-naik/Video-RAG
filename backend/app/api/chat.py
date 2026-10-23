from fastapi import APIRouter, Depends, HTTPException, status

from app.api.dependencies import get_video_question_service
from app.ingestion.errors import IngestionError
from app.qa.schemas import ChatRequest, ChatResponse
from app.qa.service import (
    CollectionNotFound,
    ThreadNotFound,
    ThreadScopeMismatch,
    VideoNotFound,
    VideoNotReady,
    VideoQuestionService,
)

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
async def ask_video_question(
    request: ChatRequest,
    service: VideoQuestionService = Depends(get_video_question_service),
) -> ChatResponse:
    try:
        return await service.ask(
            question=request.question,
            collection_id=request.collection_id,
            video_id=request.video_id,
            thread_id=request.thread_id,
            top_k=request.top_k,
        )
    except CollectionNotFound as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except VideoNotFound as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ThreadNotFound as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except VideoNotReady as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except ThreadScopeMismatch as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except IngestionError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(error),
        ) from error
