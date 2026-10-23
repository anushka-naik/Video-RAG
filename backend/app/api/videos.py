from collections.abc import AsyncIterator
from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from app.api.dependencies import get_video_ingestion_service, get_video_service
from app.api.schemas import IngestionAccepted, VideoCreate, VideoDetails
from app.ingestion.queue import QueueUnavailable
from app.ingestion.service import (
    CollectionNotFound,
    VideoIngestionService,
    VideoService,
)

router = APIRouter(prefix="/videos", tags=["videos"])
UPLOAD_CHUNK_SIZE = 1_048_576


@router.post("", response_model=IngestionAccepted, status_code=status.HTTP_202_ACCEPTED)
async def create_video(
    request: VideoCreate,
    service: VideoIngestionService = Depends(get_video_ingestion_service),
) -> IngestionAccepted:
    try:
        video, job = await service.create_video(
            collection_id=request.collection_id,
            title=request.title,
            source_url=request.source_url,
            upload_chunks=None,
            upload_filename=None,
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except CollectionNotFound as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except QueueUnavailable as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    return IngestionAccepted(video_id=video.id, job_id=job.id, status=video.status)


@router.post("/upload", response_model=IngestionAccepted, status_code=status.HTTP_202_ACCEPTED)
async def upload_video(
    collection_id: UUID = Form(),
    title: str = Form(min_length=1, max_length=500),
    file: UploadFile = File(),
    service: VideoIngestionService = Depends(get_video_ingestion_service),
) -> IngestionAccepted:
    try:
        video, job = await service.create_video(
            collection_id=collection_id,
            title=title,
            source_url=None,
            upload_chunks=_read_upload(file),
            upload_filename=file.filename,
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except CollectionNotFound as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except QueueUnavailable as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    finally:
        await file.close()
    return IngestionAccepted(video_id=video.id, job_id=job.id, status=video.status)


@router.get("/{video_id}", response_model=VideoDetails)
async def get_video(
    video_id: UUID,
    service: VideoService = Depends(get_video_service),
) -> VideoDetails:
    video = await service.get_video(video_id)
    if video is None:
        raise HTTPException(status_code=404, detail=f"Video {video_id} does not exist")
    return VideoDetails.model_validate(video)


async def _read_upload(upload: UploadFile) -> AsyncIterator[bytes]:
    while chunk := await upload.read(UPLOAD_CHUNK_SIZE):
        yield chunk
