from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException

from app.api.dependencies import get_job_service
from app.api.schemas import JobDetails
from app.ingestion.service import JobService

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.get("/{job_id}", response_model=JobDetails)
async def get_job(
    job_id: UUID,
    service: JobService = Depends(get_job_service),
) -> JobDetails:
    job = await service.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} does not exist")
    return JobDetails.model_validate(job)
