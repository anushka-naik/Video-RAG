from fastapi import APIRouter, Depends

from app.api.dependencies import get_collection_service
from app.api.schemas import CollectionCreate, CollectionDetails
from app.ingestion.service import CollectionService

router = APIRouter(prefix="/collections", tags=["collections"])


@router.post("", response_model=CollectionDetails, status_code=201)
async def create_collection(
    request: CollectionCreate,
    service: CollectionService = Depends(get_collection_service),
) -> CollectionDetails:
    collection = await service.create_collection(request.name, request.description)
    return CollectionDetails.model_validate(collection)
