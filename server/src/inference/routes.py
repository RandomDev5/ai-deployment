from fastapi import APIRouter

from src.inference.brain_tumors_classification.routes import (
    brain_tumors_classification_router,
)

inference_router = APIRouter()

inference_router.include_router(
    brain_tumors_classification_router, prefix="/brain-tumors-classification"
)
