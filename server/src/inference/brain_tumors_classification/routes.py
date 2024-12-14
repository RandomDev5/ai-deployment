from fastapi import APIRouter, File, UploadFile
from celery.result import AsyncResult

from src.inference.brain_tumors_classification.service import (
    BrainTumorClassificationService,
)
from src.inference.celery_jobs import celery_app, predict_brain_tumors_task

brain_tumors_classification_router = APIRouter()

# This line will be written live
brain_tumor_service = BrainTumorClassificationService(
    "./src/inference/brain_tumors_classification/weights/brain_tumors_classification.pt"
)


# Part 1: This is the simple deployment mechanism
@brain_tumors_classification_router.post(
    "",
    summary="Predict brain tumor type",
    description="Predict the type of brain tumor from an image",
)
async def predict(file: UploadFile = File(...)):
    """
    Submit a brain tumor prediction task.

    Args:
        inferenceSchema (InferenceSchema): Schema containing the instance URL of the DICOM image.

    Returns:
        dict: Contains the task ID of the submitted Celery task.
    """
    image_bytes = await file.read()  # This line will be written live
    prediction = brain_tumor_service.predict(
        image_bytes
    )  # This line will be written live
    return prediction  # This line will be written live


# Part 2: This is the advanced deployment mechanism
@brain_tumors_classification_router.post(
    "/background",
    summary="Submit a brain tumor prediction task",
    description="Submit an image for brain tumor prediction",
)
async def submit_prediction(file: UploadFile = File(...)):
    """
    Submit a brain tumor prediction task.

    Args:
        inferenceSchema (InferenceSchema): Schema containing the instance URL of the DICOM image.

    Returns:
        dict: Contains the task ID of the submitted Celery task.
    """
    image_bytes = await file.read()  # This line will be written live
    task = predict_brain_tumors_task.delay(
        image_bytes
    )  # This line will be written live
    return {"task_id": task.id}  # This line will be written live


# Part 2, the whole block will be written in the live session
@brain_tumors_classification_router.get(
    "/results/{task_id}",
    summary="Check task status",
    description="Query the status and result of a prediction task",
)
async def get_task_status(task_id: str):
    """
    Returns the status of the background task.
    """
    task_result = AsyncResult(task_id, app=celery_app)

    if task_result.state == "SUCCESS":
        return {
            "task_id": task_id,
            "status": task_result.state,
            "result": task_result.result,
        }
    elif task_result.state == "PENDING":
        return {
            "task_id": task_id,
            "status": "PENDING",
            "message": "Task is still processing.",
        }
    elif task_result.state == "FAILURE":
        return {
            "task_id": task_id,
            "status": "FAILURE",
            "message": str(task_result.info),
        }
    else:
        return {"task_id": task_id, "status": task_result.state}
