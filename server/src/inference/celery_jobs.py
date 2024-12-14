from celery import Celery
import torch

from src.inference.brain_tumors_classification.service import (
    BrainTumorClassificationService,
)

from src.config import Config

torch.set_num_threads(1)

# This will be written live
celery_app = Celery(
    "tasks",
    broker=Config.CELERY_BROKER_URL,
    backend=Config.CELERY_RESULT_BACKEND,
)

brain_tumor_service = None


# Part 2: This will be explained live, the whole block will be written live
@celery_app.task
def predict_brain_tumors_task(image_bytes: bytes):
    """
    Celery task for predicting brain tumor type.

    Args:
        instance_url (str): The DICOM file instance URL.

    Returns:
        dict: Prediction result with class and probability.
    """
    try:
        global brain_tumor_service
        if brain_tumor_service is None:
            brain_tumor_service = BrainTumorClassificationService(
                "./src/inference/brain_tumors_classification/weights/brain_tumors_classification.pt"
            )

        prediction = brain_tumor_service.predict(image_bytes)

        return prediction
    except Exception as e:
        raise e
