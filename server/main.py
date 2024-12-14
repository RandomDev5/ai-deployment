import uvicorn
from fastapi import FastAPI

from src.inference.routes import inference_router

version = "v1"

app = FastAPI(
    title="AI Deployment with FastAPI and Celery",
    description="This is a project for deploying AI models with FastAPI and Celery.",
    version=version,
)

app.include_router(
    inference_router, prefix=f"/api/{version}/inference", tags=["Inference"]
)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
