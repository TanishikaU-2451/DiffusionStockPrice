from __future__ import annotations

import json
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse

from diffusion_finance.schemas import PredictionRequest, PredictionResponse
from diffusion_finance.service import ModelService

service = ModelService()


@asynccontextmanager
async def lifespan(_: FastAPI):
    service.ensure_ready()
    yield


app = FastAPI(
    title="Diffusion-Inspired Financial Anomaly Detection",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> dict:
    pipeline = service.ensure_ready()
    return pipeline.predict_window([point.model_dump() for point in request.window])


@app.get("/stream")
def stream(
    limit: int | None = Query(default=None, ge=1),
    sleep_ms: int = Query(default=0, ge=0, le=5_000),
) -> StreamingResponse:
    pipeline = service.ensure_ready()
    events = pipeline.simulate_stream(limit=limit)

    def event_generator():
        for event in events:
            payload = json.dumps(event, default=str)
            yield f"data: {payload}\n\n"
            if sleep_ms:
                time.sleep(sleep_ms / 1000)

    return StreamingResponse(event_generator(), media_type="text/event-stream")
