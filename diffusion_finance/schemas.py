from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class OHLCVPoint(BaseModel):
    date: datetime | None = None
    open: float
    high: float
    low: float
    close: float
    volume: float


class PredictionRequest(BaseModel):
    window: list[OHLCVPoint] = Field(..., min_length=20, max_length=20)


class PredictionResponse(BaseModel):
    timestamp: datetime | None
    anomaly_score: float
    anomaly_flag: bool
    label: str
    cluster_id: int
    reconstruction_error: float
    latent_distance: float


class StreamEvent(PredictionResponse):
    close: float
