from __future__ import annotations

from threading import Lock

from .config import Settings, settings
from .pipeline import AnomalyDetectionPipeline


class ModelService:
    def __init__(self, config: Settings = settings) -> None:
        self.config = config
        self.pipeline = AnomalyDetectionPipeline(config)
        self._lock = Lock()

    def ensure_ready(
        self,
        refresh_data: bool = False,
        retrain: bool = False,
    ) -> AnomalyDetectionPipeline:
        with self._lock:
            if (
                not retrain
                and self.pipeline.model is not None
                and self.pipeline.scaler is not None
                and self.pipeline.kmeans is not None
                and self.pipeline.state is not None
            ):
                return self.pipeline

            if retrain:
                self.pipeline.fit(refresh_data=refresh_data)
                self.pipeline.save()
                return self.pipeline

            try:
                return self.pipeline.load()
            except FileNotFoundError:
                self.pipeline.fit(refresh_data=refresh_data)
                self.pipeline.save()
                return self.pipeline
