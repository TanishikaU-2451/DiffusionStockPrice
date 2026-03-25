from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans

from .config import Settings, settings
from .data import StockDataClient
from .features import WindowBundle, create_window_bundle, fit_scaler, flatten_windows, summarize_windows
from .model import AutoencoderTrainer, DenoisingAutoencoder


@dataclass(slots=True)
class PipelineState:
    threshold: float
    score_mean: float
    score_std: float
    embedding_mean: np.ndarray
    cluster_label_map: dict[int, str]


class AnomalyDetectionPipeline:
    def __init__(self, config: Settings = settings) -> None:
        self.config = config
        self.data_client = StockDataClient(config)
        self.trainer = AutoencoderTrainer(config)
        self.scaler = None
        self.model: DenoisingAutoencoder | None = None
        self.kmeans: KMeans | None = None
        self.state: PipelineState | None = None
        self.history: pd.DataFrame | None = None
        self.training_df: pd.DataFrame | None = None

    def fit(self, refresh_data: bool = False) -> "AnomalyDetectionPipeline":
        history = self.data_client.fetch_all_history(refresh=refresh_data)
        self.history = history
        self.scaler = fit_scaler(history, self.config)
        bundle = create_window_bundle(history, self.scaler, self.config)
        train_matrix = flatten_windows(bundle.scaled_windows)

        self.model = self.trainer.fit(train_matrix)
        reconstructions, embeddings = self.trainer.reconstruct_and_embed(self.model, train_matrix)
        reconstruction_error = np.mean((reconstructions - train_matrix) ** 2, axis=1)

        embedding_mean = embeddings.mean(axis=0)
        latent_distance = np.linalg.norm(embeddings - embedding_mean, axis=1)
        anomaly_scores = (
            self.config.alpha * reconstruction_error + self.config.beta * latent_distance
        )
        score_mean = float(anomaly_scores.mean())
        score_std = float(anomaly_scores.std())
        threshold = score_mean + self.config.threshold_k * score_std

        cluster_count = max(1, min(self.config.cluster_count, len(embeddings)))
        self.kmeans = KMeans(
            n_clusters=cluster_count,
            n_init=10,
            random_state=self.config.random_state,
        )
        cluster_ids = self.kmeans.fit_predict(embeddings)
        label_map = self._build_cluster_labels(bundle.raw_windows, cluster_ids)

        self.state = PipelineState(
            threshold=float(threshold),
            score_mean=score_mean,
            score_std=score_std,
            embedding_mean=embedding_mean.astype(np.float32),
            cluster_label_map=label_map,
        )
        self.training_df = self._build_results_frame(
            bundle=bundle,
            anomaly_scores=anomaly_scores,
            reconstruction_error=reconstruction_error,
            latent_distance=latent_distance,
            cluster_ids=cluster_ids,
        )
        return self

    def predict_window(self, window_records: list[dict[str, Any]]) -> dict[str, Any]:
        self._ensure_ready()
        frame = pd.DataFrame(window_records)
        if len(frame) != self.config.window_size:
            raise ValueError(f"Window must contain exactly {self.config.window_size} rows.")

        frame = frame.copy()
        if self.config.timestamp_column not in frame.columns:
            frame[self.config.timestamp_column] = pd.NaT
        frame[self.config.timestamp_column] = pd.to_datetime(
            frame[self.config.timestamp_column],
            errors="coerce",
        )
        for column in self.config.feature_columns:
            frame[column] = pd.to_numeric(frame[column], errors="raise")

        matrix = self.scaler.transform(frame[list(self.config.feature_columns)])
        flattened = flatten_windows(np.expand_dims(matrix.astype(np.float32), axis=0))
        reconstruction, embedding = self.trainer.reconstruct_and_embed(self.model, flattened)
        reconstruction_error = float(np.mean((reconstruction - flattened) ** 2))
        latent_distance = float(np.linalg.norm(embedding[0] - self.state.embedding_mean))
        anomaly_score = float(
            self.config.alpha * reconstruction_error + self.config.beta * latent_distance
        )
        cluster_id = int(self.kmeans.predict(embedding)[0])
        label = self.state.cluster_label_map.get(cluster_id, "Stable")
        timestamp = frame[self.config.timestamp_column].iloc[-1]

        return {
            "timestamp": timestamp.isoformat() if pd.notna(timestamp) else None,
            "close": float(frame["close"].iloc[-1]),
            "anomaly_score": anomaly_score,
            "anomaly_flag": anomaly_score > self.state.threshold,
            "label": label,
            "cluster_id": cluster_id,
            "reconstruction_error": reconstruction_error,
            "latent_distance": latent_distance,
        }

    def simulate_stream(self, limit: int | None = None) -> list[dict[str, Any]]:
        self._ensure_ready()
        records = self.training_df.to_dict(orient="records")
        if limit is not None:
            records = records[:limit]
        return records

    def save(self) -> None:
        self._ensure_ready()
        torch.save(
            {
                "input_dim": self.model.encoder[0].in_features,
                "latent_dim": self.config.latent_dim,
                "state_dict": self.model.state_dict(),
            },
            self.config.artifact_dir / "autoencoder.pt",
        )
        joblib.dump(self.scaler, self.config.artifact_dir / "scaler.joblib")
        joblib.dump(self.kmeans, self.config.artifact_dir / "kmeans.joblib")
        joblib.dump(self.training_df, self.config.artifact_dir / "stream_results.joblib")
        state_payload = {
            "threshold": self.state.threshold,
            "score_mean": self.state.score_mean,
            "score_std": self.state.score_std,
            "embedding_mean": self.state.embedding_mean.tolist(),
            "cluster_label_map": self.state.cluster_label_map,
        }
        (self.config.artifact_dir / "pipeline_state.json").write_text(
            json.dumps(state_payload, indent=2),
            encoding="utf-8",
        )

    def load(self) -> "AnomalyDetectionPipeline":
        state_path = self.config.artifact_dir / "pipeline_state.json"
        model_path = self.config.artifact_dir / "autoencoder.pt"
        scaler_path = self.config.artifact_dir / "scaler.joblib"
        kmeans_path = self.config.artifact_dir / "kmeans.joblib"
        results_path = self.config.artifact_dir / "stream_results.joblib"

        if not all(
            path.exists()
            for path in (state_path, model_path, scaler_path, kmeans_path, results_path)
        ):
            raise FileNotFoundError("Saved artifacts are incomplete or missing.")

        payload = torch.load(model_path, map_location=self.trainer.device)
        self.model = DenoisingAutoencoder(
            input_dim=payload["input_dim"],
            latent_dim=payload["latent_dim"],
        ).to(self.trainer.device)
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval()
        self.scaler = joblib.load(scaler_path)
        self.kmeans = joblib.load(kmeans_path)
        self.training_df = joblib.load(results_path)

        state_payload = json.loads(state_path.read_text(encoding="utf-8"))
        self.state = PipelineState(
            threshold=float(state_payload["threshold"]),
            score_mean=float(state_payload["score_mean"]),
            score_std=float(state_payload["score_std"]),
            embedding_mean=np.asarray(state_payload["embedding_mean"], dtype=np.float32),
            cluster_label_map={
                int(key): value for key, value in state_payload["cluster_label_map"].items()
            },
        )
        return self

    def _ensure_ready(self) -> None:
        if self.model is None or self.scaler is None or self.kmeans is None or self.state is None:
            raise RuntimeError("Pipeline is not initialized. Call fit() or load() first.")

    def _build_results_frame(
        self,
        bundle: WindowBundle,
        anomaly_scores: np.ndarray,
        reconstruction_error: np.ndarray,
        latent_distance: np.ndarray,
        cluster_ids: np.ndarray,
    ) -> pd.DataFrame:
        labels = [self.state.cluster_label_map.get(int(cluster_id), "Stable") for cluster_id in cluster_ids]
        frame = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(bundle.timestamps),
                "close": bundle.closes.astype(float),
                "anomaly_score": anomaly_scores.astype(float),
                "anomaly_flag": anomaly_scores > self.state.threshold,
                "label": labels,
                "cluster_id": cluster_ids.astype(int),
                "reconstruction_error": reconstruction_error.astype(float),
                "latent_distance": latent_distance.astype(float),
            }
        )
        return frame

    def _build_cluster_labels(
        self,
        raw_windows: np.ndarray,
        cluster_ids: np.ndarray,
    ) -> dict[int, str]:
        summary = summarize_windows(raw_windows)
        summary["cluster_id"] = cluster_ids
        cluster_stats = summary.groupby("cluster_id").mean(numeric_only=True)
        available_clusters = list(cluster_stats.index.astype(int))

        label_map: dict[int, str] = {}
        if not available_clusters:
            return label_map

        spike_cluster = int(cluster_stats["spike"].idxmax())
        label_map[spike_cluster] = "Sudden Spike/Drop"

        remaining = [cluster for cluster in available_clusters if cluster != spike_cluster]
        if remaining:
            trend_series = cluster_stats.loc[remaining, "trend"]
            uptrend_cluster = int(trend_series.idxmax())
            label_map[uptrend_cluster] = "Uptrend"
            remaining = [cluster for cluster in remaining if cluster != uptrend_cluster]

        if remaining:
            trend_series = cluster_stats.loc[remaining, "trend"]
            downtrend_cluster = int(trend_series.idxmin())
            label_map[downtrend_cluster] = "Downtrend"
            remaining = [cluster for cluster in remaining if cluster != downtrend_cluster]

        if remaining:
            volatility_cluster = int(cluster_stats.loc[remaining, "volatility"].idxmax())
            label_map[volatility_cluster] = "Volatile"
            remaining = [cluster for cluster in remaining if cluster != volatility_cluster]

        for cluster in remaining:
            label_map[int(cluster)] = "Stable"
        return label_map
