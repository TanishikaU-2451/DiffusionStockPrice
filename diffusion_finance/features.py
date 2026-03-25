from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .config import Settings, settings


@dataclass(slots=True)
class WindowBundle:
    scaled_windows: np.ndarray
    raw_windows: np.ndarray
    timestamps: np.ndarray
    closes: np.ndarray


def fit_scaler(df: pd.DataFrame, config: Settings = settings) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(df[list(config.feature_columns)])
    return scaler


def create_window_bundle(
    df: pd.DataFrame,
    scaler: StandardScaler,
    config: Settings = settings,
) -> WindowBundle:
    scaled_rows = scaler.transform(df[list(config.feature_columns)])
    raw_rows = df[list(config.feature_columns)].to_numpy(dtype=np.float32)
    timestamps = df[config.timestamp_column].to_numpy()

    window_count = len(df) - config.window_size + 1
    if window_count <= 0:
        raise ValueError(
            f"Not enough rows ({len(df)}) to build windows of size {config.window_size}."
        )

    scaled_windows = np.stack(
        [scaled_rows[index : index + config.window_size] for index in range(window_count)]
    ).astype(np.float32)
    raw_windows = np.stack(
        [raw_rows[index : index + config.window_size] for index in range(window_count)]
    ).astype(np.float32)
    end_timestamps = timestamps[config.window_size - 1 :]
    closes = raw_windows[:, -1, 3]
    return WindowBundle(
        scaled_windows=scaled_windows,
        raw_windows=raw_windows,
        timestamps=end_timestamps,
        closes=closes,
    )


def flatten_windows(windows: np.ndarray) -> np.ndarray:
    return windows.reshape(windows.shape[0], -1).astype(np.float32)


def summarize_windows(raw_windows: np.ndarray) -> pd.DataFrame:
    close_paths = raw_windows[:, :, 3]
    returns = np.diff(close_paths, axis=1) / np.clip(close_paths[:, :-1], a_min=1e-6, a_max=None)
    slopes = np.array([np.polyfit(np.arange(path.size), path, deg=1)[0] for path in close_paths])
    normalized_slopes = slopes / np.clip(close_paths[:, 0], a_min=1e-6, a_max=None)
    volatility = returns.std(axis=1)
    max_abs_return = np.abs(returns).max(axis=1)

    return pd.DataFrame(
        {
            "trend": normalized_slopes,
            "volatility": volatility,
            "spike": max_abs_return,
        }
    )
