from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import requests

from .config import Settings, settings


class StockDataClient:
    def __init__(self, config: Settings = settings) -> None:
        self.config = config

    def fetch_all_history(self, refresh: bool = False) -> pd.DataFrame:
        if self.config.cache_file.exists() and not refresh:
            return self._load_cache(self.config.cache_file)

        all_records: list[dict[str, Any]] = []
        offset = 0

        while True:
            params = {
                "identifier": self.config.identifier,
                "limit": self.config.page_size,
                "offset": offset,
            }
            headers = self._build_headers()
            if self.config.api_key and self.config.api_key_query_param:
                params[self.config.api_key_query_param] = self.config.api_key

            response = requests.get(
                self.config.base_url,
                params=params,
                headers=headers,
                timeout=30,
            )
            if response.status_code == 401:
                raise RuntimeError(
                    "Financial API rejected the request with 401 Unauthorized. "
                    "Set FINANCIAL_API_KEY in a .env file or environment variable, and if your "
                    "provider expects a different field name also set FINANCIAL_API_KEY_HEADER "
                    "or FINANCIAL_API_KEY_QUERY_PARAM."
                )
            response.raise_for_status()
            batch = self._extract_records(response.json())
            if not batch:
                break

            all_records.extend(batch)
            if len(batch) < self.config.page_size:
                break
            offset += self.config.page_size

        if not all_records:
            raise ValueError("No stock records were returned by the API.")

        df = self._normalize_frame(pd.DataFrame(all_records))
        df.to_csv(self.config.cache_file, index=False)
        return df

    def _load_cache(self, path: Path) -> pd.DataFrame:
        df = pd.read_csv(path, parse_dates=[self.config.timestamp_column])
        return self._normalize_frame(df)

    def _build_headers(self) -> dict[str, str]:
        if not self.config.api_key:
            return {}
        if self.config.api_key_as_bearer:
            return {"Authorization": f"Bearer {self.config.api_key}"}
        return {self.config.api_key_header: self.config.api_key}

    def _extract_records(self, payload: Any) -> list[dict[str, Any]]:
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            for key in ("data", "results", "prices", "items"):
                value = payload.get(key)
                if isinstance(value, list):
                    return value
        raise ValueError("Unable to find a list of price records in the API response.")

    def _normalize_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        column_aliases = {
            "timestamp": "date",
            "datetime": "date",
            "time": "date",
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
        df = df.rename(columns=column_aliases)

        required = {self.config.timestamp_column, *self.config.feature_columns}
        missing = [column for column in required if column not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns in stock data: {missing}")

        normalized = df[[self.config.timestamp_column, *self.config.feature_columns]].copy()
        normalized[self.config.timestamp_column] = pd.to_datetime(
            normalized[self.config.timestamp_column],
            utc=False,
            errors="coerce",
        )
        normalized = normalized.dropna(subset=[self.config.timestamp_column])
        for column in self.config.feature_columns:
            normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
        normalized = normalized.dropna(subset=list(self.config.feature_columns))
        normalized = normalized.sort_values(self.config.timestamp_column).reset_index(drop=True)
        normalized = normalized.drop_duplicates(subset=[self.config.timestamp_column], keep="last")
        return normalized
