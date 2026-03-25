from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    def load_dotenv() -> bool:
        return False


load_dotenv()


@dataclass(slots=True)
class Settings:
    base_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    data_dir: Path = field(init=False)
    raw_data_dir: Path = field(init=False)
    artifact_dir: Path = field(init=False)
    cache_file: Path = field(init=False)
    base_url: str = "https://financialdata.net/api/v1/stock-prices"
    identifier: str = "MSFT"
    api_key: str | None = field(default_factory=lambda: os.getenv("FINANCIAL_API_KEY"))
    api_key_header: str = field(
        default_factory=lambda: os.getenv("FINANCIAL_API_KEY_HEADER", "X-API-Key")
    )
    api_key_query_param: str = field(
        default_factory=lambda: os.getenv("FINANCIAL_API_KEY_QUERY_PARAM", "key")
    )
    api_key_as_bearer: bool = field(
        default_factory=lambda: os.getenv("FINANCIAL_API_KEY_AS_BEARER", "false").lower()
        == "true"
    )
    feature_columns: tuple[str, ...] = ("open", "high", "low", "close", "volume")
    timestamp_column: str = "date"
    window_size: int = 20
    page_size: int = 300
    latent_dim: int = 16
    alpha: float = 0.7
    beta: float = 0.3
    threshold_k: float = 2.0
    cluster_count: int = 5
    epochs: int = 60
    batch_size: int = 64
    learning_rate: float = 1e-3
    noise_std: float = 0.08
    random_state: int = 42

    def __post_init__(self) -> None:
        self.data_dir = self.base_dir / "data"
        self.raw_data_dir = self.data_dir / "raw"
        self.artifact_dir = self.base_dir / "artifacts"
        self.cache_file = self.raw_data_dir / f"{self.identifier.lower()}_prices.csv"

        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
