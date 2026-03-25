# Diffusion-Inspired Financial Anomaly Detection MVP

This project builds a lightweight anomaly detection and pattern labeling pipeline for historical MSFT OHLCV data. It fetches paginated stock history, trains a denoising autoencoder as a diffusion-inspired encoder, scores anomalies from reconstruction and latent distance, clusters learned embeddings into interpretable pattern groups, and exposes the results through FastAPI and Streamlit.

## What the System Does

- Fetches complete historical MSFT price data from `https://financialdata.net/api/v1/stock-prices` using `limit=300` and `offset` pagination.
- Sorts the data chronologically and uses `open`, `high`, `low`, `close`, and `volume`.
- Applies `StandardScaler` normalization and creates sliding windows of size `20`.
- Trains a denoising autoencoder on flattened windows to learn compact embeddings and reconstructions.
- Computes anomaly scores as:

```text
anomaly_score = alpha * reconstruction_error + beta * latent_distance
```

- Uses a dynamic threshold of `mean + k * std` to flag anomalies.
- Clusters embeddings with KMeans and maps clusters to:
  - `Stable`
  - `Uptrend`
  - `Downtrend`
  - `Volatile`
  - `Sudden Spike/Drop`
- Replays the historical series sequentially to simulate real-time inference.

## Project Layout

```text
diffusion_finance/
  config.py
  data.py
  features.py
  model.py
  pipeline.py
  schemas.py
  service.py
main.py
dashboard.py
train_pipeline.py
requirements.txt
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## API Key

Create a `.env` file in the project root and add your key there:

```env
FINANCIAL_API_KEY=your_real_key_here
FINANCIAL_API_KEY_HEADER=X-API-Key
FINANCIAL_API_KEY_QUERY_PARAM=key
FINANCIAL_API_KEY_AS_BEARER=false
```

The client sends the key both as a header and, by default, as the `key` query parameter. If your provider expects a different field name, update `FINANCIAL_API_KEY_HEADER` or `FINANCIAL_API_KEY_QUERY_PARAM`.

## Train the Pipeline

```bash
python train_pipeline.py --refresh-data --retrain
```

Artifacts are written to `artifacts/` and cached price history is stored in `data/raw/`.

## Run the API

```bash
uvicorn main:app --reload
```

Available endpoints:

- `POST /predict`
- `GET /stream`
- `GET /health`

Example `POST /predict` body:

```json
{
  "window": [
    {
      "date": "2024-01-02T00:00:00",
      "open": 375.18,
      "high": 376.35,
      "low": 366.50,
      "close": 370.87,
      "volume": 25258600
    }
  ]
}
```

The `window` array must contain exactly 20 OHLCV rows. Repeat the same object structure for each timestep in the latest 20-day window.

## Run the Dashboard

```bash
streamlit run dashboard.py
```

The dashboard shows:

- Price trajectory with anomaly markers
- Label assignments per window
- Anomaly score trend
- Historical replay controls for simulated real-time behavior

## Notes

- The code assumes the financial API returns records under one of these keys: `data`, `results`, `prices`, or `items`.
- If the API schema differs, update `StockDataClient._extract_records()` and the column aliases in `StockDataClient._normalize_frame()`.
- The training flow assumes the historical data is mostly normal, which is standard for unsupervised anomaly detection.
