from __future__ import annotations

import argparse

from diffusion_finance.service import ModelService


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and persist the financial anomaly pipeline."
    )
    parser.add_argument(
        "--refresh-data",
        action="store_true",
        help="Refetch paginated stock history from the API.",
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Force a fresh model fit even if artifacts exist.",
    )
    args = parser.parse_args()

    service = ModelService()
    pipeline = service.ensure_ready(
        refresh_data=args.refresh_data,
        retrain=args.retrain,
    )
    print(
        "Pipeline ready:",
        {
            "windows": len(pipeline.training_df),
            "threshold": pipeline.state.threshold,
            "labels": pipeline.state.cluster_label_map,
        },
    )


if __name__ == "__main__":
    main()
