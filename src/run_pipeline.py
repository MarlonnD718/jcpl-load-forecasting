"""
Run the baseline pipeline for one (or all) rate classes:
  load data -> engineer features -> exclude COVID window -> train decision
  trees at 7/15/30-day horizons -> report MAE/RMSE/MAPE + feature importance
  -> pickle the cleaned dataset and trained models.

Usage:
    python -m src.run_pipeline              # runs the default rate class (Res)
    python -m src.run_pipeline --rate GPC    # runs a specific rate class
    python -m src.run_pipeline --all         # runs every rate class, prints a summary table
"""

import argparse

import numpy as np
import pandas as pd

from . import config
from . import features
from . import model as model_mod


def run_for_rate_class(rate_class: str) -> dict:
    print(f"\n{'=' * 60}\nRate class: {rate_class}\n{'=' * 60}")
    df, feature_cols = features.build_dataset(rate_class)
    print(f"Modeling dataset: {df.shape[0]} rows, {len(feature_cols)} features")

    model_mod.save_pickle(df, rate_class, "cleaned_features", out_dir=config.PROCESSED_DIR)

    results = model_mod.evaluate_horizons(df, feature_cols)
    for h, res in results.items():
        model_mod.save_pickle(res["model"], rate_class, f"model_{h}day")

    avg_mape = float(np.mean([r["mape"] for r in results.values()])) if results else float("nan")
    print(f"\nAverage MAPE across horizons: {avg_mape:.4f}")
    return {"rate_class": rate_class, "avg_mape": avg_mape, "results": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rate", default="Res", help="Rate class to run (default: Res)")
    parser.add_argument("--all", action="store_true", help="Run every rate class")
    args = parser.parse_args()

    if args.all:
        summary = []
        for rc in config.RATE_CLASSES:
            out = run_for_rate_class(rc)
            summary.append(out)

        print(f"\n{'=' * 60}\nSummary (average MAPE per rate class)\n{'=' * 60}")
        summary_df = pd.DataFrame(
            [{"rate_class": s["rate_class"], "avg_mape": s["avg_mape"]} for s in summary]
        ).sort_values("avg_mape")
        print(summary_df.to_string(index=False))
    else:
        run_for_rate_class(args.rate)


if __name__ == "__main__":
    main()
