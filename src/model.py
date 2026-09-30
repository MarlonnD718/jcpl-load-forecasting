"""Train and evaluate a decision tree regressor per out-of-sample horizon."""

import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.tree import DecisionTreeRegressor

from . import config


def evaluate_horizons(df: pd.DataFrame, feature_cols: list[str], target_col: str = "load_mw",
                       horizons_days: tuple = config.HORIZONS_DAYS) -> dict:
    """Chronological train/test split at each horizon; trains a fresh tree per horizon.

    Returns {horizon_days: {"model", "mae", "rmse", "mape", "importances", "n_test_rows"}}
    """
    results = {}
    last_date = df.index.max()

    for h in horizons_days:
        cutoff = last_date - pd.Timedelta(days=h)
        train = df.loc[df.index <= cutoff]
        test = df.loc[(df.index > cutoff) & (df.index <= last_date)]

        if train.empty or test.empty:
            print(f"[evaluate_horizons] skipping {h}-day horizon - not enough data")
            continue

        model = DecisionTreeRegressor(max_depth=8, min_samples_leaf=20, random_state=42)
        model.fit(train[feature_cols], train[target_col])
        preds = model.predict(test[feature_cols])

        mae = mean_absolute_error(test[target_col], preds)
        rmse = mean_squared_error(test[target_col], preds) ** 0.5
        mape = mean_absolute_percentage_error(test[target_col], preds)
        importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)

        results[h] = {
            "model": model, "mae": mae, "rmse": rmse, "mape": mape,
            "importances": importances, "n_test_rows": len(test),
        }

        print(f"\n[{h}-day horizon] train={len(train)} test={len(test)}")
        print(f"  MAE={mae:.2f} MW | RMSE={rmse:.2f} MW | MAPE={mape:.4f}")
        print(f"  Top features: {importances.head(5).to_dict()}")

    return results


def save_pickle(obj, rate_class: str, name: str, out_dir=config.MODELS_DIR):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{rate_class}_{name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"[save_pickle] saved {path}")
    return path
