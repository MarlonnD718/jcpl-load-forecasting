# JCP&L Hourly Load Forecasting

Semester project: predicting hourly electricity load (MW) for JCP&L (Jersey
Central Power & Light, a New Jersey utility) by rate class, using weather and
calendar features.

## Project structure

```
src/
  config.py          paths, rate class list, COVID window, horizons
  data_loaders.py     loaders for load, weather, and date-category data
  features.py          COVID exclusion, time features, dataset assembly
  model.py              train/evaluate a decision tree per horizon
  run_pipeline.py       CLI entry point - ties it all together
data/
  raw/                 unzipped source data (gitignored - see Setup)
  processed/            pickled cleaned datasets (gitignored, regenerated on run)
models/                pickled trained models (gitignored, regenerated on run)
```

## Setup

1. Clone the repo and install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Unzip the provided `Data-Load_Prediction.zip` so the folder structure is:
   ```
   data/raw/Data/Predict - y/...
   data/raw/Data/Predict With - X/...
   ```
   (The raw data is not committed to the repo - it's listed in `.gitignore`
   since it's large and not ours to publish. Each teammate unzips it locally.)

## Running

```bash
# Single rate class (default: Res)
python -m src.run_pipeline --rate Res

# A specific rate class
python -m src.run_pipeline --rate GPC

# All 11 rate classes, with a summary table at the end
python -m src.run_pipeline --all
```

Each run prints MAE/RMSE/MAPE and the top feature importances at the 7/15/30-day
out-of-sample horizons, and pickles the cleaned dataset (`data/processed/`) and
trained models (`models/`) for that rate class.

## Approach

- **Target**: cleaned hourly load per rate class (`Predict - y/Cleaned Up - JCPL/`)
- **Features**: Newark hourly temperature/dew point/humidity, plus calendar
  features (weekday/weekend, month, holiday)
- **Modeling**: a separate `DecisionTreeRegressor` per rate class (load
  behavior differs meaningfully by class - see `--all` summary)
- **COVID exclusion**: March 2020 - June 2021 is excluded from training as a
  non-representative demand period (adjust in `src/config.py` if the group
  agrees on a different window)
- **Evaluation**: chronological (not random) train/test splits, tested
  out-of-sample at 7, 15, and 30 days, to mimic real forecasting conditions

## Known data quirks (already handled in `data_loaders.py`)

- **Weather timestamp formats differ by year**: 2019/2022/2023 use
  `YYYY-MM-DD H:MM AM/PM`, while 2020/2021/2024 use `M/D/YYYY H:MM` (24h).
  Parsing all years together with one inferred format silently drops
  whichever years don't match - each file is parsed individually instead.
- **DST**: load data is resampled to hourly with `.mean()` (averages the
  duplicate "fall back" hour) and `.interpolate()` (fills the missing
  "spring forward" hour).
- **Weather station readings** land at irregular minutes (e.g. `:51`) and
  are resampled to the top of each hour.

## Next steps

- Wire in the `Energy/` files (monthly generation by source, customer
  account counts) as additional features for longer-horizon / trend context
- Try `RandomForestRegressor` / gradient boosting as a drop-in upgrade over
  the single decision tree
- Compare against a CNN/CRNN sequence model as a stretch goal (see project notes)
