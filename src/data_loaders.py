"""Loaders for each raw data source: target load, weather, and date-category features.

Each loader returns a DataFrame indexed by an hourly DatetimeIndex, so they can
be joined directly with `.join()`.
"""

from pathlib import Path

import pandas as pd

from . import config


def load_target(rate_class: str) -> pd.DataFrame:
    """Load a cleaned hourly load series for one rate class.

    Resamples to hourly (averaging any DST fall-back duplicate timestamps)
    and interpolates any gap hours (e.g. DST spring-forward).
    """
    path = config.CLEANED_Y_DIR / f"{rate_class}_cleaned.xlsx"
    df = pd.read_excel(path)
    df = df.rename(columns={rate_class: "load_mw"})
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    df = df.resample("h").mean()

    n_missing = df["load_mw"].isna().sum()
    if n_missing:
        print(f"[load_target:{rate_class}] filling {n_missing} missing hour(s)")
    df["load_mw"] = df["load_mw"].interpolate(method="time")
    return df


def load_weather(weather_dir: Path = config.WEATHER_DIR) -> pd.DataFrame:
    """Load and combine the per-year Newark hourly weather CSVs.

    IMPORTANT: the per-year files are NOT all in the same timestamp format -
    some years use "YYYY-MM-DD H:MM AM/PM", others "M/D/YYYY H:MM" (24h).
    Each file is parsed on its own with format='mixed' (per-row inference)
    rather than inferring one format across the whole concatenated set -
    doing the latter silently drops whichever years don't match the
    inferred format.
    """
    files = sorted(weather_dir.glob("hourly_weather_newark_*.csv"))
    if not files:
        raise FileNotFoundError(f"No weather files found in {weather_dir}")

    frames = []
    for f in files:
        d = pd.read_csv(f)
        d["Date"] = pd.to_datetime(d["Time"], format="mixed", errors="coerce")
        n_fail = d["Date"].isna().sum()
        if n_fail:
            print(f"[load_weather] {f.name}: {n_fail}/{len(d)} timestamps failed to parse")
        frames.append(d)

    wx = pd.concat(frames, ignore_index=True)
    wx = wx.dropna(subset=["Date"])

    for col in ["Temperature", "Dew Point"]:
        wx[col] = wx[col].astype(str).str.replace("F", "", regex=False)
        wx[col] = pd.to_numeric(wx[col], errors="coerce")

    wx = wx.rename(columns={"Temperature": "temp_f", "Dew Point": "dew_point_f", "Humidity": "humidity"})
    wx["humidity"] = pd.to_numeric(wx["humidity"].astype(str).str.replace("%", "", regex=False), errors="coerce")

    wx = wx.set_index("Date").sort_index()
    wx = wx[["temp_f", "dew_point_f", "humidity"]]

    # Station readings land at irregular minutes (e.g. :51) - snap to the top of each hour
    wx_hourly = wx.resample("h").mean()
    wx_hourly = wx_hourly.interpolate(method="time", limit=6)  # fill small gaps only
    return wx_hourly


def load_date_categories(path: Path = config.DATE_CATS_FILE) -> pd.DataFrame:
    """Load daily calendar features (weekday/weekend/month/holiday) and upsample to hourly."""
    dc = pd.read_excel(path)
    dc["Date"] = pd.to_datetime(dc["Date"])
    dc = dc.set_index("Date").sort_index()
    return dc.resample("h").ffill()
