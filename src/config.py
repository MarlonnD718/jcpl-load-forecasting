"""Paths and constants shared across the pipeline."""

from pathlib import Path

# Root of the unzipped dataset - see README for setup instructions
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "Data"

CLEANED_Y_DIR = DATA_DIR / "Predict - y" / "Cleaned Up - JCPL"
DATE_CATS_FILE = DATA_DIR / "Predict With - X" / "Date_Catagories.xlsx"
WEATHER_DIR = DATA_DIR / "Predict With - X" / "Weather" / "Newark"

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

# Rate classes available in Cleaned Up - JCPL (file stem -> column name)
RATE_CLASSES = ["Res", "GPC", "GPI", "GSC", "GSI", "GSTC", "GSTI", "GTC", "GTI", "CIEP", "RSCP"]

# COVID-affected window to exclude from training (adjust as a group if needed)
COVID_START = "2020-03-01"
COVID_END = "2021-06-30"

# Out-of-sample evaluation horizons, in days
HORIZONS_DAYS = (7, 15, 30)
