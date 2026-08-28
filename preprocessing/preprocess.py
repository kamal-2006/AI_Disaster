from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common import resolve_path

RAW_CSV = resolve_path("data", "raw", "erode_weather_2000_2026.csv")
DAILY_CSV = resolve_path("data", "processed", "erode_daily.csv")

RAW_COLUMNS = [
    "time",
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "precipitation",
    "vapour_pressure_deficit",
    "wind_gusts_10m",
    "cloud_cover",
    "dew_point_2m",
    "apparent_temperature",
]


def load_raw_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {path}. Place erode_weather_2000_2026.csv in ai/data/raw/."
        )

    encodings = ["utf-8-sig", "utf-8", "latin1"]
    last_error: Exception | None = None
    for encoding in encodings:
        try:
            with path.open("r", encoding=encoding, errors="replace") as handle:
                lines = handle.readlines()

            header_index = 0
            for idx, line in enumerate(lines):
                normalized = line.strip().lower().replace(" ", "")
                if normalized.startswith("time,"):
                    header_index = idx
                    break

            return pd.read_csv(path, encoding=encoding, skiprows=header_index, low_memory=False)
        except Exception as exc:  # pragma: no cover - encoding fallback path
            last_error = exc
    raise RuntimeError(f"Unable to read CSV {path}. Last error: {last_error}")


def clean_column_name(name: str) -> str:
    cleaned = str(name).replace("Â", "").replace("°", "").strip()
    cleaned = re.sub(r"\(.*?\)", "", cleaned)
    cleaned = cleaned.strip().lower()
    cleaned = re.sub(r"\s+", "_", cleaned)
    cleaned = re.sub(r"[^a-z0-9_]+", "", cleaned)
    cleaned = cleaned.strip("_")
    return cleaned


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [clean_column_name(column) for column in df.columns]
    return df


def parse_time_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "time" not in df.columns:
        raise ValueError(f"Expected a time column. Found: {list(df.columns)}")

    time_values = df["time"].astype(str).str.replace("Â", "", regex=False).str.strip()
    parsed = pd.to_datetime(time_values, errors="coerce", utc=False, format="mixed")
    if parsed.isna().mean() > 0.5:
        parsed = pd.to_datetime(time_values, errors="coerce", utc=False, dayfirst=True, format="mixed")

    if parsed.isna().all():
        raise ValueError("Could not parse the time column into datetimes.")

    df["time"] = parsed
    df = df.dropna(subset=["time"])
    df = df.sort_values("time").reset_index(drop=True)
    return df


def convert_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    numeric_columns = [column for column in RAW_COLUMNS if column != "time" and column in df.columns]
    for column in numeric_columns:
        series = df[column].astype(str).str.replace("Â", "", regex=False).str.replace(",", "", regex=False)
        series = series.str.replace(r"[^0-9eE+\-.]", "", regex=True)
        df[column] = pd.to_numeric(series, errors="coerce")
    return df


def report_missing_values(df: pd.DataFrame) -> pd.Series:
    missing_percent = df.isna().mean().mul(100).round(2)
    print("Missing values (%):")
    print(missing_percent.to_string())
    return missing_percent


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    duplicate_rows = int(df.duplicated().sum())
    duplicate_times = int(df.duplicated(subset=["time"]).sum()) if "time" in df.columns else 0
    print(f"Duplicate full rows: {duplicate_rows}")
    print(f"Duplicate timestamps: {duplicate_times}")
    if duplicate_rows:
        df = df.drop_duplicates()
    if duplicate_times:
        df = df.drop_duplicates(subset=["time"], keep="first")
    return df


def detect_and_fix_invalid_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    invalid_rules = {
        "relative_humidity_2m": (0, 100),
        "wind_speed_10m": (0, None),
        "precipitation": (0, None),
        "vapour_pressure_deficit": (0, None),
        "wind_gusts_10m": (0, None),
        "cloud_cover": (0, 100),
        "temperature_2m": (-20, 60),
        "dew_point_2m": (-20, 40),
        "apparent_temperature": (-25, 65),
    }

    for column, (lower, upper) in invalid_rules.items():
        if column not in df.columns:
            continue
        series = df[column]
        invalid_mask = series.isna()
        if lower is not None:
            invalid_mask |= series < lower
        if upper is not None:
            invalid_mask |= series > upper
        invalid_count = int(invalid_mask.sum())
        if invalid_count:
            print(f"Invalid values replaced with NaN in {column}: {invalid_count}")
            df.loc[invalid_mask, column] = np.nan
    return df


def impute_hourly_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.set_index("time")
    numeric_columns = [column for column in df.columns if pd.api.types.is_numeric_dtype(df[column])]
    if numeric_columns:
        df[numeric_columns] = df[numeric_columns].interpolate(method="time", limit_direction="both")
        df[numeric_columns] = df[numeric_columns].ffill().bfill()
        for column in numeric_columns:
            if column == "precipitation":
                df[column] = df[column].clip(lower=0)
            if column in {"relative_humidity_2m", "cloud_cover"}:
                df[column] = df[column].clip(lower=0, upper=100)
    df = df.reset_index()
    return df


def hourly_to_daily(df: pd.DataFrame) -> pd.DataFrame:
    required_columns = [column for column in RAW_COLUMNS if column != "time"]
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    df["date"] = df["time"].dt.floor("D")

    aggregations = {
        "temperature_2m": ["mean", "max", "min"],
        "relative_humidity_2m": ["mean", "max"],
        "wind_speed_10m": ["mean", "max"],
        "wind_gusts_10m": ["max"],
        "precipitation": ["sum"],
        "vapour_pressure_deficit": ["mean", "max"],
        "cloud_cover": ["mean"],
        "dew_point_2m": ["mean"],
        "apparent_temperature": ["mean", "max", "min"],
    }

    grouped = df.groupby("date").agg(aggregations)
    grouped.columns = [
        "mean_temperature",
        "max_temperature",
        "min_temperature",
        "mean_relative_humidity",
        "max_relative_humidity",
        "mean_wind_speed",
        "max_wind_speed",
        "max_wind_gust",
        "precipitation_sum",
        "mean_vpd",
        "max_vpd",
        "mean_cloud_cover",
        "mean_dew_point",
        "mean_apparent_temperature",
        "max_apparent_temperature",
        "min_apparent_temperature",
    ]
    grouped = grouped.reset_index().rename(columns={"date": "date"})
    return grouped


def preprocess_raw_file(raw_path: Path = RAW_CSV, output_path: Path = DAILY_CSV) -> pd.DataFrame:
    print(f"Loading raw data from: {raw_path}")
    raw_df = load_raw_csv(raw_path)
    original_rows = len(raw_df)
    print(f"Original rows: {original_rows}")

    raw_df = standardize_columns(raw_df)
    raw_df = parse_time_column(raw_df)
    raw_df = remove_duplicates(raw_df)
    raw_df = convert_numeric_columns(raw_df)
    raw_df = detect_and_fix_invalid_values(raw_df)

    print("Missing values before imputation:")
    report_missing_values(raw_df)
    raw_df = impute_hourly_values(raw_df)
    print("Missing values after imputation:")
    report_missing_values(raw_df)

    raw_df = raw_df.sort_values("time").reset_index(drop=True)
    daily_df = hourly_to_daily(raw_df)
    daily_df = daily_df.sort_values("date").reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    daily_df.to_csv(output_path, index=False)

    print(f"Daily rows: {len(daily_df)}")
    print(f"Date range: {daily_df['date'].min()} to {daily_df['date'].max()}")
    print("Final columns:")
    print(list(daily_df.columns))
    print("First 5 rows:")
    print(daily_df.head().to_string(index=False))
    return daily_df


def main() -> None:
    preprocess_raw_file()


if __name__ == "__main__":
    main()
