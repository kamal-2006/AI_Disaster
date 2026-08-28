from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common import resolve_path, save_pickle  # noqa: E402
from preprocessing.preprocess import preprocess_raw_file  # noqa: E402


PROCESSED_DAILY_CSV = resolve_path("data", "processed", "erode_daily.csv")
FEATURES_CSV = resolve_path("data", "processed", "erode_features.csv")
FEATURE_COLUMNS_PATH = resolve_path("models", "feature_columns.pkl")
HEATWAVE_CONFIG_PATH = resolve_path("models", "heatwave_config.pkl")

HEATWAVE_MIN_CONSECUTIVE_DAYS = 2
HEATWAVE_FORECAST_HORIZON_DAYS = 1
DEFAULT_TRAIN_END = pd.Timestamp("2021-12-31")
DEFAULT_VALIDATION_END = pd.Timestamp("2024-12-31")
BASELINE_PERCENTILE = 95


@dataclass(frozen=True)
class HeatwaveConfig:
    min_consecutive_days: int = HEATWAVE_MIN_CONSECUTIVE_DAYS
    forecast_horizon_days: int = HEATWAVE_FORECAST_HORIZON_DAYS
    baseline_percentile: int = BASELINE_PERCENTILE
    train_end: str = str(DEFAULT_TRAIN_END.date())
    validation_end: str = str(DEFAULT_VALIDATION_END.date())


def load_daily_data(path: Path = PROCESSED_DAILY_CSV) -> pd.DataFrame:
    if not path.exists():
        print(f"Daily dataset not found at {path}. Running preprocessing first.")
        return preprocess_raw_file()
    daily_df = pd.read_csv(path, parse_dates=["date"])
    return daily_df.sort_values("date").reset_index(drop=True)


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day_of_year"] = df["date"].dt.dayofyear
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    temperature_series = df["max_temperature"]
    df["temperature_lag_1"] = temperature_series.shift(1)
    df["temperature_lag_2"] = temperature_series.shift(2)
    df["temperature_lag_3"] = temperature_series.shift(3)
    df["temperature_lag_7"] = temperature_series.shift(7)
    return df


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    previous_temperature = df["max_temperature"].shift(1)
    df["temperature_rolling_3"] = previous_temperature.rolling(3, min_periods=3).mean()
    df["temperature_rolling_5"] = previous_temperature.rolling(5, min_periods=5).mean()
    df["temperature_rolling_7"] = previous_temperature.rolling(7, min_periods=7).mean()

    previous_humidity = df["mean_relative_humidity"].shift(1)
    previous_apparent = df["mean_apparent_temperature"].shift(1)
    previous_vpd = df["mean_vpd"].shift(1)

    df["relative_humidity_rolling_3"] = previous_humidity.rolling(3, min_periods=3).mean()
    df["relative_humidity_rolling_5"] = previous_humidity.rolling(5, min_periods=5).mean()
    df["relative_humidity_rolling_7"] = previous_humidity.rolling(7, min_periods=7).mean()

    df["apparent_temperature_rolling_3"] = previous_apparent.rolling(3, min_periods=3).mean()
    df["apparent_temperature_rolling_5"] = previous_apparent.rolling(5, min_periods=5).mean()
    df["apparent_temperature_rolling_7"] = previous_apparent.rolling(7, min_periods=7).mean()

    df["vpd_rolling_3"] = previous_vpd.rolling(3, min_periods=3).mean()
    df["vpd_rolling_5"] = previous_vpd.rolling(5, min_periods=5).mean()
    df["vpd_rolling_7"] = previous_vpd.rolling(7, min_periods=7).mean()
    return df


def create_heatwave_event_labels(
    df: pd.DataFrame,
    train_end: pd.Timestamp = DEFAULT_TRAIN_END,
    percentile: int = BASELINE_PERCENTILE,
    min_consecutive_days: int = HEATWAVE_MIN_CONSECUTIVE_DAYS,
) -> tuple[pd.DataFrame, dict[str, float]]:
    df = df.copy().sort_values("date").reset_index(drop=True)
    training_mask = df["date"] <= train_end
    if not training_mask.any():
        raise ValueError("Training mask is empty. Check the configured training end date.")

    baseline = df.loc[training_mask].copy()
    baseline["month"] = baseline["date"].dt.month
    monthly_thresholds = baseline.groupby("month")["max_temperature"].quantile(percentile / 100.0).to_dict()

    monthly_series = df["date"].dt.month.map(monthly_thresholds)
    fallback_threshold = float(baseline["max_temperature"].quantile(percentile / 100.0))
    monthly_series = monthly_series.fillna(fallback_threshold)

    df["heatwave_hot_day"] = (df["max_temperature"] >= monthly_series).astype(int)
    hot_group_id = (df["heatwave_hot_day"] != df["heatwave_hot_day"].shift()).cumsum()
    run_length = df["heatwave_hot_day"].groupby(hot_group_id).transform("size")
    df["heatwave_event"] = ((df["heatwave_hot_day"] == 1) & (run_length >= min_consecutive_days)).astype(int)

    thresholds = {f"month_{month}": float(value) for month, value in monthly_thresholds.items()}
    thresholds["fallback_threshold"] = fallback_threshold
    return df, thresholds


def create_forecast_target(df: pd.DataFrame, horizon_days: int = HEATWAVE_FORECAST_HORIZON_DAYS) -> pd.DataFrame:
    df = df.copy().sort_values("date").reset_index(drop=True)
    future_flags = [df["heatwave_event"].shift(-day_offset) for day_offset in range(1, horizon_days + 1)]
    future_matrix = pd.concat(future_flags, axis=1)
    df["heatwave"] = future_matrix.max(axis=1).fillna(0).astype(int)
    return df


def build_feature_dataset(
    daily_df: pd.DataFrame,
    train_end: pd.Timestamp = DEFAULT_TRAIN_END,
    validation_end: pd.Timestamp = DEFAULT_VALIDATION_END,
    config: HeatwaveConfig | None = None,
) -> tuple[pd.DataFrame, dict[str, float], list[str]]:
    if config is None:
        config = HeatwaveConfig()

    df = daily_df.copy().sort_values("date").reset_index(drop=True)
    df = add_calendar_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df, thresholds = create_heatwave_event_labels(
        df,
        train_end=train_end,
        percentile=config.baseline_percentile,
        min_consecutive_days=config.min_consecutive_days,
    )
    df = create_forecast_target(df, horizon_days=config.forecast_horizon_days)

    feature_columns = [
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
        "year",
        "month",
        "day_of_year",
        "temperature_lag_1",
        "temperature_lag_2",
        "temperature_lag_3",
        "temperature_lag_7",
        "temperature_rolling_3",
        "temperature_rolling_5",
        "temperature_rolling_7",
        "relative_humidity_rolling_3",
        "relative_humidity_rolling_5",
        "relative_humidity_rolling_7",
        "apparent_temperature_rolling_3",
        "apparent_temperature_rolling_5",
        "apparent_temperature_rolling_7",
        "vpd_rolling_3",
        "vpd_rolling_5",
        "vpd_rolling_7",
    ]

    return df, thresholds, feature_columns


def finalize_dataset(df: pd.DataFrame) -> pd.DataFrame:
    required_features = [
        "temperature_lag_1",
        "temperature_lag_2",
        "temperature_lag_3",
        "temperature_lag_7",
        "temperature_rolling_3",
        "temperature_rolling_5",
        "temperature_rolling_7",
        "relative_humidity_rolling_3",
        "relative_humidity_rolling_5",
        "relative_humidity_rolling_7",
        "apparent_temperature_rolling_3",
        "apparent_temperature_rolling_5",
        "apparent_temperature_rolling_7",
        "vpd_rolling_3",
        "vpd_rolling_5",
        "vpd_rolling_7",
        "heatwave",
    ]
    return df.dropna(subset=required_features).reset_index(drop=True)


def save_feature_dataset(
    features_df: pd.DataFrame,
    feature_columns: list[str],
    thresholds: dict[str, float],
    output_path: Path = FEATURES_CSV,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(output_path, index=False)
    save_pickle(feature_columns, FEATURE_COLUMNS_PATH)
    save_pickle(thresholds, HEATWAVE_CONFIG_PATH)


def main() -> None:
    daily_df = load_daily_data()
    features_df, thresholds, feature_columns = build_feature_dataset(daily_df)
    features_df = finalize_dataset(features_df)
    save_feature_dataset(features_df, feature_columns, thresholds)
    print(f"Saved feature dataset to {FEATURES_CSV}")
    print(f"Feature columns: {feature_columns}")
    print(f"Heatwave thresholds saved to {HEATWAVE_CONFIG_PATH}")


if __name__ == "__main__":
    main()
