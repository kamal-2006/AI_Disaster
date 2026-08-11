# SafeGraph AI - Erode Heatwave Prediction Module

This folder contains a standalone Python ML module for predicting heatwave risk for Erode, Tamil Nadu, India.

## Purpose

The goal is to transform the raw hourly weather history for Erode into a reproducible daily forecasting dataset, train time-series-safe classifiers, and output a heatwave probability for the next day by default.

## Dataset

Place the raw hourly CSV here:

`ai/data/raw/erode_weather_2000_2026.csv`

The current implementation expects these columns after cleaning:

- `time`
- `temperature_2m`
- `relative_humidity_2m`
- `wind_speed_10m`
- `precipitation`
- `vapour_pressure_deficit`
- `wind_gusts_10m`
- `cloud_cover`
- `dew_point_2m`
- `apparent_temperature`

The preprocessing code validates the schema at runtime and will stop with a clear error if the file is missing or columns do not match.

## Location and Date Range

This module is tailored for Erode, Tamil Nadu, India. The raw file is expected to cover roughly 2000 to the present.

## Data Preprocessing

`preprocessing/preprocess.py`:

- reads the raw hourly CSV
- parses the `time` column into pandas datetimes
- sorts chronologically
- removes duplicate timestamps and duplicate rows
- reports missing values by column
- converts numeric fields safely
- handles obvious invalid values by converting them to missing values rather than zero
- imputes hourly gaps using time interpolation and forward/backward fill only where needed
- aggregates hourly observations into a daily dataset

The daily aggregation uses the statistics requested in the task, including mean, max, min, and sum where appropriate.

## Hourly to Daily Aggregation

The processed daily file is saved to:

`ai/data/processed/erode_daily.csv`

## Heatwave Definition

Heatwave labeling is based on local climatology rather than a blind `temperature > 40C` rule.

Implementation summary:

- Daily heat stress is derived from the daily maximum temperature.
- A month-specific 95th percentile threshold is computed from the historical training period only.
- A day is treated as a heatwave day only if it meets the threshold and is part of a run of at least `HEATWAVE_MIN_CONSECUTIVE_DAYS` hot days.

This makes the label location-specific, persistence-based, and less sensitive to single isolated spikes.

## Target Construction

The model predicts whether a heatwave will occur in the near future, not whether the current day is already a heatwave.

Default setup:

- target horizon: next 1 day
- minimum consecutive hot days: 2

So each feature row uses information available on or before day `t`, while the label is `1` if a heatwave occurs on day `t+1`.

## Feature Engineering

`features/feature_engineering.py` adds:

- `year`
- `month`
- `day_of_year`
- temperature lags: `temperature_lag_1`, `temperature_lag_2`, `temperature_lag_3`, `temperature_lag_7`
- temperature rolling means: `temperature_rolling_3`, `temperature_rolling_5`, `temperature_rolling_7`
- rolling humidity, apparent temperature, and VPD features

All lag and rolling features use only past values to avoid leakage.

## Time-Series Split

The preferred split is chronological:

- Training: 2000–2021
- Validation: 2022–2024
- Testing: 2025–2026

If the latest available date is different, the code falls back to a chronological date-based split without shuffling.

## Models

The training script compares:

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost

If XGBoost is not installed, the script prints a clear installation requirement.

## Evaluation

The module reports:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

The selected model is chosen mainly for recall and F1-score, because missed heatwaves are more costly than false alarms.

Outputs are written to `ai/outputs/`, including the confusion matrix and feature importance plot.

## Installation

From inside `ai/`:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Pipeline

Preprocess:

```bash
python preprocessing/preprocess.py
```

Feature engineering:

```bash
python features/feature_engineering.py
```

Train:

```bash
python training/train.py
```

Evaluate:

```bash
python evaluation/evaluate.py
```

Predict:

```bash
python prediction/predict.py --input-csv path\to\recent_daily_history.csv
```

Prediction input should contain enough recent daily history to build the lag and rolling features, ideally at least 8 days.

## Saved Artifacts

- `models/heatwave_model.pkl`
- `models/feature_columns.pkl`
- `models/training_metadata.pkl`
- `outputs/model_results.csv`
- `outputs/confusion_matrix.png`
- `outputs/feature_importance.png`

## Notes

The raw CSV is not committed automatically. Place it locally in `ai/data/raw/` before running preprocessing or training.
