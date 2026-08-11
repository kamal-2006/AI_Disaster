from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common import load_pickle, resolve_path  # noqa: E402
from evaluation import __init__  # noqa: F401,E402
from features.feature_engineering import DEFAULT_TRAIN_END, DEFAULT_VALIDATION_END  # noqa: E402
from training.train import chronological_split, evaluate_pipeline, get_feature_importance, plot_confusion_matrix  # noqa: E402


FEATURES_CSV = resolve_path("data", "processed", "erode_features.csv")
MODEL_PATH = resolve_path("models", "heatwave_model.pkl")
FEATURE_COLUMNS_PATH = resolve_path("models", "feature_columns.pkl")
CONFUSION_MATRIX_PATH = resolve_path("outputs", "confusion_matrix.png")
FEATURE_IMPORTANCE_PATH = resolve_path("outputs", "feature_importance.png")


def load_feature_dataset() -> pd.DataFrame:
    if not FEATURES_CSV.exists():
        raise FileNotFoundError(
            f"Feature dataset not found at {FEATURES_CSV}. Run python features/feature_engineering.py first."
        )
    return pd.read_csv(FEATURES_CSV, parse_dates=["date"])


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run python training/train.py first.")

    model = joblib.load(MODEL_PATH)
    feature_columns = load_pickle(FEATURE_COLUMNS_PATH)
    df = load_feature_dataset()
    _, _, test_df = chronological_split(df)
    x_test, y_test = test_df[feature_columns], test_df["heatwave"]

    metrics = evaluate_pipeline(model, x_test, y_test)
    print("Test metrics:")
    print(metrics)

    y_pred = model.predict(x_test)
    plot_confusion_matrix(y_test, y_pred, CONFUSION_MATRIX_PATH)

    feature_importance = get_feature_importance(model, feature_columns)
    if feature_importance is not None:
        import matplotlib.pyplot as plt

        top_features = feature_importance.head(15)
        plt.figure(figsize=(10, 7))
        top_features.sort_values().plot(kind="barh", color="#d97706")
        plt.title("Top Feature Importance")
        plt.xlabel("Importance")
        plt.tight_layout()
        plt.savefig(FEATURE_IMPORTANCE_PATH, dpi=200)
        plt.close()


if __name__ == "__main__":
    main()
