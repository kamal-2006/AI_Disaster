from __future__ import annotations

import sys
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common import resolve_path, save_pickle  # noqa: E402
from features.feature_engineering import (  # noqa: E402
    DEFAULT_TRAIN_END,
    DEFAULT_VALIDATION_END,
    HeatwaveConfig,
    build_feature_dataset,
    finalize_dataset,
    load_daily_data,
    save_feature_dataset,
)

try:  # pragma: no cover - optional dependency guard
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    XGBClassifier = None


FEATURES_CSV = resolve_path("data", "processed", "erode_features.csv")
RESULTS_CSV = resolve_path("outputs", "model_results.csv")
CONFUSION_MATRIX_PATH = resolve_path("outputs", "confusion_matrix.png")
FEATURE_IMPORTANCE_PATH = resolve_path("outputs", "feature_importance.png")
MODEL_PATH = resolve_path("models", "heatwave_model.pkl")
FEATURE_COLUMNS_PATH = resolve_path("models", "feature_columns.pkl")
TRAINED_MODEL_META_PATH = resolve_path("models", "training_metadata.pkl")

RANDOM_STATE = 42


def chronological_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    latest_date = pd.Timestamp(df["date"].max())
    if latest_date >= pd.Timestamp("2025-12-31"):
        train_mask = df["date"] <= DEFAULT_TRAIN_END
        validation_mask = (df["date"] > DEFAULT_TRAIN_END) & (df["date"] <= DEFAULT_VALIDATION_END)
        test_mask = df["date"] > DEFAULT_VALIDATION_END
        if not test_mask.any():
            test_start = latest_date - pd.DateOffset(months=12)
            validation_start = test_start - pd.DateOffset(months=24)
            train_mask = df["date"] <= validation_start
            validation_mask = (df["date"] > validation_start) & (df["date"] <= test_start)
            test_mask = df["date"] > test_start
    else:
        dates = df["date"].sort_values().drop_duplicates().reset_index(drop=True)
        train_cutoff = dates.iloc[max(1, int(len(dates) * 0.7)) - 1]
        validation_cutoff = dates.iloc[max(2, int(len(dates) * 0.85)) - 1]
        train_mask = df["date"] <= train_cutoff
        validation_mask = (df["date"] > train_cutoff) & (df["date"] <= validation_cutoff)
        test_mask = df["date"] > validation_cutoff

    train_df = df.loc[train_mask].copy()
    validation_df = df.loc[validation_mask].copy()
    test_df = df.loc[test_mask].copy()
    if train_df.empty or validation_df.empty or test_df.empty:
        raise ValueError("One of the chronological splits is empty. Check the available date range.")
    return train_df, validation_df, test_df


def compute_class_balance(y: pd.Series) -> dict[str, float]:
    counts = y.value_counts().sort_index()
    normal = int(counts.get(0, 0))
    heatwave = int(counts.get(1, 0))
    imbalance_ratio = float(normal / heatwave) if heatwave else float("inf")
    print(f"Normal samples: {normal}")
    print(f"Heatwave samples: {heatwave}")
    print(f"Imbalance ratio (normal/heatwave): {imbalance_ratio:.2f}")
    return {"normal": normal, "heatwave": heatwave, "imbalance_ratio": imbalance_ratio}


def build_model_pipelines(feature_columns: list[str], imbalance_ratio: float) -> dict[str, Pipeline]:
    logistic_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced" if imbalance_ratio > 1.5 else None,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    tree_class_weight = "balanced" if imbalance_ratio > 1.5 else None
    decision_tree_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "classifier",
                DecisionTreeClassifier(
                    class_weight=tree_class_weight,
                    random_state=RANDOM_STATE,
                    max_depth=8,
                    min_samples_leaf=10,
                ),
            ),
        ]
    )

    random_forest_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    class_weight="balanced_subsample" if imbalance_ratio > 1.5 else None,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                    max_depth=None,
                    min_samples_leaf=5,
                ),
            ),
        ]
    )

    pipelines = {
        "Logistic Regression": logistic_pipeline,
        "Decision Tree": decision_tree_pipeline,
        "Random Forest": random_forest_pipeline,
    }

    if XGBClassifier is not None:
        xgb_scale_pos_weight = imbalance_ratio if np.isfinite(imbalance_ratio) and imbalance_ratio > 1 else 1.0
        xgb_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "classifier",
                    XGBClassifier(
                        n_estimators=400,
                        learning_rate=0.05,
                        max_depth=4,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        reg_lambda=1.0,
                        random_state=RANDOM_STATE,
                        scale_pos_weight=xgb_scale_pos_weight,
                        eval_metric="logloss",
                        tree_method="hist",
                    ),
                ),
            ]
        )
        pipelines["XGBoost"] = xgb_pipeline
    else:
        print("XGBoost is not installed. Install it with `pip install xgboost` to enable this model.")

    return pipelines


def evaluate_pipeline(model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    predictions = model.predict(x_test)
    probability_scores = model.predict_proba(x_test)[:, 1] if hasattr(model, "predict_proba") else None
    return {
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions, zero_division=0),
        "Recall": recall_score(y_test, predictions, zero_division=0),
        "F1": f1_score(y_test, predictions, zero_division=0),
        "ROC_AUC": roc_auc_score(y_test, probability_scores) if probability_scores is not None and len(np.unique(y_test)) > 1 else np.nan,
    }


def choose_best_model(results: pd.DataFrame) -> str:
    scoring = results.copy()
    scoring["selection_score"] = scoring["Recall"] * 0.45 + scoring["F1"] * 0.40 + scoring["ROC_AUC"].fillna(0) * 0.15
    best_row = scoring.sort_values(["selection_score", "Recall", "F1", "ROC_AUC"], ascending=False).iloc[0]
    return str(best_row["Model"])


def plot_confusion_matrix(y_true: pd.Series, y_pred: pd.Series, path: Path) -> None:
    matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="YlOrRd", cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def get_feature_importance(model: Pipeline, feature_columns: list[str]) -> pd.Series | None:
    classifier = model.named_steps["classifier"]
    if hasattr(classifier, "feature_importances_"):
        return pd.Series(classifier.feature_importances_, index=feature_columns).sort_values(ascending=False)
    if hasattr(classifier, "coef_"):
        return pd.Series(np.abs(classifier.coef_[0]), index=feature_columns).sort_values(ascending=False)
    return None


def plot_feature_importance(importances: pd.Series, path: Path, top_n: int = 15) -> None:
    top_features = importances.head(top_n).sort_values()
    plt.figure(figsize=(10, 7))
    top_features.plot(kind="barh", color="#d97706")
    plt.title("Top Feature Importance")
    plt.xlabel("Importance")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=200)
    plt.close()


def plot_supporting_trends(df: pd.DataFrame) -> None:
    outputs_dir = resolve_path("outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 5))
    df.sort_values("date").set_index("date")["max_temperature"].rolling(30, min_periods=10).mean().plot(color="#b45309")
    plt.title("30-Day Rolling Temperature Trend")
    plt.ylabel("Max Temperature (C)")
    plt.tight_layout()
    plt.savefig(outputs_dir / "temperature_trend.png", dpi=200)
    plt.close()

    heatwave_by_year = df.groupby(df["date"].dt.year)["heatwave"].mean().mul(100)
    plt.figure(figsize=(10, 5))
    heatwave_by_year.plot(kind="bar", color="#2563eb")
    plt.title("Heatwave Frequency by Year")
    plt.ylabel("Heatwave Rate (%)")
    plt.tight_layout()
    plt.savefig(outputs_dir / "heatwave_frequency_by_year.png", dpi=200)
    plt.close()

    heatwave_by_month = df.groupby(df["date"].dt.month)["heatwave"].mean().mul(100)
    plt.figure(figsize=(10, 5))
    heatwave_by_month.plot(kind="bar", color="#059669")
    plt.title("Heatwave Frequency by Month")
    plt.ylabel("Heatwave Rate (%)")
    plt.tight_layout()
    plt.savefig(outputs_dir / "heatwave_frequency_by_month.png", dpi=200)
    plt.close()


def main() -> None:
    warnings.filterwarnings("ignore")
    daily_df = load_daily_data()
    features_df, thresholds, feature_columns = build_feature_dataset(
        daily_df,
        train_end=DEFAULT_TRAIN_END,
        validation_end=DEFAULT_VALIDATION_END,
        config=HeatwaveConfig(),
    )
    features_df = finalize_dataset(features_df)
    save_feature_dataset(features_df, feature_columns, thresholds)

    train_df, validation_df, test_df = chronological_split(features_df)
    x_train, y_train = train_df[feature_columns], train_df["heatwave"]
    x_validation, y_validation = validation_df[feature_columns], validation_df["heatwave"]
    x_test, y_test = test_df[feature_columns], test_df["heatwave"]

    balance_info = compute_class_balance(y_train)
    imbalance_ratio = balance_info["imbalance_ratio"]
    pipelines = build_model_pipelines(feature_columns, imbalance_ratio)

    results = []
    trained_models: dict[str, Pipeline] = {}
    for model_name, pipeline in pipelines.items():
        print(f"Training {model_name}...")
        pipeline.fit(x_train, y_train)
        trained_models[model_name] = pipeline
        validation_metrics = evaluate_pipeline(pipeline, x_validation, y_validation)
        validation_metrics["Model"] = model_name
        results.append(validation_metrics)
        print(validation_metrics)

    results_df = pd.DataFrame(results)[["Model", "Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]]
    results_df.to_csv(RESULTS_CSV, index=False)
    print(f"Saved model comparison to {RESULTS_CSV}")

    best_model_name = choose_best_model(results_df)
    best_model = trained_models[best_model_name]
    test_metrics = evaluate_pipeline(best_model, x_test, y_test)
    print(f"Selected best model: {best_model_name}")
    print(f"Test metrics: {test_metrics}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    save_pickle(feature_columns, FEATURE_COLUMNS_PATH)
    save_pickle(
        {
            "best_model": best_model_name,
            "test_metrics": test_metrics,
            "thresholds": thresholds,
            "train_end": str(DEFAULT_TRAIN_END.date()),
            "validation_end": str(DEFAULT_VALIDATION_END.date()),
            "imbalance_ratio": imbalance_ratio,
        },
        TRAINED_MODEL_META_PATH,
    )

    y_test_pred = best_model.predict(x_test)
    plot_confusion_matrix(y_test, y_test_pred, CONFUSION_MATRIX_PATH)

    feature_importance = get_feature_importance(best_model, feature_columns)
    if feature_importance is not None:
        plot_feature_importance(feature_importance, FEATURE_IMPORTANCE_PATH)

    plot_supporting_trends(features_df)
    print(f"Saved final model to {MODEL_PATH}")
    print(f"Saved confusion matrix to {CONFUSION_MATRIX_PATH}")
    if feature_importance is not None:
        print(f"Saved feature importance to {FEATURE_IMPORTANCE_PATH}")


if __name__ == "__main__":
    main()
