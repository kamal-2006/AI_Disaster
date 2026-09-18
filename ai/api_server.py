from __future__ import annotations

import sys
import logging
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Ensure project root is in sys.path for internal module imports
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from heatwave_prediction.date_parser import parse_date_input
from heatwave_prediction.predictor import predict_heatwave

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("heatwave-ai-api")

app = FastAPI(
    title="Heatwave AI REST API",
    description="Standalone REST API service for Erode Heatwave Prediction model inference.",
    version="1.0.0",
)

# Enable CORS for ResQLearn local development frontend (and standard localhost origins)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictHeatwaveRequest(BaseModel):
    date: str = Field(
        ...,
        description="Target prediction date (e.g. 'today', 'tomorrow', '2026-09-20', '20/09/2026', 'September 20, 2026')",
        example="2026-09-20",
    )


@app.get("/health", status_code=status.HTTP_200_OK)
def health_check() -> Dict[str, str]:
    """Basic health check endpoint."""
    return {
        "status": "ok",
        "service": "heatwave-ai",
    }


@app.post("/api/predict-heatwave", status_code=status.HTTP_200_OK)
def predict_heatwave_endpoint(payload: PredictHeatwaveRequest) -> Dict[str, Any]:
    """
    Predict heatwave risk and temperature for a given date.
    Returns heatwave risk level, probability, model output, and weather information.
    """
    raw_date = payload.date.strip() if payload.date else ""
    if not raw_date:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing 'date' parameter in request payload.",
        )

    # 1. Validate date input
    parsed = parse_date_input(raw_date)
    if not parsed.get("valid"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid date format or value: {parsed.get('error')}",
        )

    # 2. Invoke prediction engine
    try:
        result = predict_heatwave(raw_date)
    except Exception as exc:
        logger.error(f"Prediction execution failure for date '{raw_date}': {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while processing the heatwave prediction.",
        )

    # 3. Format response payload preserving required fields
    if not isinstance(result, dict):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction engine returned an invalid data structure.",
        )

    # Clean response dictionary for client output
    response_data = {
        "available": result.get("available", False),
        "date": result.get("date"),
        "formatted_date": result.get("formatted_date"),
        "location": result.get("location"),
        "temperature": result.get("temperature"),
        "risk": result.get("risk"),
        "probability": result.get("probability"),
        "prediction_class": result.get("prediction_class"),
        "prediction": result.get("prediction"),
        "source": result.get("source"),
        "weather_source": result.get("weather_source"),
        "prediction_source": result.get("prediction_source", "forecast"),
        "date_type": result.get("date_type"),
        "model_name": result.get("model_name"),
    }

    # Include explanation or error message if data is unavailable (e.g. out of forecast range)
    if not response_data["available"]:
        response_data["explanation"] = result.get("explanation") or result.get("error") or "Data unavailable for requested date."

    return response_data


@app.get("/api/heatwave/analytics", status_code=status.HTTP_200_OK)
def heatwave_analytics_endpoint() -> Dict[str, Any]:
    """
    Retrieve historical climate analytics, heatwave frequency, monthly distributions, and summary statistics.
    Migrated from SafeGraph Streamlit historical service.
    """
    try:
        from services.historical_service import (
            load_historical_daily_data,
            load_historical_features_data,
            get_yearly_heatwave_summary,
            get_monthly_heatwave_summary,
        )

        daily_df = load_historical_daily_data()
        features_df = load_historical_features_data()

        if daily_df.empty:
            return {"error": "Historical climate dataset unavailable."}

        total_days = len(daily_df)
        min_date = daily_df["date"].min().strftime("%Y-%m-%d")
        max_date = daily_df["date"].max().strftime("%Y-%m-%d")
        peak_temp = float(daily_df["max_temperature"].max())
        avg_max_temp = float(daily_df["max_temperature"].mean())

        hw_col = "heatwave" if "heatwave" in features_df.columns else "heatwave_event" if "heatwave_event" in features_df.columns else None
        if hw_col:
            total_hw_days = int(features_df[hw_col].sum())
        else:
            total_hw_days = int((daily_df["max_temperature"] >= 38.5).sum())

        yearly_df = get_yearly_heatwave_summary(features_df if not features_df.empty else daily_df)
        monthly_df = get_monthly_heatwave_summary(features_df if not features_df.empty else daily_df)

        yearly_trends = yearly_df.to_dict(orient="records") if not yearly_df.empty else []
        monthly_distribution = monthly_df.to_dict(orient="records") if not monthly_df.empty else []

        # Find highest heatwave month
        highest_month = "April"
        if not monthly_df.empty and "heatwave_days" in monthly_df.columns:
            top_m = monthly_df.sort_values("heatwave_days", ascending=False).iloc[0]
            highest_month = str(top_m.get("month_name", "April"))

        return {
            "summary": {
                "recorded_days": total_days,
                "date_range": f"{min_date} to {max_date}",
                "total_heatwave_days": total_hw_days,
                "all_time_peak_temp": round(peak_temp, 1),
                "overall_avg_max_temp": round(avg_max_temp, 1),
                "peak_heatwave_month": highest_month,
                "location": "Erode, Tamil Nadu",
            },
            "yearly_trends": yearly_trends,
            "monthly_distribution": monthly_distribution,
        }
    except Exception as exc:
        logger.error(f"Failed to generate heatwave analytics: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to calculate heatwave analytics: {exc}",
        )

