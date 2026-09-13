# Independent Heatwave Prediction Module

This module predicts Erode heatwave risk for one user-provided date. It does
not import or modify the Disaster Assistant, Knowledge Graph, or LLM logic.

## Run

From the `ai/` directory:

```powershell
..\.venv\Scripts\python.exe -m heatwave_prediction.predictor 2026-09-20
```

Accepted dates include `Today`, `Tomorrow`, `YYYY-MM-DD`, `September 20, 2026`,
and `20/09/2026`.

## Existing artifacts reused

- Model: `models/heatwave_model.pkl`
- Feature order: `models/feature_columns.pkl` (35 saved columns)
- Historical data: `data/processed/erode_daily.csv`
- Feature preprocessing: `features.feature_engineering.add_calendar_features`,
  `add_lag_features`, and `add_rolling_features`
- Risk classification: `services.prediction_service.calculate_risk_level`
- Weather: Open-Meteo forecast API for Erode (`11.3410`, `77.7172`)

The module never substitutes a fabricated temperature or a threshold-based
prediction. Dates outside the available Open-Meteo range return an unavailable
result.