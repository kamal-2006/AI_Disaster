# Heatwave Prediction Module

Independent, date-aware heatwave risk prediction system for **Erode, Tamil Nadu**.

## Overview
This module integrates live Open-Meteo weather forecast data with a trained machine learning classification pipeline (`models/heatwave_model.pkl`) to predict temperature and heatwave risk for any given target date.

## Module Structure
```
heatwave_prediction/
│
├── __init__.py         # Package entry points (predict_heatwave, parse_date_input)
├── predictor.py        # Main prediction controller & response formatter
├── weather_service.py  # Open-Meteo forecast & current weather service
├── preprocessing.py    # 35-feature engineering matching ML training pipeline
├── date_parser.py      # Flexible date parser (ISO, DD-MM-YYYY, DD/MM/YYYY, relative)
├── config.py           # Erode coordinates, API URLs, and model configurations
└── README.md           # Documentation
```

## Workflow Architecture
```
              USER
               │
               ▼
          ENTER DATE
               │
               ▼
         DATE PARSER
               │
      ┌────────┴────────┐
      │                 │
    TODAY             FUTURE
      │                 │
      ▼                 ▼
Current Weather      Forecast API
      │                 │
      └────────┬────────┘
               ▼
          WEATHER DATA
               │
               ▼
      FEATURE ENGINEERING
               │
               ▼
      EXISTING PREPROCESSING
               │
               ▼
         HEATWAVE MODEL
       (heatwave_model.pkl)
               │
               ▼
          HEATWAVE RISK
               │
               ▼
       TEMPERATURE + RISK
```

## Supported Input Date Formats
- `2026-09-20` (YYYY-MM-DD)
- `20-09-2026` (DD-MM-YYYY)
- `20/09/2026` (DD/MM/YYYY)
- `September 20, 2026`
- `20 September 2026`
- `today`
- `tomorrow`
- `next week`

## Programmatic Usage
```python
from heatwave_prediction import predict_heatwave

result = predict_heatwave("2026-09-20")
print(result["user_response"])
```