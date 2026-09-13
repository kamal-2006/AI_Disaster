# SafeGraph AI - Real-Time Heatwave Prediction & Explainable Disaster Preparedness Frontend

A **Streamlit web application** for **“SafeGraph AI: A Knowledge Graph and Agentic AI Framework for Explainable Disaster Preparedness and Emergency Response Education”**, focused on real-time heatwave prediction and explainable disaster preparedness education for **Erode, Tamil Nadu, India**.

---

## 🌟 Key Features

1. **Real-Time Live Weather Telemetry**:
   - Integrates live weather metrics from the **Open-Meteo Weather API** (`https://api.open-meteo.com/v1/forecast`) for Erode (Lat: `11.3410`, Lon: `77.7172`, Timezone: `Asia/Kolkata`).
   - Fetches live temperature, relative humidity, wind speed, precipitation, vapour pressure deficit (VPD), wind gusts, cloud cover, dew point, and apparent temperature.

2. **Leakage-Free ML Feature Pipeline**:
   - Combines past daily history (`erode_daily.csv`) with live weather API data to compute **35 exact features** used during model training:
     - 16 Daily Weather Aggregates
     - 3 Calendar Features (`year`, `month`, `day_of_year`)
     - 4 Temperature Lag Features (`lag_1`, `lag_2`, `lag_3`, `lag_7`)
     - 12 Rolling Mean Features (3/5/7 day means for temperature, humidity, apparent temp, and VPD)

3. **Real-Time ML Inference & Risk Classification**:
   - Loads the existing trained machine learning model (`heatwave_model.pkl`) without retraining.
   - Outputs heatwave probability (0.0 to 1.0) and assigns visual risk badges:
     - 🟢 **LOW RISK**: Probability < 50%
     - 🟠 **MEDIUM RISK**: 50% ≤ Probability < 80%
     - 🔴 **HIGH RISK**: Probability ≥ 80%

4. **Explainable AI (XAI)**:
   - Renders actual model feature importance weights and driver values using Plotly horizontal bar charts.
   - Provides plain-English decision narratives explaining why a specific risk level was assigned.

5. **Interactive "What-If" Scenario Simulator**:
   - Allows users to tweak temperature, humidity, VPD, and wind speed using sliders to test hypothetical heatwave conditions against the trained ML model in real time.

6. **Historical Climate Analytics (2000–Present)**:
   - Visualizes long-term Erode dataset trends:
     - Yearly heatwave frequency
     - Monthly heatwave distribution (identifying April–June peak stress)
     - Multi-year 30-day rolling temperature trends
     - Year × Month heatwave calendar heatmap matrix

7. **Multi-Persona Disaster Preparedness Education**:
   - Actionable guidelines dynamically tailored for **Students**, **Teachers**, **School/City Administrators**, and the **General Public**.
   - Interactive safety completion checklists with downloadable `.txt` action guides.

8. **Optional MongoDB Logging**:
   - Logs prediction outputs to a MongoDB collection (`safegraph_ai` DB).
   - Operates with graceful fallback if MongoDB is offline or unconfigured.

9. **Performance & Caching**:
   - Streamlit `@st.cache_data(ttl=300)` for 5-minute weather API caching.
   - Manual **"🔄 Refresh Weather"** button to force API data updates.

## Disaster Assistant Retrieval Pipeline

The Disaster Assistant follows this sequence:

```text
User question
   -> question understanding (disaster, intent, topic, urgency, entities)
   -> Knowledge Graph retrieval (Disaster -> Topic -> Knowledge nodes)
   -> relevance ranking and filtering
   -> LLM reasoning over retrieved context (when configured)
   -> explainable answer
```

The current repository does not contain a Neo4j dependency, connection, schema, or
persisted graph. `services/knowledge_graph.py` therefore materializes the existing
curated knowledge base as a local graph with explicit nodes and relationships. It
is the retrieval boundary used by the assistant and can be replaced by a Neo4j
adapter later without changing the assistant or LLM contract. Unsupported
disasters return a clear no-data response instead of falling back to unrelated
generic guidance.

Set `SAFEGRAPH_AI_DEBUG=1` during development to include question understanding,
retrieved graph nodes and relationships, final context, final LLM prompt, and
answer in the service result. The Streamlit UI does not render these internals.

---

## 📂 Modular Code Structure

```text
ai/
├── app.py                      # Main Streamlit application entry point & navigation
├── pages/
│   ├── 1_Dashboard.py          # Executive dashboard view
│   ├── 2_Live_Weather.py       # Real-time Erode weather observatory
│   ├── 3_Heatwave_Prediction.py# ML prediction breakdown, XAI, and What-If simulator
│   ├── 4_Historical_Analysis.py# Long-term climate analytics (2000-present)
│   ├── 5_Preparedness.py       # Persona-based disaster preparedness guidelines
│   └── 6_About.py              # Framework mission, model specs & evaluation metrics
├── components/
│   ├── header.py               # Top bar banner, location pill, and refresh button
│   ├── metrics_cards.py        # Weather metrics grid & risk badges
│   ├── charts.py               # Plotly interactive forecast & historical charts
│   ├── explainability.py       # XAI feature importance chart & narrative
│   ├── alerts.py               # Color-coded heatwave risk alert banners
│   └── recommendations.py      # Persona recommendations & checklist downloader
├── services/
│   ├── weather_api.py          # Open-Meteo live API fetcher & caching
│   ├── prediction_service.py   # ML inference, feature engineering, and simulation
│   ├── historical_service.py   # Climate dataset aggregations
│   └── db_service.py           # Optional MongoDB logging with graceful fallback
├── utils/
│   ├── constants.py            # Location constants, feature maps, preparedness text
│   └── helpers.py              # Formatting & HTML badge helpers
├── models/
│   ├── heatwave_model.pkl      # Trained ML model pipeline
│   ├── feature_columns.pkl     # Saved 35 feature names
│   ├── heatwave_config.pkl     # Baseline thresholds
│   └── training_metadata.pkl  # Model performance metadata
├── data/
│   ├── raw/
│   │   └── erode_weather_2000_2026.csv
│   └── processed/
│       ├── erode_daily.csv
│       └── erode_features.csv
├── .streamlit/
│   ├── config.toml             # Custom dark mode UI theme
│   └── secrets.toml.example    # Secrets template for MongoDB / API config
├── requirements.txt            # Project dependencies
└── README.md                   # Detailed project documentation
```

---

## 🚀 Execution & Quick Start Guide

### 1. Environment Setup

From inside the `ai/` directory:

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Windows (CMD):
.venv\Scripts\activate.bat
# Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Streamlit Frontend App

```bash
streamlit run app.py
```

The application will launch in your browser at `http://localhost:8501`.

---

## ⚙️ Configuration & Secrets Management

To configure optional MongoDB logging, copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml`:

```toml
[mongodb]
uri = "mongodb://localhost:27017"
database = "safegraph_ai"
```

> **Note**: Secrets in `.streamlit/secrets.toml` are excluded by `.gitignore` to prevent credential exposure.

---

## 🧪 Pipeline Verification

To test individual service modules:

```bash
# Test Open-Meteo Live API fetching
python -c "from services.weather_api import fetch_erode_weather; print(fetch_erode_weather())"

# Test ML real-time inference
python -c "from services.prediction_service import get_realtime_prediction; print(get_realtime_prediction())"

# Test Historical Climate Service
python -c "from services.historical_service import load_historical_daily_data; print(load_historical_daily_data().head())"
```

---

## 🎓 Citation & Project Context

Part of the **SafeGraph AI** research framework for explainable AI in disaster risk management and climate adaptation education.
