from __future__ import annotations

import sys
from pathlib import Path
import streamlit as st

# Ensure project root is on sys.path for internal imports
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Configure Streamlit page settings
st.set_page_config(
    page_title="SafeGraph AI - Erode Heatwave Monitoring",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Register Pages cleanly
dashboard_page = st.Page("pages/1_Dashboard.py", title="Dashboard", icon="📊", default=True)
live_weather_page = st.Page("pages/2_Live_Weather.py", title="Live Weather", icon="🌡️")
prediction_page = st.Page("pages/3_Heatwave_Prediction.py", title="Heatwave Prediction", icon="🎯")
historical_page = st.Page("pages/4_Historical_Analysis.py", title="Historical Analysis", icon="📈")
preparedness_page = st.Page("pages/5_Preparedness.py", title="Preparedness", icon="🛡️")
about_page = st.Page("pages/6_About.py", title="About", icon="ℹ️")

pg = st.navigation(
    {
        "Navigation": [
            dashboard_page,
            live_weather_page,
            prediction_page,
            historical_page,
            preparedness_page,
            about_page,
        ]
    }
)

# Single concise sidebar footer
with st.sidebar:
    st.divider()
    st.caption("🔥 **SafeGraph AI** | Erode, TN")

pg.run()

