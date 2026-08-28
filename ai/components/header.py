from __future__ import annotations

from datetime import datetime
import streamlit as st
from services.db_service import get_db_status
from utils.constants import LOCATION_NAME


def render_app_header(title: str = "Heatwave Monitoring", is_live: bool = True):
    """Render single clean application header."""
    st.markdown(
        """
        <style>
        .header-container {
            background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
            border: 1px solid #334155;
            border-radius: 12px;
            padding: 16px 20px;
            margin-bottom: 20px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
        }
        .header-title {
            color: #F8FAFC;
            font-weight: 800;
            font-size: 1.5rem;
            margin: 0;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .header-sub {
            color: #94A3B8;
            font-size: 0.88rem;
            margin-top: 4px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([3, 1])

    status_pill = "🟢 Live weather connected" if is_live else "⚠️ Live weather offline"

    with col1:
        st.markdown(
            f"""
            <div class="header-container">
                <div class="header-title">
                    🔥 SafeGraph AI – {title}
                </div>
                <div class="header-sub">
                    AI-powered heatwave prediction and disaster preparedness for Erode, Tamil Nadu
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.caption(status_pill)
        st.caption(f"🕒 **Updated**: {datetime.now().strftime('%H:%M IST')}")
        if st.button("🔄 Refresh Weather", help="Fetch fresh live data from Open-Meteo API", use_container_width=True):
            st.cache_data.clear()
            st.rerun()


