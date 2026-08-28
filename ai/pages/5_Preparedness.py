from __future__ import annotations

import streamlit as st
from components.header import render_app_header
from components.recommendations import render_preparedness_section
from services.prediction_service import get_realtime_prediction


def show_preparedness_page():
    render_app_header(title="Disaster Preparedness Hub")

    with st.spinner("Checking active risk status..."):
        prediction_data = get_realtime_prediction()

    current_risk = prediction_data.get("risk_level", "LOW")

    st.markdown(
        f"""
        ### 🛡️ Heatwave Emergency Preparedness Guidelines
        Evidence-based disaster risk reduction guidelines for Erode district.
        Currently active model prediction risk level: **{current_risk}**.
        """
    )

    st.divider()

    # Persona Preparedness Tabs & Downloadable Checklist
    render_preparedness_section(current_risk_level=current_risk)

    st.divider()

    # Emergency Contact Protocols
    st.markdown("### 📞 Emergency Helplines (Erode District)")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.info("🚑 **Medical Emergency**\n\nCall **108** (TN Ambulance)")
    with col2:
        st.warning("🔥 **District Control Room**\n\nCall **1077** (Collectorate)")
    with col3:
        st.success("💧 **Water Supply**\n\nCall **1913** (Corporation)")
    with col4:
        st.error("⚡ **Power Outage**\n\nCall **1912** (Electricity)")


if __name__ == "__main__":
    show_preparedness_page()

