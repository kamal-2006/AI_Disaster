from __future__ import annotations

import streamlit as st
from components.charts import (
    plot_heatwave_calendar_heatmap,
    plot_monthly_heatwave_distribution,
    plot_rolling_temperature_trend,
    plot_yearly_heatwave_frequency,
)
from components.header import render_app_header
from services.historical_service import (
    get_heatwave_calendar_heatmap,
    get_monthly_heatwave_summary,
    get_yearly_heatwave_summary,
    load_historical_daily_data,
    load_historical_features_data,
)


def show_historical_analysis_page():
    render_app_header(title="Historical Climate Analysis")

    with st.spinner("Loading Erode historical climate records..."):
        daily_df = load_historical_daily_data()
        features_df = load_historical_features_data()

    if daily_df.empty:
        st.error("Historical dataset `erode_daily.csv` not found in `data/processed/`. Run preprocessing first.")
        return

    # Key Historical KPIs
    total_days = len(daily_df)
    min_date = daily_df["date"].min().strftime("%Y-%m-%d")
    max_date = daily_df["date"].max().strftime("%Y-%m-%d")
    peak_temp = daily_df["max_temperature"].max()
    avg_max_temp = daily_df["max_temperature"].mean()

    hw_col = "heatwave" if "heatwave" in features_df.columns else "heatwave_event" if "heatwave_event" in features_df.columns else None
    total_hw_days = int(features_df[hw_col].sum()) if hw_col and hw_col in features_df.columns else int((daily_df["max_temperature"] >= 38.5).sum())

    st.markdown("### 📊 Climate Records Summary")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Recorded Days", f"{total_days:,}", help=f"Date Range: {min_date} to {max_date}")
    with col2:
        st.metric("Total Heatwave Days", f"{total_hw_days:,}")
    with col3:
        st.metric("All-Time Peak Temp", f"{peak_temp:.1f} °C")
    with col4:
        st.metric("Avg Max Temp", f"{avg_max_temp:.1f} °C")

    st.divider()

    # 1. Yearly Heatwave Frequency
    st.markdown("### 📅 Yearly Heatwave Frequency")
    yearly_summary = get_yearly_heatwave_summary(features_df if not features_df.empty else daily_df)
    plot_yearly_heatwave_frequency(yearly_summary)

    st.divider()

    # 2. Monthly Distribution
    st.markdown("### 🗓️ Monthly Heatwave Distribution")
    monthly_summary = get_monthly_heatwave_summary(features_df if not features_df.empty else daily_df)
    plot_monthly_heatwave_distribution(monthly_summary)

    st.divider()

    # 3. Rolling Temperature Trend
    st.markdown("### 📈 Multi-Year Temperature Trend (30-Day Rolling Avg)")
    plot_rolling_temperature_trend(daily_df)

    st.divider()

    # 4. Heatwave Calendar Matrix
    st.markdown("### 🗓️ Heatwave Calendar Matrix")
    heatmap_df = get_heatwave_calendar_heatmap(features_df if not features_df.empty else daily_df)
    plot_heatwave_calendar_heatmap(heatmap_df)

    st.divider()

    # 5. Dataset Explorer
    with st.expander("🔍 Historical Dataset Table Explorer"):
        st.dataframe(daily_df, use_container_width=True)


if __name__ == "__main__":
    show_historical_analysis_page()

