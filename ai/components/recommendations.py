from __future__ import annotations

import streamlit as st
from utils.constants import PREPAREDNESS_DATA


def render_preparedness_section(current_risk_level: str = "MEDIUM"):
    """
    Render persona-based disaster preparedness recommendations tabs:
    [Students] [Teachers] [Administrators] [General Public]
    """
    st.markdown("### 🛡️ Disaster Preparedness & Action Recommendations")
    st.caption("Tailored emergency response education guidelines powered by SafeGraph AI Framework.")

    # Filter selector
    selected_risk = st.radio(
        "Select Risk Level View:",
        options=["LOW", "MEDIUM", "HIGH"],
        index=["LOW", "MEDIUM", "HIGH"].index(current_risk_level),
        horizontal=True,
        key="prep_risk_selector",
    )

    personas = ["Students", "Teachers", "Administrators", "General Public"]
    tab_student, tab_teacher, tab_admin, tab_public = st.tabs(
        ["🎒 Students", "👩‍🏫 Teachers", "🏛️ School/City Administrators", "🏡 General Public"]
    )

    tabs_map = {
        "Students": tab_student,
        "Teachers": tab_teacher,
        "Administrators": tab_admin,
        "General Public": tab_public,
    }

    checklist_items = []

    for persona in personas:
        tab = tabs_map[persona]
        with tab:
            st.markdown(f"#### Guidelines for **{persona}** ({selected_risk} Risk Level)")
            recommendations = PREPAREDNESS_DATA.get(persona, {}).get(selected_risk, [])

            for idx, rec in enumerate(recommendations, 1):
                st.markdown(f"- **{idx}.** {rec}")
                checklist_items.append(f"[{persona}] {rec}")

            # Safety checklist interactive toggles
            with st.expander(f"📋 {persona} Action Verification Checklist"):
                for idx, rec in enumerate(recommendations, 1):
                    st.checkbox(f"Completed: {rec}", key=f"chk_{persona}_{selected_risk}_{idx}")

    # Exportable checklist download button
    checklist_text = f"SafeGraph AI - Erode Heatwave Preparedness Checklist ({selected_risk} Risk)\n"
    checklist_text += "=" * 60 + "\n\n" + "\n".join(checklist_items)

    st.download_button(
        label="📥 Download Preparedness Action Checklist (.txt)",
        data=checklist_text,
        file_name=f"Erode_Heatwave_Preparedness_Checklist_{selected_risk}.txt",
        mime="text/plain",
        use_container_width=True,
    )
