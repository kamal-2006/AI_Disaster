from __future__ import annotations

import streamlit as st
from services.disaster_assistant import get_disaster_assistant_response


def render_disaster_assistant_section():
    """
    Renders the AI Disaster Assistant UI section on the Streamlit Dashboard.
    Manages session state, suggested quick questions, chat bubbles, and service responses.
    """
    st.markdown("## 🤖 AI Disaster Assistant")
    st.markdown("##### *Ask questions about disasters, preparedness, safety, and emergency response.*")

    # Initialize chat history in session state
    if "disaster_chat_messages" not in st.session_state:
        st.session_state.disaster_chat_messages = [
            {
                "role": "assistant",
                "content": (
                    "👋 **Hello! I am your AI Disaster Safety Assistant.**\n\n"
                    "You can ask me questions about **heatwaves, floods, cyclones, earthquakes, landslides, "
                    "lightning, emergency kits, or school preparedness**. I also provide live Erode heatwave precautions!"
                ),
                "source": "SafeGraph AI Assistant",
            }
        ]

    # Handle quick suggested question clicks
    suggested_query = None

    st.markdown("**Suggested Questions:**")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("🔥 Heatwave Precautions", key="btn_q1", use_container_width=True):
            suggested_query = "What should I do during a heatwave?"
    with col2:
        if st.button("📍 Today in Erode", key="btn_q2", use_container_width=True):
            suggested_query = "What precautions should I take today in Erode?"
    with col3:
        if st.button("🌊 Flood Safety", key="btn_q3", use_container_width=True):
            suggested_query = "What should I do during a flood?"
    with col4:
        if st.button("🎒 Emergency Kit", key="btn_q4", use_container_width=True):
            suggested_query = "What should be in an emergency kit?"

    col5, col6, col7, col8 = st.columns(4)
    with col5:
        if st.button("🌀 Cyclone Prep", key="btn_q5", use_container_width=True):
            suggested_query = "How can I prepare for a cyclone?"
    with col6:
        if st.button("🏚️ Earthquake Safety", key="btn_q6", use_container_width=True):
            suggested_query = "What should I do during an earthquake?"
    with col7:
        if st.button("🏫 School Safety", key="btn_q7", use_container_width=True):
            suggested_query = "How can students stay safe during extreme heat?"
    with col8:
        if st.button("⚡ Lightning Tips", key="btn_q8", use_container_width=True):
            suggested_query = "How can I stay safe during lightning?"

    st.write("")

    # Chat Input Box
    user_input = st.chat_input("Type your disaster or emergency safety question here...")

    # Determine prompt to process
    active_prompt = suggested_query or user_input

    if active_prompt:
        # Append user message
        st.session_state.disaster_chat_messages.append({"role": "user", "content": active_prompt})

        # Process via disaster assistant service
        with st.spinner("Analyzing query & generating safety response..."):
            res = get_disaster_assistant_response(active_prompt, st.session_state.disaster_chat_messages)

        # Append assistant response
        st.session_state.disaster_chat_messages.append({
            "role": "assistant",
            "content": res["response"],
            "source": res.get("source", "SafeGraph AI Service"),
        })

    # Render Chat Container
    chat_container = st.container(height=520, border=True)
    with chat_container:
        for msg in st.session_state.disaster_chat_messages:
            if msg["role"] == "user":
                with st.chat_message("user", avatar="👤"):
                    st.markdown(msg["content"])
            else:
                with st.chat_message("assistant", avatar="🛡️"):
                    st.markdown(msg["content"])
                    if "source" in msg:
                        st.caption(f"🔍 Source: {msg['source']}")

    # Clear Chat History Controls
    col_clear, col_space = st.columns([1, 4])
    with col_clear:
        if st.button("🗑️ Clear Chat History", key="btn_clear_chat", use_container_width=True):
            st.session_state.disaster_chat_messages = [
                {
                    "role": "assistant",
                    "content": "Chat history cleared. How can I assist you with disaster safety today?",
                    "source": "SafeGraph AI Assistant",
                }
            ]
            st.rerun()
