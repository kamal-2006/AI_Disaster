from __future__ import annotations

import sys
import os
from pathlib import Path

# Ensure UTF-8 output on Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from services.disaster_assistant import get_disaster_assistant_response

os.environ.setdefault("SAFEGRAPH_AI_DEBUG", "1")

TEST_QUERIES = [
    "What are the symptoms of heat exhaustion?",
    "What should I do if flood water enters my house?",
    "Why do earthquakes happen?",
    "How can I prepare my home for a cyclone?",
    "What are the warning signs of a possible landslide?",
    "How can communities prepare for drought?",
    "What should I do during a thunderstorm?",
    "What should I do after receiving a lightning warning?",
    "What should I do after receiving a tsunami warning?",
    "What should be in an emergency kit?",
    "Which symptom is the most serious?",
    "Tell me about cricket.",
]

EXPECTED_CATEGORIES = {
    "What are the symptoms of heat exhaustion?": "heatwave",
    "What should I do if flood water enters my house?": "flood",
    "Why do earthquakes happen?": "earthquake",
    "How can I prepare my home for a cyclone?": "cyclone",
    "What are the warning signs of a possible landslide?": "landslide",
    "How can communities prepare for drought?": "drought",
    "What should I do during a thunderstorm?": "thunderstorm",
    "What should I do after receiving a lightning warning?": "lightning",
    "What should I do after receiving a tsunami warning?": "tsunami",
    "What should be in an emergency kit?": "emergency_kit",
}

PIPELINE_EXPECTATIONS = {
    "I am in a forest fire area, what should I do?": {"category": "forest_fire", "intent": "immediate_action", "graph_nodes": 0},
    "Why do earthquakes happen?": {"category": "earthquake", "intent": "cause"},
    "What should I do during a flood?": {"category": "flood", "intent": "immediate_action"},
    "What are the warning signs of heat exhaustion?": {"category": "heatwave", "intent": "symptoms"},
    "How can I prepare my house before a cyclone?": {"category": "cyclone", "intent": "preparedness"},
    "Why should I avoid going outside during extreme heat?": {"category": "heatwave", "intent": "prevention"},
    "What should I do if a landslide is happening near me?": {"category": "landslide", "intent": "immediate_action", "urgency": "active_emergency"},
    "How can I prepare for a tsunami?": {"category": "tsunami", "intent": "preparedness"},
}

WEATHER_EXPECTATIONS = {
    "What is the current temperature in Erode?": ("today", "current_weather"),
    "Is there a heatwave in Erode today?": ("today", "current_heatwave_risk"),
    "What will the temperature be in Erode tomorrow?": ("future", "future_weather"),
    "Will there be a heatwave in Erode tomorrow?": ("future", "future_heatwave_risk"),
    "What will the temperature be in Erode on September 20, 2026?": ("future", "future_weather"),
    "Will there be a heatwave in Erode on September 20, 2026?": ("future", "future_heatwave_risk"),
    "What is the heatwave risk for Erode on September 25, 2026?": ("future", "future_heatwave_risk"),
    "What was the temperature in Erode on January 10, 2026?": ("past", "historical_weather"),
}

def run_tests():
    print("=" * 80)
    print("RUNNING AI DISASTER ASSISTANT VERIFICATION TESTS")
    print("=" * 80)

    for idx, query in enumerate(TEST_QUERIES, 1):
        print(f"\n[Test {idx}] Query: '{query}'")
        res = get_disaster_assistant_response(query)
        expected_category = EXPECTED_CATEGORIES.get(query)
        if expected_category:
            assert res["category"] == expected_category, (query, res["category"])
            assert "couldn't find enough specific information" not in res["response"]
        print(f"Is Disaster Query: {res['is_disaster']}")
        print(f"Category: {res.get('category')}")
        print(f"Source: {res.get('source')}")
        if 'live_context' in res and res['live_context'].get('available'):
            ctx = res['live_context']
            print(f"Live Erode Context: Risk={ctx.get('risk_level')}, MaxTemp={ctx.get('max_temperature')}°C, Prob={ctx.get('probability')}")
        print("-" * 40)
        print("Response Snippet:")
        response_preview = res['response'][:250].replace('\n', ' ')
        print(f"{response_preview}...")
        print("=" * 80)

    follow_up_history = [
        {"role": "user", "content": "What are the symptoms of heat exhaustion?"},
        {"role": "assistant", "content": "Heavy sweating, weakness, dizziness, and nausea are warning signs."},
        {"role": "user", "content": "Which symptom is the most serious?"},
    ]
    follow_up = get_disaster_assistant_response("Which symptom is the most serious?", follow_up_history)
    assert follow_up["category"] == "heatwave"
    assert "Heatstroke" in follow_up["response"]
    print("Follow-up context: passed")

    for query, expected in PIPELINE_EXPECTATIONS.items():
        result = get_disaster_assistant_response(query)
        assert result["category"] == expected["category"], (query, result["category"])
        assert result["understanding"]["intent"] == expected["intent"], (query, result["understanding"])
        if "urgency" in expected:
            assert result["understanding"]["urgency"] == expected["urgency"]
        assert "debug" in result
        assert len(result["debug"]["retrieved_graph_nodes"]) == expected.get("graph_nodes", len(result["debug"]["retrieved_graph_nodes"]))
    print("Eight-question Knowledge Graph pipeline: passed")

    for query, (expected_date_type, expected_intent) in WEATHER_EXPECTATIONS.items():
        result = get_disaster_assistant_response(query)
        assert result["understanding"]["date_type"] == expected_date_type
        assert result["understanding"]["weather_intent"] == expected_intent
        assert result["weather_result"] is not None
        assert result["source"]
    print("Current, future, and historical weather routing: passed")

if __name__ == "__main__":
    run_tests()
