from __future__ import annotations

import sys
from pathlib import Path

# Ensure UTF-8 output on Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from services.disaster_assistant import get_disaster_assistant_response

TEST_QUERIES = [
    "What should I do during a heatwave?",
    "How can students stay safe during extreme heat?",
    "What precautions should I take today in Erode?",
    "What should I do during a flood?",
    "How can I prepare for a cyclone?",
    "What should I do during an earthquake?",
    "What should be in an emergency kit?",
    "Tell me about cricket.",
]

def run_tests():
    print("=" * 80)
    print("RUNNING AI DISASTER ASSISTANT VERIFICATION TESTS")
    print("=" * 80)

    for idx, query in enumerate(TEST_QUERIES, 1):
        print(f"\n[Test {idx}] Query: '{query}'")
        res = get_disaster_assistant_response(query)
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

if __name__ == "__main__":
    run_tests()
