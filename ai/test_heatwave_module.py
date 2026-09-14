from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from heatwave_prediction import predict_heatwave
from services.disaster_assistant import get_disaster_assistant_response


def run_tests() -> None:
    print("=" * 60)
    print("RUNNING HEATWAVE MODULE VERIFICATION SUITE")
    print("=" * 60)

    tests = [
        ("TEST 1: Today", "today"),
        ("TEST 2: Tomorrow", "tomorrow"),
        ("TEST 3: ISO Date (2026-09-20)", "2026-09-20"),
        ("TEST 4: Text Date (September 20, 2026)", "September 20, 2026"),
        ("TEST 5: Slash Date (20/09/2026)", "20/09/2026"),
        ("TEST 6: Out of Forecast Range Date (2026-12-20)", "2026-12-20"),
        ("TEST 7: Past Date (2026-08-10)", "2026-08-10"),
    ]

    for title, input_val in tests:
        print(f"\n--- {title} ---")
        print(f"Input: {input_val}")
        result = predict_heatwave(input_val)
        print("Structured Result:")
        print(f"  Available: {result.get('available')}")
        print(f"  Date: {result.get('date')}")
        print(f"  Temperature: {result.get('temperature')}")
        print(f"  Risk: {result.get('risk')}")
        print(f"  Prediction: {result.get('prediction')}")
        print(f"  Source: {result.get('source')}")
        if result.get("error"):
            print(f"  Error/Message: {result.get('error')}")

        print("\nFormatted User Response:")
        print(result.get("user_response", result.get("explanation")))

        # Test Disaster Assistant integration
        assistant_res = get_disaster_assistant_response(input_val)
        print("\nDisaster Assistant Integration Output:")
        print(assistant_res.get("response"))

    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED SUCCESSFULLY.")
    print("=" * 60)


if __name__ == "__main__":
    run_tests()
