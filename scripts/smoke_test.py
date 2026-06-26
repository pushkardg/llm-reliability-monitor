"""
smoke_test.py
-------------
Fast no-GPU sanity check. Run this on any machine (including the GPU
machine you're about to rent) before starting the real benchmark to
catch environment / logic problems without spending GPU-hours.

Usage:
    python scripts/smoke_test.py

All checks should print OK. Any FAILED message means something is broken.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.baselines import ADWIN, compute_error_rate, compute_p95_latency
from src.injection import SCENARIOS, SYSTEM_PROMPT_BASELINE, get_turn_config
from src.metrics import GAS_FALLBACK_SCORE, parse_gas_score
from src.tasks import generate_tasks, task_to_user_message


def check(label: str, condition: bool, detail: str = "") -> bool:
    status = "OK" if condition else "FAILED"
    msg = f"  [{status}] {label}"
    if detail and not condition:
        msg += f"\n         {detail}"
    print(msg)
    return condition


def main():
    all_ok = True

    print("\n=== smoke_test.py ===\n")

    # 1. Task generation is deterministic
    a = generate_tasks(50, seed=42)
    b = generate_tasks(50, seed=42)
    all_ok &= check("Task generation is deterministic", a == b)

    # 2. Task generation with different seeds differs
    c = generate_tasks(50, seed=99)
    all_ok &= check("Different seeds produce different tasks", a != c)

    # 3. task_to_user_message works
    msg = task_to_user_message(a[0])
    all_ok &= check("task_to_user_message produces non-empty string", len(msg) > 10)

    # 4. Injection turn-gating: no injection at turn 0, injection at turn 999
    for scenario_id in SCENARIOS:
        _, _, before = get_turn_config(scenario_id, 0)
        _, _, after  = get_turn_config(scenario_id, 999)
        all_ok &= check(
            f"Injection gating correct for {scenario_id}",
            (not before) and after,
            f"before={before}, after={after}",
        )

    # 5. GAS fallback score is 0.3 (not 0.5)
    all_ok &= check(
        "GAS_FALLBACK_SCORE is 0.3",
        GAS_FALLBACK_SCORE == 0.3,
        f"actual: {GAS_FALLBACK_SCORE}",
    )

    # 6. parse_gas_score parses valid scores
    all_ok &= check("parse_gas_score parses 'Score: 0.8'",   parse_gas_score("Score: 0.8")   == 0.8)
    all_ok &= check("parse_gas_score parses 'score: 0.123'", parse_gas_score("score: 0.123") == 0.123)
    all_ok &= check("parse_gas_score falls back on junk",    parse_gas_score("unparseable")  == 0.3)

    # 7. ADWIN detects an obvious step-change
    adwin = ADWIN()
    fired = any(adwin.add(v) for v in [0.9] * 30 + [0.1] * 30)
    all_ok &= check("ADWIN detects obvious step-change in 60 values", fired)

    # 8. ADWIN does not fire on a flat signal
    adwin2 = ADWIN()
    fired2 = any(adwin2.add(0.8) for _ in range(60))
    all_ok &= check("ADWIN does not fire on flat signal", not fired2)

    # 9. Error rate compute
    records = [{"error": True}] * 10 + [{"error": False}] * 90
    er = compute_error_rate(records, window=100)
    all_ok &= check("Error rate computes correctly (10%)", abs(er - 0.10) < 0.001)

    # 10. P95 latency computes correctly
    records_lat = [{"latency_s": float(i)} for i in range(1, 101)]
    p95 = compute_p95_latency(records_lat, window=100)
    all_ok &= check("P95 latency computes correctly (~95s)", abs(p95 - 95.0) < 1.0)

    print()
    if all_ok:
        print("All checks passed. Safe to proceed to GPU experiments.\n")
    else:
        print("One or more checks FAILED. Fix issues before running GPU experiments.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
