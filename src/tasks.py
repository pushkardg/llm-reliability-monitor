"""
tasks.py
--------
Synthetic industrial predictive-maintenance task generator.

Generates deterministic (seeded) tasks so the same --seed always produces
the same dataset, which matters for comparing detection runs across
models and scenarios.
"""
import json
import random
from pathlib import Path

EQUIPMENT_TYPES = [
    "centrifugal_pump",
    "induction_motor",
    "hydraulic_cylinder",
    "heat_exchanger",
    "reciprocating_compressor",
    "conveyor_belt_drive",
]

MAINTENANCE_HISTORY_TEMPLATES = [
    "No issues found at last inspection.",
    "Bearing replaced {days_ago} days ago.",
    "Seal replaced {days_ago} days ago; minor leakage noted post-repair.",
    "Alignment corrected {days_ago} days ago.",
    "Lubrication interval extended after last service.",
    "Vibration analysis flagged borderline readings {days_ago} days ago, no action taken.",
]

OBJECTIVE = (
    "Analyze the equipment telemetry and maintenance history below. "
    "Provide: (1) an assessment of current equipment condition, "
    "(2) a root-cause analysis of any anomalies, "
    "(3) a recommended maintenance action with priority level, and "
    "(4) reasoning that traces from the telemetry to the recommendation. "
    "Prioritize minimizing unplanned downtime."
)

SYSTEM_PROMPT_BASE = (
    "You are an industrial predictive-maintenance AI assistant. "
    + OBJECTIVE
)


def _make_telemetry(rng: random.Random) -> dict:
    """Generate plausible (not necessarily physically accurate) sensor readings."""
    return {
        "vibration_mm_s": round(rng.uniform(0.5, 12.0), 2),
        "temperature_c": round(rng.uniform(40.0, 95.0), 1),
        "pressure_bar": round(rng.uniform(1.0, 8.0), 2),
        "current_draw_a": round(rng.uniform(5.0, 40.0), 1),
        "runtime_hours": rng.randint(100, 25000),
    }


def _make_history(rng: random.Random) -> str:
    template = rng.choice(MAINTENANCE_HISTORY_TEMPLATES)
    # Guard: only format days_ago if the template actually uses it
    if "{days_ago}" in template:
        max_days = 365
        days_ago = rng.randint(1, max_days)
        return template.format(days_ago=days_ago)
    return template


def generate_tasks(n: int, seed: int = 0) -> list[dict]:
    """
    Return a list of n task dicts, each containing:
      equipment_type, equipment_id, telemetry, maintenance_history,
      objective (fixed string), system_prompt (fixed string)
    """
    rng = random.Random(seed)
    tasks = []
    for i in range(n):
        eq_type = rng.choice(EQUIPMENT_TYPES)
        task = {
            "task_id": i,
            "equipment_type": eq_type,
            "equipment_id": f"{eq_type[:3].upper()}-{rng.randint(1000, 9999)}",
            "telemetry": _make_telemetry(rng),
            "maintenance_history": _make_history(rng),
            "objective": OBJECTIVE,
            "system_prompt": SYSTEM_PROMPT_BASE,
        }
        tasks.append(task)
    return tasks


def task_to_user_message(task: dict) -> str:
    """Format a task dict into the user turn text sent to the model."""
    t = task["telemetry"]
    return (
        f"Equipment: {task['equipment_id']} ({task['equipment_type']})\n"
        f"Telemetry:\n"
        f"  Vibration: {t['vibration_mm_s']} mm/s\n"
        f"  Temperature: {t['temperature_c']} °C\n"
        f"  Pressure: {t['pressure_bar']} bar\n"
        f"  Current draw: {t['current_draw_a']} A\n"
        f"  Runtime: {t['runtime_hours']} hours\n"
        f"Maintenance history: {task['maintenance_history']}\n"
    )


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    tasks = generate_tasks(n, seed=0)
    for t in tasks[:3]:
        print(json.dumps(t, indent=2))
