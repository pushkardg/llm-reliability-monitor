"""
run_canary_rollback.py
----------------------
Canary rollback simulation.

Runs two model instances in parallel (same model, same tasks) simulating
a stable production deployment and a canary deployment with injected drift.

  - Stable: no injection, all traffic
  - Canary: failure injection enabled from turn INJECTION_TURN onward

The simulation uses SDI thresholds from configs/thresholds.yaml to decide
when to fire a rollback signal. Records:
  - Turn at which each SDI crosses the rollback threshold
  - Whether rollback was triggered before vs. after a simulated
    downstream quality alert (which we define as turn 400 post-injection)

Usage:
    python scripts/run_canary_rollback.py \\
        --model meta-llama/Meta-Llama-3-8B-Instruct \\
        --tasks data/tasks.json \\
        --scenarios F1 F2 F3 F4 F5 F6 \\
        --seeds 42 43 44 45 46 \\
        --out results/rollback.csv
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.injection import INJECTION_TURN, get_turn_config, SCENARIOS
from src.metrics import compute_scd, compute_rtci, compute_gas, compute_ctc
from src.repro import seed_everything, save_run_config
from src.tasks import task_to_user_message
from src.injection import SYSTEM_PROMPT_BASELINE

# Downstream quality alert fires at this many post-injection turns
# (simulates a human noticing operational degradation)
DOWNSTREAM_ALERT_TURN = 400

WINDOW_SIZE       = 50
BASELINE_REF_SIZE = 100
MAX_POST_INJECT   = 500


def load_tasks(path):
    with open(path) as f:
        return json.load(f)


def load_thresholds(path="configs/thresholds.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def build_pipeline(model_name):
    from transformers import pipeline
    import torch
    pipe = pipeline(
        "text-generation",
        model=model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    def generate(sp, user_msg):
        messages = [
            {"role": "system", "content": sp},
            {"role": "user",   "content": user_msg},
        ]
        t0 = time.time()
        try:
            result = pipe(messages, max_new_tokens=512, do_sample=False)
            text = result[0]["generated_text"][-1]["content"]
            return text, time.time() - t0, False
        except Exception as e:
            return "", time.time() - t0, True
    return generate


def run_canary_single(scenario, seed, tasks, generate_fn, thresholds, objective):
    seed_everything(seed)
    import random
    rng = random.Random(seed)
    indices = list(range(len(tasks)))
    rng.shuffle(indices)

    stable_outputs  = []
    canary_outputs  = []
    baseline_ref    = []
    session_outputs = []
    thr = thresholds["sdi"]

    rollback_turns = {"scd": None, "rtci": None, "gas": None, "ctc": None}

    def evaluator_fn(prompt):
        out, _, _ = generate_fn(
            "You are an objective evaluator. Follow the scoring format exactly.",
            prompt
        )
        return out

    session_tokens = 0

    for turn in range(INJECTION_TURN + MAX_POST_INJECT):
        task = tasks[indices[turn % len(indices)]]
        user_msg = task_to_user_message(task)

        # Stable arm: never inject
        stable_out, _, _ = generate_fn(SYSTEM_PROMPT_BASELINE, user_msg)
        stable_outputs.append(stable_out)

        # Canary arm: inject after INJECTION_TURN
        sp, extra, _ = get_turn_config(scenario, turn, session_tokens)
        canary_user = user_msg + ("\n" + extra if extra else "")
        canary_out, _, _ = generate_fn(sp, canary_user)
        canary_outputs.append(canary_out)
        session_outputs.append(canary_out)
        session_tokens += len(canary_out.split())

        if turn < INJECTION_TURN:
            if len(baseline_ref) < BASELINE_REF_SIZE:
                baseline_ref.append(stable_out)
            continue

        post_turn = turn - INJECTION_TURN

        if len(canary_outputs) < WINDOW_SIZE:
            continue

        window = canary_outputs[-WINDOW_SIZE:]

        if rollback_turns["scd"] is None:
            scd = compute_scd(baseline_ref, window)
            if scd > thr["scd"]["rollback_above"]:
                rollback_turns["scd"] = post_turn

        if rollback_turns["rtci"] is None:
            rtci = compute_rtci(baseline_ref, window)
            if rtci < thr["rtci"]["rollback_below"]:
                rollback_turns["rtci"] = post_turn

        if rollback_turns["gas"] is None and post_turn % 10 == 0:
            gas = compute_gas(window[-10:], objective, evaluator_fn)
            if gas < thr["gas"]["rollback_below"]:
                rollback_turns["gas"] = post_turn

        if rollback_turns["ctc"] is None:
            ctc = compute_ctc(session_outputs)
            if ctc < thr["ctc"]["rollback_below"]:
                rollback_turns["ctc"] = post_turn

        if all(v is not None for v in rollback_turns.values()):
            break

    # Summarise
    first_rollback = min(
        (v for v in rollback_turns.values() if v is not None),
        default=None
    )
    return {
        "rollback_turns": rollback_turns,
        "first_rollback": first_rollback,
        "before_downstream": (
            first_rollback is not None and first_rollback < DOWNSTREAM_ALERT_TURN
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",     required=True)
    parser.add_argument("--tasks",     required=True)
    parser.add_argument("--scenarios", nargs="+", default=list(SCENARIOS))
    parser.add_argument("--seeds",     nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--out",       required=True)
    parser.add_argument("--thresholds-file", default="configs/thresholds.yaml")
    args = parser.parse_args()

    tasks      = load_tasks(args.tasks)
    thresholds = load_thresholds(args.thresholds_file)
    objective  = tasks[0]["objective"]

    save_run_config(
        args.out,
        model=args.model,
        scenarios=args.scenarios,
        seeds=args.seeds,
        n_tasks=len(tasks),
        thresholds=thresholds,
        extra={"downstream_alert_turn": DOWNSTREAM_ALERT_TURN},
    )

    generate_fn = build_pipeline(args.model)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "scenario", "seed",
        "rollback_scd", "rollback_rtci", "rollback_gas", "rollback_ctc",
        "first_rollback", "before_downstream_alert"
    ]

    with open(out_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for scenario in args.scenarios:
            for seed in args.seeds:
                print(f"\n=== Canary rollback: {scenario}, seed {seed} ===")
                res = run_canary_single(
                    scenario, seed, tasks, generate_fn, thresholds, objective
                )
                rt = res["rollback_turns"]
                row = {
                    "scenario": scenario,
                    "seed": seed,
                    "rollback_scd":  rt["scd"]  if rt["scd"]  is not None else "N/D",
                    "rollback_rtci": rt["rtci"] if rt["rtci"] is not None else "N/D",
                    "rollback_gas":  rt["gas"]  if rt["gas"]  is not None else "N/D",
                    "rollback_ctc":  rt["ctc"]  if rt["ctc"]  is not None else "N/D",
                    "first_rollback": res["first_rollback"] if res["first_rollback"] is not None else "N/D",
                    "before_downstream_alert": res["before_downstream"],
                }
                writer.writerow(row)
                csvfile.flush()
                print(f"  First rollback: {res['first_rollback']}  "
                      f"Before downstream alert: {res['before_downstream']}")

    print(f"\nRollback results saved → {out_path}")


if __name__ == "__main__":
    main()
