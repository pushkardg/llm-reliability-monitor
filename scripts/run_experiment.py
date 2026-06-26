"""
run_experiment.py
-----------------
Main detection-latency experiment. For each (scenario, seed) pair:

  1. Warm-up: run 200 turns with no injection to build baseline windows.
  2. Injection: enable the failure protocol from turn 200 onward.
  3. Detection: compute SDI and baseline metrics on rolling windows;
     record the first turn at which each detector crosses its threshold.

Writes one CSV row per (scenario, seed, detector) with detection latency
(in turns from injection start) or "N/D" if not detected within 500 turns.

Usage:
    python scripts/run_experiment.py \\
        --model meta-llama/Meta-Llama-3-8B-Instruct \\
        --tasks data/tasks.json \\
        --scenarios F1 F2 F3 F4 F5 F6 \\
        --seeds 42 43 44 45 46 \\
        --out results/llama3_8b.csv
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
from src.metrics import compute_scd, compute_rtci, compute_gas, compute_ctc, GAS_PROMPT
from src.baselines import ADWIN, compute_error_rate, compute_p95_latency, compute_bertscore_f1
from src.repro import seed_everything, save_run_config
from src.tasks import task_to_user_message

WARM_UP_TURNS   = INJECTION_TURN       # 200 turns of clean baseline
MAX_POST_INJECT = 500                  # stop scanning at this many post-injection turns
WINDOW_SIZE     = 50                   # rolling window for metric computation
BASELINE_REF_SIZE = 100               # how many warm-up outputs form the reference


def load_tasks(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def load_thresholds(path: str = "configs/thresholds.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_pipeline(model_name: str):
    """Return a simple generate() callable wrapping the HF pipeline."""
    from transformers import pipeline
    import torch
    pipe = pipeline(
        "text-generation",
        model=model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    def generate(system_prompt: str, user_message: str) -> tuple[str, float, bool]:
        """Returns (output_text, latency_seconds, error)."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message},
        ]
        t0 = time.time()
        try:
            result = pipe(messages, max_new_tokens=512, do_sample=False)
            text = result[0]["generated_text"][-1]["content"]
            return text, time.time() - t0, False
        except Exception as e:
            print(f"[generate] Error: {e}")
            return "", time.time() - t0, True

    return generate


def run_single(
    scenario: str,
    seed: int,
    tasks: list[dict],
    generate_fn,
    thresholds: dict,
    objective: str,
) -> dict:
    """
    Run one (scenario, seed) experiment. Returns a dict of detection latencies.
    """
    seed_everything(seed)
    rng_indices = list(range(len(tasks)))
    import random
    random.Random(seed).shuffle(rng_indices)

    per_turn_records = []   # for error rate and P95 latency baselines
    all_outputs = []        # all output texts so far
    session_outputs = []    # outputs for CTC (within-session context)
    baseline_ref = []       # warm-up outputs used as the SDI reference window
    adwin_scd  = ADWIN(delta=thresholds["baselines"]["adwin"]["delta"])
    adwin_rtci = ADWIN(delta=thresholds["baselines"]["adwin"]["delta"])
    adwin_gas  = ADWIN(delta=thresholds["baselines"]["adwin"]["delta"])

    detection = {
        "scd": None, "rtci": None, "gas": None, "ctc": None,
        "error_rate": None, "p95_latency": None,
        "bertscore": None, "adwin": None,
    }

    # GAS evaluator reuses the same generate_fn pointed at the same model.
    # In a real deployment the evaluator would be a separate model/endpoint.
    def evaluator_fn(prompt: str) -> str:
        out, _, _ = generate_fn(
            "You are an objective evaluator. Follow the scoring format exactly.",
            prompt,
        )
        return out

    session_tokens = 0
    total_turns = WARM_UP_TURNS + MAX_POST_INJECT

    for turn in range(total_turns):
        task = tasks[rng_indices[turn % len(rng_indices)]]
        user_msg = task_to_user_message(task)

        # Determine injection state
        sp, extra_suffix, injecting = get_turn_config(scenario, turn, session_tokens)
        if extra_suffix:
            user_msg = user_msg + "\n" + extra_suffix

        output, latency, error = generate_fn(sp, user_msg)
        session_tokens += len(output.split())

        per_turn_records.append({"error": error, "latency_s": latency})
        all_outputs.append(output)
        session_outputs.append(output)

        # Build baseline reference from warm-up
        if turn < WARM_UP_TURNS:
            if len(baseline_ref) < BASELINE_REF_SIZE:
                baseline_ref.append(output)
            continue  # don't check detectors during warm-up

        post_injection_turn = turn - WARM_UP_TURNS

        # Need a full window before checking
        if len(all_outputs) < WINDOW_SIZE:
            continue

        current_window_outputs = all_outputs[-WINDOW_SIZE:]
        thr = thresholds["sdi"]

        # --- SCD ---
        if detection["scd"] is None:
            scd = compute_scd(baseline_ref, current_window_outputs)
            if scd > thr["scd"]["alert_above"]:
                detection["scd"] = post_injection_turn
            adwin_scd.add(scd)

        # --- RTCI ---
        if detection["rtci"] is None:
            rtci = compute_rtci(baseline_ref, current_window_outputs)
            if rtci < thr["rtci"]["alert_below"]:
                detection["rtci"] = post_injection_turn
            adwin_rtci.add(rtci)

        # --- GAS ---
        if detection["gas"] is None:
            # Only compute GAS every 10 turns to reduce evaluator calls
            if post_injection_turn % 10 == 0:
                gas = compute_gas(current_window_outputs[-10:], objective, evaluator_fn)
                if gas < thr["gas"]["alert_below"]:
                    detection["gas"] = post_injection_turn
                adwin_gas.add(gas)

        # --- CTC ---
        if detection["ctc"] is None:
            ctc = compute_ctc(session_outputs)
            if ctc < thr["ctc"]["alert_below"]:
                detection["ctc"] = post_injection_turn

        # --- Error rate ---
        if detection["error_rate"] is None:
            er = compute_error_rate(per_turn_records)
            if er > thresholds["baselines"]["error_rate"]["alert_above"]:
                detection["error_rate"] = post_injection_turn

        # --- P95 latency ---
        if detection["p95_latency"] is None:
            p95 = compute_p95_latency(per_turn_records)
            if p95 > thresholds["baselines"]["p95_latency_s"]["alert_above"]:
                detection["p95_latency"] = post_injection_turn

        # --- BERTScore (every 20 turns, expensive) ---
        if detection["bertscore"] is None and post_injection_turn % 20 == 0:
            bs = compute_bertscore_f1(current_window_outputs, baseline_ref[:WINDOW_SIZE])
            if bs < thresholds["baselines"]["bertscore_f1"]["alert_below"]:
                detection["bertscore"] = post_injection_turn

        # --- ADWIN (on SCD stream) ---
        if detection["adwin"] is None:
            if adwin_scd._detect():  # fires if SCD stream shows a step-change
                detection["adwin"] = post_injection_turn

        # Early exit if all detectors have fired
        if all(v is not None for v in detection.values()):
            break

    # Replace None with "N/D" for not-detected
    return {k: v if v is not None else "N/D" for k, v in detection.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",     required=True)
    parser.add_argument("--tasks",     required=True)
    parser.add_argument("--scenarios", nargs="+", default=list(SCENARIOS))
    parser.add_argument("--seeds",     nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--out",       required=True)
    parser.add_argument("--thresholds-file", default="configs/thresholds.yaml")
    args = parser.parse_args()

    tasks = load_tasks(args.tasks)
    thresholds = load_thresholds(args.thresholds_file)
    objective = tasks[0]["objective"]

    save_run_config(
        args.out,
        model=args.model,
        scenarios=args.scenarios,
        seeds=args.seeds,
        n_tasks=len(tasks),
        thresholds=thresholds,
    )

    generate_fn = build_pipeline(args.model)

    fieldnames = ["scenario", "seed", "detector", "detection_turn"]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for scenario in args.scenarios:
            for seed in args.seeds:
                print(f"\n=== Scenario {scenario}, seed {seed} ===")
                result = run_single(scenario, seed, tasks, generate_fn, thresholds, objective)
                for detector, latency in result.items():
                    writer.writerow({
                        "scenario": scenario,
                        "seed": seed,
                        "detector": detector,
                        "detection_turn": latency,
                    })
                    print(f"  {detector:15s}: {latency}")
                csvfile.flush()

    print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    main()
