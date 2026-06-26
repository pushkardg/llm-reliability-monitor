"""
run_fpr.py
----------
False-positive-rate experiment.

Runs the model on clean (no injection) turns and counts how often each
detector fires a false alert. Uses the same rolling window and thresholds
as run_experiment.py.

Usage:
    python scripts/run_fpr.py \\
        --model meta-llama/Meta-Llama-3-8B-Instruct \\
        --tasks data/tasks.json \\
        --n-windows 100 \\
        --out results/fpr.csv
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.metrics import compute_scd, compute_rtci, compute_gas, compute_ctc
from src.baselines import ADWIN, compute_error_rate, compute_p95_latency, compute_bertscore_f1
from src.repro import seed_everything, save_run_config
from src.tasks import task_to_user_message
from src.injection import SYSTEM_PROMPT_BASELINE

WINDOW_SIZE      = 50
BASELINE_REF_SIZE = 100
SEED             = 42


def load_tasks(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def load_thresholds(path: str = "configs/thresholds.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_pipeline(model_name: str):
    from transformers import pipeline
    import torch
    pipe = pipeline(
        "text-generation",
        model=model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    def generate(prompt: str) -> tuple[str, float, bool]:
        t0 = time.time()
        try:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT_BASELINE},
                {"role": "user",   "content": prompt},
            ]
            result = pipe(messages, max_new_tokens=512, do_sample=False)
            text = result[0]["generated_text"][-1]["content"]
            return text, time.time() - t0, False
        except Exception as e:
            return "", time.time() - t0, True
    return generate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",    required=True)
    parser.add_argument("--tasks",    required=True)
    parser.add_argument("--n-windows", type=int, default=100,
                        help="Number of non-overlapping windows to evaluate")
    parser.add_argument("--out",      required=True)
    parser.add_argument("--thresholds-file", default="configs/thresholds.yaml")
    args = parser.parse_args()

    seed_everything(SEED)
    tasks = load_tasks(args.tasks)
    thresholds = load_thresholds(args.thresholds_file)
    objective = tasks[0]["objective"]
    thr = thresholds["sdi"]

    save_run_config(
        args.out,
        model=args.model,
        scenarios=["FPR_CLEAN_BASELINE"],
        seeds=[SEED],
        n_tasks=len(tasks),
        thresholds=thresholds,
    )

    generate_fn = build_pipeline(args.model)

    def evaluator_fn(prompt: str) -> str:
        out, _, _ = generate_fn(prompt)
        return out

    # Generate baseline reference outputs
    print("Generating baseline reference outputs ...")
    baseline_ref = []
    for i in range(BASELINE_REF_SIZE):
        task = tasks[i % len(tasks)]
        out, _, _ = generate_fn(task_to_user_message(task))
        baseline_ref.append(out)

    # Now run n_windows windows and check for false alerts
    false_alerts = {
        "scd": 0, "rtci": 0, "gas": 0, "ctc": 0,
        "error_rate": 0, "p95_latency": 0, "bertscore": 0,
    }
    per_turn_records = []
    session_outputs = []
    total_turns = args.n_windows * WINDOW_SIZE

    print(f"Running {total_turns} clean turns across {args.n_windows} windows ...")
    all_outputs = []
    for turn in range(total_turns):
        task = tasks[turn % len(tasks)]
        out, latency, error = generate_fn(task_to_user_message(task))
        all_outputs.append(out)
        session_outputs.append(out)
        per_turn_records.append({"error": error, "latency_s": latency})

        if len(all_outputs) < WINDOW_SIZE:
            continue

        # Only check at window boundaries to avoid overcounting
        if (turn + 1) % WINDOW_SIZE != 0:
            continue

        window = all_outputs[-WINDOW_SIZE:]
        window_idx = (turn + 1) // WINDOW_SIZE

        scd  = compute_scd(baseline_ref, window)
        rtci = compute_rtci(baseline_ref, window)
        gas  = compute_gas(window[-10:], objective, evaluator_fn)
        ctc  = compute_ctc(session_outputs)
        er   = compute_error_rate(per_turn_records)
        p95  = compute_p95_latency(per_turn_records)
        bs   = compute_bertscore_f1(window, baseline_ref)

        if scd  > thr["scd"]["alert_above"]:   false_alerts["scd"]       += 1
        if rtci < thr["rtci"]["alert_below"]:   false_alerts["rtci"]      += 1
        if gas  < thr["gas"]["alert_below"]:    false_alerts["gas"]       += 1
        if ctc  < thr["ctc"]["alert_below"]:    false_alerts["ctc"]       += 1
        if er   > thresholds["baselines"]["error_rate"]["alert_above"]:
            false_alerts["error_rate"] += 1
        if p95  > thresholds["baselines"]["p95_latency_s"]["alert_above"]:
            false_alerts["p95_latency"] += 1
        if bs   < thresholds["baselines"]["bertscore_f1"]["alert_below"]:
            false_alerts["bertscore"] += 1

        print(f"  Window {window_idx:3d}: SCD={scd:.3f} RTCI={rtci:.3f} "
              f"GAS={gas:.3f} CTC={ctc:.3f}  false_alerts so far: {false_alerts}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["detector", "false_alerts", "n_windows", "fpr"])
        writer.writeheader()
        for det, count in false_alerts.items():
            writer.writerow({
                "detector": det,
                "false_alerts": count,
                "n_windows": args.n_windows,
                "fpr": round(count / args.n_windows, 4),
            })

    print(f"\n=== False Positive Rates ({args.n_windows} clean windows) ===")
    for det, count in false_alerts.items():
        print(f"  {det:15s}: {count}/{args.n_windows} = {count/args.n_windows:.3f}")
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
