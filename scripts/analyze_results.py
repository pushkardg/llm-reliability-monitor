"""
analyze_results.py
------------------
Reads the CSVs produced by run_experiment.py, run_fpr.py, and
run_canary_rollback.py and produces:

  - Table II: Detection latency (mean ± std per scenario × detector)
  - Table III: False positive rates
  - Rollback summary

Usage:
    python scripts/analyze_results.py --results-dir results/ --out results/summary.md
"""
import argparse
import csv
import math
import statistics
from pathlib import Path


def load_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def mean_std(values: list) -> str:
    """Format as 'mean ± std', or 'N/D' if all values are N/D."""
    numeric = [int(v) for v in values if str(v) != "N/D"]
    if not numeric:
        return "N/D"
    if len(numeric) == 1:
        return str(numeric[0])
    m = statistics.mean(numeric)
    s = statistics.stdev(numeric)
    return f"{m:.0f}±{s:.0f}"


DETECTORS = ["error_rate", "p95_latency", "bertscore", "adwin", "scd", "rtci", "gas", "ctc"]
SCENARIOS = ["F1", "F2", "F3", "F4", "F5", "F6"]
DETECTOR_LABELS = {
    "error_rate":  "Error Rate",
    "p95_latency": "P95 Latency",
    "bertscore":   "BERTScore",
    "adwin":       "ADWIN",
    "scd":         "SCD (ours)",
    "rtci":        "RTCI (ours)",
    "gas":         "GAS (ours)",
    "ctc":         "CTC (ours)",
}


def build_table_ii(experiment_csvs: list[Path]) -> str:
    """Table II: detection latency mean ± std."""
    rows = []
    for p in experiment_csvs:
        rows.extend(load_csv(p))

    # Group by (scenario, detector)
    data: dict[tuple, list] = {}
    for row in rows:
        key = (row["scenario"], row["detector"])
        data.setdefault(key, []).append(row["detection_turn"])

    lines = ["## Table II — Detection Latency (Agent Turns from Injection, Mean ± Std)\n"]
    header = ["Detector"] + SCENARIOS + ["Mean"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")

    for det in DETECTORS:
        cells = [DETECTOR_LABELS.get(det, det)]
        per_scenario = []
        for sc in SCENARIOS:
            vals = data.get((sc, det), [])
            cell = mean_std(vals) if vals else "—"
            cells.append(cell)
            per_scenario.append([v for v in vals if str(v) != "N/D"])

        # Overall mean (numeric only)
        all_numeric = [v for vs in per_scenario for v in vs]
        if all_numeric:
            cells.append(f"{statistics.mean(all_numeric):.0f}")
        else:
            cells.append("N/D")

        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines)


def build_table_iii(fpr_csv: Path) -> str:
    """Table III: false positive rates."""
    rows = load_csv(fpr_csv)
    lines = ["## Table III — False Positive Rates (Clean Baseline)\n"]
    lines.append("| Detector | False Alerts | N Windows | FPR |")
    lines.append("| --- | --- | --- | --- |")
    for row in rows:
        det = DETECTOR_LABELS.get(row["detector"], row["detector"])
        lines.append(f"| {det} | {row['false_alerts']} | {row['n_windows']} | {row['fpr']} |")
    return "\n".join(lines)


def build_rollback_summary(rollback_csv: Path) -> str:
    rows = load_csv(rollback_csv)
    lines = ["## Canary Rollback Summary\n"]
    lines.append("| Scenario | Seed | First Rollback Turn | Before Downstream Alert? |")
    lines.append("| --- | --- | --- | --- |")
    for row in rows:
        lines.append(
            f"| {row['scenario']} | {row['seed']} "
            f"| {row['first_rollback']} | {row['before_downstream_alert']} |"
        )

    # Aggregate: % of runs where rollback fired before downstream alert
    total = len(rows)
    before = sum(1 for r in rows if str(r["before_downstream_alert"]).lower() == "true")
    lines.append(f"\n**Rollback before downstream alert: {before}/{total} runs "
                 f"({100*before/max(total,1):.0f}%)**")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--out",         required=True)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_path    = Path(args.out)

    # Find experiment CSVs (all CSVs that are not fpr or rollback)
    experiment_csvs = [
        p for p in results_dir.glob("*.csv")
        if "fpr" not in p.name and "rollback" not in p.name and "summary" not in p.name
    ]
    fpr_csvs      = list(results_dir.glob("fpr*.csv"))
    rollback_csvs = list(results_dir.glob("rollback*.csv"))

    sections = ["# LLM Reliability Monitor — Results Summary\n"]

    if experiment_csvs:
        sections.append(build_table_ii(experiment_csvs))
    else:
        sections.append("*(No experiment CSVs found — run run_experiment.py first)*")

    sections.append("")

    if fpr_csvs:
        sections.append(build_table_iii(fpr_csvs[0]))
    else:
        sections.append("*(No FPR CSV found — run run_fpr.py first)*")

    sections.append("")

    if rollback_csvs:
        sections.append(build_rollback_summary(rollback_csvs[0]))
    else:
        sections.append("*(No rollback CSV found — run run_canary_rollback.py first)*")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary = "\n".join(sections)
    out_path.write_text(summary)
    print(summary)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
