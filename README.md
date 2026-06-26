# LLM Reliability Monitor — Failure Injection & Detection Benchmark

Companion code for:

> **"A Kubernetes-Native Monitoring Framework for Drift and Failure Detection
> in Industrial LLM Services"**  
> UEMCON 2026 (empirical paper)

This repo implements four Semantic Drift Indicators (SCD, RTCI, GAS, CTC),
six controlled failure injection protocols (F1–F6), and the baseline
comparison methods (error rate, P95 latency, BERTScore, ADWIN) used to
evaluate them. It also includes the canary rollback simulation and the
false-positive-rate experiment.

> **Note:** For the IECON 2026 companion paper (reliability patterns framework),
> see the separate `iecon-synthetic-drift-study` repo — that one is pure Python,
> no GPU required, and only covers the Section V-D sanity check.

## What this does NOT do

This is a synthetic benchmark. It does not connect to a real Kubernetes
cluster, KServe, Prometheus, or Argo Rollouts — those integrations are
described in the paper but are not required to reproduce the detection
latency / FPR / ablation numbers, which only need a local GPU.

## Requirements

- Python 3.10+
- A GPU with at least 24GB VRAM (Llama-3-8B-Instruct in fp16 needs ~16GB;
  GAS evaluator and embedding models add a few GB)
- A Hugging Face account with access to `meta-llama/Meta-Llama-3-8B-Instruct`
  (accept the license at https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)

## Setup

```bash
git clone <this-repo>
cd llm-reliability-monitor
pip install -r requirements.txt
huggingface-cli login          # paste your HF token
python scripts/download_models.py
```

## Running the full benchmark

```bash
# Step 1: generate the deterministic synthetic task dataset
python scripts/generate_tasks.py --n-tasks 1000 --seed 0 --out data/tasks.json

# Step 2: run all six failure scenarios x five seeds on Llama-3-8B (~48-72 GPU-hrs)
python scripts/run_experiment.py \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --tasks data/tasks.json \
    --scenarios F1 F2 F3 F4 F5 F6 \
    --seeds 42 43 44 45 46 \
    --out results/llama3_8b.csv

# Step 3: generalizability check on Mistral (F1 and F3 only)
python scripts/run_experiment.py \
    --model mistralai/Mistral-7B-Instruct-v0.3 \
    --tasks data/tasks.json \
    --scenarios F1 F3 \
    --seeds 42 43 44 \
    --out results/mistral_7b.csv

# Step 4: false-positive-rate experiment (clean baseline, no injection)
python scripts/run_fpr.py \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --tasks data/tasks.json \
    --n-windows 100 \
    --out results/fpr.csv

# Step 5: canary rollback simulation
python scripts/run_canary_rollback.py \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --tasks data/tasks.json \
    --scenarios F1 F2 F3 F4 F5 F6 \
    --seeds 42 43 44 45 46 \
    --out results/rollback.csv

# Step 6: generate Table II / Table III / ablation summary from CSVs
python scripts/analyze_results.py --results-dir results/ --out results/summary.md
```

## Reduced run (time-constrained)

Two scenarios, three seeds, one model — roughly 16 GPU-hours, honestly
reportable as "preliminary results on two of six failure modes":

```bash
python scripts/run_experiment.py \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --tasks data/tasks.json \
    --scenarios F1 F3 \
    --seeds 42 43 44 \
    --out results/llama3_8b_reduced.csv
```

## Smoke test (no GPU needed)

Run this first to catch logic/environment problems before spending GPU-hours:

```bash
python scripts/smoke_test.py
```

## Reproducibility notes

- All randomness is seeded (`--seed` / `--seeds`); results are deterministic
  given the same seed, model checkpoint, and hardware.
- Model checkpoints are pinned by HF repo string; `src/repro.py` logs the
  exact commit hash at the start of every run — include this in your paper.
- The GAS evaluator prompt is fixed in `src/metrics.py::GAS_PROMPT`. Do not
  change it between runs you intend to compare.
- Every run writes a `<output>.config.json` alongside the CSV with model,
  scenario, seeds, thresholds, and timestamp.

## Repo layout

```
src/
  __init__.py
  metrics.py          # SCD, RTCI, GAS, CTC implementations
  baselines.py        # error rate, P95 latency, BERTScore, ADWIN
  injection.py        # F1-F6 failure injection protocols
  tasks.py            # synthetic task generator
  repro.py            # seeding, config logging, model commit hashes
scripts/
  download_models.py
  generate_tasks.py
  run_experiment.py   # main detection-latency loop
  run_fpr.py          # false-positive-rate experiment
  run_canary_rollback.py
  analyze_results.py  # produces Table II/III/ablation from CSVs
  smoke_test.py       # fast no-GPU sanity check
configs/
  thresholds.yaml     # alert/rollback thresholds for each SDI + baseline
data/                 # generated task JSON lands here
results/              # CSV outputs land here
```