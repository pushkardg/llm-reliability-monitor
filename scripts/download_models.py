"""
download_models.py
------------------
Pre-download all model checkpoints needed for the benchmark.
Run once before starting GPU experiments.

Usage:
    python scripts/download_models.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


MODELS = [
    "meta-llama/Meta-Llama-3-8B-Instruct",   # primary agent model
    "mistralai/Mistral-7B-Instruct-v0.3",     # generalizability check
    "all-MiniLM-L6-v2",                        # SentenceTransformer for SCD/RTCI/CTC
    "microsoft/deberta-xlarge-mnli",           # BERTScore baseline
]


def download_hf_model(name: str):
    print(f"  Downloading {name} ...")
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        AutoTokenizer.from_pretrained(name)
        # Only cache the config/tokenizer here; full weights load in run_experiment
        print(f"  ✓ {name} tokenizer cached")
    except Exception as e:
        print(f"  ✗ Failed to download {name}: {e}")


def download_sentence_transformer(name: str):
    print(f"  Downloading SentenceTransformer: {name} ...")
    try:
        from sentence_transformers import SentenceTransformer
        SentenceTransformer(name)
        print(f"  ✓ {name} cached")
    except Exception as e:
        print(f"  ✗ Failed: {e}")


def main():
    print("=== Downloading model checkpoints ===\n")
    for model in MODELS:
        if "/" not in model:
            download_sentence_transformer(model)
        else:
            download_hf_model(model)
    print("\nDone. Run python scripts/smoke_test.py to verify the environment.")


if __name__ == "__main__":
    main()
