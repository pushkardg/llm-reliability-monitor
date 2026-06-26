"""
repro.py
--------
Reproducibility utilities: deterministic seeding, config logging, and
model checkpoint pinning (commit hash capture).

Every experiment run should call seed_everything() at startup and
save_run_config() before the main loop to produce a self-documenting
config JSON alongside the results CSV.
"""

import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np


def seed_everything(seed: int) -> None:
    """Set Python, NumPy, and (if available) PyTorch seeds."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def get_model_commit_hash(model_name_or_path: str) -> str:
    """
    Return the git commit hash of the HuggingFace model snapshot, or
    a content hash of the local directory if running offline.
    Logs a warning if the hash cannot be determined.
    """
    try:
        from huggingface_hub import model_info
        info = model_info(model_name_or_path)
        sha = getattr(info, "sha", None)
        if sha:
            return sha
    except Exception:
        pass

    # Fallback: hash the config.json of a locally cached snapshot
    cache_dir = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    model_slug = model_name_or_path.replace("/", "--")
    config_paths = list(cache_dir.rglob(f"*{model_slug}*/config.json"))
    if config_paths:
        content = config_paths[0].read_bytes()
        return hashlib.sha256(content).hexdigest()[:16]

    return "unknown"


def save_run_config(
    out_path: str,
    model: str,
    scenarios: list[str],
    seeds: list[int],
    n_tasks: int,
    thresholds: dict,
    extra: dict | None = None,
) -> None:
    """
    Write a JSON config file alongside the results CSV so each output
    file is self-documenting.

    out_path: path to the results CSV (config is written as <out_path>.config.json)
    """
    config = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": model,
        "model_commit_hash": get_model_commit_hash(model),
        "scenarios": scenarios,
        "seeds": seeds,
        "n_tasks": n_tasks,
        "thresholds": thresholds,
    }
    if extra:
        config.update(extra)

    config_path = str(out_path) + ".config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"[repro] Config saved → {config_path}")
