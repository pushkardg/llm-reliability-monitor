"""
generate_tasks.py
-----------------
Generate the synthetic task dataset and save to JSON.

Usage:
    python scripts/generate_tasks.py --n-tasks 1000 --seed 0 --out data/tasks.json
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.tasks import generate_tasks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-tasks", type=int, default=1000)
    parser.add_argument("--seed",    type=int, default=0)
    parser.add_argument("--out",     type=str, default="data/tasks.json")
    args = parser.parse_args()

    print(f"Generating {args.n_tasks} tasks (seed={args.seed}) ...")
    tasks = generate_tasks(args.n_tasks, seed=args.seed)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(tasks, f, indent=2)
    print(f"Saved {len(tasks)} tasks → {out_path}")


if __name__ == "__main__":
    main()
