#!/usr/bin/env python
"""
Run all data generation scripts in sequence.

Usage:
    python generate_data/run_all.py
    python generate_data/run_all.py --test
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

STEPS = [
    {
        "name": "Full manifold evaluation",
        "script": "evaluate_dpt.py",
        "extra_args": [],
    },
    {
        "name": "Null distribution",
        "script": "generate_null_distribution.py",
        "extra_args": [],
    },
    {
        "name": "Stage permutation test",
        "script": "evaluate_stage_permutations.py",
        "extra_args": [],
    },
    {
        "name": "Generalizability analysis",
        "script": "evaluate_generalizability.py",
        "extra_args": [],
    },
    {
        "name": "Hyperparameter sweep",
        "script": "hyper_param_sweep.py",
        "extra_args": [],
    },
    {
        "name": "Trajectory cell counts",
        "script": "analyze_trajectory_cellcounts.py",
        "extra_args": ["--progression", "CRC-Conventional", "--model", "uni2"],
    },
]


def fmt_elapsed(seconds):
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def main():
    parser = argparse.ArgumentParser(description="Run all data generation scripts.")
    parser.add_argument(
        "--test", action="store_true",
        help="Pass --test to every sub-script (minimal samples, output to results/test/)"
    )
    args = parser.parse_args()

    results = []

    for i, step in enumerate(STEPS, 1):
        label = step["name"]
        script = SCRIPT_DIR / step["script"]
        cmd = [sys.executable, str(script)] + step["extra_args"]
        if args.test:
            cmd.append("--test")

        print(f"\n{'=' * 60}")
        print(f"[{i}/{len(STEPS)}] {label}")
        print(f"  cmd: {' '.join(cmd)}")
        print("=" * 60)

        t0 = time.time()
        proc = subprocess.run(cmd)
        elapsed = time.time() - t0

        ok = proc.returncode == 0
        status = "OK" if ok else f"FAILED (exit {proc.returncode})"
        print(f"\n  -> {status}  ({fmt_elapsed(elapsed)})")
        results.append((label, ok, elapsed))

    # Summary
    print(f"\n{'=' * 60}")
    print("Summary")
    print("=" * 60)
    for label, ok, elapsed in results:
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}]  {label}  ({fmt_elapsed(elapsed)})")

    n_fail = sum(1 for _, ok, _ in results if not ok)
    if n_fail:
        print(f"\n{n_fail} step(s) failed.")
        sys.exit(1)
    else:
        print("\nAll steps completed successfully.")


if __name__ == "__main__":
    main()
