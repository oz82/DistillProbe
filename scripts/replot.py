"""Regenerate plots for an existing run directory.

Usage:
    python scripts/replot.py results/morphology/<run_id>/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from distillprobe.modules import REGISTRY


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", help="path to a run directory containing metrics.json")
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        print(f"missing: {metrics_path}", file=sys.stderr)
        return 2

    result = json.loads(metrics_path.read_text(encoding="utf-8"))
    module_name = result.get("module")
    if module_name not in REGISTRY:
        print(f"unknown module in metrics.json: {module_name!r}", file=sys.stderr)
        return 2

    module = REGISTRY[module_name]()
    out = module.plot(result, run_dir / "plots")
    for path in out:
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
