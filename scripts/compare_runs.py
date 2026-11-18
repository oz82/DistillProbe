"""Compare two run directories' metrics.json field by field.

Provenance fields (run id, timestamp, git revision, dependency versions) are
expected to differ and are ignored; every other leaf value must match exactly
(or within --atol when both are floats). Exit code 0 = identical, 1 = differs.

Usage:
    python scripts/compare_runs.py results/sts/<canonical> results/sts/<rerun> [--atol 0]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

IGNORED_PREFIXES = ("manifest.",)


def flatten(d: dict, prefix: str = "") -> dict[str, object]:
    out: dict[str, object] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        elif isinstance(v, list):
            out[key] = json.dumps(v, sort_keys=True, ensure_ascii=False)
        else:
            out[key] = v
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_a")
    ap.add_argument("run_b")
    ap.add_argument("--atol", type=float, default=0.0)
    args = ap.parse_args()

    a = flatten(json.loads((Path(args.run_a) / "metrics.json").read_text(encoding="utf-8")))
    b = flatten(json.loads((Path(args.run_b) / "metrics.json").read_text(encoding="utf-8")))
    keys = sorted((set(a) | set(b)))
    n_diff = 0
    n_cmp = 0
    for k in keys:
        if k.startswith(IGNORED_PREFIXES):
            continue
        n_cmp += 1
        va, vb = a.get(k, "<missing>"), b.get(k, "<missing>")
        same = va == vb
        if not same and isinstance(va, float) and isinstance(vb, float):
            same = (math.isnan(va) and math.isnan(vb)) or math.isclose(
                va, vb, abs_tol=args.atol, rel_tol=0
            )
        if not same:
            n_diff += 1
            print(f"DIFF {k}: {va!r} != {vb!r}")
    print(f"compared {n_cmp} fields, {n_diff} differ")
    return 1 if n_diff else 0


if __name__ == "__main__":
    sys.exit(main())
