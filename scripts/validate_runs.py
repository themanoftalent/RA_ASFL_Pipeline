"""Audit completed experiment files without creating measurements."""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def validate(root, require_matched=False):
    root = Path(root)
    paths = sorted(root.rglob("results.json"))
    if not paths:
        raise ValueError(f"No experiment results in {root}")
    records = defaultdict(dict)
    errors = []
    for path in paths:
        try:
            item = json.loads(path.read_text(encoding="utf-8"))
            seed = item["seed"]
            method = item["method"]
            metrics = item["test_metrics"]
            if not isinstance(metrics, dict) or not metrics:
                raise ValueError("Test metrics are absent")
            for key, value in metrics.items():
                if isinstance(value, (int, float)) and not math.isfinite(value):
                    raise ValueError(f"Non-finite {key}")
            key = (item["dataset"], seed)
            if method in records[key]:
                raise ValueError(f"Duplicate method {method} for {key}")
            records[key][method] = item
            if not (path.parent / "history.json").is_file():
                raise ValueError("History file missing")
            if not (path.parent / "checkpoint.pt").is_file():
                raise ValueError("Checkpoint missing")
        except (KeyError, ValueError, TypeError, json.JSONDecodeError) as exc:
            errors.append(f"{path}: {exc}")
    if require_matched:
        method_sets = [set(group) for group in records.values()]
        if method_sets and any(s != method_sets[0] for s in method_sets[1:]):
            errors.append("Incomplete method-by-seed experiment matrix")
        for key, group in records.items():
            signatures = {
                (r["train_records"], r["validation_records"], r["test_records"])
                for r in group.values()
            }
            if len(signatures) != 1:
                errors.append(f"Data partition counts differ across methods for {key}")
    summary = {
        "root": str(root), "experiments": len(paths),
        "datasets_and_seeds": len(records),
        "methods": sorted({m for g in records.values() for m in g}),
        "errors": errors,
    }
    print(json.dumps(summary, indent=2))
    if errors:
        raise SystemExit(1)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("--require-matched", action="store_true")
    args = parser.parse_args()
    validate(args.root, args.require_matched)
