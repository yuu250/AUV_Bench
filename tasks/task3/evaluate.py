#!/usr/bin/env python3
"""Score a Task 3 result JSONL into the frozen Repair Pass summary.

Usage::

    python -m tasks.task3.evaluate outputs/task3/results/repair_<model>.jsonl

The summary is fixed-denominator over every record in the file: the primary
metric is ``repair_pass_rate`` (F ∧ C ∧ P ∧ S per sample), with the four
component rates and the status breakdown alongside.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tasks.task3 import grading
from tasks.task3.grading import Task3Error


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, help="Task 3 result JSONL")
    parser.add_argument("--out", type=Path, default=None,
                        help="write the summary JSON here (default: print only)")
    args = parser.parse_args(argv)

    ids = _record_ids(args.results)
    records_by_id = grading.load_result_jsonl(args.results, selected_ids=ids)
    records = list(records_by_id.values())
    if not records:
        raise Task3Error(f"no records in {args.results}")
    summary = grading.expected_summary(out_path=args.results, records=records)
    if args.out is not None:
        grading.atomic_write_json(args.out, summary)
    print(f"n={summary['n']} status={summary['status_counts']}")
    print(
        f"repair_pass_rate: {summary['repair_pass_count']}/{summary['n']} "
        f"= {summary['repair_pass_rate']:.4f}"
    )
    for component in grading.METRIC_COMPONENTS:
        print(
            f"  {component}: {summary[f'{component}_count']}/{summary['n']} "
            f"= {summary[f'{component}_rate']:.4f}"
        )
    return 0


def _record_ids(path: Path) -> list[str]:
    import json

    ids: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                ids.append(json.loads(line)["sample_id"])
    return ids


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Task3Error as exc:
        print(f"[FATAL] {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
