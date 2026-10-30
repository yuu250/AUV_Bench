#!/usr/bin/env python3
"""Project a source unified-GT JSONL into environment-independent defect specs.

The spec keeps only the fields that are properties of the page and the
injection recipe: identities (xpaths, peer groups, evidence anchors), the
rule parameters, and the human-annotation fields.  Everything the scoring
environment can influence — bounding boxes, victim fingerprints, page
context, scan counts — is dropped here and re-derived by ``build_gt.py``
against the shipped reference cache.

Usage::

    python3 -m tasks.task3.export_defect_specs \
        --source-gt /path/to/unified_gt.jsonl \
        --out data/task3/defect_specs.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SPEC_SCHEMA_VERSION = "task3-defect-spec-v1"

ROW_FIELDS = (
    "sample_id",
    "layer",
    "viewport",
    "dpr",
    "target_rule_ids",
    "scoring_rule_ids",
    "safety_gate_rule_ids",
    "safety_report_only_rule_ids",
)

DEFECT_FIELDS = (
    "defect_id",
    "rule_id",
    "dim",
    "prop",
    "base_value",
    "new_value",
    "severity",
    "severity_metric",
    "difficulty",
    "evidence",
    "victim_xpaths_baseline",
    "victim_xpaths_degraded",
    "peer_xpaths",
)


class SpecExportError(RuntimeError):
    pass


def export_specs(source_gt: Path, out_path: Path) -> int:
    rows = []
    with source_gt.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise SpecExportError(f"source GT line {line_number} must be an object")
            defects = row.get("defects")
            if not isinstance(defects, list) or not defects:
                raise SpecExportError(f"source GT line {line_number} lacks defects")
            spec = {key: row[key] for key in ROW_FIELDS if key in row}
            for key in ROW_FIELDS:
                if key not in spec:
                    raise SpecExportError(
                        f"source GT line {line_number} lacks {key!r}"
                    )
            spec_defects = []
            for defect in defects:
                projected = {key: defect[key] for key in DEFECT_FIELDS if key in defect}
                missing = [
                    key
                    for key in ("defect_id", "rule_id", "victim_xpaths_degraded")
                    if key not in projected
                ]
                if missing:
                    raise SpecExportError(
                        f"source GT line {line_number} defect lacks {missing}"
                    )
                spec_defects.append(projected)
            spec["defects"] = spec_defects
            rows.append(spec)
    payload = "".join(
        json.dumps(
            {"schema_version": SPEC_SCHEMA_VERSION, **row},
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n"
        for row in rows
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(payload, encoding="utf-8")
    return len(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-gt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    n_rows = export_specs(args.source_gt, args.out)
    print(f"exported {n_rows} defect spec(s) -> {args.out}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SpecExportError as exc:
        print(f"spec export failed: {exc}", file=sys.stderr)
        sys.exit(1)
