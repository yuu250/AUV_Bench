#!/usr/bin/env python3
"""Formal-cohort loading and fail-closed validation for Task 4.

The released data pack (``data/task4/``) pins the shared 1,395-item dedup
cohort.  All identity checks are direct comparisons (full ID lists, exact
counts, protocol strings) — no digests are computed anywhere.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tasks.task4.prompts import (
    INPUT_PROTOCOL,
    OUTPUT_AUDIT_PROTOCOL,
    RENDER_PROTOCOL,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "task4"
DEFAULT_QUERIES_PATH = DEFAULT_DATA_ROOT / "queries.jsonl"

FORMAL_DENOMINATOR = 1395
RUN_COHORT_SCHEMA = "task4-run-cohort-v1"
EXPECTED_FREEZE_VERSION = "task4-topk-v5-dedup-v1"


class EvaluationError(ValueError):
    pass


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise EvaluationError(f"{path}: expected a JSON object")
    return value


def load_formal_cohort(
    data_root: Path = DEFAULT_DATA_ROOT,
) -> tuple[list[str], dict[str, Any], dict[str, Any]]:
    """Load queries + freeze + cohort and fail closed on any drift."""

    data_root = Path(data_root)
    freeze = read_json(data_root / "freeze.json")
    cohort = read_json(data_root / "cohort.json")
    if freeze.get("status") != "active_default":
        raise EvaluationError("Task 4 freeze is not active_default")
    if freeze.get("freeze_version") != EXPECTED_FREEZE_VERSION:
        raise EvaluationError("unexpected Task 4 freeze version")
    if cohort.get("schema_version") != freeze["cohort"]["schema_version"]:
        raise EvaluationError("cohort schema does not match the Task 4 freeze")
    if cohort.get("cohort") != freeze["cohort"]["name"]:
        raise EvaluationError("cohort name does not match the Task 4 freeze")
    included_ids = [str(sample_id) for sample_id in cohort.get("included_ids", [])]
    expected_count = int(freeze["cohort"]["formal_items"])
    if len(included_ids) != expected_count or len(included_ids) != len(set(included_ids)):
        raise EvaluationError("formal cohort count or uniqueness check failed")
    query_rows = [
        json.loads(line)
        for line in (data_root / "queries.jsonl").read_bytes().splitlines()
        if line.strip()
    ]
    ids = [str(row["sample_id"]) for row in query_rows]
    if len(ids) != expected_count or len(ids) != len(set(ids)) or set(ids) != set(included_ids):
        raise EvaluationError("Task 4 query IDs do not match the formal cohort")
    if freeze["protocols"]["input"] != INPUT_PROTOCOL:
        raise EvaluationError("input protocol drift")
    if freeze["protocols"]["output_audit"] != OUTPUT_AUDIT_PROTOCOL:
        raise EvaluationError("output audit protocol drift")
    if freeze["protocols"]["render"] != RENDER_PROTOCOL:
        raise EvaluationError("render protocol drift")
    return ids, freeze, cohort


def load_query_records(
    queries_path: Path,
    *,
    default_queries_path: Path = DEFAULT_QUERIES_PATH,
    cohort_path: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load generation queries and fail closed if the default cohort drifts.

    Returns the query records plus the run-level ``_cohort.json`` sidecar
    payload.  The sidecar carries the full included-ID list so downstream
    validation compares ID sets directly.
    """

    queries_path = Path(queries_path)
    records = [
        json.loads(line)
        for line in queries_path.read_bytes().splitlines()
        if line.strip()
    ]
    ids = [str(record["sample_id"]) for record in records]
    if len(ids) != len(set(ids)):
        raise EvaluationError(f"duplicate sample_id in query file: {queries_path}")
    if queries_path.resolve() != Path(default_queries_path).resolve():
        return records, {
            "schema_version": RUN_COHORT_SCHEMA,
            "cohort": f"custom:{queries_path.name}",
            "queries_path": str(queries_path.resolve()),
            "sample_count": len(records),
        }
    if cohort_path is None:
        raise EvaluationError("the default queries require a cohort file")
    cohort = read_json(Path(cohort_path))
    included_ids = [str(sample_id) for sample_id in cohort["included_ids"]]
    if len(records) != FORMAL_DENOMINATOR or set(ids) != set(included_ids):
        raise EvaluationError("default Task 4 dedup queries must contain the shared 1,395 IDs")
    return records, {
        "schema_version": RUN_COHORT_SCHEMA,
        "cohort": cohort["cohort"],
        "cohort_schema_version": cohort["schema_version"],
        "cohort_path": str(Path(cohort_path).resolve()),
        "included_ids": included_ids,
        "queries_path": str(queries_path.resolve()),
        "sample_count": len(records),
    }


def validate_run_cohort(
    run_dir: Path, freeze: dict[str, Any], cohort: dict[str, Any]
) -> None:
    """Fail closed when a run directory was produced for a different cohort."""

    sidecar = read_json(Path(run_dir) / "_cohort.json")
    expected = {
        "schema_version": RUN_COHORT_SCHEMA,
        "cohort": cohort["cohort"],
        "cohort_schema_version": cohort["schema_version"],
        "included_ids": [str(sample_id) for sample_id in cohort["included_ids"]],
        "sample_count": int(freeze["cohort"]["formal_items"]),
    }
    mismatched = [key for key, value in expected.items() if sidecar.get(key) != value]
    if mismatched:
        raise EvaluationError(f"run cohort metadata mismatch: {mismatched}")


def select_ids(
    formal_ids: list[str], all_items: bool, sample_ids: list[str]
) -> list[str]:
    if all_items == bool(sample_ids):
        raise EvaluationError("choose exactly one of --all or --sample-id")
    if all_items:
        return list(formal_ids)
    requested = set(sample_ids)
    unknown = sorted(requested - set(formal_ids))
    if unknown:
        raise EvaluationError(f"sample IDs are outside the formal cohort: {unknown}")
    if len(requested) != len(sample_ids):
        raise EvaluationError("duplicate --sample-id values are not allowed")
    return [sample_id for sample_id in formal_ids if sample_id in requested]


__all__ = [
    "DEFAULT_DATA_ROOT",
    "DEFAULT_QUERIES_PATH",
    "EXPECTED_FREEZE_VERSION",
    "FORMAL_DENOMINATOR",
    "REPO_ROOT",
    "RUN_COHORT_SCHEMA",
    "EvaluationError",
    "load_formal_cohort",
    "load_query_records",
    "read_json",
    "select_ids",
    "validate_run_cohort",
]
