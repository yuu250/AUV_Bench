#!/usr/bin/env python3
"""Evaluate Task 2 v8 diagnosis records and emit the formal metrics artifact.

    python -m tasks.task2.evaluate outputs/task2/predictions/diagnosis_MODEL.jsonl \
        --manifest data/task2/manifest.jsonl

The predictions file must cover the eligible manifest cohort exactly: every
record (API errors, parse failures, wrong version decisions) stays in the
fixed denominator. Four headline metrics are exported at the top level; the
full evaluator summary is embedded under ``context.summary``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
from pathlib import Path
from typing import Any, Sequence

from tasks.task2 import coordinate_contract
from tasks.task2.metrics import label_macro_metrics
from tasks.task2.prompts import ALL_DIMS
from tasks.task2.scoring import (
    EVAL_VERSION,
    PROTOCOL,
    SUMMARY_SCHEMA_VERSION,
    aggregate_location_records,
    complete_cohort_records,
    load_manifest_pairs,
    load_records,
    pair_is_eligible,
)

EVALUATOR_VERSION = "1.0.0"
SCHEMA_VERSION = "design-bench-formal-metrics-v2"

METRIC_DEFINITIONS = {
    "degradation_detection_balanced_accuracy": {
        "formal_name": "Degradation Detection Balanced Accuracy",
        "abbreviation": "Det-BAcc",
        "direction": "higher_is_better",
        "role": "main",
    },
    "detection_gated_attribution_macro_j": {
        "formal_name": "Detection-Gated Attribution Macro Youden's J",
        "abbreviation": "Attr-MacroJ",
        "direction": "higher_is_better",
        "role": "main",
    },
    "version_aware_localization_micro_f1_at_0_5": {
        "formal_name": "Version-Aware Localization Micro F1 at IoU 0.5",
        "abbreviation": "Loc-F1@0.5",
        "direction": "higher_is_better",
        "role": "main",
    },
    "exact_chain_success_rate": {
        "formal_name": "Exact Chain Success Rate",
        "abbreviation": "Chain",
        "direction": "higher_is_better",
        "role": "main",
    },
}
_METRIC_RANGES = {
    "degradation_detection_balanced_accuracy": (0.0, 1.0),
    "detection_gated_attribution_macro_j": (-1.0, 1.0),
    "version_aware_localization_micro_f1_at_0_5": (0.0, 1.0),
    "exact_chain_success_rate": (0.0, 1.0),
}

DEFAULT_MANIFEST = Path("data/task2/manifest.jsonl")


class EvaluationError(ValueError):
    pass


def _load_and_check_records(
    prediction_path: Path, cohort_ids: set[str]
) -> tuple[list[dict], str, dict]:
    records = load_records(prediction_path)
    if not records:
        raise EvaluationError(f"{prediction_path} contains no records")
    rows = list(records.values())
    for row in rows:
        if row.get("eval_version") != EVAL_VERSION:
            raise EvaluationError(
                f"record {row.get('sample_id')} has eval_version "
                f"{row.get('eval_version')!r}, expected {EVAL_VERSION!r}")
        if not isinstance(row.get("run_meta"), dict):
            raise EvaluationError(
                f"record {row.get('sample_id')} lacks run_meta")
    keys = ("model", "eval_version", "protocol", "single_call_version")
    first_meta = rows[0]["run_meta"]
    for row in rows[1:]:
        if any(row["run_meta"].get(key) != first_meta.get(key) for key in keys):
            raise EvaluationError(
                "records disagree on model/protocol/version; refusing to mix runs")
    model = first_meta.get("model")
    if not isinstance(model, str) or not model:
        raise EvaluationError("run_meta lacks a model name")
    try:
        ordered = complete_cohort_records(records, cohort_ids)
    except RuntimeError as error:
        raise EvaluationError(str(error)) from error
    return ordered, model, first_meta


def aggregate(rows: list[dict]) -> dict[str, Any]:
    """Fixed-denominator aggregation; identical definitions to the frozen
    evaluator's summary section."""
    n_total = len(rows)
    n_api_err = sum(r.get("status") == "error" for r in rows)
    n_det_parse = sum(r.get("det_parse_fail", False) for r in rows)
    det_acc = sum(r.get("det_correct", False) for r in rows) / n_total
    recalls = []
    for version in (1, 2):
        subset = [r for r in rows if r["degraded_version"] == version]
        if subset:
            recalls.append(sum(r.get("det_pred") == version for r in subset) / len(subset))
    det_bacc = sum(recalls) / len(recalls) if len(recalls) == 2 else None
    det_score = (det_bacc - 0.5) * 200 if det_bacc is not None else None

    correct = [r for r in rows if r.get("det_correct")]
    attr_e2e = label_macro_metrics(rows, ALL_DIMS)
    raw_attr_records = [
        {**r, "attr_pred": list(r.get("attr_pred_raw", []))}
        for r in rows
    ]
    attr_raw_diagnostic = label_macro_metrics(raw_attr_records, ALL_DIMS)
    attr_cond = label_macro_metrics(correct, ALL_DIMS) if correct else None
    attr_exact_e2e = sum(r.get("det_correct") and r.get("attr_exact")
                         for r in rows) / n_total
    attr_exact_cond = (sum(r.get("attr_exact", False) for r in correct) / len(correct)
                       if correct else 0.0)
    n_attr_parse = sum(r.get("attr_parse_fail", False) for r in rows)

    loc50 = aggregate_location_records(rows, "50")
    loc30 = aggregate_location_records(rows, "30")
    loc50_raw_diagnostic = aggregate_location_records(
        rows, "50", prefix="loc_ungated_loc_iou")
    loc30_raw_diagnostic = aggregate_location_records(
        rows, "30", prefix="loc_ungated_loc_iou")
    loc50_cond = aggregate_location_records(correct, "50") if correct else None
    loc_exact = sum(r.get("loc_exact", False) for r in rows) / n_total
    n_loc_parse = sum(r.get("loc_parse_fail", False) for r in rows)
    chain = sum(r.get("chain_success", False) for r in rows) / n_total

    joint_object_valid_rate = sum(
        bool(r.get("joint_object_valid")) for r in rows) / n_total
    joint_schema_valid_rate = sum(
        bool(r.get("joint_schema_valid")) for r in rows) / n_total
    joint_semantic_valid_rate = sum(
        bool(r.get("joint_valid")) for r in rows) / n_total
    attribution_schema_valid_rate = sum(
        bool(r.get("attr_schema_valid")) for r in rows) / n_total
    location_schema_valid_rate = sum(
        bool(r.get("loc_schema_valid")) for r in rows) / n_total
    coordinate_declaration_valid_rate = sum(
        bool(r.get("coordinate_declaration_valid")) for r in rows) / n_total
    location_coordinate_contract_valid_rate = sum(
        bool(r.get("loc_coordinate_contract_valid")) for r in rows) / n_total
    coordinate_local_failure_count = sum(
        bool(r.get("loc_schema_valid"))
        and not bool(r.get("loc_coordinate_contract_valid"))
        for r in rows)
    coordinate_space_counts = dict(sorted({
        value: sum(r.get("coordinate_space") == value for r in rows)
        for value in coordinate_contract.COORDINATE_SPACES
    }.items()))
    box_format_counts = dict(sorted({
        value: sum(r.get("box_format") == value for r in rows)
        for value in coordinate_contract.BOX_FORMATS
    }.items()))

    layers = {}
    for layer in sorted({r["layer"] for r in rows}):
        subset = [r for r in rows if r["layer"] == layer]
        layers[layer] = {
            "n": len(subset),
            "det_acc": round(sum(r.get("det_correct", False) for r in subset) / len(subset), 4),
            "chain_success": round(sum(r.get("chain_success", False) for r in subset) / len(subset), 4),
        }

    return {
        "summary_schema_version": SUMMARY_SCHEMA_VERSION,
        "eval_version": EVAL_VERSION,
        "protocol": PROTOCOL,
        "n": n_total,
        "n_api_or_stage_errors": n_api_err,
        "parse_failures": {
            "detection": n_det_parse,
            "attribution": n_attr_parse,
            "location": n_loc_parse,
        },
        "joint_object_valid_rate": joint_object_valid_rate,
        "joint_schema_valid_rate": joint_schema_valid_rate,
        "joint_semantic_valid_rate": joint_semantic_valid_rate,
        "attribution_schema_valid_rate": attribution_schema_valid_rate,
        "location_schema_valid_rate": location_schema_valid_rate,
        "coordinate_declaration_valid_rate": coordinate_declaration_valid_rate,
        "location_coordinate_contract_valid_rate": (
            location_coordinate_contract_valid_rate),
        "coordinate_local_failure_count": coordinate_local_failure_count,
        "coordinate_space_counts": coordinate_space_counts,
        "box_format_counts": box_format_counts,
        "detection": {
            "accuracy": det_acc,
            "balanced_accuracy": det_bacc,
            "chance_corrected_score": det_score,
            "parse_failures": n_det_parse,
        },
        "attribution_detection_gated": attr_e2e,
        "attribution_e2e": attr_e2e,
        "attribution_raw_diagnostic": attr_raw_diagnostic,
        "attribution_conditional_on_detection": attr_cond,
        "attribution_exact_e2e": attr_exact_e2e,
        "location_version_aware_iou50_e2e": loc50,
        "location_iou50_e2e": loc50,
        "location_version_aware_iou30_diagnostic": loc30,
        "location_raw_iou50_diagnostic": loc50_raw_diagnostic,
        "location_raw_iou30_diagnostic": loc30_raw_diagnostic,
        "location_iou50_conditional_on_detection": loc50_cond,
        "location_exact_e2e": loc_exact,
        "overall_exact": chain,
        "layers": layers,
    }


def _build_artifact(
    metrics: dict[str, Any],
    *,
    context: dict[str, Any],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    if set(metrics) != set(_METRIC_RANGES):
        raise EvaluationError(
            f"metric keys must be {sorted(_METRIC_RANGES)}, got {sorted(metrics)}"
        )
    for metric, value in metrics.items():
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise EvaluationError(f"{metric} must be numeric or null")
        number = float(value)
        if not math.isfinite(number):
            raise EvaluationError(f"{metric} must be finite")
        lower, upper = _METRIC_RANGES[metric]
        if not lower <= number <= upper:
            raise EvaluationError(f"{metric} must be within [{lower}, {upper}]")
    return {
        "schema_version": SCHEMA_VERSION,
        "task": "task2",
        "metrics": dict(metrics),
        "metric_definitions": {
            metric: dict(METRIC_DEFINITIONS[metric]) for metric in _METRIC_RANGES
        },
        "context": dict(context),
        "provenance": dict(provenance),
    }


def _write_artifact(path: Path, artifact: dict[str, Any]) -> None:
    serialized = json.dumps(
        artifact, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
    ) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def build_summary(prediction_path: Path, manifest_path: Path) -> dict[str, Any]:
    pairs = load_manifest_pairs(manifest_path)
    cohort_ids = {pair[0]["sample_id"] for pair in pairs if pair_is_eligible(pair)}
    rows, model, run_meta = _load_and_check_records(prediction_path, cohort_ids)
    summary = aggregate(rows)
    summary["model"] = model
    summary["run_meta"] = run_meta

    metrics = {
        "degradation_detection_balanced_accuracy":
            summary["detection"]["balanced_accuracy"],
        "detection_gated_attribution_macro_j":
            summary["attribution_detection_gated"]["macro_j"],
        "version_aware_localization_micro_f1_at_0_5":
            summary["location_version_aware_iou50_e2e"]["f1"],
        "exact_chain_success_rate": summary["overall_exact"],
    }
    return _build_artifact(
        metrics,
        context={
            "evaluator_version": EVALUATOR_VERSION,
            "model": model,
            "summary": summary,
            "versions": {"python": platform.python_version()},
        },
        provenance={
            "manifest": {"path": str(manifest_path.resolve())},
            "predictions": {"path": str(prediction_path.resolve())},
        },
    )


def default_output_path(prediction_path: Path) -> Path:
    if prediction_path.suffix == ".jsonl":
        return prediction_path.with_suffix(".formal_metrics.json")
    return Path(str(prediction_path) + ".formal_metrics.json")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_path = args.out or default_output_path(args.predictions)
    try:
        artifact = build_summary(args.predictions, args.manifest)
        _write_artifact(output_path, artifact)
    except (EvaluationError, OSError, ValueError, KeyError, RuntimeError) as error:
        print(
            json.dumps(
                {"status": "error", "error": f"{type(error).__name__}: {error}"},
                ensure_ascii=False,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(
        json.dumps(
            {
                "status": "ok",
                "output": str(output_path),
                "metrics": artifact["metrics"],
                "n": artifact["context"]["summary"]["n"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
