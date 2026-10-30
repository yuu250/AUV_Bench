#!/usr/bin/env python3
"""Evaluate Task 1 predictions against cleaned human MOS labels.

Metrics: per-dimension Spearman rank correlation (SRCC) and mean absolute
error (MAE), macro-averaged with equal weight over the 8 dimensions. The
denominator is always the full manifest; failed samples are reported in the
coverage counts but never enter the metric computation.

    python -m tasks.task1.evaluate outputs/task1/predictions/scores_MODEL_en.jsonl \
        --gt data/task1/human_gt_cleaned_mos.csv --manifest data/task1/manifest.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import scipy
from scipy.stats import spearmanr

EVALUATOR_VERSION = "1.2.0"
SCHEMA_VERSION = "design-bench-formal-metrics-v2"
DIMENSIONS = (
    "color",
    "typography",
    "graphics",
    "layout",
    "component_consistency",
    "visual_style_consistency",
    "copy_quality",
    "image_text_fit",
)

METRIC_DEFINITIONS = {
    "spearman_rank_correlation": {
        "formal_name": "Spearman Rank Correlation",
        "abbreviation": "SRCC",
        "direction": "higher_is_better",
        "role": "main",
    },
    "mean_absolute_error": {
        "formal_name": "Mean Absolute Error",
        "abbreviation": "MAE",
        "direction": "lower_is_better",
        "role": "main",
    },
}
_METRIC_RANGES = {
    "spearman_rank_correlation": (-1.0, 1.0),
    "mean_absolute_error": (0.0, 4.0),
}

DEFAULT_GT = Path("data/task1/human_gt_cleaned_mos.csv")
DEFAULT_MANIFEST = Path("data/task1/manifest.jsonl")


class EvaluationError(ValueError):
    pass


def _require_object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise EvaluationError(f"{label} must be a JSON object")
    return value


def _parse_score(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvaluationError(f"{label} must be a number")
    score = float(value)
    if not math.isfinite(score):
        raise EvaluationError(f"{label} must be finite")
    if not 1.0 <= score <= 5.0:
        raise EvaluationError(f"{label} must be within [1, 5]")
    return score


def _parse_csv_score(value: str | None, label: str) -> float:
    if value is None or not value.strip():
        raise EvaluationError(f"{label} is missing")
    try:
        score = float(value)
    except ValueError as error:
        raise EvaluationError(f"{label} must be numeric") from error
    if not math.isfinite(score):
        raise EvaluationError(f"{label} must be finite")
    if not 1.0 <= score <= 5.0:
        raise EvaluationError(f"{label} must be within [1, 5]")
    return score


def load_manifest(path: Path) -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = _require_object(
                    json.loads(line), f"manifest line {line_number}"
                )
            except json.JSONDecodeError as error:
                raise EvaluationError(
                    f"manifest line {line_number} is invalid JSON"
                ) from error
            sample_id = record.get("id")
            if not isinstance(sample_id, str) or not sample_id:
                raise EvaluationError(
                    f"manifest line {line_number} has invalid id"
                )
            if sample_id in seen:
                raise EvaluationError(f"manifest contains duplicate id: {sample_id}")
            seen.add(sample_id)
            ids.append(sample_id)
    if not ids:
        raise EvaluationError("manifest is empty")
    return ids


def load_ground_truth(
    path: Path, manifest_ids: Sequence[str]
) -> dict[str, dict[str, float]]:
    expected = set(manifest_ids)
    ground_truth: dict[str, dict[str, float]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"output_folder", *DIMENSIONS}
        missing_columns = sorted(required - set(reader.fieldnames or ()))
        if missing_columns:
            raise EvaluationError(
                "ground-truth CSV is missing columns: " + ", ".join(missing_columns)
            )
        for row_number, row in enumerate(reader, 2):
            output_folder = row.get("output_folder")
            if not isinstance(output_folder, str) or not output_folder.endswith(
                "_output"
            ):
                raise EvaluationError(
                    f"ground-truth row {row_number} has invalid output_folder"
                )
            sample_id = output_folder[: -len("_output")]
            if not sample_id:
                raise EvaluationError(
                    f"ground-truth row {row_number} has empty sample id"
                )
            if sample_id in ground_truth:
                raise EvaluationError(
                    f"ground-truth CSV contains duplicate id: {sample_id}"
                )
            if sample_id not in expected:
                raise EvaluationError(
                    f"ground-truth CSV contains unknown id: {sample_id}"
                )
            ground_truth[sample_id] = {
                dimension: _parse_csv_score(
                    row.get(dimension),
                    f"ground-truth row {row_number} {dimension}",
                )
                for dimension in DIMENSIONS
            }
    missing_ids = [sample_id for sample_id in manifest_ids if sample_id not in ground_truth]
    if missing_ids:
        raise EvaluationError(
            f"ground-truth CSV is missing {len(missing_ids)} manifest ids; "
            f"first: {missing_ids[0]}"
        )
    return ground_truth


def load_predictions(
    path: Path, manifest_ids: Sequence[str]
) -> tuple[dict[str, dict[str, float] | None], str | None, Counter[str]]:
    expected = set(manifest_ids)
    predictions: dict[str, dict[str, float] | None] = {}
    models: set[str] = set()
    errors: Counter[str] = Counter()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = _require_object(
                    json.loads(line), f"prediction line {line_number}"
                )
            except json.JSONDecodeError as error:
                raise EvaluationError(
                    f"prediction line {line_number} is invalid JSON"
                ) from error
            sample_id = record.get("id")
            if not isinstance(sample_id, str) or not sample_id:
                raise EvaluationError(
                    f"prediction line {line_number} has invalid id"
                )
            if sample_id not in expected:
                raise EvaluationError(
                    f"prediction file contains unknown id: {sample_id}"
                )
            if sample_id in predictions:
                raise EvaluationError(
                    f"prediction file contains duplicate id: {sample_id}"
                )
            model = record.get("model")
            if not isinstance(model, str) or not model:
                raise EvaluationError(
                    f"prediction line {line_number} has invalid model"
                )
            models.add(model)
            if len(models) > 1:
                raise EvaluationError("prediction file contains multiple models")

            if "scores" not in record:
                raise EvaluationError(
                    f"prediction line {line_number} is missing scores"
                )
            scores = record["scores"]
            if scores is None:
                error_name = record.get("error")
                if not isinstance(error_name, str) or not error_name:
                    raise EvaluationError(
                        f"prediction line {line_number} has null scores without error"
                    )
                errors[error_name] += 1
                predictions[sample_id] = None
                continue

            score_object = _require_object(
                scores, f"prediction line {line_number} scores"
            )
            parsed: dict[str, float] = {}
            for dimension in DIMENSIONS:
                dimension_object = _require_object(
                    score_object.get(dimension),
                    f"prediction line {line_number} scores.{dimension}",
                )
                if "score" not in dimension_object:
                    raise EvaluationError(
                        f"prediction line {line_number} scores.{dimension} "
                        "is missing score"
                    )
                parsed[dimension] = _parse_score(
                    dimension_object["score"],
                    f"prediction line {line_number} scores.{dimension}.score",
                )
            predictions[sample_id] = parsed

    model = next(iter(models)) if models else None
    return predictions, model, errors


def _dimension_metrics(
    gt_values: Sequence[float], prediction_values: Sequence[float]
) -> dict[str, Any]:
    count = len(gt_values)
    if count != len(prediction_values):
        raise EvaluationError("metric inputs have different lengths")
    mae = (
        float(np.mean(np.abs(np.asarray(prediction_values) - np.asarray(gt_values))))
        if count
        else None
    )
    if count < 2:
        return {
            "n": count,
            "spearman_rank_correlation": 0.0,
            "raw_spearman_rank_correlation": None,
            "mean_absolute_error": mae,
            "status": "insufficient_samples",
        }
    if len(set(gt_values)) == 1:
        status = "constant_ground_truth"
        raw_srcc = None
    elif len(set(prediction_values)) == 1:
        status = "constant_prediction"
        raw_srcc = None
    else:
        statistic = float(spearmanr(gt_values, prediction_values).statistic)
        if math.isfinite(statistic):
            status = "ok"
            raw_srcc = statistic
        else:
            status = "undefined_srcc"
            raw_srcc = None
    return {
        "n": count,
        "spearman_rank_correlation": (
            raw_srcc if raw_srcc is not None else 0.0
        ),
        "raw_spearman_rank_correlation": raw_srcc,
        "mean_absolute_error": mae,
        "status": status,
    }


def _build_artifact(
    metrics: Mapping[str, Any],
    *,
    context: Mapping[str, Any],
    provenance: Mapping[str, Any],
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
        "task": "task1",
        "metrics": dict(metrics),
        "metric_definitions": {
            metric: dict(METRIC_DEFINITIONS[metric]) for metric in _METRIC_RANGES
        },
        "context": dict(context),
        "provenance": dict(provenance),
    }


def _write_artifact(path: Path, artifact: Mapping[str, Any]) -> None:
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


def build_summary(
    prediction_path: Path,
    gt_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    manifest_ids = load_manifest(manifest_path)
    ground_truth = load_ground_truth(gt_path, manifest_ids)
    predictions, model, errors = load_predictions(prediction_path, manifest_ids)
    scored_ids = [
        sample_id
        for sample_id in manifest_ids
        if predictions.get(sample_id) is not None
    ]

    dimension_metrics: dict[str, dict[str, Any]] = {}
    for dimension in DIMENSIONS:
        gt_values = [ground_truth[sample_id][dimension] for sample_id in scored_ids]
        prediction_values = [
            predictions[sample_id][dimension]  # type: ignore[index]
            for sample_id in scored_ids
        ]
        dimension_metrics[dimension] = _dimension_metrics(
            gt_values, prediction_values
        )

    macro_mae_values = [
        result["mean_absolute_error"]
        for result in dimension_metrics.values()
        if result["mean_absolute_error"] is not None
    ]
    formal_count = len(manifest_ids)
    submitted_count = len(predictions)
    scored_count = len(scored_ids)
    metrics = {
        "spearman_rank_correlation": float(
            np.mean(
                [
                    result["spearman_rank_correlation"]
                    for result in dimension_metrics.values()
                ]
            )
        ),
        "mean_absolute_error": (
            float(np.mean(macro_mae_values)) if macro_mae_values else None
        ),
    }
    return _build_artifact(
        metrics,
        context={
            "evaluator_version": EVALUATOR_VERSION,
            "model": model,
            "dimension_metrics": dimension_metrics,
            "coverage": {
                "formal_count": formal_count,
                "submitted_count": submitted_count,
                "scored_count": scored_count,
                "failed_count": submitted_count - scored_count,
                "missing_count": formal_count - submitted_count,
                "scored_fraction": scored_count / formal_count,
            },
            "error_counts": dict(sorted(errors.items())),
            "versions": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "scipy": scipy.__version__,
            },
            "prediction_overall_used": False,
        },
        provenance={
            "manifest": {"path": str(manifest_path.resolve())},
            "ground_truth": {"path": str(gt_path.resolve())},
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
    parser.add_argument("--gt", type=Path, default=DEFAULT_GT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_path = args.out or default_output_path(args.predictions)
    try:
        summary = build_summary(args.predictions, args.gt, args.manifest)
        _write_artifact(output_path, summary)
    except (EvaluationError, OSError) as error:
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
                "metrics": summary["metrics"],
                "coverage": summary["context"]["coverage"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
