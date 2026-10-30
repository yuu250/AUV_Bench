"""Tests for the Task 1 evaluator on a small hand-computed fixture."""

import csv
import json
from pathlib import Path

import pytest

from tasks.task1.evaluate import DIMENSIONS, EvaluationError, build_summary, main

IDS = ["a", "b", "c"]


def write_manifest(path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for sample_id in IDS:
            handle.write(json.dumps(
                {"id": sample_id, "image": f"{sample_id}_output/x.png"}
            ) + "\n")


def write_gt(path: Path, values: dict[str, float]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["output_folder", *DIMENSIONS])
        for sample_id in IDS:
            if sample_id in values:
                writer.writerow([f"{sample_id}_output"] + [values[sample_id]] * 8)


def scores_dict(value: float) -> dict:
    return {dim: {"score": value, "reason": "x"} for dim in DIMENSIONS}


def write_predictions(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def scored_record(sample_id: str, value: float) -> dict:
    return {
        "id": sample_id,
        "image": f"{sample_id}_output/x.png",
        "model": "test-model",
        "scores": scores_dict(value),
    }


def make_fixture(tmp_path: Path, records: list[dict],
                 gt_values: dict[str, float] | None = None):
    manifest = tmp_path / "manifest.jsonl"
    gt = tmp_path / "gt.csv"
    preds = tmp_path / "preds.jsonl"
    write_manifest(manifest)
    write_gt(gt, gt_values if gt_values is not None else {"a": 1, "b": 2, "c": 3})
    write_predictions(preds, records)
    return manifest, gt, preds


def test_known_metric_values(tmp_path):
    # GT per dim: [1, 2, 3]; predictions: [1, 3, 2]
    # Spearman = 1 - 6*(0+1+1)/(3*8) = 0.5; MAE = (0+1+1)/3 = 2/3
    manifest, gt, preds = make_fixture(tmp_path, [
        scored_record("a", 1), scored_record("b", 3), scored_record("c", 2),
    ])
    summary = build_summary(preds, gt, manifest)
    metrics = summary["metrics"]
    assert metrics["spearman_rank_correlation"] == pytest.approx(0.5)
    assert metrics["mean_absolute_error"] == pytest.approx(2 / 3)
    for dim in DIMENSIONS:
        result = summary["context"]["dimension_metrics"][dim]
        assert result["status"] == "ok"
        assert result["n"] == 3
        assert result["raw_spearman_rank_correlation"] == pytest.approx(0.5)
    coverage = summary["context"]["coverage"]
    assert coverage == {
        "formal_count": 3,
        "submitted_count": 3,
        "scored_count": 3,
        "failed_count": 0,
        "missing_count": 0,
        "scored_fraction": 1.0,
    }
    assert summary["context"]["model"] == "test-model"
    assert summary["context"]["error_counts"] == {}
    assert summary["context"]["prediction_overall_used"] is False
    assert summary["schema_version"] == "design-bench-formal-metrics-v2"
    assert summary["task"] == "task1"


def test_constant_prediction_fills_zero_srcc(tmp_path):
    manifest, gt, preds = make_fixture(tmp_path, [
        scored_record("a", 3), scored_record("b", 3), scored_record("c", 3),
    ])
    summary = build_summary(preds, gt, manifest)
    assert summary["metrics"]["spearman_rank_correlation"] == 0.0
    assert summary["metrics"]["mean_absolute_error"] == pytest.approx(1.0)
    result = summary["context"]["dimension_metrics"]["color"]
    assert result["status"] == "constant_prediction"
    assert result["raw_spearman_rank_correlation"] is None


def test_failed_sample_excluded_from_metrics(tmp_path):
    failed = {
        "id": "c",
        "image": "c_output/x.png",
        "model": "test-model",
        "scores": None,
        "error": "max_retries_exceeded",
    }
    manifest, gt, preds = make_fixture(tmp_path, [
        scored_record("a", 1), scored_record("b", 3), failed,
    ])
    summary = build_summary(preds, gt, manifest)
    # Scored: GT [1, 2] vs pred [1, 3] -> SRCC 1.0, MAE 0.5
    assert summary["metrics"]["spearman_rank_correlation"] == pytest.approx(1.0)
    assert summary["metrics"]["mean_absolute_error"] == pytest.approx(0.5)
    coverage = summary["context"]["coverage"]
    assert coverage["scored_count"] == 2
    assert coverage["failed_count"] == 1
    assert coverage["missing_count"] == 0
    assert coverage["scored_fraction"] == pytest.approx(2 / 3)
    assert summary["context"]["error_counts"] == {"max_retries_exceeded": 1}


def test_duplicate_prediction_id_rejected(tmp_path):
    manifest, gt, preds = make_fixture(tmp_path, [
        scored_record("a", 1), scored_record("a", 2),
    ])
    with pytest.raises(EvaluationError, match="duplicate id"):
        build_summary(preds, gt, manifest)


def test_unknown_prediction_id_rejected(tmp_path):
    manifest, gt, preds = make_fixture(tmp_path, [scored_record("zzz", 1)])
    with pytest.raises(EvaluationError, match="unknown id"):
        build_summary(preds, gt, manifest)


def test_null_scores_without_error_rejected(tmp_path):
    bad = {"id": "a", "image": "a_output/x.png", "model": "test-model",
           "scores": None}
    manifest, gt, preds = make_fixture(tmp_path, [bad])
    with pytest.raises(EvaluationError, match="null scores without error"):
        build_summary(preds, gt, manifest)


def test_missing_ground_truth_id_rejected(tmp_path):
    manifest, gt, preds = make_fixture(tmp_path, [scored_record("a", 1)],
                                       gt_values={"a": 1, "b": 2})
    with pytest.raises(EvaluationError, match="missing 1 manifest ids"):
        build_summary(preds, gt, manifest)


def test_main_writes_artifact(tmp_path):
    manifest, gt, preds = make_fixture(tmp_path, [
        scored_record("a", 1), scored_record("b", 3), scored_record("c", 2),
    ])
    out = tmp_path / "metrics.json"
    assert main([str(preds), "--gt", str(gt), "--manifest", str(manifest),
                 "--out", str(out)]) == 0
    artifact = json.loads(out.read_text(encoding="utf-8"))
    assert artifact["schema_version"] == "design-bench-formal-metrics-v2"
    assert artifact["metrics"]["spearman_rank_correlation"] == pytest.approx(0.5)
