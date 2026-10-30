"""Task 2 protocol constants, record-level scoring, and cohort helpers.

The fixed-denominator policies are part of the frozen protocol:
- attribution is detection-gated: a wrong or invalid version decision scores
  against the empty prediction set;
- localization on a wrong/invalid version treats every valid box as an FP
  and every GT region as an FN;
- malformed stage output never leaves the denominator.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from tasks.task2.geometry import (
    MAX_SLICES,
    SLICE_H,
    build_pair_geometry,
)
from tasks.task2.metrics import (
    clipped_iou_matrix,
    localization_metrics,
    max_visible_slice_numbers,
    merge_gt_regions,
)
from tasks.task2.prompts import ALL_DIMS

EVAL_VERSION = "task2-v8-single-call-declared-coordinate"
PROTOCOL = "single-call"
SINGLE_CALL_VERSION = "v8"
MESSAGE_FORMAT_VERSION = "task2-v8-symmetric-pair-images-declared-coordinate-v1"
SUMMARY_SCHEMA_VERSION = "task2-summary-v8"

LOC_PRIMARY_IOU = 0.5
LOC_SOFT_IOU = 0.3
GT_REGION_MERGE_IOU = 0.9


def attr_item_scores(pred: set[str], gt: set[str]) -> dict:
    """Per-sample exact/F1 is an end-to-end aid only; the headline attribution
    metric is the label-wise macro average computed at aggregation time."""
    tp = len(pred & gt)
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(gt) if gt else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"attr_exact": pred == gt, "attr_item_precision": round(precision, 4),
            "attr_item_recall": round(recall, 4), "attr_item_f1": round(f1, 4)}


def gt_boxes_for(deg: dict) -> list[tuple[float, float, float, float]]:
    """Manifest rule-level bboxes -> deduplicated/merged visual GT regions."""
    regions = []
    for defect in deg["defects"]:
        raw_boxes = defect.get("bboxes", [])
        paths = defect.get("localization_xpaths", defect.get("victim_xpaths", []))
        for index, raw in enumerate(raw_boxes):
            if not isinstance(raw, list) or len(raw) != 4:
                raise ValueError(f"{deg.get('sample_id')}: malformed GT bbox")
            x, y, width, height = (float(v) for v in raw)
            box = (x, y, x + width, y + height)
            # Usually one xpath per bbox; if counts mismatch, keep all paths.
            box_paths = [paths[index]] if len(paths) == len(raw_boxes) else paths
            regions.append((box, box_paths))
    return merge_gt_regions(regions, GT_REGION_MERGE_IOU)


def pair_is_eligible(pair: tuple[dict, dict]) -> bool:
    """Formal full-chain cohort: inputtable length and every localization GT
    exists, is finite, and lies inside the degraded page canvas."""
    deg, clean = pair
    if deg["image_size"][0] != clean["image_size"][0]:
        return False
    if not set(deg.get("gt_dims", [])) or not set(deg["gt_dims"]) <= set(ALL_DIMS):
        return False
    try:
        geometry = build_pair_geometry(deg, clean)
    except (TypeError, ValueError, KeyError):
        return False
    if geometry.n_slices > MAX_SLICES:
        return False
    try:
        boxes = gt_boxes_for(deg)
    except (TypeError, ValueError, OverflowError):
        return False
    if not boxes:
        return False
    width, height = deg["image_size"]
    return all(all(math.isfinite(v) for v in box)
               and 0 <= box[0] < box[2] <= width
               and 0 <= box[1] < box[3] <= height
               for box in boxes)


def _slice_rects(width: int, height: int, offsets: list[int]) -> list[tuple]:
    return [(0.0, float(y), float(width), float(min(y + SLICE_H, height)))
            for y in offsets]


def score_locations(rec: dict, locs: list[tuple[int, list[float]]],
                    deg: dict, offsets) -> None:
    """Slice boxes -> page boxes; GT clipped to each prediction's slice, then
    maximum one-to-one IoU matching under the max-visible-area slice policy."""
    width, height = deg["image_size"]
    offsets = list(offsets)
    predictions = []
    prediction_slices = []
    for slice_no, box in locs:
        y_off = offsets[slice_no - 1]
        pred_box = (box[0] / 1000 * width,
                    y_off + box[1] / 1000 * SLICE_H,
                    box[2] / 1000 * width,
                    y_off + box[3] / 1000 * SLICE_H)
        slice_rect = (0.0, float(y_off), float(width), float(min(y_off + SLICE_H, height)))
        predictions.append((pred_box, slice_rect))
        prediction_slices.append(slice_no)

    gt_boxes = gt_boxes_for(deg)
    scores = clipped_iou_matrix(predictions, gt_boxes)
    allowed_slices = max_visible_slice_numbers(
        gt_boxes, _slice_rects(width, height, offsets))
    for pred_index, slice_no in enumerate(prediction_slices):
        for gt_index, allowed in enumerate(allowed_slices):
            if slice_no not in allowed:
                scores[pred_index][gt_index] = 0.0
    rec["loc_n_pred"] = len(predictions)
    rec["loc_gt"] = len(gt_boxes)
    rec["loc_gt_regions"] = len(gt_boxes)
    for suffix, threshold in (("50", LOC_PRIMARY_IOU), ("30", LOC_SOFT_IOU)):
        metrics = localization_metrics(scores, len(predictions), len(gt_boxes), threshold)
        for key, value in metrics.items():
            rec[f"loc_iou{suffix}_{key}"] = round(value, 4) if isinstance(value, float) else value

    rec["loc_exact"] = (bool(gt_boxes)
                        and rec["loc_iou50_tp"] == len(gt_boxes)
                        and rec["loc_iou50_fp"] == 0)


def score_locations_no_match(rec: dict,
                             locs: list[tuple[int, list[float]]],
                             deg: dict) -> None:
    """Score valid boxes on a wrong/invalid selected Version as FP-only.

    A box attached to the wrong page cannot match a degraded-page GT region.
    Every valid prediction is therefore an FP and every GT region remains an
    FN. Malformed location output is handled by the caller as zero predictions.
    """
    n_pred = len(locs)
    n_gt = len(gt_boxes_for(deg))
    rec["loc_n_pred"] = n_pred
    rec["loc_gt"] = n_gt
    rec["loc_gt_regions"] = n_gt
    for suffix, threshold in (("50", LOC_PRIMARY_IOU), ("30", LOC_SOFT_IOU)):
        metrics = localization_metrics([], n_pred, n_gt, threshold)
        for key, value in metrics.items():
            rec[f"loc_iou{suffix}_{key}"] = (
                round(value, 4) if isinstance(value, float) else value
            )
    rec["loc_exact"] = False


def aggregate_location_records(rows: list[dict], suffix: str,
                               prefix: str = "loc_iou") -> dict:
    """Micro-average frozen per-record localization counts."""
    tp = sum(record.get(f"{prefix}{suffix}_tp", 0) for record in rows)
    fp = sum(record.get(f"{prefix}{suffix}_fp", 0) for record in rows)
    fn = sum(record.get(
        f"{prefix}{suffix}_fn", record.get("loc_gt_regions", 0)
    ) for record in rows)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
    }


def load_manifest_pairs(manifest_path: Path) -> list[tuple[dict, dict]]:
    """Load (degraded, clean) pairs in manifest order; strip provenance-only
    entry fields that are not part of the scientific record."""
    entries = []
    with manifest_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{manifest_path}:{line_number}: invalid JSON") from error
            if not isinstance(entry, dict):
                raise ValueError(f"{manifest_path}:{line_number}: entry must be an object")
            entries.append(entry)
    by_id = {}
    for entry in entries:
        sample_id = entry.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"{manifest_path}: entry missing sample_id")
        if sample_id in by_id:
            raise ValueError(f"{manifest_path}: duplicate sample_id {sample_id}")
        by_id[sample_id] = entry
    pairs = []
    for entry in entries:
        if entry.get("source") != "degraded":
            continue
        clean_id = entry["sample_id"] + "_clean"
        if clean_id not in by_id:
            raise ValueError(f"{manifest_path}: missing paired control {clean_id}")
        pairs.append((entry, by_id[clean_id]))
    return pairs


def load_records(path: Path) -> dict[str, dict]:
    """Load a predictions JSONL keyed by sample_id; duplicates are an error."""
    latest: dict[str, dict] = {}
    if not path.exists():
        return latest
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(
                    f"{path}:{line_number}: invalid JSONL record") from error
            if not isinstance(record, dict):
                raise RuntimeError(f"{path}:{line_number}: record must be an object")
            sample_id = record.get("sample_id")
            if not isinstance(sample_id, str) or not sample_id:
                raise RuntimeError(f"{path}:{line_number}: invalid sample_id")
            if sample_id in latest:
                raise RuntimeError(
                    f"{path}:{line_number}: duplicate sample_id {sample_id}")
            latest[sample_id] = record
    return latest


def complete_cohort_records(latest: dict[str, dict], cohort_ids: set[str]) -> list[dict]:
    """Formal aggregation is only allowed when the output ID set matches the
    planned cohort exactly."""
    actual_ids = set(latest)
    if actual_ids != cohort_ids:
        missing = sorted(cohort_ids - actual_ids)
        extra = sorted(actual_ids - cohort_ids)
        raise RuntimeError(
            f"output records do not match the planned cohort: "
            f"missing {len(missing)}, extra {len(extra)}; "
            f"missing={missing[:5]} extra={extra[:5]}")
    return [latest[sample_id] for sample_id in sorted(cohort_ids)]


__all__ = [
    "ALL_DIMS",
    "EVAL_VERSION",
    "GT_REGION_MERGE_IOU",
    "LOC_PRIMARY_IOU",
    "LOC_SOFT_IOU",
    "MESSAGE_FORMAT_VERSION",
    "PROTOCOL",
    "SINGLE_CALL_VERSION",
    "SLICE_H",
    "SUMMARY_SCHEMA_VERSION",
    "aggregate_location_records",
    "attr_item_scores",
    "complete_cohort_records",
    "gt_boxes_for",
    "load_manifest_pairs",
    "load_records",
    "pair_is_eligible",
    "score_locations",
    "score_locations_no_match",
]
