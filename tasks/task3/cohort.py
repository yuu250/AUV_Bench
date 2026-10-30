"""Frozen formal cohort for the Task 3 direct chain.

The cohort binds three shipped artifacts:

* ``cohort.json`` — the frozen, ordered list of formal sample ids (660);
* ``unified_gt.jsonl`` + ``metadata.json`` — the authoritative ground truth
  (schema ``design-bench-unified-gt-v3``);
* ``pages/<sample_id>_combo_output/`` — the frozen degraded/baseline HTML.

A Task 2 prediction JSONL may be attached as *advisory* context.  Advisory
rows are diagnostic-only: a missing or invalid row never changes the page,
the editable targets, or the scoring scope.  Following the direct chain
contract, a sample whose advisory row is missing, invalid, or reports
``det_valid=false`` still goes to the model — the five-field handoff falls
back to ``{}`` in the prompt, exactly like a Task 2 "no trusted answer".
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from tasks.task2.coordinate_contract import (
    BOX_FORMATS,
    COORDINATE_SPACES,
    convert_box,
)
from tasks.task3.grading import (
    EXPECTED_FORMAL_COHORT,
    GT_SCHEMA_VERSION,
    RequestInput,
    Task3Error,
    validate_scoring_gt_row,
)

COHORT_SCHEMA_VERSION = "task3-repair-pass-cohort-v1"
ADVISORY_EVAL_VERSION = "task2-v8-single-call-declared-coordinate"
ADVISORY_PROTOCOL = "single-call"
ADVISORY_CANONICAL_REPRESENTATION = "normalized_1000_xyxy_float"

ALLOWED_DIMENSIONS = frozenset({
    "Typography",
    "Color",
    "Spacing",
    "Layout & Reading Flow",
    "Visual Consistency",
})

HANDOFF_FIELDS = frozenset({
    "degraded_version",
    "dimensions",
    "coordinate_space",
    "box_format",
    "locations",
})


class CohortError(Task3Error):
    pass


class _AdvisoryUnusable(ValueError):
    pass


@dataclass
class FormalItem:
    sample_id: str
    layer: str
    selection: dict
    viewport: tuple[int, int]
    dpr: int
    n_lesions: int
    call_eligible: bool
    upstream_failure_reason: str | None
    request: RequestInput | None
    task2_handoff: dict | None
    task2_provenance: dict
    task2_metric_gates: dict
    task3_record: dict
    scoring_gt: Mapping


class _DuplicateJSONKey(ValueError):
    pass


class _NonStandardJSONConstant(ValueError):
    pass


def _fail(sample_id: str, message: str) -> None:
    raise CohortError(f"{sample_id}: {message}")


def load_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise CohortError(
                        f"{path.name} line {line_number} is invalid JSON"
                    ) from exc
                if not isinstance(record, dict):
                    raise CohortError(
                        f"{path.name} line {line_number} must be a JSON object"
                    )
                records.append(record)
    except OSError as exc:
        raise CohortError(f"cannot read {path}") from exc
    return records


def load_cohort_ids(path: Path) -> list[str]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise CohortError(f"cannot read frozen cohort file {path}") from exc
    if not isinstance(value, dict):
        raise CohortError("frozen cohort file must be a JSON object")
    if value.get("schema_version") != COHORT_SCHEMA_VERSION:
        raise CohortError("frozen cohort schema mismatch")
    sample_ids = value.get("sample_ids")
    if (
        not isinstance(sample_ids, list)
        or not sample_ids
        or any(not isinstance(item, str) or not item for item in sample_ids)
    ):
        raise CohortError("frozen cohort sample_ids must be a non-empty string array")
    if len(set(sample_ids)) != len(sample_ids):
        raise CohortError("frozen cohort contains duplicate sample ids")
    if value.get("n_samples") != len(sample_ids):
        raise CohortError("frozen cohort n_samples drifted")
    return list(sample_ids)


def load_gt_index(gt_path: Path, metadata_path: Path) -> dict[str, Mapping]:
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise CohortError(f"cannot read unified GT metadata {metadata_path}") from exc
    if not isinstance(metadata, dict):
        raise CohortError("unified GT metadata must be a JSON object")
    if metadata.get("schema_version") != GT_SCHEMA_VERSION:
        raise CohortError("unified GT metadata schema mismatch")
    rows = load_jsonl(gt_path)
    index: dict[str, Mapping] = {}
    for row in rows:
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise CohortError("unified GT row lacks a valid sample_id")
        if sample_id in index:
            raise CohortError(f"unified GT contains duplicate sample id {sample_id}")
        if row.get("schema_version") != GT_SCHEMA_VERSION:
            _fail(sample_id, "unified GT schema mismatch")
        index[sample_id] = row
    if metadata.get("n_rows") != len(rows):
        raise CohortError("unified GT metadata n_rows drifted")
    return index


def _resolve_input_file(root: Path, relative: str, sample_id: str, label: str) -> tuple[str, str]:
    root_resolved = root.resolve()
    candidate = (root_resolved / relative).resolve()
    if not str(candidate).startswith(str(root_resolved) + "/"):
        _fail(sample_id, f"{label} path escapes the frozen pages root")
    if not candidate.is_file():
        _fail(sample_id, f"missing frozen {label}")
    if candidate.stat().st_size == 0:
        _fail(sample_id, f"empty frozen {label}")
    return relative, str(candidate)


def _page_sources(pages_root: Path, sample_id: str) -> dict:
    page_dir = Path("pages") / f"{sample_id}_combo_output"
    degraded_rel, degraded_abs = _resolve_input_file(
        pages_root, (page_dir / "index.html").as_posix(), sample_id, "degraded HTML"
    )
    clean_rel, clean_abs = _resolve_input_file(
        pages_root, (page_dir / "baseline.html").as_posix(), sample_id, "clean HTML"
    )
    return {
        "degraded": {"html": degraded_rel, "html_path": degraded_abs},
        "clean": {"html": clean_rel, "html_path": clean_abs},
    }


def actual_degraded_version(sample_id: str) -> int:
    return random.Random(sample_id).choice([1, 2])


def _task3_record(gt_row: Mapping, sources: Mapping, sample_id: str) -> dict:
    return {
        "sample_id": sample_id,
        "viewport": [int(gt_row["viewport"][0]), int(gt_row["viewport"][1])],
        "dpr": int(gt_row["dpr"]),
        "layer": gt_row["layer"],
        "n_defects": len(gt_row["defects"]),
        "artifact_paths": {
            "degraded_html": sources["degraded"]["html_path"],
            "clean_html": sources["clean"]["html_path"],
        },
    }


def _zero_metric_gates(n_regions: int) -> dict:
    return {
        "det_correct": False,
        "loc_exact_iou50": False,
        "loc_iou50_tp": 0,
        "loc_iou50_fp": 0,
        "loc_iou50_fn": n_regions,
        "loc_gt_regions": n_regions,
    }


def _extract_task2_metric_gates(
    row: Mapping, *, actual_degraded_version: int
) -> dict:
    """Detach the frozen Task 2 gates for evaluator diagnostics only.

    These fields must never be merged into ``task2_handoff`` or any model
    message.
    """

    det_correct = row.get("det_correct")
    if type(det_correct) is not bool:
        raise _AdvisoryUnusable("det_correct must be a boolean")
    expected_det_correct = bool(
        type(row.get("det_pred")) is int
        and row["det_pred"] in (1, 2)
        and row["det_pred"] == actual_degraded_version
    )
    if det_correct != expected_det_correct:
        raise _AdvisoryUnusable("det_correct disagrees with deterministic pairing")
    counts: dict[str, int] = {}
    for key in ("loc_iou50_tp", "loc_iou50_fp", "loc_iou50_fn", "loc_gt_regions"):
        value = row.get(key)
        if type(value) is not int or value < 0:
            raise _AdvisoryUnusable(f"{key} must be a non-negative integer")
        counts[key] = value
    if counts["loc_iou50_tp"] + counts["loc_iou50_fn"] != counts["loc_gt_regions"]:
        raise _AdvisoryUnusable("IoU50 TP/FN counts disagree with GT region count")
    loc_n_pred = row.get("loc_n_pred")
    if (
        type(loc_n_pred) is not int
        or loc_n_pred < 0
        or counts["loc_iou50_tp"] + counts["loc_iou50_fp"] != loc_n_pred
    ):
        raise _AdvisoryUnusable("IoU50 TP/FP counts disagree with prediction count")
    loc_exact = row.get("loc_exact")
    if type(loc_exact) is not bool:
        raise _AdvisoryUnusable("loc_exact must be a boolean")
    expected_loc_exact = bool(
        counts["loc_iou50_tp"] == counts["loc_gt_regions"]
        and counts["loc_iou50_fp"] == 0
        and counts["loc_iou50_fn"] == 0
    )
    if loc_exact != expected_loc_exact:
        raise _AdvisoryUnusable("loc_exact disagrees with IoU50 counts")
    return {
        "det_correct": det_correct,
        "loc_exact_iou50": loc_exact,
        **counts,
    }


def _strict_json_object(text: object) -> dict:
    if not isinstance(text, str) or not text.strip():
        raise _AdvisoryUnusable("joint_raw is missing or not a string")

    def object_pairs(pairs: list[tuple[str, object]]) -> dict:
        result: dict = {}
        for key, value in pairs:
            if key in result:
                raise _DuplicateJSONKey(key)
            result[key] = value
        return result

    def reject_constant(token: str) -> object:
        raise _NonStandardJSONConstant(token)

    try:
        value = json.loads(
            text,
            object_pairs_hook=object_pairs,
            parse_constant=reject_constant,
        )
    except (
        json.JSONDecodeError,
        _DuplicateJSONKey,
        _NonStandardJSONConstant,
        TypeError,
        OverflowError,
        RecursionError,
    ) as exc:
        raise _AdvisoryUnusable("joint_raw is not strict JSON") from exc
    if not isinstance(value, dict) or set(value) != HANDOFF_FIELDS:
        raise _AdvisoryUnusable("joint_raw must contain exactly the five handoff fields")
    return value


def _unwrap_quoted_token(value: object) -> object:
    if not isinstance(value, str):
        return value
    token = value.strip()
    if len(token) >= 2 and token[0] == token[-1] and token[0] in ("'", '"'):
        token = token[1:-1].strip()
    return token


def _raw_locations_have_shape(value: object) -> bool:
    if not isinstance(value, list):
        return False
    for location in value:
        if not isinstance(location, Mapping) or set(location) != {"slice", "box"}:
            return False
        if type(location["slice"]) is not int:
            return False
        box = location["box"]
        if (
            not isinstance(box, list)
            or len(box) != 4
            or any(type(component) is not int for component in box)
        ):
            return False
    return True


def _validate_raw_locations(value: object) -> list[dict]:
    if not isinstance(value, list):
        raise _AdvisoryUnusable("raw locations must be an array")
    locations: list[dict] = []
    for location in value:
        if not isinstance(location, Mapping) or set(location) != {"slice", "box"}:
            raise _AdvisoryUnusable("raw location has invalid fields")
        slice_number = location["slice"]
        box = location["box"]
        if type(slice_number) is not int:
            raise _AdvisoryUnusable("raw location has invalid slice type")
        if (
            not isinstance(box, list)
            or len(box) != 4
            or any(type(component) is not int for component in box)
        ):
            raise _AdvisoryUnusable("raw location has invalid coordinate types")
        locations.append({"slice": slice_number, "box": list(box)})
    return locations


def _validate_canonical_locations(value: object, *, n_slices: int) -> list[dict]:
    if not isinstance(value, list):
        raise _AdvisoryUnusable("canonical locations must be an array")
    result: list[dict] = []
    for location in value:
        if not isinstance(location, Mapping) or set(location) != {"slice", "box"}:
            raise _AdvisoryUnusable("canonical location has invalid fields")
        slice_number = location["slice"]
        box = location["box"]
        if type(slice_number) is not int or not 1 <= slice_number <= n_slices:
            raise _AdvisoryUnusable("canonical location has invalid slice")
        if (
            not isinstance(box, list)
            or len(box) != 4
            or any(
                isinstance(component, bool)
                or not isinstance(component, (int, float))
                or not 0 <= float(component) <= 1000
                for component in box
            )
        ):
            raise _AdvisoryUnusable("canonical location is out of range")
        values = [float(component) for component in box]
        if values[0] >= values[2] or values[1] >= values[3]:
            raise _AdvisoryUnusable("canonical location is degenerate")
        result.append({"slice": slice_number, "box": values})
    return result


def _locations_close(left: Sequence[Mapping], right: Sequence[Mapping]) -> bool:
    if len(left) != len(right):
        return False
    for actual, expected in zip(left, right):
        if actual.get("slice") != expected.get("slice"):
            return False
        actual_box = actual.get("box")
        expected_box = expected.get("box")
        if not isinstance(actual_box, list) or not isinstance(expected_box, list):
            return False
        if len(actual_box) != 4 or len(expected_box) != 4:
            return False
        if any(
            not math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=1e-9)
            for a, b in zip(actual_box, expected_box)
        ):
            return False
    return True


def _require_bool(row: Mapping, key: str) -> bool:
    value = row.get(key)
    if type(value) is not bool:
        raise _AdvisoryUnusable(f"advisory field {key!r} must be a boolean")
    return value


def _advisory_input_geometry(row: Mapping) -> dict:
    geometry = row.get("input_geometry")
    if not isinstance(geometry, Mapping):
        raise _AdvisoryUnusable("advisory row lacks input_geometry")
    canvas_width = geometry.get("canvas_width")
    common_height = geometry.get("common_height")
    n_slices = geometry.get("n_slices")
    if (
        type(canvas_width) is not int
        or canvas_width < 1
        or type(common_height) is not int
        or common_height < 1
        or type(n_slices) is not int
        or n_slices < 1
    ):
        raise _AdvisoryUnusable("advisory input_geometry contains invalid values")
    if row.get("n_slices") != n_slices:
        raise _AdvisoryUnusable("advisory n_slices disagrees with input_geometry")
    return {
        "canvas_width": canvas_width,
        "common_height": common_height,
        "n_slices": n_slices,
    }


def _extract_advisory(
    row: Mapping,
    sample_id: str,
    n_regions: int,
    *,
    actual_degraded_version: int,
) -> tuple[dict | None, dict, dict]:
    """Return ``(handoff, metric_gates, provenance)`` for one Task 2 row.

    Any protocol violation degrades the sample to "no trusted advisory"; it
    never removes the sample from the Task 3 call set.
    """

    provenance: dict = {
        "source": "task2-open-predictions",
        "eval_version": row.get("eval_version"),
        "protocol": row.get("protocol"),
        "model": row.get("model"),
    }
    try:
        if row.get("eval_version") != ADVISORY_EVAL_VERSION:
            raise _AdvisoryUnusable("advisory eval_version mismatch")
        if row.get("protocol") != ADVISORY_PROTOCOL:
            raise _AdvisoryUnusable("advisory protocol mismatch")
        run_meta = row.get("run_meta")
        if not isinstance(run_meta, Mapping):
            raise _AdvisoryUnusable("advisory row lacks run_meta")
        model = row.get("model")
        if not isinstance(model, str) or not model:
            raise _AdvisoryUnusable("advisory model must be a non-empty string")
        if run_meta.get("model") != model:
            raise _AdvisoryUnusable("advisory model differs from run_meta")
        slice_h = run_meta.get("slice_h")
        if type(slice_h) is not int or slice_h < 1:
            raise _AdvisoryUnusable("advisory run_meta lacks a valid slice_h")
        if row.get("status") not in {"complete", "error"}:
            raise _AdvisoryUnusable("advisory row lacks a terminal status")
        row_version = row.get("degraded_version")
        if type(row_version) is not int or row_version not in (1, 2):
            raise _AdvisoryUnusable("advisory row lacks degraded_version 1 or 2")
        if row_version != actual_degraded_version:
            raise _AdvisoryUnusable("advisory pairing differs from deterministic pairing")
        gates = _extract_task2_metric_gates(
            row, actual_degraded_version=actual_degraded_version
        )
        det_valid = _require_bool(row, "det_valid")
        selected_version = row.get("det_pred")
        if not det_valid:
            if type(selected_version) is int and selected_version in (1, 2):
                raise _AdvisoryUnusable("det_valid=false contradicts det_pred")
            provenance["advisory_status"] = "det_invalid"
            return None, gates, provenance
        if type(selected_version) is not int or selected_version not in (1, 2):
            raise _AdvisoryUnusable("det_valid=true requires det_pred in {1, 2}")
        if row.get("status") != "complete":
            raise _AdvisoryUnusable("valid advisory row must be complete")
        geometry = _advisory_input_geometry(row)
        n_slices = geometry["n_slices"]

        raw = _strict_json_object(row.get("joint_raw"))
        if raw["degraded_version"] != selected_version:
            raise _AdvisoryUnusable("joint_raw degraded_version differs from det_pred")
        dimensions = raw["dimensions"]
        if (
            not isinstance(dimensions, list)
            or len(dimensions) > len(ALLOWED_DIMENSIONS)
            or any(
                not isinstance(value, str) or value not in ALLOWED_DIMENSIONS
                for value in dimensions
            )
        ):
            raise _AdvisoryUnusable("joint_raw dimensions violate the contract")
        coordinate_space = raw["coordinate_space"]
        box_format = raw["box_format"]
        declaration_failed = False
        if coordinate_space not in COORDINATE_SPACES or box_format not in BOX_FORMATS:
            normalized_space = _unwrap_quoted_token(coordinate_space)
            normalized_format = _unwrap_quoted_token(box_format)
            if (
                normalized_space not in COORDINATE_SPACES
                or normalized_format not in BOX_FORMATS
            ):
                raise _AdvisoryUnusable("coordinate declaration is invalid")
            if _require_bool(row, "coordinate_declaration_valid"):
                raise _AdvisoryUnusable(
                    "quote-wrapped declaration contradicts coordinate_declaration_valid"
                )
            if _require_bool(row, "joint_schema_valid"):
                raise _AdvisoryUnusable(
                    "quote-wrapped declaration contradicts joint_schema_valid"
                )
            # Task 2 rejected this quote-wrapped declaration and zeroed its
            # canonical locations.  Mirror the parse-failure semantics: keep
            # the normalized declaration as metadata, trust no locations, and
            # let the sample stay callable for Task 3 with detection and
            # attributes intact.
            declaration_failed = True
            coordinate_space = normalized_space
            box_format = normalized_format
        if declaration_failed or not _raw_locations_have_shape(raw["locations"]):
            # Task 2 scored this answer a location parse failure, recording
            # empty declared/canonical predictions.  Mirror that
            # all-or-nothing semantics: no trusted locations, sample stays
            # callable for Task 3, and the raw text remains in joint_raw.
            raw_locations: list[dict] = []
        else:
            raw_locations = _validate_raw_locations(raw["locations"])

        for key in ("joint_object_valid", "attr_schema_valid", "attr_valid"):
            if not _require_bool(row, key):
                raise _AdvisoryUnusable(f"parsed advisory field {key} is not true")
        declaration_flag = _require_bool(row, "coordinate_declaration_valid")
        if declaration_flag == declaration_failed:
            raise _AdvisoryUnusable("coordinate-declaration flag is inconsistent")
        if row.get("attr_pred_raw_labels") != sorted(set(dimensions)):
            raise _AdvisoryUnusable("joint_raw dimensions differ from parsed labels")
        if declaration_failed:
            if (
                row.get("coordinate_space") is not None
                or row.get("box_format") is not None
            ):
                raise _AdvisoryUnusable(
                    "parsed coordinate fields contradict rejected declaration"
                )
        else:
            if row.get("coordinate_space") != coordinate_space:
                raise _AdvisoryUnusable(
                    "joint_raw coordinate_space differs from parsed field"
                )
            if row.get("box_format") != box_format:
                raise _AdvisoryUnusable("joint_raw box_format differs from parsed field")
        loc_schema_valid = _require_bool(row, "loc_schema_valid")
        if declaration_failed:
            # Task 2 recorded the declared boxes verbatim from joint_raw even
            # though it refused the declaration and withheld the canonical ones.
            expected_declared = raw["locations"] if loc_schema_valid else []
        else:
            expected_declared = raw_locations if loc_schema_valid else []
        if row.get("loc_pred_declared") != expected_declared:
            raise _AdvisoryUnusable("joint_raw locations differ from loc_pred_declared")
        if row.get("loc_pred_raw_representation") != ADVISORY_CANONICAL_REPRESENTATION:
            raise _AdvisoryUnusable("canonical location representation drifted")

        canonical = _validate_canonical_locations(
            row.get("loc_pred_canonical"), n_slices=n_slices
        )
        if row.get("loc_pred_raw") != row.get("loc_pred_canonical"):
            raise _AdvisoryUnusable("canonical location compatibility alias drifted")

        expected_canonical: list[dict] = []
        conversion_valid = loc_schema_valid and not declaration_failed
        for location in raw_locations:
            if not 1 <= location["slice"] <= n_slices:
                conversion_valid = False
                expected_canonical = []
                break
            converted = convert_box(
                location["box"],
                coordinate_space,
                box_format,
                geometry["canvas_width"],
                slice_h,
            )
            if converted is None:
                conversion_valid = False
                expected_canonical = []
                break
            expected_canonical.append({
                "slice": location["slice"],
                "box": [float(value) for value in converted],
            })
        loc_valid = _require_bool(row, "loc_valid")
        if (
            _require_bool(row, "loc_coordinate_contract_valid") != conversion_valid
            or loc_valid != conversion_valid
        ):
            raise _AdvisoryUnusable("coordinate-validity flags are inconsistent")
        if not _locations_close(canonical, expected_canonical):
            raise _AdvisoryUnusable("stored canonical locations differ from conversion")
        joint_valid = _require_bool(row, "joint_valid")
        if joint_valid != bool(det_valid and row["attr_valid"] and loc_valid):
            raise _AdvisoryUnusable("joint_valid flag is inconsistent")

        handoff = {
            "degraded_version": selected_version,
            "dimensions": list(dimensions),
            "coordinate_space": coordinate_space,
            "box_format": box_format,
            "locations": raw_locations,
        }
        if set(handoff) != HANDOFF_FIELDS:
            raise AssertionError("internal handoff field drift")
        provenance["advisory_status"] = "valid"
        return handoff, gates, provenance
    except _AdvisoryUnusable as exc:
        provenance["advisory_status"] = f"unusable: {exc}"
        return None, _zero_metric_gates(n_regions), provenance


def _absent_advisory(n_regions: int) -> tuple[dict | None, dict, dict]:
    provenance = {"source": "none", "advisory_status": "absent"}
    return None, _zero_metric_gates(n_regions), provenance


def build_formal_items(
    *,
    cohort_path: Path,
    gt_path: Path,
    gt_metadata_path: Path,
    pages_root: Path,
    advisory_path: Path | None = None,
) -> tuple[list[FormalItem], dict]:
    sample_ids = load_cohort_ids(cohort_path)
    gt_index = load_gt_index(gt_path, gt_metadata_path)

    advisory_index: dict[str, Mapping] = {}
    if advisory_path is not None:
        for row in load_jsonl(advisory_path):
            row_id = row.get("sample_id")
            if not isinstance(row_id, str) or not row_id:
                raise CohortError("advisory prediction row lacks a valid sample_id")
            if row_id in advisory_index:
                raise CohortError(
                    f"advisory predictions contain duplicate sample id {row_id}"
                )
            advisory_index[row_id] = row

    items: list[FormalItem] = []
    for sample_id in sample_ids:
        gt_row = gt_index.get(sample_id)
        if gt_row is None:
            _fail(sample_id, "missing from the unified GT")
        validate_scoring_gt_row(gt_row, sample_id)
        scoring_gt = gt_row
        defects = scoring_gt["defects"]
        viewport = (int(scoring_gt["viewport"][0]), int(scoring_gt["viewport"][1]))
        dpr = int(scoring_gt["dpr"])
        selected_version = actual_degraded_version(sample_id)
        if (
            type(gt_row.get("degraded_version")) is not int
            or gt_row["degraded_version"] != selected_version
        ):
            _fail(sample_id, "unified GT degraded_version mismatch")
        sources = _page_sources(pages_root, sample_id)
        task3_record = _task3_record(scoring_gt, sources, sample_id)
        selection = {
            "kind": "degraded",
            "is_degraded": True,
            "selected_version": selected_version,
            "source": {
                "html": sources["degraded"]["html"],
                "html_path": sources["degraded"]["html_path"],
            },
        }

        advisory_row = advisory_index.get(sample_id)
        if advisory_row is None:
            handoff, gates, provenance = _absent_advisory(len(defects))
        else:
            handoff, gates, provenance = _extract_advisory(
                advisory_row,
                sample_id,
                len(defects),
                actual_degraded_version=selected_version,
            )

        geometry = None
        if advisory_row is not None and isinstance(
            advisory_row.get("input_geometry"), Mapping
        ):
            raw_geometry = advisory_row["input_geometry"]
            canvas_width = raw_geometry.get("canvas_width")
            common_height = raw_geometry.get("common_height")
            n_slices = raw_geometry.get("n_slices")
            if (
                type(canvas_width) is int
                and type(common_height) is int
                and type(n_slices) is int
            ):
                geometry = {
                    "canvas_width": canvas_width,
                    "common_height": common_height,
                    "n_slices": n_slices,
                }

        request = RequestInput(
            sample_id=sample_id,
            task2_handoff=handoff,
            geometry=geometry,
            selected_version=selected_version,
            selected_source=selection["source"],
            task2_provenance=provenance,
            viewport=viewport,
            dpr=dpr,
        )
        items.append(FormalItem(
            sample_id=sample_id,
            layer=str(scoring_gt["layer"]),
            selection=selection,
            viewport=viewport,
            dpr=dpr,
            n_lesions=len(defects),
            call_eligible=True,
            upstream_failure_reason=None,
            request=request,
            task2_handoff=handoff,
            task2_provenance=provenance,
            task2_metric_gates=gates,
            task3_record=task3_record,
            scoring_gt=scoring_gt,
        ))

    audit = {
        "schema_version": COHORT_SCHEMA_VERSION,
        "n_samples": len(items),
        "formal_expected": EXPECTED_FORMAL_COHORT,
        "is_formal": len(items) == EXPECTED_FORMAL_COHORT,
        "advisory": {
            "path": str(advisory_path) if advisory_path is not None else None,
            "rows": len(advisory_index),
            "valid_handoffs": sum(
                1 for item in items if item.task2_handoff is not None
            ),
        },
    }
    return items, audit


def build_source_items(
    *,
    cohort_path: Path,
    specs_path: Path,
    pages_root: Path,
) -> list[FormalItem]:
    """Prebuild-only items from the cohort and the defect spec projection.

    The reference cache is an input to GT construction, so it cannot be
    built from GT-backed formal items.  The defect spec rows carry every
    environment-independent field the cache build needs (viewport, dpr,
    layer, defect count).  ``scoring_gt`` here is the spec row itself —
    these items must never reach the runtime scoring path.
    """

    from tasks.task3.export_defect_specs import SPEC_SCHEMA_VERSION

    sample_ids = load_cohort_ids(cohort_path)
    spec_index: dict[str, Mapping] = {}
    for row in load_jsonl(specs_path):
        row_id = row.get("sample_id")
        if not isinstance(row_id, str) or not row_id:
            raise CohortError("defect spec row lacks a valid sample_id")
        if row.get("schema_version") != SPEC_SCHEMA_VERSION:
            _fail(row_id, "defect spec schema mismatch")
        if row_id in spec_index:
            _fail(row_id, "duplicate defect spec row")
        spec_index[row_id] = row

    items: list[FormalItem] = []
    for sample_id in sample_ids:
        spec = spec_index.get(sample_id)
        if spec is None:
            _fail(sample_id, "missing from the defect spec projection")
        viewport_raw = spec.get("viewport")
        dpr_raw = spec.get("dpr")
        defects = spec.get("defects")
        if (
            not isinstance(viewport_raw, list)
            or len(viewport_raw) != 2
            or any(type(value) is not int or value <= 0 for value in viewport_raw)
        ):
            _fail(sample_id, "defect spec viewport is invalid")
        if not isinstance(dpr_raw, (int, float)) or not float(dpr_raw) > 0:
            _fail(sample_id, "defect spec dpr is invalid")
        if not float(dpr_raw).is_integer():
            _fail(sample_id, "defect spec dpr must be integral")
        if not isinstance(spec.get("layer"), str) or not spec["layer"]:
            _fail(sample_id, "defect spec layer is invalid")
        if not isinstance(defects, list) or not defects:
            _fail(sample_id, "defect spec must carry at least one defect")
        viewport = (int(viewport_raw[0]), int(viewport_raw[1]))
        dpr = int(dpr_raw)
        selected_version = actual_degraded_version(sample_id)
        sources = _page_sources(pages_root, sample_id)
        selection = {
            "kind": "degraded",
            "is_degraded": True,
            "selected_version": selected_version,
            "source": {
                "html": sources["degraded"]["html"],
                "html_path": sources["degraded"]["html_path"],
            },
        }
        request = RequestInput(
            sample_id=sample_id,
            task2_handoff=None,
            geometry=None,
            selected_version=selected_version,
            selected_source=selection["source"],
            task2_provenance={"source": "none", "advisory_status": "absent"},
            viewport=viewport,
            dpr=dpr,
        )
        items.append(FormalItem(
            sample_id=sample_id,
            layer=str(spec["layer"]),
            selection=selection,
            viewport=viewport,
            dpr=dpr,
            n_lesions=len(defects),
            call_eligible=True,
            upstream_failure_reason=None,
            request=request,
            task2_handoff=None,
            task2_provenance={"source": "none", "advisory_status": "absent"},
            task2_metric_gates=_zero_metric_gates(len(defects)),
            task3_record=_task3_record(spec, sources, sample_id),
            scoring_gt=spec,
        ))
    return items


def select_items(
    items: Sequence[FormalItem],
    *,
    sample_ids: Sequence[str] | None = None,
    limit: int | None = None,
) -> list[FormalItem]:
    selected = list(items)
    if sample_ids:
        wanted = set(sample_ids)
        selected = [item for item in selected if item.sample_id in wanted]
        missing = wanted - {item.sample_id for item in selected}
        if missing:
            raise CohortError(
                f"requested sample ids missing from cohort: {sorted(missing)}"
            )
    if limit is not None:
        if limit < 1:
            raise CohortError("--limit must be positive")
        selected = selected[:limit]
    return selected
