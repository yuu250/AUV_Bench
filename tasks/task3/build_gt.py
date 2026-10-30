#!/usr/bin/env python3
"""Re-derive the Task 3 scoring GT (schema v3) from defect specs + the cache.

The shipped defect specs carry only environment-independent identities and
rule parameters.  This tool resolves every anchor against the frozen
reference-cache element indexes and recomputes everything the render
environment influences: victim bounding boxes (CSS and physical px), victim
fingerprints, the baseline page context (peer groups, image natural sizes),
and the hybrid scan counts.

Every rebuilt defect is checked with the exact scorer the benchmark runs:
the baseline page must be clean and the degraded page must violate at the
injected victim.  A defect that does not reproduce is dropped from the GT and
recorded in ``metadata.dropped_defects`` (a borderline geometry-dependent
injection can vanish under the frozen environment); a sample left with zero
defects aborts the build, as does any other failure — a partial GT is never
written.

Must run after ``prebuild.py``: the reference cache is the source of truth
for the rendered element indexes.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Mapping

from tasks.task3 import cohort as cohort_mod
from tasks.task3 import grading
from tasks.task3 import reference_cache
from tasks.task3.grading import GT_SCHEMA_VERSION, Task3Error
from tasks.task3.lesion_scorer import (
    GROUP_RULES,
    build_hybrid_ei,
    grade_lesions,
    scan_seed,
)
from tasks.task3.export_defect_specs import SPEC_SCHEMA_VERSION

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent

DEFAULT_SPECS = REPO_ROOT / "data" / "task3" / "defect_specs.jsonl"
DEFAULT_COHORT = REPO_ROOT / "data" / "task3" / "cohort.json"
DEFAULT_GT_OUT = REPO_ROOT / "data" / "task3" / "unified_gt.jsonl"
DEFAULT_METADATA_OUT = REPO_ROOT / "data" / "task3" / "metadata.json"


class GTBuildError(Task3Error):
    pass


def _fail(sample_id: str, message: str) -> None:
    raise GTBuildError(f"{sample_id}: {message}")


def load_specs(path: Path) -> dict[str, dict]:
    specs: dict[str, dict] = {}
    for row in cohort_mod.load_jsonl(path):
        if row.get("schema_version") != SPEC_SCHEMA_VERSION:
            raise GTBuildError("defect spec schema mismatch")
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise GTBuildError("defect spec row lacks a valid sample_id")
        if sample_id in specs:
            raise GTBuildError(f"duplicate defect spec for {sample_id}")
        specs[sample_id] = row
    if not specs:
        raise GTBuildError("defect spec file is empty")
    return specs


def _elements_by_xpath(element_index: Mapping, sample_id: str) -> dict[str, Mapping]:
    elements = element_index.get("elements")
    if not isinstance(elements, list):
        _fail(sample_id, "reference element index lacks elements")
    by_xpath: dict[str, Mapping] = {}
    for element in elements:
        if not isinstance(element, Mapping) or not isinstance(element.get("xpath"), str):
            _fail(sample_id, "reference element index contains an invalid element")
        by_xpath[element["xpath"]] = element
    return by_xpath


def _bboxes(by_xpath: Mapping[str, Mapping], xpaths: object, sample_id: str, label: str) -> list[list[float]]:
    if not isinstance(xpaths, list):
        _fail(sample_id, f"{label} must be an xpath array")
    boxes: list[list[float]] = []
    for xpath in xpaths:
        element = by_xpath.get(xpath)
        if element is None:
            _fail(sample_id, f"{label} xpath missing from render: {xpath}")
        bbox = element.get("bbox")
        if (
            not isinstance(bbox, list)
            or len(bbox) != 4
            or any(not isinstance(value, (int, float)) for value in bbox)
        ):
            _fail(sample_id, f"{label} element lacks a valid bbox: {xpath}")
        boxes.append([float(value) for value in bbox])
    return boxes


def _to_physical(css_boxes: list[list[float]], dpr: float) -> list[list[float]]:
    return [[round(value * dpr, 1) for value in box] for box in css_boxes]


def _fingerprint(element: Mapping) -> dict:
    return {
        "tag": str(element.get("tag") or ""),
        "text": str(element.get("text") or ""),
        "id": str(element.get("id") or ""),
        "class": str(element.get("cls") or ""),
    }


def _fingerprints(
    by_xpath: Mapping[str, Mapping], xpaths: object, sample_id: str, label: str
) -> list[dict]:
    if not isinstance(xpaths, list):
        _fail(sample_id, f"{label} must be an xpath array")
    result = []
    for xpath in xpaths:
        element = by_xpath.get(xpath)
        if element is None:
            _fail(sample_id, f"{label} xpath missing from render: {xpath}")
        result.append(_fingerprint(element))
    return result


def _baseline_context(baseline_ei: Mapping, sample_id: str) -> dict:
    peer_groups = baseline_ei.get("peer_groups")
    if not isinstance(peer_groups, list):
        _fail(sample_id, "baseline element index lacks peer_groups")
    natural: dict[str, list] = {}
    for element in baseline_ei.get("elements") or []:
        if isinstance(element, Mapping) and element.get("tag") == "img":
            value = element.get("natural")
            if isinstance(value, list):
                natural[element["xpath"]] = list(value)
    return {
        "peer_groups": copy.deepcopy(peer_groups),
        "image_natural_by_xpath": natural,
    }


def _scan_counts(scan: Mapping, rule_ids: list[str], sample_id: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for rule_id in rule_ids:
        record = scan.get(rule_id)
        if not isinstance(record, Mapping) or record.get("na"):
            _fail(sample_id, f"scan rule {rule_id} is unavailable")
        n_hard = record.get("n_hard")
        if type(n_hard) is not int or n_hard < 0:
            _fail(sample_id, f"scan rule {rule_id} lacks n_hard")
        counts[rule_id] = n_hard
    return counts


def _scan_status(scan: Mapping, rule_ids: list[str]) -> dict[str, str]:
    status: dict[str, str] = {}
    for rule_id in rule_ids:
        record = scan.get(rule_id)
        if not isinstance(record, Mapping):
            status[rule_id] = "missing"
        elif record.get("na"):
            status[rule_id] = f"na:{record['na']}"
        else:
            status[rule_id] = "ok"
    return status


def build_gt_row(
    spec: Mapping,
    *,
    baseline_ei: dict,
    degraded_ei: dict,
    image_size: list[int],
) -> dict:
    sample_id = str(spec["sample_id"])
    viewport = spec["viewport"]
    if (
        not isinstance(viewport, list)
        or len(viewport) != 2
        or any(type(value) is not int or value < 1 for value in viewport)
    ):
        _fail(sample_id, "spec viewport must be two positive integers")
    dpr = spec["dpr"]
    if not isinstance(dpr, (int, float)) or float(dpr) <= 0:
        _fail(sample_id, "spec dpr must be positive")
    degraded_version = cohort_mod.actual_degraded_version(sample_id)

    baseline_by_xpath = _elements_by_xpath(baseline_ei, sample_id)
    degraded_by_xpath = _elements_by_xpath(degraded_ei, sample_id)

    defects = []
    for defect in spec["defects"]:
        for key in ("defect_id", "rule_id", "dim", "prop", "severity"):
            if not isinstance(defect.get(key), str) or not defect[key]:
                _fail(sample_id, f"spec defect lacks {key!r}")
        victim_degraded = _bboxes(
            degraded_by_xpath, defect["victim_xpaths_degraded"], sample_id,
            "victim_xpaths_degraded",
        )
        victim_baseline = _bboxes(
            baseline_by_xpath,
            defect.get("victim_xpaths_baseline") or [],
            sample_id,
            "victim_xpaths_baseline",
        )
        peer_xpaths = defect.get("peer_xpaths") or []
        if not isinstance(peer_xpaths, list):
            _fail(sample_id, "spec peer_xpaths must be an array")
        if defect["rule_id"] in GROUP_RULES:
            for xpath in peer_xpaths:
                if xpath not in degraded_by_xpath:
                    _fail(
                        sample_id,
                        f"peer xpath missing from degraded render: {xpath}",
                    )
        rebuilt = {
            key: copy.deepcopy(defect[key])
            for key in (
                "defect_id", "rule_id", "dim", "prop", "base_value", "new_value",
                "severity", "severity_metric", "difficulty", "evidence",
            )
            if key in defect
        }
        rebuilt["victim_xpaths_degraded"] = list(defect["victim_xpaths_degraded"])
        rebuilt["victim_xpaths_baseline"] = list(
            defect.get("victim_xpaths_baseline") or []
        )
        rebuilt["peer_xpaths"] = list(peer_xpaths)
        rebuilt["degraded_bboxes_css"] = victim_degraded
        rebuilt["degraded_bboxes_px"] = _to_physical(victim_degraded, float(dpr))
        rebuilt["baseline_bboxes_css"] = victim_baseline
        rebuilt["baseline_bboxes_px"] = _to_physical(victim_baseline, float(dpr))
        rebuilt["victim_fingerprints_baseline"] = _fingerprints(
            baseline_by_xpath, rebuilt["victim_xpaths_baseline"], sample_id,
            "victim_xpaths_baseline",
        )
        rebuilt["victim_fingerprints_degraded"] = _fingerprints(
            degraded_by_xpath, rebuilt["victim_xpaths_degraded"], sample_id,
            "victim_xpaths_degraded",
        )
        defects.append(rebuilt)

    row = {
        "schema_version": GT_SCHEMA_VERSION,
        "sample_id": sample_id,
        "layer": spec["layer"],
        "viewport": [int(viewport[0]), int(viewport[1])],
        "dpr": float(dpr),
        "degraded_version": degraded_version,
        "image_size": [int(image_size[0]), int(image_size[1])],
        "target_rule_ids": copy.deepcopy(spec["target_rule_ids"]),
        "scoring_rule_ids": copy.deepcopy(spec["scoring_rule_ids"]),
        "safety_gate_rule_ids": copy.deepcopy(spec["safety_gate_rule_ids"]),
        "safety_report_only_rule_ids": copy.deepcopy(spec["safety_report_only_rule_ids"]),
        "defects": defects,
        "baseline_context": _baseline_context(baseline_ei, sample_id),
    }

    target_rule_ids = spec["target_rule_ids"]
    if not isinstance(target_rule_ids, list) or any(
        not isinstance(value, str) or not value for value in target_rule_ids
    ):
        _fail(sample_id, "spec target_rule_ids must be a string array")
    unknown = {
        defect["rule_id"] for defect in defects
    } - set(target_rule_ids)
    if unknown:
        _fail(sample_id, f"defect rules outside target_rule_ids: {sorted(unknown)}")
    safety_ids = list(row["safety_gate_rule_ids"]) + list(
        row["safety_report_only_rule_ids"]
    )
    scans = {
        "baseline": scan_seed(build_hybrid_ei(baseline_ei, row)),
        "degraded": scan_seed(build_hybrid_ei(degraded_ei, row)),
    }
    row["target_rule_counts"] = {
        name: _scan_counts(scan, list(target_rule_ids), sample_id)
        for name, scan in scans.items()
    }
    row["safety_rule_counts"] = {
        name: _scan_counts(scan, safety_ids, sample_id)
        for name, scan in scans.items()
    }
    row["scan_status"] = {
        name: _scan_status(scan, list(target_rule_ids) + safety_ids)
        for name, scan in scans.items()
    }

    lesion_details = grade_lesions(
        baseline_ei,
        degraded_ei,
        degraded_ei,
        row,
        scans={**scans, "fixed": scans["degraded"]},
    )
    dropped = []
    for defect, detail in zip(row["defects"], lesion_details):
        if not detail.get("reference_valid"):
            dropped.append({
                "sample_id": sample_id,
                "defect_id": detail.get("defect_id"),
                "rule_id": detail.get("rule_id"),
                "reason": detail.get("reason"),
                "target_bad": detail.get("target_bad"),
                "n_hard": detail.get("n_hard"),
            })
    if dropped:
        dropped_ids = {entry["defect_id"] for entry in dropped}
        row["defects"] = [
            defect for defect in row["defects"]
            if defect["defect_id"] not in dropped_ids
        ]
    return row, dropped


def build_gt(
    *,
    specs_path: Path,
    cohort_path: Path,
    cache_root: Path,
    gt_out: Path,
    metadata_out: Path,
) -> dict:
    specs = load_specs(specs_path)
    sample_ids = cohort_mod.load_cohort_ids(cohort_path)
    missing_specs = [sample_id for sample_id in sample_ids if sample_id not in specs]
    if missing_specs:
        raise GTBuildError(
            f"cohort samples lack defect specs: {missing_specs[:5]} "
            f"({len(missing_specs)} total)"
        )

    manifest_path = cache_root / "manifest.json"
    manifest = reference_cache.strict_json(manifest_path, "reference cache manifest")
    records = manifest.get("records")
    if not isinstance(records, list):
        raise GTBuildError("reference cache manifest lacks records")
    records_by_id = {
        record.get("sample_id"): record
        for record in records
        if isinstance(record, Mapping)
    }
    missing_cache = [
        sample_id for sample_id in sample_ids if sample_id not in records_by_id
    ]
    if missing_cache:
        raise GTBuildError(
            f"cohort samples missing from reference cache: {missing_cache[:5]} "
            f"({len(missing_cache)} total)"
        )

    rows: list[dict] = []
    dropped_all: list[dict] = []
    for sample_id in sample_ids:
        record = records_by_id[sample_id]
        loaded = reference_cache.load_bundle(
            reference_cache.resolved_repo_path(record["bundle"]["path"]),
            manifest_record=record,
        )
        roles = loaded["bundle"]["roles"]
        degraded_size = roles["degraded"]["grading_render"]["screenshot"]["size_px"]
        row, dropped = build_gt_row(
            specs[sample_id],
            baseline_ei=roles["baseline"]["element_index"],
            degraded_ei=roles["degraded"]["element_index"],
            image_size=[int(degraded_size[0]), int(degraded_size[1])],
        )
        if not row["defects"]:
            raise GTBuildError(
                f"{sample_id}: every injected defect failed to reproduce on the "
                "frozen renders; the sample is ungradeable"
            )
        dropped_all.extend(dropped)
        grading.validate_scoring_gt_row(row, sample_id)
        rows.append(row)

    gt_out.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    )
    grading.atomic_write_text(gt_out, payload)
    layers = Counter(str(row["layer"]) for row in rows)
    metadata = {
        "schema_version": GT_SCHEMA_VERSION,
        "n_rows": len(rows),
        "layers": dict(sorted(layers.items())),
        "n_defects_total": sum(len(row["defects"]) for row in rows),
        "target_rule_ids": sorted({
            defect["rule_id"] for row in rows for defect in row["defects"]
        }),
        "dropped_defects": dropped_all,
    }
    grading.atomic_write_json(metadata_out, metadata)
    return metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--specs", type=Path, default=DEFAULT_SPECS)
    parser.add_argument("--cohort", type=Path, default=DEFAULT_COHORT)
    parser.add_argument(
        "--cache-root", type=Path, default=reference_cache.DEFAULT_CACHE_ROOT
    )
    parser.add_argument("--gt-out", type=Path, default=DEFAULT_GT_OUT)
    parser.add_argument("--metadata-out", type=Path, default=DEFAULT_METADATA_OUT)
    args = parser.parse_args(argv)

    metadata = build_gt(
        specs_path=args.specs,
        cohort_path=args.cohort,
        cache_root=args.cache_root.resolve(),
        gt_out=args.gt_out,
        metadata_out=args.metadata_out,
    )
    print(
        f"build-gt: OK — {metadata['n_rows']} rows, "
        f"{metadata['n_defects_total']} defects -> {args.gt_out}"
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Task3Error as exc:
        print(f"build-gt failed: {exc}", file=sys.stderr)
        sys.exit(1)
