#!/usr/bin/env python3
"""Task 3 v3 lesion-specific repair and collateral scorer.

The scorer freezes the construction-time peer-group context.  A repair is
therefore judged on the injected lesion itself, not on an unrelated page-wide
count or on peer groups re-discovered after the model changes geometry.
"""

from __future__ import annotations

import copy
import re
from typing import Iterable

try:
    from .compat_rules import scan_seed
except ImportError:  # direct script import from the module directory
    from compat_rules import scan_seed


GROUP_RULES = {
    "TEXT_PEER_FONTSIZE",
    "TEXT_PEER_WEIGHT",
    "COLOR_PEER_ACCENT",
    "CONSIST_CONTAINER_RADIUS",
    "CONSIST_CONTAINER_GAP",
    "LAYOUT_ALIGNMENT",
    "SPACING_PEER_UNIFORM",
}
VICTIM_RULES = {
    "COLOR_SEMANTIC",
    "TEXT_SCALE_COUNT",
    "TEXT_FONT_FAMILY_COUNT",
    "TEXT_TABLE_ALIGN",
    "LAYOUT_FITTS_EDGE",
}
ORDER_RULES = {"LAYOUT_GUTENBERG_INFO"}
SUPPORTED_TARGET_RULES = GROUP_RULES | VICTIM_RULES | ORDER_RULES


class ScoringError(RuntimeError):
    pass


def build_hybrid_ei(current_ei: dict, gt_row: dict) -> dict:
    """Use current geometry/styles with the GT's frozen grouping/natural context."""
    hybrid = dict(current_ei)
    context = gt_row["baseline_context"]
    hybrid["peer_groups"] = copy.deepcopy(context["peer_groups"])
    natural = context.get("image_natural_by_xpath", {})
    elements = []
    for element in current_ei["elements"]:
        item = element
        if element.get("tag") == "img":
            item = dict(element)
            item["natural"] = copy.deepcopy(natural.get(element["xpath"], [0, 0]))
        elements.append(item)
    hybrid["elements"] = elements
    return hybrid


def scan_checked(scan: dict, rule_id: str, *, require_applicable: bool = False) -> dict:
    record = scan.get(rule_id)
    if not isinstance(record, dict):
        raise ScoringError(f"{rule_id}: scanner result missing")
    if record.get("na"):
        raise ScoringError(f"{rule_id}: scanner unavailable ({record['na']})")
    if "n_hard" not in record:
        raise ScoringError(f"{rule_id}: scanner omitted n_hard")
    if require_applicable and int(record.get("n_app") or 0) <= 0:
        raise ScoringError(f"{rule_id}: target is no longer applicable")
    return record


def _scan_hybrid(current_ei: dict, gt_row: dict) -> dict:
    return scan_seed(build_hybrid_ei(current_ei, gt_row))


def _target_group_scan(current_ei: dict, gt_row: dict, defect: dict) -> dict:
    hybrid = build_hybrid_ei(current_ei, gt_row)
    paths = list(defect["peer_xpaths"])
    present = {element["xpath"] for element in hybrid["elements"]}
    missing = sorted(set(paths) - present)
    if missing:
        raise ScoringError(
            f"{defect['defect_id']}: {len(missing)} target peer(s) missing")
    hybrid["peer_groups"] = [{"gid": "target", "size": len(paths), "members": paths}]
    return scan_checked(scan_seed(hybrid), defect["rule_id"], require_applicable=True)


def _normalise_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def resolve_victim_by_fingerprint(current_ei: dict, defect: dict) -> list[str]:
    """Resolve a DOM-moved heading; ambiguity is a hard failure."""
    resolved = []
    elements = current_ei["elements"]
    for fingerprint in defect["victim_fingerprints_baseline"]:
        tag = fingerprint.get("tag", "")
        text = _normalise_text(fingerprint.get("text"))
        element_id = fingerprint.get("id", "")
        element_class = _normalise_text(fingerprint.get("class"))
        candidates = [element for element in elements
                      if element.get("tag", "") == tag
                      and _normalise_text(element.get("text")) == text]
        if element_id:
            candidates = [element for element in candidates
                          if element.get("id", "") == element_id]
        if element_class:
            candidates = [
                element for element in candidates
                if _normalise_text(element.get("cls")) == element_class
            ]
        if len(candidates) != 1:
            raise ScoringError(
                f"{defect['defect_id']}: fingerprint matched {len(candidates)} elements")
        resolved.append(candidates[0]["xpath"])
    return resolved


def _target_paths(current_ei: dict, defect: dict) -> list[str]:
    rule_id = defect["rule_id"]
    if rule_id in ORDER_RULES:
        return resolve_victim_by_fingerprint(current_ei, defect)
    paths = (defect.get("evidence", {}).get("row_xpaths", [])
             if rule_id == "LAYOUT_FITTS_EDGE"
             else defect["victim_xpaths_degraded"])
    present = {element["xpath"] for element in current_ei["elements"]}
    missing = sorted(set(paths) - present)
    if not paths or missing:
        raise ScoringError(
            f"{defect['defect_id']}: target identity missing ({len(missing)})")
    return list(paths)


def _bad_paths(record: dict) -> set[str]:
    return set((record.get("extra") or {}).get("bad_xpaths", []))


def grade_lesions(baseline_ei: dict, degraded_ei: dict, fixed_ei: dict,
                  gt_row: dict, *, scans: dict[str, dict] | None = None) -> list[dict]:
    """Grade every injected lesion independently and validate the frozen GT."""
    unknown = {defect["rule_id"] for defect in gt_row["defects"]} - SUPPORTED_TARGET_RULES
    if unknown:
        raise ScoringError(f"unsupported target rule(s): {sorted(unknown)}")
    scans = scans or {
        "baseline": _scan_hybrid(baseline_ei, gt_row),
        "degraded": _scan_hybrid(degraded_ei, gt_row),
        "fixed": _scan_hybrid(fixed_ei, gt_row),
    }
    output = []
    for defect in gt_row["defects"]:
        rule_id = defect["rule_id"]
        detail = {"defect_id": defect["defect_id"], "rule_id": rule_id,
                  "dim": defect["dim"], "severity": defect["severity"]}
        try:
            if rule_id in GROUP_RULES:
                records = {
                    "baseline": _target_group_scan(baseline_ei, gt_row, defect),
                    "degraded": _target_group_scan(degraded_ei, gt_row, defect),
                    "fixed": _target_group_scan(fixed_ei, gt_row, defect),
                }
                counts = {name: int(record["n_hard"]) for name, record in records.items()}
                reference_valid = counts["baseline"] == 0 and counts["degraded"] > 0
                passed = counts["fixed"] == 0
                detail.update(mode="fixed_peer_group", n_hard=counts,
                              reference_valid=reference_valid, passed=passed,
                              reason="clean_group" if passed else "target_group_still_violates")
            else:
                records = {name: scan_checked(scan, rule_id, require_applicable=True)
                           for name, scan in scans.items()}
                paths = {
                    "baseline": _target_paths(baseline_ei, defect),
                    "degraded": _target_paths(degraded_ei, defect),
                    "fixed": _target_paths(fixed_ei, defect),
                }
                target_bad = {
                    name: bool(set(paths[name]) & _bad_paths(records[name]))
                    for name in records
                }
                reference_valid = not target_bad["baseline"] and target_bad["degraded"]
                passed = not target_bad["fixed"]
                detail.update(
                    mode="fingerprint" if rule_id in ORDER_RULES else "victim_xpath",
                    target_xpaths_fixed=paths["fixed"],
                    target_bad=target_bad,
                    n_hard={name: int(record["n_hard"]) for name, record in records.items()},
                    reference_valid=reference_valid,
                    passed=passed,
                    reason="target_clean" if passed else "target_still_violates",
                )
            if not detail["reference_valid"]:
                raise ScoringError(f"{defect['defect_id']}: baseline/degraded GT did not reproduce")
        except ScoringError as error:
            detail.update(passed=False, reason="scoring_error", error=str(error))
        output.append(detail)
    return output


def grade_collateral(scans: dict[str, dict], gt_row: dict) -> dict:
    """Reject regressions beyond both the clean baseline and degraded input.

    A degradation can accidentally remove an unrelated pre-existing violation.
    Restoring the clean page must not be called collateral damage merely because
    it restores that baseline count, so the permitted ceiling is max(base, deg).
    """
    gate_detail = {}
    counts = {name: {} for name in scans}
    for rule_id in gt_row["safety_gate_rule_ids"]:
        for name, scan in scans.items():
            counts[name][rule_id] = int(scan_checked(scan, rule_id)["n_hard"])
        ceiling = max(counts["baseline"][rule_id], counts["degraded"][rule_id])
        if counts["fixed"][rule_id] > ceiling:
            gate_detail[rule_id] = {
                **{name: counts[name][rule_id]
                   for name in ("baseline", "degraded", "fixed")},
                "allowed_ceiling": ceiling,
            }

    report_only = {}
    for rule_id in gt_row["safety_report_only_rule_ids"]:
        values = {name: int(scan_checked(scan, rule_id)["n_hard"])
                  for name, scan in scans.items()}
        if len(set(values.values())) > 1:
            report_only[rule_id] = values
        for name, value in values.items():
            counts[name][rule_id] = value
    return {"passed": not gate_detail, "failures": gate_detail,
            "report_only_changes": report_only, "counts": counts}


def grade_page(baseline_ei: dict, degraded_ei: dict, fixed_ei: dict, gt_row: dict) -> dict:
    scans = {
        "baseline": _scan_hybrid(baseline_ei, gt_row),
        "degraded": _scan_hybrid(degraded_ei, gt_row),
        "fixed": _scan_hybrid(fixed_ei, gt_row),
    }
    lesions = grade_lesions(baseline_ei, degraded_ei, fixed_ei, gt_row, scans=scans)
    collateral = grade_collateral(scans, gt_row)
    repaired = sum(bool(item["passed"]) for item in lesions)
    gate_ids = gt_row["safety_gate_rule_ids"]
    base_total = sum(collateral["counts"]["baseline"][rule] for rule in gate_ids)
    degraded_total = sum(collateral["counts"]["degraded"][rule] for rule in gate_ids)
    fixed_total = sum(collateral["counts"]["fixed"][rule] for rule in gate_ids)
    # Sum positive per-rule regressions so an accidental improvement in another
    # rule cannot cancel the injected repair headroom.
    headroom = sum(max(
        collateral["counts"]["degraded"][rule]
        - collateral["counts"]["baseline"][rule], 0) for rule in gate_ids)
    remaining = sum(max(
        collateral["counts"]["fixed"][rule]
        - collateral["counts"]["baseline"][rule], 0) for rule in gate_ids)
    raw_reduction = ((headroom - remaining) / headroom if headroom > 0 else None)
    return {
        "lesion_tests": lesions,
        "n_lesions": len(lesions),
        "n_lesions_fixed": repaired,
        "lesion_repair_rate": repaired / len(lesions) if lesions else 0.0,
        "fix_pass": bool(lesions) and repaired == len(lesions),
        "collateral_pass": collateral["passed"],
        "collateral_detail": collateral["failures"],
        "safety_report_only": collateral["report_only_changes"],
        "safety_counts": collateral["counts"],
        "viol": {"baseline": base_total, "degraded": degraded_total, "fixed": fixed_total},
        "net_reduction": (max(0.0, min(1.0, raw_reduction))
                          if raw_reduction is not None else None),
        "net_reduction_raw": raw_reduction,
    }
