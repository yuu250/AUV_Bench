#!/usr/bin/env python3
"""Deterministic, model-facing short target IDs for bounded Task 3 patches.

This layer is deliberately ground-truth blind.  It consumes only:

* elements extracted from the selected/degraded page's live browser DOM; and
* Task 2's frozen localization regions, converted to CSS-page coordinates.

The model sees compact IDs (``t1``, ``t2``, ...), visual fingerprints, and
whether an ID may be edited or used only as a move anchor.  Frozen XPath and
source-executor ordinals remain in the internal catalog.  A validated model
patch can then be resolved directly to ``structured_patch.apply_source_patch``
inputs without asking the model to reproduce a long XPath.
"""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Mapping, Sequence

from tasks.task3.structured_patch import (
    ALLOWED_STYLE_PROPERTIES,
    MAX_EDITS,
    StructuredPatchError,
    validate_patch,
)


CATALOG_SCHEMA_VERSION = "task3-target-catalog-v1"
MAX_TARGETS = 64
MAX_EDITABLE_TARGETS = 32
DEFAULT_PER_REGION_LIMIT = 12
DEFAULT_SIBLING_LIMIT = 16
DEFAULT_SEARCH_MARGIN_CSS = 32.0

_TARGET_ID_RE = re.compile(r"t[1-9][0-9]*\Z")
_XPATH_RE = re.compile(r"(?:/[a-z][a-z0-9:_-]*\[[1-9][0-9]*\])+\Z")
_TAG_RE = re.compile(r"[a-z][a-z0-9:_-]*\Z")
_NON_EDITABLE_TAGS = frozenset(
    {"html", "head", "body", "script", "style", "link", "meta", "title", "template"}
)


class TargetCatalogError(ValueError):
    """Raised when extraction data, a catalog, or a model ID patch is invalid."""


# The global tag ordinal is computed over the complete DOM.  We retain every
# body element so an invisible/zero-box sibling can still be named as a move
# anchor; only visible, non-internal elements may become editable candidates.
# This is the same zero-based locator convention consumed by
# structured_patch.set_style_by_ordinals and move.
COLLECT_TARGETS_JS = r"""
() => {
  function xpathOf(el) {
    const parts = [];
    let node = el;
    while (node && node.nodeType === 1) {
      let index = 1;
      let sibling = node.previousElementSibling;
      while (sibling) {
        if (sibling.tagName === node.tagName) index += 1;
        sibling = sibling.previousElementSibling;
      }
      parts.unshift(node.tagName.toLowerCase() + '[' + index + ']');
      node = node.parentElement;
    }
    return '/' + parts.join('/');
  }

  const all = Array.from(document.querySelectorAll('*'));
  const tagCounts = new Map();
  const tagIndex = new WeakMap();
  const documentIndex = new WeakMap();
  for (let index = 0; index < all.length; index += 1) {
    const el = all[index];
    const tag = el.tagName.toLowerCase();
    const ordinal = tagCounts.get(tag) || 0;
    tagCounts.set(tag, ordinal + 1);
    tagIndex.set(el, ordinal);
    documentIndex.set(el, index);
  }

  const elements = [];
  const root = document.body || document.documentElement;
  if (!root) return {elements};
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_ELEMENT);
  let node;
  while ((node = walker.nextNode())) {
    const style = getComputedStyle(node);
    const rect = node.getBoundingClientRect();
    const visible = !(style.display === 'none' || style.visibility === 'hidden' ||
        Number.parseFloat(style.opacity) === 0 || rect.width < 0.5 || rect.height < 0.5);
    let directText = '';
    for (const child of node.childNodes) {
      if (child.nodeType === Node.TEXT_NODE) directText += child.textContent || '';
    }
    directText = directText.replace(/\s+/g, ' ').trim().slice(0, 120);
    elements.push({
      xpath: xpathOf(node),
      tag: node.tagName.toLowerCase(),
      tag_index: tagIndex.get(node),
      document_index: documentIndex.get(node),
      visible,
      bbox: [
        Math.round(rect.x * 10) / 10,
        Math.round(rect.y * 10) / 10,
        Math.round(rect.width * 10) / 10,
        Math.round(rect.height * 10) / 10,
      ],
      id: (node.id || '').slice(0, 120),
      cls: (node.getAttribute('class') || '').slice(0, 200),
      text: directText,
      role: (node.getAttribute('role') || '').slice(0, 80),
      aria_label: (node.getAttribute('aria-label') || '').slice(0, 120),
    });
  }
  return {elements};
}
"""


def _finite_number(value: object, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TargetCatalogError(f"{context} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise TargetCatalogError(f"{context} must be a finite number")
    return result


def _round_geometry(value: float) -> float:
    rounded = round(value, 4)
    return 0.0 if rounded == 0 else rounded


def _validate_xyxy(box: object, context: str) -> list[float]:
    if not isinstance(box, Sequence) or isinstance(box, (str, bytes)) or len(box) != 4:
        raise TargetCatalogError(f"{context} must be a four-number xyxy array")
    values = [_finite_number(value, f"{context}[{index}]") for index, value in enumerate(box)]
    if values[0] >= values[2] or values[1] >= values[3]:
        raise TargetCatalogError(f"{context} must have x1<x2 and y1<y2")
    return [_round_geometry(value) for value in values]


def _validate_xywh(
    box: object, context: str, *, allow_zero: bool = False
) -> list[float]:
    if not isinstance(box, Sequence) or isinstance(box, (str, bytes)) or len(box) != 4:
        raise TargetCatalogError(f"{context} must be a four-number xywh array")
    values = [_finite_number(value, f"{context}[{index}]") for index, value in enumerate(box)]
    if values[2] < 0 or values[3] < 0 or (
        not allow_zero and (values[2] == 0 or values[3] == 0)
    ):
        qualifier = "non-negative" if allow_zero else "positive"
        raise TargetCatalogError(
            f"{context} must have {qualifier} width and height"
        )
    return [_round_geometry(value) for value in values]


def canonical_locations_to_css_regions(
    locations: Sequence[Mapping[str, object]],
    *,
    slice_offsets_px: Sequence[int | float],
    canvas_width_px: int | float,
    dpr: int | float,
    canvas_height_px: int | float = 1800,
) -> list[list[float]]:
    """Convert Task 2 v8 canonical normalized-1000 xyxy boxes to page CSS xyxy.

    ``slice_offsets_px`` and canvas dimensions are in the frozen Task 2 image
    coordinate system.  Page CSS coordinates are recovered by dividing both
    axes by the selected page's DPR.
    """

    if not isinstance(locations, Sequence) or isinstance(locations, (str, bytes)):
        raise TargetCatalogError("locations must be an array")
    if not isinstance(slice_offsets_px, Sequence) or isinstance(
        slice_offsets_px, (str, bytes)
    ):
        raise TargetCatalogError("slice_offsets_px must be an array")
    offsets = [
        _finite_number(value, f"slice_offsets_px[{index}]")
        for index, value in enumerate(slice_offsets_px)
    ]
    if any(value < 0 for value in offsets):
        raise TargetCatalogError("slice offsets must be non-negative")
    width = _finite_number(canvas_width_px, "canvas_width_px")
    height = _finite_number(canvas_height_px, "canvas_height_px")
    scale = _finite_number(dpr, "dpr")
    if width <= 0 or height <= 0 or scale <= 0:
        raise TargetCatalogError("canvas dimensions and dpr must be positive")

    regions: list[list[float]] = []
    for index, location in enumerate(locations):
        context = f"locations[{index}]"
        if not isinstance(location, Mapping) or set(location) != {"slice", "box"}:
            raise TargetCatalogError(f"{context} must contain exactly slice and box")
        slice_number = location["slice"]
        if isinstance(slice_number, bool) or not isinstance(slice_number, int):
            raise TargetCatalogError(f"{context}.slice must be an integer")
        if not 1 <= slice_number <= len(offsets):
            raise TargetCatalogError(f"{context}.slice is outside the frozen slice list")
        normalized = _validate_xyxy(location["box"], f"{context}.box")
        if any(value < 0 or value > 1000 for value in normalized):
            raise TargetCatalogError(f"{context}.box must remain inside normalized_1000")
        top = offsets[slice_number - 1]
        regions.append(
            [
                _round_geometry(normalized[0] * width / 1000.0 / scale),
                _round_geometry((top + normalized[1] * height / 1000.0) / scale),
                _round_geometry(normalized[2] * width / 1000.0 / scale),
                _round_geometry((top + normalized[3] * height / 1000.0) / scale),
            ]
        )
    return regions


def _parent_xpath(xpath: str) -> str:
    return xpath.rpartition("/")[0]


def _intersection_area(left: Sequence[float], right: Sequence[float]) -> float:
    width = max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
    height = max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
    return width * height


def _rect_distance(left: Sequence[float], right: Sequence[float]) -> float:
    dx = max(left[0] - right[2], right[0] - left[2], 0.0)
    dy = max(left[1] - right[3], right[1] - left[3], 0.0)
    return math.hypot(dx, dy)


def _candidate_rank(
    bbox: Sequence[float], region: Sequence[float], document_index: int
) -> tuple[float, ...]:
    intersection = _intersection_area(bbox, region)
    region_width = region[2] - region[0]
    region_height = region[3] - region[1]
    element_width = bbox[2] - bbox[0]
    element_height = bbox[3] - bbox[1]
    region_area = region_width * region_height
    element_area = element_width * element_height
    normalizer = max(region_width, region_height, 1.0)
    distance = _rect_distance(bbox, region) / normalizer
    area_mismatch = abs(math.log(max(element_area, 1e-6) / max(region_area, 1e-6)))
    edge_mismatch = (
        abs(bbox[0] - region[0])
        + abs(bbox[1] - region[1])
        + abs(bbox[2] - region[2])
        + abs(bbox[3] - region[3])
    ) / (2.0 * max(region_width + region_height, 1.0))
    return (
        0.0 if intersection > 0 else 1.0,
        distance,
        area_mismatch,
        edge_mismatch,
        float(document_index),
    )


def _bounded_text(value: object, limit: int) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.split())[:limit]


def _normalize_browser_elements(
    browser_elements: Sequence[Mapping[str, object]],
) -> list[dict]:
    if not isinstance(browser_elements, Sequence) or isinstance(
        browser_elements, (str, bytes)
    ):
        raise TargetCatalogError("browser_elements must be an array")
    normalized: list[dict] = []
    seen_xpaths: set[str] = set()
    seen_locators: set[tuple[str, int]] = set()
    seen_document_indices: set[int] = set()
    for input_index, element in enumerate(browser_elements):
        context = f"browser_elements[{input_index}]"
        if not isinstance(element, Mapping):
            raise TargetCatalogError(f"{context} must be an object")
        xpath = element.get("xpath")
        tag = element.get("tag")
        tag_index = element.get("tag_index")
        document_index = element.get("document_index", input_index)
        if not isinstance(xpath, str) or not _XPATH_RE.fullmatch(xpath):
            raise TargetCatalogError(f"{context}.xpath is not a safe absolute XPath")
        if not isinstance(tag, str) or not _TAG_RE.fullmatch(tag):
            raise TargetCatalogError(f"{context}.tag must be a lowercase HTML tag")
        if isinstance(tag_index, bool) or not isinstance(tag_index, int) or tag_index < 0:
            raise TargetCatalogError(f"{context}.tag_index must be a non-negative integer")
        if (
            isinstance(document_index, bool)
            or not isinstance(document_index, int)
            or document_index < 0
        ):
            raise TargetCatalogError(
                f"{context}.document_index must be a non-negative integer"
            )
        if xpath in seen_xpaths:
            raise TargetCatalogError(f"duplicate browser XPath: {xpath}")
        locator = (tag, tag_index)
        if locator in seen_locators:
            raise TargetCatalogError(f"duplicate browser tag ordinal: {locator}")
        if document_index in seen_document_indices:
            raise TargetCatalogError(f"duplicate browser document_index: {document_index}")
        seen_xpaths.add(xpath)
        seen_locators.add(locator)
        seen_document_indices.add(document_index)
        bbox_xywh = _validate_xywh(
            element.get("bbox", element.get("bbox_css_xywh")),
            f"{context}.bbox",
            allow_zero=True,
        )
        visible = element.get("visible")
        if visible is None:
            visible = bbox_xywh[2] >= 0.5 and bbox_xywh[3] >= 0.5
        if not isinstance(visible, bool):
            raise TargetCatalogError(f"{context}.visible must be boolean")
        bbox_xyxy = [
            bbox_xywh[0],
            bbox_xywh[1],
            _round_geometry(bbox_xywh[0] + bbox_xywh[2]),
            _round_geometry(bbox_xywh[1] + bbox_xywh[3]),
        ]
        normalized.append(
            {
                "xpath": xpath,
                "parent_xpath": _parent_xpath(xpath),
                "tag": tag,
                "tag_index": tag_index,
                "document_index": document_index,
                "visible": visible,
                "bbox_css_xywh": bbox_xywh,
                "bbox_css_xyxy": bbox_xyxy,
                "id": _bounded_text(element.get("id"), 120),
                "class": _bounded_text(element.get("cls", element.get("class")), 200),
                "text": _bounded_text(element.get("text"), 120),
                "role": _bounded_text(element.get("role"), 80),
                "aria_label": _bounded_text(element.get("aria_label"), 120),
            }
        )
    normalized.sort(key=lambda item: item["document_index"])
    return normalized


def build_target_catalog(
    browser_elements: Sequence[Mapping[str, object]] | Mapping[str, object],
    authorized_regions_css_xyxy: Sequence[Sequence[int | float]],
    *,
    per_region_limit: int = DEFAULT_PER_REGION_LIMIT,
    sibling_limit: int = DEFAULT_SIBLING_LIMIT,
    search_margin_css: float = DEFAULT_SEARCH_MARGIN_CSS,
    max_editable_targets: int = MAX_EDITABLE_TARGETS,
    max_targets: int = MAX_TARGETS,
) -> dict:
    """Build a deterministic target catalog around frozen Task 2 regions.

    For every region, the closest bbox-shaped matches inside a bounded margin
    become editable candidates.  Direct DOM siblings are then added as
    anchor-only candidates so reading-order repairs can name a ``before_id``.
    IDs are assigned in browser document order, never by GT or model content.
    """

    if isinstance(browser_elements, Mapping):
        if set(browser_elements) != {"elements"}:
            raise TargetCatalogError("browser extraction must contain exactly elements")
        browser_elements = browser_elements["elements"]  # type: ignore[assignment]
    elements = _normalize_browser_elements(browser_elements)  # type: ignore[arg-type]
    if not isinstance(authorized_regions_css_xyxy, Sequence) or isinstance(
        authorized_regions_css_xyxy, (str, bytes)
    ):
        raise TargetCatalogError("authorized_regions_css_xyxy must be an array")
    regions = [
        _validate_xyxy(region, f"authorized_regions_css_xyxy[{index}]")
        for index, region in enumerate(authorized_regions_css_xyxy)
    ]
    for name, value in {
        "per_region_limit": per_region_limit,
        "sibling_limit": sibling_limit,
        "max_editable_targets": max_editable_targets,
        "max_targets": max_targets,
    }.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise TargetCatalogError(f"{name} must be a non-negative integer")
    if max_targets > MAX_TARGETS:
        raise TargetCatalogError(f"max_targets cannot exceed protocol limit {MAX_TARGETS}")
    if max_editable_targets > max_targets:
        raise TargetCatalogError("max_editable_targets cannot exceed max_targets")
    margin = _finite_number(search_margin_css, "search_margin_css")
    if margin < 0:
        raise TargetCatalogError("search_margin_css must be non-negative")

    editable_best_rank: dict[int, tuple[float, ...]] = {}
    editable_regions: dict[int, set[int]] = {}
    for region_index, region in enumerate(regions):
        ranked: list[tuple[tuple[float, ...], int]] = []
        for element_index, element in enumerate(elements):
            if not element["visible"] or element["tag"] in _NON_EDITABLE_TAGS:
                continue
            if _rect_distance(element["bbox_css_xyxy"], region) > margin:
                continue
            ranked.append(
                (
                    _candidate_rank(
                        element["bbox_css_xyxy"], region, element["document_index"]
                    ),
                    element_index,
                )
            )
        ranked.sort()
        for rank, element_index in ranked[:per_region_limit]:
            editable_regions.setdefault(element_index, set()).add(region_index)
            previous = editable_best_rank.get(element_index)
            if previous is None or rank < previous:
                editable_best_rank[element_index] = rank

    if len(editable_best_rank) > max_editable_targets:
        retained = {
            index
            for _, index in sorted(
                (rank, index) for index, rank in editable_best_rank.items()
            )[:max_editable_targets]
        }
        editable_best_rank = {
            index: rank for index, rank in editable_best_rank.items() if index in retained
        }
        editable_regions = {
            index: values for index, values in editable_regions.items() if index in retained
        }

    editable_indices = set(editable_best_rank)
    by_parent: dict[str, list[int]] = {}
    for index, element in enumerate(elements):
        by_parent.setdefault(element["parent_xpath"], []).append(index)

    anchor_priority: dict[int, tuple[float, ...]] = {}
    for editable_index in sorted(
        editable_indices, key=lambda index: elements[index]["document_index"]
    ):
        target = elements[editable_index]
        parent_children = by_parent[target["parent_xpath"]]
        target_sibling_index = parent_children.index(editable_index)
        siblings = [index for index in parent_children if index != editable_index]
        siblings.sort(
            key=lambda index: (
                abs(parent_children.index(index) - target_sibling_index),
                elements[index]["document_index"],
            )
        )
        for sibling in siblings[:sibling_limit]:
            priority = (
                *editable_best_rank[editable_index],
                float(abs(parent_children.index(sibling) - target_sibling_index)),
                float(elements[sibling]["document_index"]),
            )
            previous = anchor_priority.get(sibling)
            if previous is None or priority < previous:
                anchor_priority[sibling] = priority

    candidate_indices = set(editable_indices)
    remaining_capacity = max_targets - len(candidate_indices)
    for index, _ in sorted(anchor_priority.items(), key=lambda item: item[1]):
        if index in candidate_indices:
            continue
        if remaining_capacity <= 0:
            break
        candidate_indices.add(index)
        remaining_capacity -= 1

    ordered_indices = sorted(
        candidate_indices, key=lambda index: elements[index]["document_index"]
    )
    targets: list[dict] = []
    for ordinal, element_index in enumerate(ordered_indices, 1):
        element = elements[element_index]
        targets.append(
            {
                "target_id": f"t{ordinal}",
                "editable": element_index in editable_indices,
                "anchor": True,
                "authorized_region_indices": sorted(
                    editable_regions.get(element_index, set())
                ),
                "document_index": element["document_index"],
                "locator": {
                    "frozen_xpath": element["xpath"],
                    "tag": element["tag"],
                    "tag_index": element["tag_index"],
                },
                "fingerprint": {
                    "tag": element["tag"],
                    "id": element["id"],
                    "class": element["class"],
                    "text": element["text"],
                    "role": element["role"],
                    "aria_label": element["aria_label"],
                    "visible": element["visible"],
                    "bbox_css_xywh": element["bbox_css_xywh"],
                },
            }
        )

    catalog = {
        "schema_version": CATALOG_SCHEMA_VERSION,
        "coordinate_space": "page_css_xyxy",
        "authorization_regions_css_xyxy": regions,
        "selection_policy": {
            "ground_truth_used": False,
            "per_region_limit": per_region_limit,
            "sibling_limit": sibling_limit,
            "search_margin_css": _round_geometry(margin),
            "max_editable_targets": max_editable_targets,
            "max_targets": max_targets,
            "id_order": "browser_document_order",
        },
        "targets": targets,
    }
    validate_target_catalog(catalog)
    return catalog


def validate_target_catalog(catalog: object) -> dict:
    """Strictly validate a catalog loaded from memory or disk."""

    if not isinstance(catalog, Mapping):
        raise TargetCatalogError("target catalog must be an object")
    expected_top = {
        "schema_version",
        "coordinate_space",
        "authorization_regions_css_xyxy",
        "selection_policy",
        "targets",
    }
    if set(catalog) != expected_top:
        raise TargetCatalogError("target catalog has unexpected or missing fields")
    if catalog["schema_version"] != CATALOG_SCHEMA_VERSION:
        raise TargetCatalogError("unsupported target catalog schema_version")
    if catalog["coordinate_space"] != "page_css_xyxy":
        raise TargetCatalogError("unsupported target catalog coordinate_space")
    regions_value = catalog["authorization_regions_css_xyxy"]
    if not isinstance(regions_value, list):
        raise TargetCatalogError("catalog authorization regions must be an array")
    regions = [
        _validate_xyxy(region, f"authorization_regions_css_xyxy[{index}]")
        for index, region in enumerate(regions_value)
    ]
    if not isinstance(catalog["selection_policy"], Mapping):
        raise TargetCatalogError("catalog selection_policy must be an object")
    targets_value = catalog["targets"]
    if not isinstance(targets_value, list) or len(targets_value) > MAX_TARGETS:
        raise TargetCatalogError(f"catalog targets must be an array of at most {MAX_TARGETS}")

    clean_targets: list[dict] = []
    seen_xpaths: set[str] = set()
    seen_locators: set[tuple[str, int]] = set()
    previous_document_index = -1
    for index, target in enumerate(targets_value, 1):
        context = f"targets[{index - 1}]"
        if not isinstance(target, Mapping) or set(target) != {
            "target_id",
            "editable",
            "anchor",
            "authorized_region_indices",
            "document_index",
            "locator",
            "fingerprint",
        }:
            raise TargetCatalogError(f"{context} has unexpected or missing fields")
        if target["target_id"] != f"t{index}":
            raise TargetCatalogError("target IDs must be gap-free browser-order t1,t2,...")
        if not isinstance(target["editable"], bool) or target["anchor"] is not True:
            raise TargetCatalogError(f"{context} has invalid roles")
        region_indices = target["authorized_region_indices"]
        if (
            not isinstance(region_indices, list)
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 0 <= value < len(regions)
                for value in region_indices
            )
            or region_indices != sorted(set(region_indices))
        ):
            raise TargetCatalogError(f"{context} has invalid authorized_region_indices")
        if target["editable"] != bool(region_indices):
            raise TargetCatalogError(f"{context} editable role disagrees with authorization")
        document_index = target["document_index"]
        if (
            isinstance(document_index, bool)
            or not isinstance(document_index, int)
            or document_index <= previous_document_index
        ):
            raise TargetCatalogError("catalog targets must have increasing document_index")
        previous_document_index = document_index
        locator = target["locator"]
        if not isinstance(locator, Mapping) or set(locator) != {
            "frozen_xpath",
            "tag",
            "tag_index",
        }:
            raise TargetCatalogError(f"{context}.locator has invalid fields")
        xpath = locator["frozen_xpath"]
        tag = locator["tag"]
        tag_index = locator["tag_index"]
        if not isinstance(xpath, str) or not _XPATH_RE.fullmatch(xpath):
            raise TargetCatalogError(f"{context} has an unsafe frozen XPath")
        if not isinstance(tag, str) or not _TAG_RE.fullmatch(tag):
            raise TargetCatalogError(f"{context} has an invalid locator tag")
        if isinstance(tag_index, bool) or not isinstance(tag_index, int) or tag_index < 0:
            raise TargetCatalogError(f"{context} has an invalid locator tag_index")
        if xpath in seen_xpaths or (tag, tag_index) in seen_locators:
            raise TargetCatalogError("catalog stable locators must be unique")
        seen_xpaths.add(xpath)
        seen_locators.add((tag, tag_index))
        fingerprint = target["fingerprint"]
        if not isinstance(fingerprint, Mapping) or set(fingerprint) != {
            "tag",
            "id",
            "class",
            "text",
            "role",
            "aria_label",
            "visible",
            "bbox_css_xywh",
        }:
            raise TargetCatalogError(f"{context}.fingerprint has invalid fields")
        if fingerprint["tag"] != tag:
            raise TargetCatalogError(f"{context} fingerprint tag disagrees with locator")
        for key in ("id", "class", "text", "role", "aria_label"):
            if not isinstance(fingerprint[key], str):
                raise TargetCatalogError(f"{context}.fingerprint.{key} must be a string")
        if not isinstance(fingerprint["visible"], bool):
            raise TargetCatalogError(f"{context}.fingerprint.visible must be boolean")
        _validate_xywh(
            fingerprint["bbox_css_xywh"],
            f"{context}.fingerprint.bbox",
            allow_zero=True,
        )
        clean_targets.append(json.loads(json.dumps(target, ensure_ascii=False)))

    clean = json.loads(json.dumps(catalog, ensure_ascii=False))
    clean["authorization_regions_css_xyxy"] = regions
    clean["targets"] = clean_targets
    return clean


def model_target_view(catalog: object) -> list[dict]:
    """Return the bounded target table safe to place in the model prompt."""

    clean = validate_target_catalog(catalog)
    parent_groups: dict[str, str] = {}
    output: list[dict] = []
    for target in clean["targets"]:
        parent = _parent_xpath(target["locator"]["frozen_xpath"])
        if parent not in parent_groups:
            parent_groups[parent] = f"g{len(parent_groups) + 1}"
        fingerprint = target["fingerprint"]
        output.append(
            {
                "target_id": target["target_id"],
                "editable": target["editable"],
                "can_be_before_anchor": target["anchor"],
                "sibling_group": parent_groups[parent],
                "tag": fingerprint["tag"],
                "bbox_css_xywh": list(fingerprint["bbox_css_xywh"]),
                "text": fingerprint["text"],
                "id": fingerprint["id"],
                "class": fingerprint["class"],
                "role": fingerprint["role"],
                "aria_label": fingerprint["aria_label"],
                "visible": fingerprint["visible"],
            }
        )
    return output


def model_patch_json_schema(catalog: object) -> dict:
    """Build the strict JSON Schema for short-ID model output."""

    clean = validate_target_catalog(catalog)
    editable_ids = [target["target_id"] for target in clean["targets"] if target["editable"]]
    anchor_ids = [target["target_id"] for target in clean["targets"] if target["anchor"]]
    operation_schemas: list[dict] = []
    if editable_ids:
        operation_schemas.append(
            {
                "type": "object",
                "additionalProperties": False,
                "required": ["op", "target_id", "declarations"],
                "properties": {
                    "op": {"const": "set_style"},
                    "target_id": {"type": "string", "enum": editable_ids},
                    "declarations": {
                        "type": "object",
                        "additionalProperties": False,
                        "minProperties": 1,
                        "maxProperties": len(ALLOWED_STYLE_PROPERTIES),
                        "properties": {
                            prop: {"type": "string", "minLength": 1, "maxLength": 512}
                            for prop in sorted(ALLOWED_STYLE_PROPERTIES)
                        },
                    },
                },
            }
        )
        if anchor_ids:
            operation_schemas.append(
                {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["op", "target_id", "before_id"],
                    "properties": {
                        "op": {"const": "move_element"},
                        "target_id": {"type": "string", "enum": editable_ids},
                        "before_id": {"type": "string", "enum": anchor_ids},
                    },
                }
            )
    item_schema = {"oneOf": operation_schemas} if operation_schemas else {"not": {}}
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["edits"],
        "properties": {
            "edits": {
                "type": "array",
                "maxItems": MAX_EDITS,
                "items": item_schema,
            }
        },
    }


def _target_by_id(clean_catalog: Mapping[str, object]) -> dict[str, dict]:
    return {
        target["target_id"]: target
        for target in clean_catalog["targets"]  # type: ignore[index]
    }


def resolve_model_patch(
    payload: object, catalog: object
) -> tuple[dict, dict[str, dict], list[dict]]:
    """Resolve a strict short-ID patch to source-patch payload and locators."""

    clean = validate_target_catalog(catalog)
    targets = _target_by_id(clean)
    if not isinstance(payload, Mapping) or set(payload) != {"edits"}:
        raise TargetCatalogError("model patch must contain exactly edits")
    edits = payload["edits"]
    if not isinstance(edits, list) or len(edits) > MAX_EDITS:
        raise TargetCatalogError(f"model patch edits must be an array of at most {MAX_EDITS}")

    resolved_edits: list[dict] = []
    resolution_log: list[dict] = []
    seen_target_ids: set[str] = set()
    used_targets: list[dict] = []
    for index, edit in enumerate(edits):
        context = f"edits[{index}]"
        if not isinstance(edit, Mapping):
            raise TargetCatalogError(f"{context} must be an object")
        op = edit.get("op")
        if op == "set_style":
            if set(edit) != {"op", "target_id", "declarations"}:
                raise TargetCatalogError(f"{context} has invalid set_style fields")
            target_id = edit["target_id"]
            if not isinstance(target_id, str) or not _TARGET_ID_RE.fullmatch(target_id):
                raise TargetCatalogError(f"{context}.target_id is invalid")
            target = targets.get(target_id)
            if target is None or not target["editable"]:
                raise TargetCatalogError(f"{context}.target_id is not editable")
            resolved = {
                "op": "set_style",
                "target_xpath": target["locator"]["frozen_xpath"],
                "declarations": edit["declarations"],
            }
            used_targets.append(target)
            log = {
                "op": "set_style",
                "target_id": target_id,
                "target_xpath": target["locator"]["frozen_xpath"],
            }
        elif op == "move_element":
            if set(edit) != {"op", "target_id", "before_id"}:
                raise TargetCatalogError(f"{context} has invalid move_element fields")
            target_id = edit["target_id"]
            before_id = edit["before_id"]
            if not isinstance(target_id, str) or not _TARGET_ID_RE.fullmatch(target_id):
                raise TargetCatalogError(f"{context}.target_id is invalid")
            if not isinstance(before_id, str) or not _TARGET_ID_RE.fullmatch(before_id):
                raise TargetCatalogError(f"{context}.before_id is invalid")
            target = targets.get(target_id)
            before = targets.get(before_id)
            if target is None or not target["editable"]:
                raise TargetCatalogError(f"{context}.target_id is not editable")
            if before is None or not before["anchor"]:
                raise TargetCatalogError(f"{context}.before_id is not an anchor")
            target_xpath = target["locator"]["frozen_xpath"]
            before_xpath = before["locator"]["frozen_xpath"]
            if _parent_xpath(target_xpath) != _parent_xpath(before_xpath):
                raise TargetCatalogError(
                    f"{context} target_id and before_id are not frozen DOM siblings"
                )
            resolved = {
                "op": "move_element",
                "target_xpath": target_xpath,
                "before_xpath": before_xpath,
            }
            used_targets.extend((target, before))
            log = {
                "op": "move_element",
                "target_id": target_id,
                "target_xpath": target_xpath,
                "before_id": before_id,
                "before_xpath": before_xpath,
            }
        else:
            raise TargetCatalogError(f"{context}.op is unsupported: {op!r}")

        if target_id in seen_target_ids:
            raise TargetCatalogError(f"model patch target_id must be unique: {target_id}")
        seen_target_ids.add(target_id)
        resolved_edits.append(resolved)
        resolution_log.append(log)

    source_payload = {"edits": resolved_edits}
    try:
        source_payload = {"edits": validate_patch(source_payload)}
    except StructuredPatchError as error:
        raise TargetCatalogError(f"resolved source patch is invalid: {error}") from error

    locator_by_xpath: dict[str, dict] = {}
    for target in used_targets:
        locator = target["locator"]
        locator_by_xpath[locator["frozen_xpath"]] = {
            "tag": locator["tag"],
            "tag_index": locator["tag_index"],
        }
    return source_payload, locator_by_xpath, resolution_log


def save_target_catalog(path: str | Path, catalog: object) -> None:
    clean = validate_target_catalog(catalog)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(clean, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
        temporary_name = None
    finally:
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass


def load_target_catalog(path: str | Path) -> dict:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TargetCatalogError(f"could not load target catalog: {error}") from error
    return validate_target_catalog(value)


__all__ = [
    "CATALOG_SCHEMA_VERSION",
    "COLLECT_TARGETS_JS",
    "TargetCatalogError",
    "build_target_catalog",
    "canonical_locations_to_css_regions",
    "load_target_catalog",
    "model_patch_json_schema",
    "model_target_view",
    "resolve_model_patch",
    "save_target_catalog",
    "validate_target_catalog",
]
