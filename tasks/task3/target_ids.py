#!/usr/bin/env python3
"""Page-wide, hint-ranked short target IDs for Task 3 repair patches.

This is the versioned successor to :mod:`tasks.task3.target_ids_v1`. Task 2
boxes influence bounded candidate membership and ranking, but they are not an
exclusive spatial authorization gate: a fixed page-wide reserve keeps targets
outside every hint eligible, and only editable catalog IDs form the executor
permission boundary. If hints are absent or malformed, selection falls back
to deterministic whole-page coverage.

The browser collector and element normalizer are intentionally shared with
v1.  Consequently frozen XPath and global tag ordinals retain exactly the
locator convention consumed by ``structured_patch.apply_source_patch``.
Ground truth is neither accepted nor consulted.
"""

from __future__ import annotations

import copy
import json
import math
import os
import re
import tempfile
from collections import Counter, defaultdict
from html.parser import HTMLParser
from pathlib import Path
from typing import Mapping, Sequence

from tasks.task3.render_gate import validate_source_render_gate_audit
from tasks.task3.structured_patch import (
    ALLOWED_STYLE_PROPERTIES,
    MAX_EDITS,
    StructuredPatchError,
    source_element_inventory,
    validate_patch,
)
from tasks.task3.target_ids_v1 import (
    TargetCatalogError,
    _NON_EDITABLE_TAGS,
    _TARGET_ID_RE,
    _candidate_rank,
    _normalize_browser_elements,
    _parent_xpath,
    _validate_xyxy,
    canonical_locations_to_css_regions,
)


CATALOG_SCHEMA_VERSION = "task3-target-catalog-page-wide-v16"
MAX_TARGETS = 256
MAX_EDITABLE_TARGETS = 128
DEFAULT_MAX_TARGETS = 256
DEFAULT_MAX_EDITABLE_TARGETS = 128
DEFAULT_PAGEWIDE_RESERVE = 24
DEFAULT_SIBLING_LIMIT = 16
MAX_COMPUTED_STYLE_VALUE_CHARS = 128
SOURCE_MARKER_ATTRIBUTE = "data-task3-source-locator"
SOURCE_MAPPING_VERSION = "source-opening-tag-marker-v3"
CHARSET_PREFIX_GUARD_VERSION = "preserve-early-meta-byte-position-v1"
SELECTION_STRATEGY = "pagewide-coverage-style-anomaly-salience-source-mapped-v6"
_SOURCE_MARKER_RE = re.compile(r"([a-z][a-z0-9:_-]*):([0-9]+)\Z")

_EDITABLE_SELECTION_REASONS = frozenset(
    {
        "inline_style",
        "style_anomaly",
        "visual_salience",
        "pagewide_coverage",
    }
)
_SELECTION_REASONS = _EDITABLE_SELECTION_REASONS | {"same_parent_anchor"}


# This repeats v1's locator computation verbatim, then adds bounded context
# required by the page-wide prompt.  Exact source locators stay internal.
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

  const styleProperties = [
    'border-radius', 'color', 'font-family', 'font-size',
    'font-weight', 'text-align', 'transform'
  ];
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
    // A form control whose name/id is e.g. ``style`` or ``id`` can shadow
    // the convenience properties on its containing form.  Read attributes
    // through Element.prototype so hostile/odd-but-valid fixture HTML cannot
    // change the collector's types.
    const attribute = (name) =>
        String(Element.prototype.getAttribute.call(node, name) || '');
    const rawInlineCss = attribute('style');
    const inlineParser = document.createElement('span');
    if (rawInlineCss) inlineParser.setAttribute('style', rawInlineCss);
    const visible = !(style.display === 'none' || style.visibility === 'hidden' ||
        Number.parseFloat(style.opacity) === 0 || rect.width < 0.5 || rect.height < 0.5);
    let directText = '';
    for (const child of node.childNodes) {
      if (child.nodeType === Node.TEXT_NODE) directText += child.textContent || '';
    }
    // iframe fallback markup is raw text, so remove temporary marker literals.
    directText = directText
        .replace(/\sdata-task3-source-locator="[^"]*"/g, '')
        .replace(/\s+/g, ' ').trim().slice(0, 120);
    const computedStyle = {};
    const inlineStyle = {};
    for (const property of styleProperties) {
      computedStyle[property] = String(
          style.getPropertyValue(property) || '').slice(0, 256);
      const inlineValue = String(
          inlineParser.style.getPropertyValue(property) || '');
      if (inlineValue) inlineStyle[property] = inlineValue.slice(0, 256);
    }
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
      id: attribute('id').slice(0, 120),
      cls: attribute('class').slice(0, 200),
      text: directText,
      role: attribute('role').slice(0, 80),
      aria_label: attribute('aria-label').slice(0, 120),
      source_marker: attribute('data-task3-source-locator').slice(0, 160),
      computed_style: computedStyle,
      inline_style: inlineStyle,
      child_element_count: node.childElementCount,
      depth: xpathOf(node).split('/').length - 1,
    });
  }
  return {elements};
}
"""


class _MetaCharsetOpeningTagProbe(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self.declares_charset = False

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        if tag.lower() != "meta":
            return
        values = {
            key.lower(): value.strip() if isinstance(value, str) else ""
            for key, value in attrs
        }
        if values.get("charset", "").strip(" \t\r\n\f\"'"):
            self.declares_charset = True
            return
        if values.get("http-equiv", "").lower() == "content-type" and re.search(
            r"(?i)(?:^|;)\s*charset\s*=\s*([^;\s]+)",
            values.get("content", ""),
        ):
            self.declares_charset = True


def _early_charset_declaration_guard_end(
    raw_html: str, inventory: Sequence[Mapping[str, object]]
) -> int:
    """Keep the first browser-relevant 1024-byte charset declaration fixed."""

    for record in inventory:
        if record["tag"] != "meta":
            continue
        start = int(record["opening_tag_start"])
        end = int(record["opening_tag_end"])
        # Browser charset prescan is byte bounded, while the source inventory
        # uses character offsets. Source is strict UTF-8, so convert only the
        # handful of early meta offsets explicitly.
        if len(raw_html[:end].encode("utf-8")) > 1024:
            continue
        probe = _MetaCharsetOpeningTagProbe()
        probe.feed(raw_html[start:end])
        probe.close()
        if probe.declares_charset:
            return end
    return 0


def build_source_marker_transport(raw_html: str) -> dict:
    """Instrument a temporary rendering copy with source-safe locators.

    Every source opening tag after any early charset declaration receives a
    marker whose value is the executor's
    exact ``tag:global-ordinal``.  Chromium carries the marker with the source
    element even when it inserts ``tbody`` or reparents malformed markup.
    Tags ending at or before an early (first 1024-byte) charset declaration
    are deliberately left unmarked so instrumentation cannot move that
    declaration outside Chromium's prescan window;
    browser-created elements have no marker and can never become targets.
    The original string is not mutated and the marker attribute is rejected if
    the page already uses it.
    """

    if not isinstance(raw_html, str):
        raise TargetCatalogError("raw_html must be a string")
    # Reject even selector/comment/string occurrences, not only an existing
    # opening-tag attribute.  This conservative rule prevents the temporary
    # marker from activating page CSS such as ``[data-task3-source-locator]``.
    if SOURCE_MARKER_ATTRIBUTE.lower() in raw_html.lower():
        raise TargetCatalogError(
            f"source already references reserved attribute {SOURCE_MARKER_ATTRIBUTE}"
        )
    try:
        inventory = source_element_inventory(raw_html)
    except StructuredPatchError as error:
        raise TargetCatalogError(f"cannot inventory source HTML: {error}") from error

    marker_by_locator: dict[tuple[str, int], str] = {}
    marker_map: dict[str, dict] = {}
    for record in inventory:
        marker = f"{record['tag']}:{record['tag_index']}"
        locator = (record["tag"], record["tag_index"])
        if locator in marker_by_locator or marker in marker_map:
            raise TargetCatalogError("source locator inventory is not unique")
        marker_by_locator[locator] = marker
        marker_map[marker] = {
            "source_xpath": record["source_xpath"],
            "tag": record["tag"],
            "tag_index": record["tag_index"],
            "parent_marker": None,
        }
    for record in inventory:
        marker = marker_by_locator[(record["tag"], record["tag_index"])]
        parent = record["parent_locator"]
        if parent is not None:
            marker_map[marker]["parent_marker"] = marker_by_locator[
                (parent["tag"], parent["tag_index"])
            ]

    charset_guard_end = _early_charset_declaration_guard_end(raw_html, inventory)
    attribute_pattern = re.compile(
        rf"(?i)(?:\s){re.escape(SOURCE_MARKER_ATTRIBUTE)}(?=\s|=|/?>)"
    )
    insertions: list[tuple[int, str]] = []
    n_charset_prefix_skipped = 0
    for record in inventory:
        start = record["opening_tag_start"]
        end = record["opening_tag_end"]
        opening_tag = raw_html[start:end]
        if attribute_pattern.search(opening_tag):
            raise TargetCatalogError(
                f"source already contains reserved attribute {SOURCE_MARKER_ATTRIBUTE}"
            )
        if start < charset_guard_end:
            n_charset_prefix_skipped += 1
            continue
        close = opening_tag.rfind(">")
        if close < 0:
            raise TargetCatalogError("source inventory contains an unterminated opening tag")
        insertion = close
        probe = close - 1
        while probe >= 0 and opening_tag[probe].isspace():
            probe -= 1
        if probe >= 0 and opening_tag[probe] == "/":
            insertion = probe
        absolute = start + insertion
        marker = marker_by_locator[(record["tag"], record["tag_index"])]
        separator = "" if absolute > 0 and raw_html[absolute - 1].isspace() else " "
        insertions.append(
            (
                absolute,
                f'{separator}{SOURCE_MARKER_ATTRIBUTE}="{marker}"',
            )
        )

    source_override = raw_html
    for position, insertion in sorted(insertions, reverse=True):
        source_override = (
            source_override[:position] + insertion + source_override[position:]
        )
    audit = {
        "mapping_version": SOURCE_MAPPING_VERSION,
        "marker_attribute": SOURCE_MARKER_ATTRIBUTE,
        "n_source_elements": len(marker_map),
        "charset_prefix_guard_version": CHARSET_PREFIX_GUARD_VERSION,
        "charset_declaration_guard_end_character_offset": charset_guard_end,
        "n_charset_prefix_source_elements_skipped": n_charset_prefix_skipped,
        "n_source_elements_instrumented": len(insertions),
    }
    return {
        "source_override": source_override,
        "source_marker_map": marker_map,
        "instrumentation_audit": audit,
    }


def _bounded_prompt_text(value: object, limit: int) -> str:
    if not isinstance(value, str):
        return ""
    # Collapse whitespace/control characters so one DOM node cannot expand or
    # structurally confuse the JSON-delimited prompt payload.
    return " ".join(value.split())[:limit]


def _normalize_v2_browser_elements(
    browser_elements: Sequence[Mapping[str, object]],
) -> list[dict]:
    """Extend v1 locator normalization with bounded model context."""

    normalized = _normalize_browser_elements(browser_elements)
    raw_by_xpath = {
        element.get("xpath"): element
        for element in browser_elements
        if isinstance(element, Mapping)
    }
    output: list[dict] = []
    for element in normalized:
        raw = raw_by_xpath[element["xpath"]]
        raw_style = raw.get("computed_style", {})
        if raw_style is None:
            raw_style = {}
        if not isinstance(raw_style, Mapping):
            raise TargetCatalogError(
                f"browser element {element['xpath']} computed_style must be an object"
            )
        clean = dict(element)
        clean["computed_style"] = {
            prop: _bounded_prompt_text(
                raw_style.get(prop), MAX_COMPUTED_STYLE_VALUE_CHARS
            )
            for prop in sorted(ALLOWED_STYLE_PROPERTIES)
        }
        raw_inline_style = raw.get("inline_style", {})
        if raw_inline_style is None:
            raw_inline_style = {}
        if not isinstance(raw_inline_style, Mapping):
            raise TargetCatalogError(
                f"browser element {element['xpath']} inline_style must be an object"
            )
        if any(prop not in ALLOWED_STYLE_PROPERTIES for prop in raw_inline_style):
            raise TargetCatalogError(
                f"browser element {element['xpath']} inline_style has unsupported property"
            )
        clean["inline_style"] = {
            prop: _bounded_prompt_text(value, MAX_COMPUTED_STYLE_VALUE_CHARS)
            for prop, value in sorted(raw_inline_style.items())
            if _bounded_prompt_text(value, MAX_COMPUTED_STYLE_VALUE_CHARS)
        }
        raw_marker = raw.get("source_marker", "")
        if not isinstance(raw_marker, str):
            raise TargetCatalogError(
                f"browser element {element['xpath']} source_marker must be a string"
            )
        clean["source_marker"] = raw_marker[:160]
        child_count = raw.get("child_element_count", 0)
        depth = raw.get("depth", element["xpath"].count("/"))
        if (
            isinstance(child_count, bool)
            or not isinstance(child_count, int)
            or child_count < 0
            or isinstance(depth, bool)
            or not isinstance(depth, int)
            or depth < 1
        ):
            raise TargetCatalogError(
                f"browser element {element['xpath']} has invalid DOM salience fields"
            )
        clean["child_element_count"] = child_count
        clean["depth"] = depth
        output.append(clean)
    return output


def _bounded_non_negative_int(value: object, name: str, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TargetCatalogError(f"{name} must be a non-negative integer")
    if value > maximum:
        raise TargetCatalogError(f"{name} cannot exceed protocol limit {maximum}")
    return value


def _sanitize_ranking_hints(value: object) -> tuple[list[list[float]], int]:
    """Return valid boxes and an audit count instead of failing on bad hints."""

    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return [], 1 if value is not None else 0
    valid: list[list[float]] = []
    invalid_count = 0
    for index, hint in enumerate(value):
        try:
            valid.append(_validate_xyxy(hint, f"ranking_hints_css_xyxy[{index}]"))
        except (TargetCatalogError, TypeError, ValueError):
            invalid_count += 1
    return valid, invalid_count


def _validate_source_marker_inputs(
    marker_map_value: object, audit_value: object
) -> tuple[dict[str, dict], dict]:
    if not isinstance(marker_map_value, Mapping):
        raise TargetCatalogError("source_marker_map must be an object")
    marker_map: dict[str, dict] = {}
    seen_xpaths: set[str] = set()
    seen_locators: set[tuple[str, int]] = set()
    for marker, record in marker_map_value.items():
        if not isinstance(marker, str) or _SOURCE_MARKER_RE.fullmatch(marker) is None:
            raise TargetCatalogError("source_marker_map has an invalid marker key")
        if not isinstance(record, Mapping) or set(record) != {
            "source_xpath",
            "tag",
            "tag_index",
            "parent_marker",
        }:
            raise TargetCatalogError(f"source marker {marker!r} has invalid fields")
        tag = record["tag"]
        tag_index = record["tag_index"]
        source_xpath = record["source_xpath"]
        parent_marker = record["parent_marker"]
        match = _SOURCE_MARKER_RE.fullmatch(marker)
        assert match is not None
        if (
            not isinstance(tag, str)
            or match.group(1) != tag
            or isinstance(tag_index, bool)
            or not isinstance(tag_index, int)
            or int(match.group(2)) != tag_index
            or not isinstance(source_xpath, str)
            or not source_xpath.startswith("/")
            or not source_xpath
            or len(source_xpath) > 4096
            or (parent_marker is not None and not isinstance(parent_marker, str))
        ):
            raise TargetCatalogError(f"source marker {marker!r} is inconsistent")
        locator = (tag, tag_index)
        if source_xpath in seen_xpaths or locator in seen_locators:
            raise TargetCatalogError("source_marker_map locators are not unique")
        seen_xpaths.add(source_xpath)
        seen_locators.add(locator)
        marker_map[marker] = {
            "source_xpath": source_xpath,
            "tag": tag,
            "tag_index": tag_index,
            "parent_marker": parent_marker,
        }
    if any(
        record["parent_marker"] is not None
        and record["parent_marker"] not in marker_map
        for record in marker_map.values()
    ):
        raise TargetCatalogError("source_marker_map references an unknown parent marker")

    expected_audit_fields = {
        "mapping_version",
        "marker_attribute",
        "n_source_elements",
        "charset_prefix_guard_version",
        "charset_declaration_guard_end_character_offset",
        "n_charset_prefix_source_elements_skipped",
        "n_source_elements_instrumented",
    }
    if not isinstance(audit_value, Mapping) or set(audit_value) != expected_audit_fields:
        raise TargetCatalogError("source_marker_audit has invalid fields")
    if (
        audit_value["mapping_version"] != SOURCE_MAPPING_VERSION
        or audit_value["marker_attribute"] != SOURCE_MARKER_ATTRIBUTE
        or audit_value["n_source_elements"] != len(marker_map)
        or audit_value["charset_prefix_guard_version"]
        != CHARSET_PREFIX_GUARD_VERSION
        or type(audit_value["charset_declaration_guard_end_character_offset"])
        is not int
        or audit_value["charset_declaration_guard_end_character_offset"] < 0
        or type(audit_value["n_charset_prefix_source_elements_skipped"]) is not int
        or audit_value["n_charset_prefix_source_elements_skipped"] < 0
        or type(audit_value["n_source_elements_instrumented"]) is not int
        or audit_value["n_source_elements_instrumented"] < 0
        or audit_value["n_charset_prefix_source_elements_skipped"]
        + audit_value["n_source_elements_instrumented"]
        != len(marker_map)
    ):
        raise TargetCatalogError("source_marker_audit disagrees with marker map")
    return marker_map, json.loads(json.dumps(audit_value))


def _coverage_sample(indices: Sequence[int], limit: int) -> list[int]:
    """Choose deterministic document-order quantiles, including both ends."""

    if limit <= 0 or not indices:
        return []
    if limit >= len(indices):
        return list(indices)
    if limit == 1:
        return [indices[(len(indices) - 1) // 2]]
    last = len(indices) - 1
    denominator = limit - 1
    # Half-up integer rounding avoids Python's banker-rounding surprises.
    positions = [
        (2 * slot * last + denominator) // (2 * denominator)
        for slot in range(limit)
    ]
    return [indices[position] for position in positions]


def _best_hint_rank(element: Mapping[str, object], hints: Sequence[Sequence[float]]) -> tuple:
    ranked = [
        (
            _candidate_rank(
                element["bbox_css_xyxy"],  # type: ignore[arg-type]
                hint,
                element["document_index"],  # type: ignore[arg-type]
            ),
            hint_index,
        )
        for hint_index, hint in enumerate(hints)
    ]
    return min(ranked)


def _balanced_hint_indices(
    elements: Sequence[Mapping[str, object]],
    eligible_indices: Sequence[int],
    hints: Sequence[Sequence[float]],
    limit: int,
) -> tuple[list[int], dict[int, int]]:
    """Round-robin nearest elements so one large region cannot starve another."""

    if limit <= 0 or not hints:
        return [], {}
    ranked_by_hint = [
        sorted(
            eligible_indices,
            key=lambda index: _candidate_rank(
                elements[index]["bbox_css_xyxy"],  # type: ignore[arg-type]
                hint,
                elements[index]["document_index"],  # type: ignore[arg-type]
            ),
        )
        for hint in hints
    ]
    cursors = [0] * len(hints)
    selected: list[int] = []
    selected_set: set[int] = set()
    selected_hint: dict[int, int] = {}
    while len(selected) < limit:
        progressed = False
        for hint_index, ranked in enumerate(ranked_by_hint):
            while cursors[hint_index] < len(ranked):
                index = ranked[cursors[hint_index]]
                cursors[hint_index] += 1
                if index in selected_set:
                    continue
                selected.append(index)
                selected_set.add(index)
                selected_hint[index] = hint_index
                progressed = True
                break
            if len(selected) >= limit:
                break
        if not progressed:
            break
    return selected, selected_hint


def _style_salience(
    elements: Sequence[Mapping[str, object]], eligible_indices: Sequence[int]
) -> tuple[dict[int, float], dict[int, int]]:
    """Rank suspicious/style-relevant DOM nodes without consulting GT.

    The signal favors explicit editable inline declarations, non-default
    transforms, and computed-style outliers among same-class or same-parent
    peers.  Semantic headings and leaf/text nodes provide a deterministic
    fallback for DOM-order repairs.
    """

    scores = {index: 0.0 for index in eligible_indices}
    anomaly_counts = {index: 0 for index in eligible_indices}
    groups: list[tuple[float, list[int]]] = []
    by_parent_tag: defaultdict[tuple[str, str], list[int]] = defaultdict(list)
    by_class_tag: defaultdict[tuple[str, str], list[int]] = defaultdict(list)
    by_tag: defaultdict[str, list[int]] = defaultdict(list)
    for index in eligible_indices:
        element = elements[index]
        tag = str(element["tag"])
        by_parent_tag[(str(element["parent_xpath"]), tag)].append(index)
        class_name = " ".join(str(element["class"]).split())
        if class_name:
            by_class_tag[(class_name, tag)].append(index)
        by_tag[tag].append(index)
    groups.extend((70.0, indices) for indices in by_class_tag.values())
    groups.extend((55.0, indices) for indices in by_parent_tag.values())

    for weight, indices in groups:
        if len(indices) < 3:
            continue
        for prop in ALLOWED_STYLE_PROPERTIES:
            counts = Counter(
                str(elements[index]["computed_style"].get(prop, ""))
                for index in indices
            )
            mode_count = max(counts.values(), default=0)
            if mode_count < 2:
                continue
            for index in indices:
                value = str(elements[index]["computed_style"].get(prop, ""))
                value_count = counts[value]
                if value_count < mode_count:
                    scores[index] += weight * (mode_count - value_count) / mode_count
                    anomaly_counts[index] += 1

    # A lone class signature among otherwise repeated same-tag siblings is a
    # strong local style signal even when the final computed values differ
    # only slightly.  Keep this weaker than computed-style peer disagreement
    # and require a repeated local mode so heterogeneous navigation lists do
    # not receive arbitrary rarity bonuses.
    for indices in by_parent_tag.values():
        if len(indices) < 3:
            continue
        class_counts = Counter(
            " ".join(str(elements[index]["class"]).split()) for index in indices
        )
        mode_count = max(class_counts.values(), default=0)
        if mode_count < 2:
            continue
        for index in indices:
            class_name = " ".join(str(elements[index]["class"]).split())
            value_count = class_counts[class_name]
            if value_count < mode_count:
                scores[index] += 50.0 * (mode_count - value_count) / mode_count
                anomaly_counts[index] += 1

    # Tag-level rarity is weaker than local peer disagreement but helps when
    # a peer group is distributed across different containers.
    for indices in by_tag.values():
        if len(indices) < 3:
            continue
        for prop in ALLOWED_STYLE_PROPERTIES:
            counts = Counter(
                str(elements[index]["computed_style"].get(prop, ""))
                for index in indices
            )
            group_size = len(indices)
            for index in indices:
                value = str(elements[index]["computed_style"].get(prop, ""))
                frequency = counts[value]
                scores[index] += min(
                    24.0,
                    6.0 * math.log2((group_size + 1.0) / (frequency + 1.0)),
                )

    semantic_tags = {
        # Headings, labels, and table headers are disproportionately likely
        # to carry reading-order defects.  These are generic DOM semantics,
        # not benchmark or ground-truth identifiers.
        "h1": 220.0,
        "h2": 190.0,
        "h3": 150.0,
        "h4": 120.0,
        "h5": 105.0,
        "h6": 105.0,
        "label": 160.0,
        "th": 110.0,
        "button": 20.0,
    }
    inline_property_weights = {
        # A visually wrong font or transform can be a lone inline outlier in
        # a very large page.  Count those declarations more strongly than a
        # generic inline declaration so they survive the bounded catalog.
        "font-family": 240.0,
        "transform": 240.0,
        "font-size": 30.0,
        "font-weight": 30.0,
        "line-height": 30.0,
        "text-align": 30.0,
    }
    inline_value_counts = {
        prop: Counter(
            str(elements[index]["inline_style"][prop])
            for index in eligible_indices
            if prop in elements[index]["inline_style"]
        )
        for prop in ALLOWED_STYLE_PROPERTIES
    }
    for index in eligible_indices:
        element = elements[index]
        inline_style = element["inline_style"]
        scores[index] += 45.0 * len(inline_style)
        scores[index] += sum(
            inline_property_weights.get(str(prop), 0.0)
            for prop in inline_style
        )
        # Synthetic style defects usually create a value that is rare among
        # the page's explicit declarations.  Page-local rarity is a generic,
        # GT-blind signal and prevents a lone anomalous declaration from being
        # drowned out by thousands of ordinary DOM nodes.
        scores[index] += sum(
            220.0 / math.sqrt(inline_value_counts[prop][str(value)])
            for prop, value in inline_style.items()
            if prop in inline_value_counts
            and inline_value_counts[prop][str(value)] > 0
        )
        transform = str(element["computed_style"].get("transform", "")).strip().lower()
        if transform and transform not in {"none", "matrix(1, 0, 0, 1, 0, 0)"}:
            scores[index] += 100.0
        scores[index] += semantic_tags.get(str(element["tag"]), 0.0)
        if element["text"]:
            scores[index] += 8.0
        if element["child_element_count"] == 0:
            scores[index] += 5.0
            if element["text"]:
                # Text-bearing leaves are the smallest safe units for both
                # style edits and semantic moves.  Area only breaks ties
                # among visible leaves; it does not promote large containers.
                width = max(0.0, float(element["bbox_css_xywh"][2]))
                height = max(0.0, float(element["bbox_css_xywh"][3]))
                scores[index] += min(
                    120.0,
                    20.0 * math.log2(1.0 + (width * height) / 1000.0),
                )
                font_size_match = re.fullmatch(
                    r"([0-9]+(?:\.[0-9]+)?)px",
                    str(element["computed_style"].get("font-size", "")).strip(),
                )
                if font_size_match:
                    font_size = float(font_size_match.group(1))
                    scores[index] += min(80.0, max(0.0, font_size - 16.0) * 5.0)
                font_weight = str(
                    element["computed_style"].get("font-weight", "")
                ).strip()
                if font_weight.isdigit() and int(font_weight) >= 600:
                    scores[index] += 30.0
        scores[index] += min(float(element["depth"]), 20.0) * 0.5
    return scores, anomaly_counts


def build_pagewide_target_catalog(
    browser_elements: Sequence[Mapping[str, object]] | Mapping[str, object],
    ranking_hints_css_xyxy: object = (),
    *,
    source_marker_map: object,
    source_marker_audit: object,
    source_render_gate_audit: object,
    pagewide_reserve: int = DEFAULT_PAGEWIDE_RESERVE,
    sibling_limit: int = DEFAULT_SIBLING_LIMIT,
    max_editable_targets: int = DEFAULT_MAX_EDITABLE_TARGETS,
    max_targets: int = DEFAULT_MAX_TARGETS,
) -> dict:
    """Build a bounded diagnosis-independent catalog from the visible page."""

    if isinstance(browser_elements, Mapping):
        if set(browser_elements) != {"elements"}:
            raise TargetCatalogError("browser extraction must contain exactly elements")
        browser_elements = browser_elements["elements"]  # type: ignore[assignment]
    elements = _normalize_v2_browser_elements(browser_elements)  # type: ignore[arg-type]
    hints: list[list[float]] = []
    invalid_hint_count = 0
    source_markers, clean_source_audit = _validate_source_marker_inputs(
        source_marker_map, source_marker_audit
    )
    try:
        clean_source_render_gate = validate_source_render_gate_audit(
            source_render_gate_audit
        )
    except Exception as error:  # noqa: BLE001
        raise TargetCatalogError(f"source render gate audit is invalid: {error}") from error

    source_binding_by_index: dict[int, dict] = {}
    observed_marker_counts: Counter[str] = Counter()
    for index, element in enumerate(elements):
        marker = element["source_marker"]
        if not marker:
            continue
        source_record = source_markers.get(marker)
        if source_record is None or source_record["tag"] != element["tag"]:
            raise TargetCatalogError(
                f"browser source marker {marker!r} has no tag-matching source element"
            )
        # Invalid source nesting can make Chromium's HTML parser create more
        # than one DOM element from a single source opening tag (the adoption
        # agency algorithm is a common example).  Every clone still resolves
        # to exactly one source element, so retain each browser geometry and
        # expose their shared-source relationship to the model.
        observed_marker_counts[marker] += 1
        source_binding_by_index[index] = source_record

    max_targets = _bounded_non_negative_int(max_targets, "max_targets", MAX_TARGETS)
    max_editable_targets = _bounded_non_negative_int(
        max_editable_targets, "max_editable_targets", MAX_EDITABLE_TARGETS
    )
    pagewide_reserve = _bounded_non_negative_int(
        pagewide_reserve, "pagewide_reserve", MAX_EDITABLE_TARGETS
    )
    sibling_limit = _bounded_non_negative_int(
        sibling_limit, "sibling_limit", MAX_TARGETS
    )
    if max_editable_targets > max_targets:
        raise TargetCatalogError("max_editable_targets cannot exceed max_targets")

    eligible_indices = [
        index
        for index, element in enumerate(elements)
        if index in source_binding_by_index
        and element["visible"]
        and element["tag"] not in _NON_EDITABLE_TAGS
        and element["bbox_css_xywh"][2] >= 0.5
        and element["bbox_css_xywh"][3] >= 0.5
    ]
    capacity = min(max_editable_targets, len(eligible_indices))
    editable_indices: set[int] = set()
    selection_reason: dict[int, str] = {}

    reserve_count = min(pagewide_reserve, capacity)
    for index in _coverage_sample(eligible_indices, reserve_count):
        editable_indices.add(index)
        selection_reason[index] = "pagewide_coverage"

    salience, anomaly_counts = _style_salience(elements, eligible_indices)
    for index in sorted(
        eligible_indices,
        key=lambda value: (
            -salience[value],
            elements[value]["document_index"],
        ),
    ):
        if len(editable_indices) >= capacity:
            break
        if index in editable_indices:
            continue
        editable_indices.add(index)
        if elements[index]["inline_style"]:
            selection_reason[index] = "inline_style"
        elif anomaly_counts[index]:
            selection_reason[index] = "style_anomaly"
        else:
            selection_reason[index] = "visual_salience"

    by_parent: dict[str | None, list[int]] = {}
    for index, source_record in source_binding_by_index.items():
        by_parent.setdefault(source_record["parent_marker"], []).append(index)

    # Anchor priority is independent of hint authorization.  Small sibling
    # groups are cheapest to make completely expressible, so after the one-
    # anchor-per-isolated-editable phase they precede broad container groups.
    anchor_priority: dict[int, tuple[int, int, int, int, int]] = {}
    for editable_order, editable_index in enumerate(
        sorted(editable_indices, key=lambda index: elements[index]["document_index"])
    ):
        siblings = by_parent[source_binding_by_index[editable_index]["parent_marker"]]
        position = siblings.index(editable_index)
        already_has_editable_anchor = any(
            index != editable_index and index in editable_indices for index in siblings
        )
        ranked_siblings = sorted(
            (index for index in siblings if index != editable_index),
            key=lambda index: (
                abs(siblings.index(index) - position),
                elements[index]["document_index"],
            ),
        )
        for sibling_rank, sibling in enumerate(ranked_siblings[:sibling_limit]):
            # Phase zero guarantees one same-parent anchor for every editable
            # target that does not already have an editable sibling in the
            # catalog.  Extra contextual siblings are filled only afterwards.
            phase = (
                0
                if sibling_rank == 0 and not already_has_editable_anchor
                else 1
            )
            priority = (
                phase,
                len(siblings),
                abs(siblings.index(sibling) - position),
                editable_order,
                elements[sibling]["document_index"],
            )
            if sibling not in anchor_priority or priority < anchor_priority[sibling]:
                anchor_priority[sibling] = priority

    candidate_indices = set(editable_indices)
    for index, _ in sorted(anchor_priority.items(), key=lambda item: item[1]):
        if len(candidate_indices) >= max_targets:
            break
        candidate_indices.add(index)

    ordered_indices = sorted(
        candidate_indices, key=lambda index: elements[index]["document_index"]
    )
    targets: list[dict] = []
    for ordinal, element_index in enumerate(ordered_indices, 1):
        element = elements[element_index]
        source_record = source_binding_by_index[element_index]
        editable = element_index in editable_indices
        targets.append(
            {
                "target_id": f"t{ordinal}",
                "editable": editable,
                "anchor": True,
                "selection_reason": selection_reason.get(
                    element_index, "same_parent_anchor"
                ),
                "ranking_hint_indices": [],
                "document_index": element["document_index"],
                "locator": {
                    "frozen_xpath": element["xpath"],
                    "tag": source_record["tag"],
                    "tag_index": source_record["tag_index"],
                },
                "source_mapping": {
                    "marker": element["source_marker"],
                    "source_xpath": source_record["source_xpath"],
                    "source_parent_marker": source_record["parent_marker"],
                    "browser_alias_count": observed_marker_counts[
                        element["source_marker"]
                    ],
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
                    "computed_style": element["computed_style"],
                    "inline_style": element["inline_style"],
                    "child_element_count": element["child_element_count"],
                    "depth": element["depth"],
                },
            }
        )

    catalog = {
        "schema_version": CATALOG_SCHEMA_VERSION,
        "coordinate_space": "page_css_xyxy",
        "ranking_hints_css_xyxy": hints,
        "source_mappability": {
            **clean_source_audit,
            "source_render_gate": clean_source_render_gate,
            "observed_source_markers_uniquely_mapped": True,
            "browser_parser_duplicate_source_bindings_supported": True,
            "unmarked_browser_nodes_never_selectable": True,
            "all_catalog_targets_source_mapped": True,
            "n_browser_elements": len(elements),
            "n_browser_elements_with_marker": len(source_binding_by_index),
            "n_browser_elements_without_marker": (
                len(elements) - len(source_binding_by_index)
            ),
            "n_unique_observed_source_markers": len(observed_marker_counts),
            "n_browser_elements_sharing_source_marker": sum(
                count - 1 for count in observed_marker_counts.values()
            ),
            "max_browser_elements_per_source_marker": max(
                observed_marker_counts.values(), default=0
            ),
            "n_source_elements_unobserved": (
                len(source_markers) - len(observed_marker_counts)
            ),
        },
        "selection_policy": {
            "ground_truth_used": False,
            "hint_membership_effect": "none_diagnosis_not_catalog_input",
            "spatial_authorization_rule": "editable_catalog_id_only",
            "outside_hints_can_be_editable": True,
            "invalid_hint_count": invalid_hint_count,
            "pagewide_fallback_used": True,
            "pagewide_reserve": pagewide_reserve,
            "sibling_limit": sibling_limit,
            "max_editable_targets": max_editable_targets,
            "max_targets": max_targets,
            "id_order": "browser_document_order",
            "editable_selection": (
                SELECTION_STRATEGY
            ),
        },
        "targets": targets,
    }
    return validate_pagewide_target_catalog(catalog)


# A concise alias eases integration while the module name keeps v1/v2 explicit.
build_target_catalog = build_pagewide_target_catalog


def validate_pagewide_target_catalog(catalog: object) -> dict:
    """Strictly validate a page-wide v2 catalog loaded from memory or disk."""

    if not isinstance(catalog, Mapping):
        raise TargetCatalogError("target catalog must be an object")
    expected_top = {
        "schema_version",
        "coordinate_space",
        "ranking_hints_css_xyxy",
        "source_mappability",
        "selection_policy",
        "targets",
    }
    if set(catalog) != expected_top:
        raise TargetCatalogError("target catalog has unexpected or missing fields")
    if catalog["schema_version"] != CATALOG_SCHEMA_VERSION:
        raise TargetCatalogError("unsupported target catalog schema_version")
    if catalog["coordinate_space"] != "page_css_xyxy":
        raise TargetCatalogError("unsupported target catalog coordinate_space")

    hints_value = catalog["ranking_hints_css_xyxy"]
    if not isinstance(hints_value, list):
        raise TargetCatalogError("catalog ranking hints must be an array")
    hints = [
        _validate_xyxy(hint, f"ranking_hints_css_xyxy[{index}]")
        for index, hint in enumerate(hints_value)
    ]

    source_mappability = catalog["source_mappability"]
    expected_source_mappability = {
        "mapping_version",
        "marker_attribute",
        "n_source_elements",
        "charset_prefix_guard_version",
        "charset_declaration_guard_end_character_offset",
        "n_charset_prefix_source_elements_skipped",
        "n_source_elements_instrumented",
        "n_browser_elements",
        "n_browser_elements_with_marker",
        "n_browser_elements_without_marker",
        "n_unique_observed_source_markers",
        "n_browser_elements_sharing_source_marker",
        "max_browser_elements_per_source_marker",
        "n_source_elements_unobserved",
        "source_render_gate",
        "observed_source_markers_uniquely_mapped",
        "browser_parser_duplicate_source_bindings_supported",
        "unmarked_browser_nodes_never_selectable",
        "all_catalog_targets_source_mapped",
    }
    if (
        not isinstance(source_mappability, Mapping)
        or set(source_mappability) != expected_source_mappability
        or source_mappability["mapping_version"] != SOURCE_MAPPING_VERSION
        or source_mappability["marker_attribute"] != SOURCE_MARKER_ATTRIBUTE
        or source_mappability["charset_prefix_guard_version"]
        != CHARSET_PREFIX_GUARD_VERSION
        or source_mappability["observed_source_markers_uniquely_mapped"] is not True
        or source_mappability[
            "browser_parser_duplicate_source_bindings_supported"
        ]
        is not True
        or source_mappability["unmarked_browser_nodes_never_selectable"] is not True
        or source_mappability["all_catalog_targets_source_mapped"] is not True
    ):
        raise TargetCatalogError("catalog source_mappability has invalid fields")
    source_counts = {}
    for key in (
        "n_source_elements",
        "n_browser_elements",
        "n_browser_elements_with_marker",
        "n_browser_elements_without_marker",
        "n_unique_observed_source_markers",
        "n_browser_elements_sharing_source_marker",
        "max_browser_elements_per_source_marker",
        "n_source_elements_unobserved",
        "charset_declaration_guard_end_character_offset",
        "n_charset_prefix_source_elements_skipped",
        "n_source_elements_instrumented",
    ):
        source_counts[key] = _bounded_non_negative_int(
            source_mappability[key], key, 10_000_000
        )
    if (
        source_counts["n_browser_elements"]
        != source_counts["n_browser_elements_with_marker"]
        + source_counts["n_browser_elements_without_marker"]
        or source_counts["n_browser_elements_with_marker"]
        != source_counts["n_unique_observed_source_markers"]
        + source_counts["n_browser_elements_sharing_source_marker"]
        or source_counts["n_source_elements"]
        != source_counts["n_unique_observed_source_markers"]
        + source_counts["n_source_elements_unobserved"]
        or source_counts["n_source_elements"]
        != source_counts["n_charset_prefix_source_elements_skipped"]
        + source_counts["n_source_elements_instrumented"]
        or (
            source_counts["n_unique_observed_source_markers"] == 0
            and source_counts["max_browser_elements_per_source_marker"] != 0
        )
        or (
            source_counts["n_unique_observed_source_markers"] > 0
            and not 1
            <= source_counts["max_browser_elements_per_source_marker"]
            <= source_counts["n_browser_elements_with_marker"]
        )
    ):
        raise TargetCatalogError("catalog source mappability counts are inconsistent")
    try:
        validate_source_render_gate_audit(source_mappability["source_render_gate"])
    except Exception as error:  # noqa: BLE001
        raise TargetCatalogError(f"catalog source render gate is invalid: {error}") from error

    policy = catalog["selection_policy"]
    expected_policy = {
        "ground_truth_used",
        "hint_membership_effect",
        "spatial_authorization_rule",
        "outside_hints_can_be_editable",
        "invalid_hint_count",
        "pagewide_fallback_used",
        "pagewide_reserve",
        "sibling_limit",
        "max_editable_targets",
        "max_targets",
        "id_order",
        "editable_selection",
    }
    if not isinstance(policy, Mapping) or set(policy) != expected_policy:
        raise TargetCatalogError("catalog selection_policy has invalid fields")
    if policy["ground_truth_used"] is not False:
        raise TargetCatalogError("ground truth must never be used")
    if policy["hint_membership_effect"] != "none_diagnosis_not_catalog_input":
        raise TargetCatalogError("diagnosis-independent membership must be explicit")
    if policy["spatial_authorization_rule"] != "editable_catalog_id_only":
        raise TargetCatalogError("catalog IDs must define mechanical edit permission")
    if policy["outside_hints_can_be_editable"] is not True:
        raise TargetCatalogError("targets outside hints must remain eligible")
    invalid_hint_count = _bounded_non_negative_int(
        policy["invalid_hint_count"], "invalid_hint_count", 1_000_000
    )
    pagewide_reserve = _bounded_non_negative_int(
        policy["pagewide_reserve"], "pagewide_reserve", MAX_EDITABLE_TARGETS
    )
    sibling_limit = _bounded_non_negative_int(
        policy["sibling_limit"], "sibling_limit", MAX_TARGETS
    )
    max_editable_targets = _bounded_non_negative_int(
        policy["max_editable_targets"],
        "max_editable_targets",
        MAX_EDITABLE_TARGETS,
    )
    max_targets = _bounded_non_negative_int(
        policy["max_targets"], "max_targets", MAX_TARGETS
    )
    if max_editable_targets > max_targets:
        raise TargetCatalogError("max_editable_targets cannot exceed max_targets")
    if hints:
        raise TargetCatalogError("diagnosis-independent catalogs cannot contain hints")
    if policy["invalid_hint_count"] != 0:
        raise TargetCatalogError("diagnosis-independent catalogs do not audit hints")
    if policy["pagewide_fallback_used"] is not True:
        raise TargetCatalogError("page-wide selection must always be active")
    if policy["editable_selection"] != SELECTION_STRATEGY:
        raise TargetCatalogError("editable_selection strategy is unsupported")
    if policy["id_order"] != "browser_document_order":
        raise TargetCatalogError("unsupported target ID order")

    targets_value = catalog["targets"]
    if not isinstance(targets_value, list) or len(targets_value) > max_targets:
        raise TargetCatalogError(
            f"catalog targets must be an array of at most {max_targets}"
        )

    clean_targets: list[dict] = []
    seen_xpaths: set[str] = set()
    source_mapping_by_marker: dict[str, tuple[object, ...]] = {}
    previous_document_index = -1
    editable_count = 0
    for index, target in enumerate(targets_value, 1):
        context = f"targets[{index - 1}]"
        expected_target = {
            "target_id",
            "editable",
            "anchor",
            "selection_reason",
            "ranking_hint_indices",
            "document_index",
            "locator",
            "source_mapping",
            "fingerprint",
        }
        if not isinstance(target, Mapping) or set(target) != expected_target:
            raise TargetCatalogError(f"{context} has unexpected or missing fields")
        if target["target_id"] != f"t{index}":
            raise TargetCatalogError("target IDs must be gap-free browser-order t1,t2,...")
        if not isinstance(target["editable"], bool) or target["anchor"] is not True:
            raise TargetCatalogError(f"{context} has invalid roles")
        reason = target["selection_reason"]
        if reason not in _SELECTION_REASONS:
            raise TargetCatalogError(f"{context} has invalid selection_reason")
        if target["editable"] != (reason in _EDITABLE_SELECTION_REASONS):
            raise TargetCatalogError(f"{context} editable role disagrees with selection")
        if target["editable"]:
            editable_count += 1

        hint_indices = target["ranking_hint_indices"]
        if (
            not isinstance(hint_indices, list)
            or hint_indices != sorted(set(hint_indices))
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 0 <= value < len(hints)
                for value in hint_indices
            )
        ):
            raise TargetCatalogError(f"{context} has invalid ranking_hint_indices")
        if target["editable"] and bool(hints) != bool(hint_indices):
            raise TargetCatalogError(f"{context} hint metadata disagrees with catalog hints")
        if not target["editable"] and hint_indices:
            raise TargetCatalogError(f"{context} anchor-only target cannot carry hints")

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
        # Reuse v1's normalizer to validate the exact locator convention.
        try:
            locator_probe = _normalize_browser_elements(
                [
                    {
                        "xpath": xpath,
                        "tag": tag,
                        "tag_index": tag_index,
                        "document_index": document_index,
                        "bbox": target["fingerprint"].get("bbox_css_xywh", []),
                        "visible": target["fingerprint"].get("visible"),
                    }
                ]
            )[0]
        except (TargetCatalogError, AttributeError) as error:
            raise TargetCatalogError(f"{context} has an invalid locator") from error
        if xpath in seen_xpaths:
            raise TargetCatalogError("catalog browser XPaths must be unique")
        seen_xpaths.add(xpath)

        source_mapping = target["source_mapping"]
        if not isinstance(source_mapping, Mapping) or set(source_mapping) != {
            "marker",
            "source_xpath",
            "source_parent_marker",
            "browser_alias_count",
        }:
            raise TargetCatalogError(f"{context}.source_mapping has invalid fields")
        marker = source_mapping["marker"]
        source_xpath = source_mapping["source_xpath"]
        source_parent_marker = source_mapping["source_parent_marker"]
        browser_alias_count = source_mapping["browser_alias_count"]
        marker_match = (
            _SOURCE_MARKER_RE.fullmatch(marker) if isinstance(marker, str) else None
        )
        if (
            marker_match is None
            or marker_match.group(1) != tag
            or int(marker_match.group(2)) != tag_index
            or not isinstance(source_xpath, str)
            or not source_xpath.startswith("/")
            or len(source_xpath) > 4096
            or isinstance(browser_alias_count, bool)
            or not isinstance(browser_alias_count, int)
            or browser_alias_count < 1
            or (
                source_parent_marker is not None
                and (
                    not isinstance(source_parent_marker, str)
                    or _SOURCE_MARKER_RE.fullmatch(source_parent_marker) is None
                )
            )
        ):
            raise TargetCatalogError(f"{context}.source_mapping is inconsistent")
        mapping_identity = (
            source_xpath,
            source_parent_marker,
            browser_alias_count,
        )
        if marker in source_mapping_by_marker:
            if source_mapping_by_marker[marker] != mapping_identity:
                raise TargetCatalogError(
                    f"{context}.source_mapping aliases disagree"
                )
        else:
            source_mapping_by_marker[marker] = mapping_identity

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
            "computed_style",
            "inline_style",
            "child_element_count",
            "depth",
        }:
            raise TargetCatalogError(f"{context}.fingerprint has invalid fields")
        if fingerprint["tag"] != tag or locator_probe["tag"] != tag:
            raise TargetCatalogError(f"{context} fingerprint tag disagrees with locator")
        for key in ("id", "class", "text", "role", "aria_label"):
            if not isinstance(fingerprint[key], str):
                raise TargetCatalogError(f"{context}.fingerprint.{key} must be a string")
        if not isinstance(fingerprint["visible"], bool):
            raise TargetCatalogError(f"{context}.fingerprint.visible must be boolean")
        computed_style = fingerprint["computed_style"]
        if not isinstance(computed_style, Mapping) or set(computed_style) != set(
            ALLOWED_STYLE_PROPERTIES
        ):
            raise TargetCatalogError(
                f"{context}.fingerprint.computed_style must contain every allowed property"
            )
        for prop, value in computed_style.items():
            if not isinstance(value, str) or len(value) > MAX_COMPUTED_STYLE_VALUE_CHARS:
                raise TargetCatalogError(
                    f"{context}.fingerprint.computed_style[{prop!r}] is invalid"
                )
        inline_style = fingerprint["inline_style"]
        if (
            not isinstance(inline_style, Mapping)
            or any(prop not in ALLOWED_STYLE_PROPERTIES for prop in inline_style)
            or any(
                not isinstance(value, str)
                or not value
                or len(value) > MAX_COMPUTED_STYLE_VALUE_CHARS
                for value in inline_style.values()
            )
        ):
            raise TargetCatalogError(f"{context}.fingerprint.inline_style is invalid")
        for key, minimum in (("child_element_count", 0), ("depth", 1)):
            value = fingerprint[key]
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise TargetCatalogError(f"{context}.fingerprint.{key} is invalid")
        if target["editable"] and (
            not fingerprint["visible"]
            or fingerprint["bbox_css_xywh"][2] < 0.5
            or fingerprint["bbox_css_xywh"][3] < 0.5
            or tag in _NON_EDITABLE_TAGS
        ):
            raise TargetCatalogError(f"{context} editable target is not visibly editable")
        clean_targets.append(json.loads(json.dumps(target, ensure_ascii=False)))

    if editable_count > max_editable_targets:
        raise TargetCatalogError(
            f"catalog has more than {max_editable_targets} editable targets"
        )
    editable_parents = {
        target["source_mapping"]["source_parent_marker"]
        for target in clean_targets
        if target["editable"]
    }
    for target in clean_targets:
        if not target["editable"] and (
            target["source_mapping"]["source_parent_marker"] not in editable_parents
        ):
            raise TargetCatalogError("anchor-only target has no editable source sibling")

    clean = json.loads(json.dumps(catalog, ensure_ascii=False))
    clean["ranking_hints_css_xyxy"] = hints
    clean["selection_policy"]["invalid_hint_count"] = invalid_hint_count
    clean["selection_policy"]["pagewide_reserve"] = pagewide_reserve
    clean["selection_policy"]["sibling_limit"] = sibling_limit
    clean["selection_policy"]["max_editable_targets"] = max_editable_targets
    clean["selection_policy"]["max_targets"] = max_targets
    clean["targets"] = clean_targets
    return clean


validate_target_catalog = validate_pagewide_target_catalog


def model_target_view(catalog: object) -> list[dict]:
    """Return the bounded target table safe to place in the model prompt."""

    clean = validate_pagewide_target_catalog(catalog)
    parent_groups: dict[str | None, str] = {}
    alias_groups: dict[str, str] = {}
    output: list[dict] = []
    for target in clean["targets"]:
        parent = target["source_mapping"]["source_parent_marker"]
        if parent not in parent_groups:
            parent_groups[parent] = f"g{len(parent_groups) + 1}"
        fingerprint = target["fingerprint"]
        marker = target["source_mapping"]["marker"]
        alias_count = target["source_mapping"]["browser_alias_count"]
        if alias_count > 1 and marker not in alias_groups:
            alias_groups[marker] = f"a{len(alias_groups) + 1}"
        row = {
            "target_id": target["target_id"],
            "editable": target["editable"],
            "can_be_before_anchor": target["anchor"],
            "sibling_group": parent_groups[parent],
            "source_alias_group": alias_groups.get(marker),
            "source_alias_count": alias_count,
            "tag": fingerprint["tag"],
            "bbox_css_xywh": list(fingerprint["bbox_css_xywh"]),
            "text": fingerprint["text"],
            "id": fingerprint["id"],
            "class": fingerprint["class"],
            "role": fingerprint["role"],
            "aria_label": fingerprint["aria_label"],
            "visible": fingerprint["visible"],
        }
        # Style context is useful only for editable nodes.  Omitting it from
        # anchor-only rows keeps large page-wide catalogs prompt-efficient.
        if target["editable"]:
            row["computed_style"] = dict(fingerprint["computed_style"])
            if fingerprint["inline_style"]:
                row["inline_style"] = dict(fingerprint["inline_style"])
        output.append(row)
    return output


def model_patch_json_schema(catalog: object, *, union_keyword: str = "oneOf") -> dict:
    """Build strict JSON Schema from page-wide editable and anchor IDs."""

    if union_keyword not in ("oneOf", "anyOf"):
        raise TargetCatalogError("union_keyword must be oneOf or anyOf")
    clean = validate_pagewide_target_catalog(catalog)
    editable_ids = [row["target_id"] for row in clean["targets"] if row["editable"]]
    anchor_ids = [row["target_id"] for row in clean["targets"] if row["anchor"]]
    # OpenAI strict mode requires "required" to cover every property key;
    # optionality is then expressed with nullable values, which the local
    # parser drops before resolution.
    declaration_value_schema: dict = {"type": "string", "minLength": 1, "maxLength": 512}
    if union_keyword == "anyOf":
        declaration_value_schema = {
            "type": ["string", "null"],
            "minLength": 1,
            "maxLength": 512,
        }
    declarations_schema: dict = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            prop: copy.deepcopy(declaration_value_schema)
            for prop in sorted(ALLOWED_STYLE_PROPERTIES)
        },
    }
    if union_keyword == "anyOf":
        declarations_schema["required"] = sorted(ALLOWED_STYLE_PROPERTIES)
    operations: list[dict] = []
    if editable_ids:
        operations.append(
            {
                "type": "object",
                "additionalProperties": False,
                "required": ["op", "target_id", "declarations"],
                "properties": {
                    "op": {"type": "string", "const": "set_style"},
                    "target_id": {"type": "string", "enum": editable_ids},
                    "declarations": declarations_schema,
                },
            }
        )
        if anchor_ids:
            operations.append(
                {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["op", "target_id", "before_id"],
                    "properties": {
                        "op": {"type": "string", "const": "move_element"},
                        "target_id": {"type": "string", "enum": editable_ids},
                        "before_id": {"type": "string", "enum": anchor_ids},
                    },
                }
            )
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["edits"],
        "properties": {
            "edits": {
                "type": "array",
                "maxItems": MAX_EDITS,
                "items": {union_keyword: operations} if operations else {"not": {}},
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
    """Resolve model IDs while retaining same-parent move enforcement."""

    clean = validate_pagewide_target_catalog(catalog)
    targets = _target_by_id(clean)
    if not isinstance(payload, Mapping) or set(payload) != {"edits"}:
        raise TargetCatalogError("model patch must contain exactly edits")
    edits = payload["edits"]
    if not isinstance(edits, list) or len(edits) > MAX_EDITS:
        raise TargetCatalogError(f"model patch edits must be an array of at most {MAX_EDITS}")

    resolved_edits: list[dict] = []
    resolution_log: list[dict] = []
    used_targets: list[dict] = []
    seen_target_ids: set[str] = set()
    seen_source_markers: set[str] = set()
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
            if (
                target["source_mapping"]["source_parent_marker"]
                != before["source_mapping"]["source_parent_marker"]
            ):
                raise TargetCatalogError(
                    f"{context} target_id and before_id are not frozen source siblings"
                )
            if (
                target["source_mapping"]["marker"]
                == before["source_mapping"]["marker"]
            ):
                raise TargetCatalogError(
                    f"{context} target_id and before_id alias the same source element"
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
        source_marker = target["source_mapping"]["marker"]
        if source_marker in seen_source_markers:
            raise TargetCatalogError(
                f"model patch cannot edit two target IDs that alias source {source_marker}"
            )
        seen_target_ids.add(target_id)
        seen_source_markers.add(source_marker)
        resolved_edits.append(resolved)
        resolution_log.append(log)

    try:
        source_payload = {"edits": validate_patch({"edits": resolved_edits})}
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
    clean = validate_pagewide_target_catalog(catalog)
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
    return validate_pagewide_target_catalog(value)


__all__ = [
    "CATALOG_SCHEMA_VERSION",
    "COLLECT_TARGETS_JS",
    "DEFAULT_MAX_EDITABLE_TARGETS",
    "DEFAULT_MAX_TARGETS",
    "DEFAULT_PAGEWIDE_RESERVE",
    "DEFAULT_SIBLING_LIMIT",
    "MAX_COMPUTED_STYLE_VALUE_CHARS",
    "MAX_EDITABLE_TARGETS",
    "MAX_TARGETS",
    "SELECTION_STRATEGY",
    "SOURCE_MAPPING_VERSION",
    "SOURCE_MARKER_ATTRIBUTE",
    "TargetCatalogError",
    "build_pagewide_target_catalog",
    "build_source_marker_transport",
    "build_target_catalog",
    "canonical_locations_to_css_regions",
    "load_target_catalog",
    "model_patch_json_schema",
    "model_target_view",
    "resolve_model_patch",
    "save_target_catalog",
    "validate_pagewide_target_catalog",
    "validate_target_catalog",
]
