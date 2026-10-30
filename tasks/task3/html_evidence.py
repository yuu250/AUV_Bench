#!/usr/bin/env python3
"""Bounded, CSS-preserving HTML evidence for Task 3 model requests.

The executor always edits the original UTF-8 source.  This module creates only
the model-facing evidence string.  Script bodies and HTML comments are always
replaced by deterministic size markers.  Stylesheets remain intact whenever
the request budget permits; only oversized data URIs are summarized eagerly.

If the transformed document is still too large, complete compacted ``style``
blocks are frozen into a separately labelled CSS-evidence segment whose whole
serialization shares a 40%-of-request (at most 120k) budget.  A DOM copy uses
ordinal relocation comments instead of style blocks, and only that DOM
segment may undergo body-aware partitioning.  Thus a later DOM truncation can
never silently discard CSS that the stylesheet audit claims was retained.
"""

from __future__ import annotations

import re
from collections.abc import Mapping


TRANSPORT_VERSION = "task3-model-html-evidence-v4-dual-segment"
DEFAULT_MAX_MODEL_HTML_CHARS = 300_000
# Kept as a public/API-compatible name. In v2 this is the largest data URI
# retained verbatim, not a generic attribute-value limit.
DEFAULT_MAX_ATTRIBUTE_CHARS = 2_048
CSS_BUDGET_FRACTION = 0.40
MAX_CSS_BUDGET_CHARS = 120_000
SMALL_STYLE_PRESERVE_CHARS = 8_000

_SCRIPT_RE = re.compile(r"(?is)(<script\b[^>]*>)(.*?)(</script\s*>)")
_STYLE_RE = re.compile(r"(?is)(<style\b[^>]*>)(.*?)(</style\s*>)")
_COMMENT_RE = re.compile(r"(?s)<!--(.*?)-->")
_CSS_DATA_URL_RE = re.compile(
    r"(?is)(url\(\s*)(?:(['\"])(data:.*?)\2|(data:[^)]*?))(\s*\))"
)
_QUOTED_ATTRIBUTE_RE = re.compile(
    r"(?s)(\s[A-Za-z_:][A-Za-z0-9_.:-]*\s*=\s*)([\"'])(.*?)\2"
)
_UNQUOTED_DATA_ATTRIBUTE_RE = re.compile(
    r"(?is)(\s[A-Za-z_:][A-Za-z0-9_.:-]*\s*=\s*)(data:[^\s>]+)"
)
_BODY_OPEN_RE = re.compile(r"(?is)<body\b[^>]*>")
_BODY_CLOSE_RE = re.compile(r"(?is)</body\s*>")

_CSS_SEGMENT_END = "<!-- TASK3_CSS_EVIDENCE_END -->\n"
_DOM_SEGMENT_BEGIN = "<!-- TASK3_DOM_EVIDENCE_BEGIN non_executable_transport=true -->\n"
_DOM_SEGMENT_END = "\n<!-- TASK3_DOM_EVIDENCE_END -->"
_DOM_MARKER_PROTECTED_GAP_JOIN_CHARS = 320
_DOM_MARKER_PARTITION_RESERVE_CHARS = 320


class ModelHTMLTransportError(ValueError):
    """The requested model-facing HTML transport is invalid."""


def _marker(kind: str, body: str) -> str:
    return f"[TASK3_{kind}_OMITTED chars={len(body)}]"


def _replace_tag_bodies(
    source: str,
    pattern: re.Pattern[str],
    kind: str,
    audit: dict,
) -> str:
    key = kind.lower()

    def replace(match: re.Match[str]) -> str:
        body = match.group(2)
        audit[f"{key}_blocks_omitted"] += 1
        audit[f"{key}_chars_omitted"] += len(body)
        return match.group(1) + _marker(kind, body) + match.group(3)

    return pattern.sub(replace, source)


def _data_uri_parts(uri: str) -> tuple[str, str] | None:
    if not uri.lower().startswith("data:"):
        return None
    comma = uri.find(",")
    if comma < 0:
        return None
    return uri[:comma + 1], uri[comma + 1:]


def _summarize_data_uri(
    uri: str,
    *,
    context: str,
    threshold: int,
    audit: dict,
) -> str:
    if len(uri) <= threshold:
        return uri
    parts = _data_uri_parts(uri)
    if parts is None:
        audit["invalid_data_uri_candidates_not_summarized"] += 1
        return uri
    header, payload = parts
    replacement = header + _marker("DATA_URI_PAYLOAD", payload)
    record = {
        "ordinal": len(audit["data_uri_records"]) + 1,
        "context": context,
        "header": header,
        "original_chars": len(uri),
        "payload_chars_omitted": len(payload),
        "payload_chars": len(payload),
        "replacement_chars": len(replacement),
    }
    audit["data_uri_records"].append(record)
    audit["data_uris_summarized"] += 1
    audit["data_uri_original_chars"] += len(uri)
    audit["data_uri_payload_chars_omitted"] += len(payload)
    audit["data_uri_replacement_chars"] += len(replacement)
    audit["data_uri_context_counts"][context] += 1
    return replacement


def _summarize_css_data_urls(source: str, threshold: int, audit: dict) -> str:
    def replace(match: re.Match[str]) -> str:
        quote = match.group(2)
        uri = match.group(3) if quote is not None else match.group(4)
        summarized = _summarize_data_uri(
            uri,
            context=("css_url_quoted" if quote is not None else "css_url_unquoted"),
            threshold=threshold,
            audit=audit,
        )
        if summarized == uri:
            return match.group(0)
        quoted = f"{quote}{summarized}{quote}" if quote is not None else summarized
        return match.group(1) + quoted + match.group(5)

    return _CSS_DATA_URL_RE.sub(replace, source)


def _summarize_html_data_attributes(
    source: str, threshold: int, audit: dict
) -> str:
    def replace_quoted(match: re.Match[str]) -> str:
        value = match.group(3)
        summarized = _summarize_data_uri(
            value,
            context="html_attribute_quoted",
            threshold=threshold,
            audit=audit,
        )
        if summarized == value:
            return match.group(0)
        return match.group(1) + match.group(2) + summarized + match.group(2)

    evidence = _QUOTED_ATTRIBUTE_RE.sub(replace_quoted, source)

    def replace_unquoted(match: re.Match[str]) -> str:
        value = match.group(2)
        summarized = _summarize_data_uri(
            value,
            context="html_attribute_unquoted",
            threshold=threshold,
            audit=audit,
        )
        if summarized == value:
            return match.group(0)
        return match.group(1) + summarized

    return _UNQUOTED_DATA_ATTRIBUTE_RE.sub(replace_unquoted, evidence)


def _allocate_proportionally(lengths: list[int], budget: int) -> list[int]:
    if not lengths:
        return []
    total = sum(lengths)
    if total <= budget:
        return list(lengths)
    if budget <= 0:
        return [0] * len(lengths)
    raw = [length * budget / total for length in lengths]
    allocated = [int(value) for value in raw]
    remainder = budget - sum(allocated)
    order = sorted(
        range(len(lengths)),
        key=lambda index: (-(raw[index] - allocated[index]), index),
    )
    for index in order[:remainder]:
        allocated[index] += 1
    return allocated


def _compact_css_body(body: str, retained_budget: int) -> tuple[str, dict]:
    if len(body) <= retained_budget:
        return body, {
            "source_chars": len(body),
            "retained_source_chars": len(body),
            "omitted_chars": 0,
            "output_chars": len(body),
        }
    retained_budget = max(0, retained_budget)
    head_chars = retained_budget * 2 // 3
    tail_chars = retained_budget - head_chars
    omitted = body[head_chars:len(body) - tail_chars if tail_chars else len(body)]
    marker = f"\n/* TASK3_CSS_OMITTED chars={len(omitted)} */\n"
    output = body[:head_chars] + marker + (body[-tail_chars:] if tail_chars else "")
    return output, {
        "source_chars": len(body),
        "retained_source_chars": head_chars + tail_chars,
        "omitted_chars": len(omitted),
        "output_chars": len(output),
    }


def _css_segment_begin(n_blocks: int) -> str:
    return (
        "<!-- TASK3_CSS_EVIDENCE_BEGIN "
        f"non_executable_transport=true blocks={n_blocks} -->\n"
    )


def _css_segment_fixed_overhead(matches: list[re.Match[str]]) -> int:
    return (
        len(_css_segment_begin(len(matches)))
        + len(_CSS_SEGMENT_END)
        + sum(len(match.group(1)) + len(match.group(3)) + 1 for match in matches)
    )


def _compact_stylesheets(source: str, max_chars: int, audit: dict) -> str:
    matches = list(_STYLE_RE.finditer(source))
    bodies = [match.group(2) for match in matches]
    total_css = sum(len(body) for body in bodies)
    css_budget = min(MAX_CSS_BUDGET_CHARS, int(max_chars * CSS_BUDGET_FRACTION))
    fixed_overhead = _css_segment_fixed_overhead(matches) if matches else 0
    body_budget = css_budget - fixed_overhead
    audit["css_budget_chars"] = css_budget
    audit["css_segment_fixed_overhead_chars"] = fixed_overhead
    audit["css_body_budget_chars"] = max(0, body_budget)
    audit["style_blocks_total"] = len(matches)
    audit["style_source_chars"] = total_css
    audit["small_style_blocks_total"] = sum(
        len(body) <= SMALL_STYLE_PRESERVE_CHARS for body in bodies
    )
    if not matches:
        audit["style_retained_source_chars"] = 0
        audit["style_output_chars"] = 0
        return source
    if body_budget < 0:
        raise ModelHTMLTransportError(
            "CSS evidence block/envelope overhead exceeds its 40%/120k budget"
        )
    if total_css <= body_budget:
        audit["style_retained_source_chars"] = total_css
        audit["style_output_chars"] = total_css
        audit["small_style_blocks_preserved_full"] = audit[
            "small_style_blocks_total"
        ]
        return source

    # Preserve compact, likely page-specific override blocks first. Large
    # framework bundles share whatever remains. Reserve a conservative marker
    # allowance for every compacted body; the exact serialized total is checked
    # below and fails closed if an extreme budget cannot represent every block.
    marker_reserve_per_block = 144
    small_indices = [
        index for index, body in enumerate(bodies)
        if len(body) <= SMALL_STYLE_PRESERVE_CHARS
    ]
    preserved_small: set[int] = set()
    available = max(0, body_budget - marker_reserve_per_block * len(matches))
    for index in small_indices:
        incremental = max(0, len(bodies[index]) - marker_reserve_per_block)
        if incremental <= available:
            preserved_small.add(index)
            available -= incremental
    compacted_indices = [
        index for index in range(len(bodies)) if index not in preserved_small
    ]
    compacted_allocations = _allocate_proportionally(
        [len(bodies[index]) for index in compacted_indices], available
    )
    allocation_by_index = {
        index: allocation
        for index, allocation in zip(compacted_indices, compacted_allocations)
    }
    allocations = [
        len(body) if index in preserved_small else allocation_by_index[index]
        for index, body in enumerate(bodies)
    ]
    compacted_bodies: list[str] = []
    block_records: list[dict] = []
    for index, (body, allocation) in enumerate(zip(bodies, allocations), 1):
        compacted, record = _compact_css_body(body, allocation)
        record.update({
            "style_block": index,
            "retained_budget_chars": allocation,
            "small_style_block": len(body) <= SMALL_STYLE_PRESERVE_CHARS,
            "preserved_by_small_style_priority": index - 1 in preserved_small,
            "source_body_chars": len(body),
            "emitted_body_chars": len(compacted),
        })
        compacted_bodies.append(compacted)
        block_records.append(record)

    style_output_chars = sum(record["output_chars"] for record in block_records)
    if style_output_chars > body_budget:
        raise ModelHTMLTransportError(
            "CSS omission markers exceed the body share of the fixed CSS segment budget"
        )
    chunks: list[str] = []
    cursor = 0
    for match, compacted in zip(matches, compacted_bodies):
        chunks.extend((source[cursor:match.start(2)], compacted))
        cursor = match.end(2)
    chunks.append(source[cursor:])
    output = "".join(chunks)
    audit["style_blocks_compacted"] = sum(
        record["omitted_chars"] > 0 for record in block_records
    )
    audit["style_retained_source_chars"] = sum(
        record["retained_source_chars"] for record in block_records
    )
    audit["style_chars_omitted"] = sum(
        record["omitted_chars"] for record in block_records
    )
    audit["style_output_chars"] = style_output_chars
    audit["style_block_records"] = block_records
    audit["small_style_blocks_total"] = len(small_indices)
    audit["small_style_blocks_preserved_full"] = len(preserved_small)
    audit["small_style_blocks_compacted_due_budget"] = (
        len(small_indices) - len(preserved_small)
    )
    return output


def _partition_marker(kind: str, position: str, omitted: str) -> str:
    return (
        f"\n<!-- TASK3_{kind.upper()}_{position.upper()}_OMITTED "
        f"chars={len(omitted)} -->\n"
    )


def _html_partition_ranges(
    source: str,
    head_chars: int,
    middle_chars: int,
    tail_chars: int,
) -> tuple[int, int, int, int]:
    """Snap retained body fragments to tag boundaries when they are available."""

    head_end = head_chars
    if head_end:
        close_tags = list(re.finditer(r"(?is)</[^>]+>", source[:head_end]))
        if close_tags:
            head_end = close_tags[-1].end()
        else:
            last_tag_end = source.rfind(">", 0, head_end)
            if last_tag_end >= 0:
                head_end = last_tag_end + 1

    raw_middle_start = (len(source) - middle_chars) // 2
    middle_start = raw_middle_start
    if middle_chars:
        next_tag = source.find("<", raw_middle_start)
        if next_tag >= 0:
            middle_start = next_tag
    raw_middle_end = min(len(source), middle_start + middle_chars)
    middle_end = raw_middle_end
    if middle_chars:
        close_tags = list(
            re.finditer(r"(?is)</[^>]+>", source[middle_start:raw_middle_end])
        )
        if close_tags:
            middle_end = middle_start + close_tags[-1].end()
        else:
            last_tag_end = source.rfind(">", middle_start, raw_middle_end)
            if last_tag_end >= 0:
                middle_end = last_tag_end + 1

    raw_tail_start = len(source) - tail_chars
    tail_start = raw_tail_start
    if tail_chars:
        next_tag = source.find("<", raw_tail_start)
        if next_tag >= 0:
            tail_start = next_tag

    # Snapping only contracts fragments.  Still guard malformed markup from
    # producing overlap and fall back to the original character boundaries.
    if not (0 <= head_end <= middle_start <= middle_end <= tail_start <= len(source)):
        return (
            head_chars,
            (len(source) - middle_chars) // 2,
            (len(source) - middle_chars) // 2 + middle_chars,
            len(source) - tail_chars,
        )
    return head_end, middle_start, middle_end, tail_start


def _partition_segment(
    source: str,
    budget: int,
    *,
    kind: str,
    html_boundaries: bool = False,
) -> tuple[str, dict]:
    if budget < 0:
        raise ModelHTMLTransportError("internal partition budget became negative")
    if len(source) <= budget:
        return source, {
            "segment": kind,
            "source_chars": len(source),
            "budget_chars": budget,
            "output_chars": len(source),
            "retained_source_chars": len(source),
            "omitted_chars": 0,
            "omissions": [],
        }
    marker_reserve = min(budget, 320)
    retained_budget = max(0, budget - marker_reserve)
    head_chars = retained_budget * 2 // 5
    middle_chars = retained_budget // 5
    tail_chars = retained_budget - head_chars - middle_chars
    if html_boundaries:
        head_end, middle_start, middle_end, tail_start = _html_partition_ranges(
            source, head_chars, middle_chars, tail_chars
        )
    else:
        head_end = head_chars
        middle_start = (len(source) - middle_chars) // 2
        middle_end = middle_start + middle_chars
        tail_start = len(source) - tail_chars
    omitted_before = source[head_end:middle_start]
    omitted_after = source[middle_end:tail_start]
    first_marker = _partition_marker(kind, "before_middle", omitted_before)
    second_marker = _partition_marker(kind, "after_middle", omitted_after)
    output = (
        source[:head_end]
        + first_marker
        + source[middle_start:middle_end]
        + second_marker
        + (source[tail_start:] if tail_chars else "")
    )
    if len(output) > budget:
        # Account exactly for marker digit lengths on pathological inputs.
        overflow = len(output) - budget
        reduce_middle = min(middle_chars, overflow)
        middle_chars -= reduce_middle
        overflow -= reduce_middle
        reduce_head = min(head_chars, overflow)
        head_chars -= reduce_head
        overflow -= reduce_head
        tail_chars -= min(tail_chars, overflow)
        if html_boundaries:
            head_end, middle_start, middle_end, tail_start = _html_partition_ranges(
                source, head_chars, middle_chars, tail_chars
            )
        else:
            head_end = head_chars
            middle_start = (len(source) - middle_chars) // 2
            middle_end = middle_start + middle_chars
            tail_start = len(source) - tail_chars
        omitted_before = source[head_end:middle_start]
        omitted_after = source[middle_end:tail_start]
        first_marker = _partition_marker(kind, "before_middle", omitted_before)
        second_marker = _partition_marker(kind, "after_middle", omitted_after)
        output = (
            source[:head_end]
            + first_marker
            + source[middle_start:middle_end]
            + second_marker
            + (source[tail_start:] if tail_chars else "")
        )
    if len(output) > budget:
        raise ModelHTMLTransportError(
            f"{kind} partition marker overhead exceeds its frozen budget"
        )
    omissions = [
        {
            "position": "before_middle",
            "chars": len(omitted_before),
        },
        {
            "position": "after_middle",
            "chars": len(omitted_after),
        },
    ]
    return output, {
        "segment": kind,
        "source_chars": len(source),
        "budget_chars": budget,
        "output_chars": len(output),
        "retained_source_chars": (
            head_end + (middle_end - middle_start) + (len(source) - tail_start)
        ),
        "omitted_chars": len(omitted_before) + len(omitted_after),
        "omissions": omissions,
    }


def _body_aware_partition(source: str, max_chars: int, audit: dict) -> str:
    open_match = _BODY_OPEN_RE.search(source)
    close_matches = list(_BODY_CLOSE_RE.finditer(source))
    close_match = close_matches[-1] if close_matches else None
    if (
        open_match is None
        or close_match is None
        or close_match.start() < open_match.end()
    ):
        output, record = _partition_segment(
            source, max_chars, kind="document", html_boundaries=True
        )
        audit["body_aware_partition_used"] = False
        audit["partition_records"] = [record]
        return output

    prefix = source[:open_match.end()]
    body = source[open_match.end():close_match.start()]
    suffix = source[close_match.start():]
    prefix_cap = max(400, int(max_chars * 0.44))
    suffix_cap = max(320, int(max_chars * 0.06))
    prefix_budget = min(len(prefix), prefix_cap)
    suffix_budget = min(len(suffix), suffix_cap)
    body_budget = max_chars - prefix_budget - suffix_budget
    if body_budget < 640:
        shortage = 640 - body_budget
        reducible_prefix = max(0, prefix_budget - 320)
        reduction = min(shortage, reducible_prefix)
        prefix_budget -= reduction
        shortage -= reduction
        suffix_budget = max(320, suffix_budget - shortage)
        body_budget = max_chars - prefix_budget - suffix_budget

    prefix_out, prefix_record = _partition_segment(
        prefix, prefix_budget, kind="document_head"
    )
    body_out, body_record = _partition_segment(
        body, body_budget, kind="body", html_boundaries=True
    )
    suffix_out, suffix_record = _partition_segment(
        suffix, suffix_budget, kind="document_tail"
    )
    output = prefix_out + body_out + suffix_out
    if len(output) > max_chars:
        raise AssertionError("body-aware partition exceeded the request budget")
    audit["body_aware_partition_used"] = True
    audit["body_source_chars"] = len(body)
    audit["body_budget_chars"] = body_budget
    audit["body_output_chars"] = len(body_out)
    audit["partition_records"] = [prefix_record, body_record, suffix_record]
    return output


def _style_movement_nonce(source: str) -> str:
    counter = 0
    while True:
        nonce = f"{counter:020d}"
        if f"TASK3_STYLE_MOVED_{nonce}" not in source:
            return nonce
        counter += 1


def _style_moved_marker(nonce: str, ordinal: int) -> str:
    return f"<!-- TASK3_STYLE_MOVED_{nonce} ordinal={ordinal} -->"


def _partition_dom_preserving_markers(
    source: str,
    budget: int,
    marker_records: list[dict],
    audit: dict,
) -> str:
    """Partition DOM gaps while copying every exact relocation marker verbatim.

    Nearby markers form immutable islands together with their short intervening
    source text.  Only the longer gaps between islands are partitioned.  This
    mechanically prevents the ordinary head/middle/tail sampler from dropping
    style-location evidence merely because a page has many ``style`` blocks.
    """

    if len(source) <= budget:
        audit["body_aware_partition_used"] = False
        audit["dom_marker_preserving_partition_used"] = False
        audit["dom_protected_range_records"] = []
        audit["dom_protected_chars"] = 0
        audit["partition_records"] = [{
            "segment": "dom_document",
            "source_chars": len(source),
            "budget_chars": budget,
            "output_chars": len(source),
            "retained_source_chars": len(source),
            "omitted_chars": 0,
            "omissions": [],
        }]
        return source

    marker_spans: list[tuple[int, int]] = []
    for record in marker_records:
        marker = record["marker"]
        start = source.find(marker)
        if start < 0 or source.count(marker) != 1:
            raise ModelHTMLTransportError(
                "DOM source does not contain each generated STYLE_MOVED marker exactly once"
            )
        marker_spans.append((start, start + len(marker)))
    marker_spans.sort()

    # The generated envelopes are immutable as well.  Without them, the final
    # transport could no longer be mechanically recognized as evidence rather
    # than a standalone executable document.
    protected_spans = [
        (0, len(_DOM_SEGMENT_BEGIN)),
        *marker_spans,
        (len(source) - len(_DOM_SEGMENT_END), len(source)),
    ]
    protected_spans.sort()
    merged: list[list[int]] = []
    for start, end in protected_spans:
        if not merged or start - merged[-1][1] > _DOM_MARKER_PROTECTED_GAP_JOIN_CHARS:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)

    gaps: list[tuple[int, int]] = []
    cursor = 0
    for start, end in merged:
        if cursor < start:
            gaps.append((cursor, start))
        cursor = end
    if cursor < len(source):
        gaps.append((cursor, len(source)))
    fixed_chars = sum(end - start for start, end in merged)
    minimum_gap_budget = _DOM_MARKER_PARTITION_RESERVE_CHARS * len(gaps)
    if fixed_chars + minimum_gap_budget > budget:
        raise ModelHTMLTransportError(
            "DOM immutable STYLE_MOVED islands and omission markers exceed the "
            "remaining evidence budget"
        )
    gap_lengths = [end - start for start, end in gaps]
    retained_budget = budget - fixed_chars - minimum_gap_budget
    allocations = _allocate_proportionally(gap_lengths, retained_budget)
    largest_gap_index = (
        max(range(len(gaps)), key=lambda index: (gap_lengths[index], -index))
        if gaps
        else None
    )
    gap_outputs: dict[tuple[int, int], str] = {}
    gap_records: list[dict] = []
    for index, ((start, end), allocation) in enumerate(
        zip(gaps, allocations)
    ):
        kind = "body" if index == largest_gap_index else f"dom_gap_{index + 1}"
        output, record = _partition_segment(
            source[start:end],
            _DOM_MARKER_PARTITION_RESERVE_CHARS + allocation,
            kind=kind,
            html_boundaries=True,
        )
        record["source_span"] = [start, end]
        gap_outputs[(start, end)] = output
        gap_records.append(record)

    output_chunks: list[str] = []
    protected_records: list[dict] = []
    source_cursor = 0
    output_cursor = 0
    for start, end in merged:
        if source_cursor < start:
            gap_output = gap_outputs[(source_cursor, start)]
            output_chunks.append(gap_output)
            output_cursor += len(gap_output)
        protected = source[start:end]
        final_start = output_cursor
        output_chunks.append(protected)
        output_cursor += len(protected)
        protected_records.append({
            "source_span": [start, end],
            "source_chars": len(protected),
            "final_span_in_dom": [final_start, output_cursor],
        })
        source_cursor = end
    if source_cursor < len(source):
        gap_output = gap_outputs[(source_cursor, len(source))]
        output_chunks.append(gap_output)
        output_cursor += len(gap_output)
    output = "".join(output_chunks)
    if len(output) > budget:
        raise AssertionError("marker-preserving DOM partition exceeded its budget")
    missing = [
        record["ordinal"]
        for record in marker_records
        if output.count(record["marker"]) != 1
    ]
    if missing:
        raise AssertionError("marker-preserving DOM partition dropped a marker")

    audit["body_aware_partition_used"] = True
    audit["dom_marker_preserving_partition_used"] = True
    audit["dom_partition_policy"] = "protected-marker-islands-gap-head-middle-tail-v1"
    audit["dom_marker_protected_gap_join_chars"] = (
        _DOM_MARKER_PROTECTED_GAP_JOIN_CHARS
    )
    audit["dom_marker_partition_reserve_chars"] = (
        _DOM_MARKER_PARTITION_RESERVE_CHARS
    )
    audit["dom_protected_chars"] = fixed_chars
    audit["dom_protected_range_records"] = protected_records
    audit["body_source_chars"] = sum(gap_lengths)
    audit["body_budget_chars"] = sum(record["budget_chars"] for record in gap_records)
    audit["body_output_chars"] = sum(record["output_chars"] for record in gap_records)
    audit["partition_records"] = [
        {
            "segment": "dom_protected_marker_islands",
            "source_chars": fixed_chars,
            "budget_chars": fixed_chars,
            "output_chars": fixed_chars,
            "retained_source_chars": fixed_chars,
            "omitted_chars": 0,
            "omissions": [],
        },
        *gap_records,
    ]
    return output


def _build_dual_segment_evidence(
    source_before_compaction: str,
    compacted_source: str,
    max_chars: int,
    audit: dict,
) -> str:
    """Freeze complete compacted CSS blocks; partition only a style-free DOM copy."""

    source_matches = list(_STYLE_RE.finditer(source_before_compaction))
    emitted_matches = list(_STYLE_RE.finditer(compacted_source))
    if not source_matches or len(source_matches) != len(emitted_matches):
        raise ModelHTMLTransportError(
            "dual-segment CSS extraction cannot establish one-to-one style ordering"
        )

    css_chunks = [_css_segment_begin(len(emitted_matches))]
    css_block_records = []
    for ordinal, (source_match, emitted_match) in enumerate(
        zip(source_matches, emitted_matches), 1
    ):
        source_block = source_match.group(0)
        source_body = source_match.group(2)
        emitted_block = emitted_match.group(0)
        emitted_body = emitted_match.group(2)
        start = sum(len(chunk) for chunk in css_chunks)
        css_chunks.extend((emitted_block, "\n"))
        end = start + len(emitted_block)
        css_block_records.append({
            "ordinal": ordinal,
            "source_block_chars": len(source_block),
            "source_body_chars": len(source_body),
            "emitted_block_chars": len(emitted_block),
            "emitted_body_chars": len(emitted_body),
            "final_span": [start, end],
        })
    css_chunks.append(_CSS_SEGMENT_END)
    css_segment = "".join(css_chunks)
    if len(css_segment) > audit["css_budget_chars"]:
        raise ModelHTMLTransportError(
            "serialized CSS evidence segment exceeds its 40%/120k budget"
        )
    expected_css_chars = (
        audit["css_segment_fixed_overhead_chars"] + audit["style_output_chars"]
    )
    if len(css_segment) != expected_css_chars:
        raise AssertionError("serialized CSS segment accounting drifted")

    nonce = _style_movement_nonce(compacted_source)
    dom_chunks = []
    marker_records = []
    cursor = 0
    for record, emitted_match in zip(css_block_records, emitted_matches):
        marker = _style_moved_marker(nonce, record["ordinal"])
        dom_chunks.extend((compacted_source[cursor:emitted_match.start()], marker))
        cursor = emitted_match.end()
        marker_records.append({
            "ordinal": record["ordinal"],
            "source_block_chars": record["source_block_chars"],
            "marker": marker,
            "marker_chars": len(marker),
        })
    dom_chunks.append(compacted_source[cursor:])
    dom_copy = "".join(dom_chunks)
    if _STYLE_RE.search(dom_copy):
        raise AssertionError("DOM copy still contains a complete style block")
    dom_source = _DOM_SEGMENT_BEGIN + dom_copy + _DOM_SEGMENT_END
    separator = "\n"
    dom_budget = max_chars - len(css_segment) - len(separator)
    mandatory_dom_overhead = (
        len(_DOM_SEGMENT_BEGIN)
        + len(_DOM_SEGMENT_END)
        + sum(record["marker_chars"] for record in marker_records)
    )
    if dom_budget < mandatory_dom_overhead:
        raise ModelHTMLTransportError(
            "DOM style-relocation marker overhead exceeds the remaining evidence budget"
        )
    dom_output = _partition_dom_preserving_markers(
        dom_source, dom_budget, marker_records, audit
    )

    missing = [
        record["ordinal"]
        for record in marker_records
        if dom_output.count(record["marker"]) != 1
    ]
    if missing:
        raise ModelHTMLTransportError(
            "DOM partition cannot retain every STYLE_MOVED marker within budget: "
            + ",".join(map(str, missing[:20]))
        )
    if _STYLE_RE.search(dom_output):
        raise AssertionError("final DOM evidence contains a complete style block")

    evidence = css_segment + separator + dom_output
    dom_start = len(css_segment) + len(separator)
    for record in marker_records:
        relative = dom_output.find(record["marker"])
        record["final_span"] = [
            dom_start + relative,
            dom_start + relative + record["marker_chars"],
        ]
    audit.update({
        "dual_segment_used": True,
        "dual_segment_separator_chars": len(separator),
        "css_segment_range": [0, len(css_segment)],
        "css_segment_output_chars": len(css_segment),
        "css_segment_chars": len(css_segment),
        "css_segment_block_count": len(css_block_records),
        "css_segment_block_records": css_block_records,
        "dom_segment_range": [dom_start, len(evidence)],
        "dom_segment_source_chars": len(dom_source),
        "dom_segment_budget_chars": dom_budget,
        "dom_segment_output_chars": len(dom_output),
        "dom_segment_chars": len(dom_output),
        "dom_segment_envelope_chars": (
            len(_DOM_SEGMENT_BEGIN) + len(_DOM_SEGMENT_END)
        ),
        "dom_style_blocks_final": 0,
        "dom_marker_nonce": nonce,
        "dom_marker_count": len(marker_records),
        "dom_marker_chars": sum(
            record["marker_chars"] for record in marker_records
        ),
        "dom_marker_records": marker_records,
        "dual_segment_fixed_overhead_chars": (
            audit["css_segment_fixed_overhead_chars"]
            + len(separator)
            + mandatory_dom_overhead
        ),
    })
    for record in audit["dom_protected_range_records"]:
        start, end = record.pop("final_span_in_dom")
        record["final_span"] = [dom_start + start, dom_start + end]
    if len(evidence) > max_chars:
        raise AssertionError("dual-segment evidence exceeded the request budget")
    return evidence


def _validate_audit(
    source_html: str,
    evidence: str,
    audit: Mapping[str, object],
) -> None:
    if audit.get("transport_version") != TRANSPORT_VERSION:
        raise AssertionError("HTML evidence audit transport version drifted")
    if audit.get("source_chars") != len(source_html):
        raise AssertionError("HTML evidence audit source size drifted")
    if audit.get("evidence_chars") != len(evidence):
        raise AssertionError("HTML evidence audit character count drifted")
    if audit.get("evidence_bytes") != len(evidence.encode("utf-8")):
        raise AssertionError("HTML evidence audit byte count drifted")
    max_chars = audit.get("max_model_html_chars")
    if type(max_chars) is not int or len(evidence) > max_chars:
        raise AssertionError("HTML evidence exceeded its frozen character cap")
    records = audit.get("data_uri_records")
    if not isinstance(records, list) or len(records) != audit.get("data_uris_summarized"):
        raise AssertionError("data-URI audit record count drifted")
    for ordinal, record in enumerate(records, 1):
        if not isinstance(record, Mapping) or record.get("ordinal") != ordinal:
            raise AssertionError("data-URI audit ordering drifted")
        if (
            not isinstance(record.get("header"), str)
            or not record["header"].lower().startswith("data:")
            or not record["header"].endswith(",")
            or type(record.get("original_chars")) is not int
            or record["original_chars"] < 0
            or type(record.get("payload_chars")) is not int
            or record["payload_chars"] < 0
        ):
            raise AssertionError("data-URI audit record is incomplete")
    if audit.get("data_uri_original_chars") != sum(
        record["original_chars"] for record in records
    ):
        raise AssertionError("data-URI original character total drifted")
    if audit.get("data_uri_payload_chars_omitted") != sum(
        record["payload_chars_omitted"] for record in records
    ):
        raise AssertionError("data-URI omitted payload total drifted")
    if audit.get("data_uri_replacement_chars") != sum(
        record["replacement_chars"] for record in records
    ):
        raise AssertionError("data-URI replacement character total drifted")
    context_counts = audit.get("data_uri_context_counts")
    if (
        not isinstance(context_counts, Mapping)
        or sum(context_counts.values()) != len(records)
        or any(
            context_counts.get(context) != sum(
                record["context"] == context for record in records
            )
            for context in context_counts
        )
    ):
        raise AssertionError("data-URI context counts drifted")
    if audit.get("style_chars_omitted") != (
        audit.get("style_source_chars", 0)
        - audit.get("style_retained_source_chars", 0)
    ):
        raise AssertionError("stylesheet retention audit does not balance")
    if audit.get("mode") == "full_css_full_dom" and audit.get("style_chars_omitted") != 0:
        raise AssertionError("full-CSS mode cannot omit stylesheet content")
    style_records = audit.get("style_block_records")
    if not isinstance(style_records, list):
        raise AssertionError("stylesheet block audit must be an array")
    if style_records:
        if len(style_records) != audit.get("style_blocks_total"):
            raise AssertionError("stylesheet block audit count drifted")
        if sum(record["source_chars"] for record in style_records) != audit.get(
            "style_source_chars"
        ):
            raise AssertionError("stylesheet source total drifted")
        if sum(record["retained_source_chars"] for record in style_records) != audit.get(
            "style_retained_source_chars"
        ):
            raise AssertionError("stylesheet retained total drifted")
        if sum(record["output_chars"] for record in style_records) != audit.get(
            "style_output_chars"
        ):
            raise AssertionError("stylesheet output total drifted")
        if any(
            type(record["omitted_chars"]) is not int
            or record["omitted_chars"] < 0
            for record in style_records
        ):
            raise AssertionError("stylesheet omission size is invalid")

    dual_segment_used = audit.get("dual_segment_used")
    if not isinstance(dual_segment_used, bool):
        raise AssertionError("dual-segment audit flag must be boolean")
    css_segment_records = audit.get("css_segment_block_records")
    dom_marker_records = audit.get("dom_marker_records")
    if not isinstance(css_segment_records, list) or not isinstance(
        dom_marker_records, list
    ):
        raise AssertionError("dual-segment block/marker audits must be arrays")
    if dual_segment_used:
        css_range = audit.get("css_segment_range")
        dom_range = audit.get("dom_segment_range")
        separator_chars = audit.get("dual_segment_separator_chars")
        if (
            not isinstance(css_range, list)
            or len(css_range) != 2
            or css_range[0] != 0
            or not isinstance(dom_range, list)
            or len(dom_range) != 2
            or type(separator_chars) is not int
            or dom_range[0] != css_range[1] + separator_chars
            or dom_range[1] != len(evidence)
        ):
            raise AssertionError("dual-segment final ranges drifted")
        css_segment = evidence[css_range[0]:css_range[1]]
        separator = evidence[css_range[1]:dom_range[0]]
        dom_segment = evidence[dom_range[0]:dom_range[1]]
        if separator != "\n":
            raise AssertionError("dual-segment separator drifted")
        if not css_segment.startswith("<!-- TASK3_CSS_EVIDENCE_BEGIN ") or not (
            css_segment.endswith(_CSS_SEGMENT_END)
        ):
            raise AssertionError("CSS evidence envelope drifted")
        if not dom_segment.startswith(_DOM_SEGMENT_BEGIN) or not dom_segment.endswith(
            _DOM_SEGMENT_END
        ):
            raise AssertionError("DOM evidence envelope drifted")
        if (
            audit.get("css_segment_output_chars") != len(css_segment)
            or audit.get("css_segment_chars") != len(css_segment)
            or audit.get("dom_segment_output_chars") != len(dom_segment)
            or audit.get("dom_segment_chars") != len(dom_segment)
        ):
            raise AssertionError("dual-segment size drifted")
        if len(css_segment) > audit.get("css_budget_chars", -1):
            raise AssertionError("serialized CSS segment exceeded its allocation")
        if len(dom_segment) > audit.get("dom_segment_budget_chars", -1):
            raise AssertionError("serialized DOM segment exceeded its allocation")
        if len(css_segment_records) != audit.get("style_blocks_total") or len(
            css_segment_records
        ) != audit.get("css_segment_block_count"):
            raise AssertionError("final CSS block count drifted")
        final_css_matches = list(_STYLE_RE.finditer(css_segment))
        if len(final_css_matches) != len(css_segment_records):
            raise AssertionError("final CSS evidence lost or gained a style block")
        previous_end = 0
        for ordinal, (record, match) in enumerate(
            zip(css_segment_records, final_css_matches), 1
        ):
            span = record.get("final_span")
            expected_span = [match.start(), match.end()]
            if (
                record.get("ordinal") != ordinal
                or span != expected_span
                or span[0] < previous_end
            ):
                raise AssertionError("final CSS block ordering/span drifted")
            emitted_block = evidence[span[0]:span[1]]
            if (
                len(emitted_block) != record.get("emitted_block_chars")
                or len(match.group(2)) != record.get("emitted_body_chars")
                or type(record.get("source_block_chars")) is not int
                or record["source_block_chars"] < 0
                or type(record.get("source_body_chars")) is not int
                or record["source_body_chars"] < 0
            ):
                raise AssertionError("final CSS block evidence size drifted")
            if style_records:
                compaction_record = style_records[ordinal - 1]
                if (
                    compaction_record.get("style_block") != ordinal
                    or record.get("source_body_chars")
                    != compaction_record.get("source_body_chars")
                    or record.get("emitted_body_chars")
                    != compaction_record.get("emitted_body_chars")
                    or record.get("source_body_chars")
                    != compaction_record.get("source_chars")
                    or record.get("emitted_body_chars")
                    != compaction_record.get("output_chars")
                ):
                    raise AssertionError(
                        "final CSS evidence differs from the compaction output audit"
                    )
            previous_end = span[1]
        if (
            len(css_segment)
            != audit.get("css_segment_fixed_overhead_chars", -1)
            + audit.get("style_output_chars", -1)
        ):
            raise AssertionError("CSS segment budget accounting drifted")
        if _STYLE_RE.search(dom_segment) or audit.get("dom_style_blocks_final") != 0:
            raise AssertionError("DOM evidence retained a complete style block")
        protected_records = audit.get("dom_protected_range_records")
        if not isinstance(protected_records, list):
            raise AssertionError("DOM protected-range audit must be an array")
        if audit.get("dom_marker_preserving_partition_used"):
            if (
                audit.get("dom_partition_policy")
                != "protected-marker-islands-gap-head-middle-tail-v1"
                or not protected_records
                or audit.get("dom_protected_chars")
                != sum(record.get("source_chars", -1) for record in protected_records)
            ):
                raise AssertionError("DOM marker-preserving policy audit drifted")
            previous_end = dom_range[0]
            for record in protected_records:
                span = record.get("final_span")
                if (
                    not isinstance(span, list)
                    or len(span) != 2
                    or not (dom_range[0] <= span[0] < span[1] <= dom_range[1])
                    or span[0] < previous_end
                ):
                    raise AssertionError("DOM protected-range span drifted")
                protected = evidence[span[0]:span[1]]
                if len(protected) != record.get("source_chars"):
                    raise AssertionError("DOM protected-range size drifted")
                previous_end = span[1]
        elif protected_records or audit.get("dom_protected_chars") != 0:
            raise AssertionError("unpartitioned DOM cannot claim protected islands")
        if len(dom_marker_records) != len(css_segment_records) or len(
            dom_marker_records
        ) != audit.get("dom_marker_count"):
            raise AssertionError("DOM relocation marker count drifted")
        if audit.get("dom_marker_chars") != sum(
            record.get("marker_chars", -1) for record in dom_marker_records
        ):
            raise AssertionError("DOM relocation marker size total drifted")
        for ordinal, (record, css_record) in enumerate(
            zip(dom_marker_records, css_segment_records), 1
        ):
            marker = record.get("marker")
            span = record.get("final_span")
            if (
                record.get("ordinal") != ordinal
                or record.get("source_block_chars")
                != css_record.get("source_block_chars")
                or not isinstance(marker, str)
                or len(marker) != record.get("marker_chars")
                or not isinstance(span, list)
                or len(span) != 2
                or evidence[span[0]:span[1]] != marker
                or not (dom_range[0] <= span[0] < span[1] <= dom_range[1])
                or dom_segment.count(marker) != 1
            ):
                raise AssertionError("DOM relocation marker evidence drifted")
    elif (
        css_segment_records
        or dom_marker_records
        or audit.get("css_segment_range") is not None
        or audit.get("dom_segment_range") is not None
    ):
        raise AssertionError("non-dual mode cannot claim final segment evidence")
    partitions = audit.get("partition_records")
    if not isinstance(partitions, list):
        raise AssertionError("partition audit must be an array")
    for record in partitions:
        omissions = record.get("omissions")
        if not isinstance(omissions, list):
            raise AssertionError("partition omission audit must be an array")
        if record.get("omitted_chars") != sum(
            omission["chars"] for omission in omissions
        ):
            raise AssertionError("partition omitted character total drifted")
        if record.get("source_chars") != (
            record.get("retained_source_chars", 0) + record.get("omitted_chars", 0)
        ):
            raise AssertionError("partition retention does not balance")
        if record.get("output_chars", 0) > record.get("budget_chars", 0):
            raise AssertionError("partition output exceeded its allocation")
        if any(
            type(omission.get("chars")) is not int
            or omission["chars"] < 0
            for omission in omissions
        ):
            raise AssertionError("partition omission size is invalid")
    for key in (
        "after_nonvisual_elision_chars",
        "after_data_uri_summary_chars",
        "after_css_compaction_chars",
    ):
        value = audit.get(key)
        if type(value) is not int or value < 0:
            raise AssertionError(f"intermediate audit size {key} is invalid")


def build_model_html_evidence(
    source_html: str,
    *,
    max_chars: int = DEFAULT_MAX_MODEL_HTML_CHARS,
    max_attribute_chars: int = DEFAULT_MAX_ATTRIBUTE_CHARS,
) -> tuple[str, dict]:
    """Return bounded evidence plus a strict deterministic transformation audit."""

    if not isinstance(source_html, str):
        raise ModelHTMLTransportError("source_html must be a string")
    if type(max_chars) is not int or max_chars < 2_000:
        raise ModelHTMLTransportError("max_chars must be an integer >= 2000")
    if type(max_attribute_chars) is not int or max_attribute_chars < 128:
        raise ModelHTMLTransportError(
            "max_attribute_chars must be an integer >= 128"
        )

    audit = {
        "transport_version": TRANSPORT_VERSION,
        "max_model_html_chars": max_chars,
        "max_attribute_chars": max_attribute_chars,
        "data_uri_threshold_chars": max_attribute_chars,
        "css_budget_fraction": CSS_BUDGET_FRACTION,
        "css_budget_cap_chars": MAX_CSS_BUDGET_CHARS,
        "small_style_preserve_chars": SMALL_STYLE_PRESERVE_CHARS,
        "style_compaction_policy": "preserve_le_8k_blocks_then_allocate_large_bundles",
        "source_chars": len(source_html),
        "source_bytes": len(source_html.encode("utf-8")),
        "script_blocks_omitted": 0,
        "script_chars_omitted": 0,
        "comment_blocks_omitted": 0,
        "comment_chars_omitted": 0,
        "data_uris_summarized": 0,
        "data_uri_original_chars": 0,
        "data_uri_payload_chars_omitted": 0,
        "data_uri_replacement_chars": 0,
        "data_uri_context_counts": {
            "css_url_quoted": 0,
            "css_url_unquoted": 0,
            "html_attribute_quoted": 0,
            "html_attribute_unquoted": 0,
        },
        "data_uri_records": [],
        "invalid_data_uri_candidates_not_summarized": 0,
        "style_blocks_total": 0,
        "style_blocks_compacted": 0,
        "style_source_chars": 0,
        "style_retained_source_chars": 0,
        "style_chars_omitted": 0,
        "style_output_chars": 0,
        "style_block_records": [],
        "small_style_blocks_total": 0,
        "small_style_blocks_preserved_full": 0,
        "small_style_blocks_compacted_due_budget": 0,
        "css_budget_chars": min(
            MAX_CSS_BUDGET_CHARS, int(max_chars * CSS_BUDGET_FRACTION)
        ),
        "css_segment_fixed_overhead_chars": 0,
        "css_body_budget_chars": 0,
        "dual_segment_used": False,
        "dual_segment_separator_chars": 0,
        "css_segment_range": None,
        "css_segment_output_chars": 0,
        "css_segment_chars": None,
        "css_segment_block_count": 0,
        "css_segment_block_records": [],
        "dom_segment_range": None,
        "dom_segment_source_chars": 0,
        "dom_segment_budget_chars": 0,
        "dom_segment_output_chars": 0,
        "dom_segment_chars": None,
        "dom_segment_envelope_chars": 0,
        "dom_style_blocks_final": None,
        "dom_marker_nonce": None,
        "dom_marker_count": 0,
        "dom_marker_chars": 0,
        "dom_marker_records": [],
        "dom_marker_preserving_partition_used": False,
        "dom_partition_policy": "none",
        "dom_marker_protected_gap_join_chars": (
            _DOM_MARKER_PROTECTED_GAP_JOIN_CHARS
        ),
        "dom_marker_partition_reserve_chars": (
            _DOM_MARKER_PARTITION_RESERVE_CHARS
        ),
        "dom_protected_chars": 0,
        "dom_protected_range_records": [],
        "dual_segment_fixed_overhead_chars": 0,
        "body_aware_partition_used": False,
        "body_source_chars": 0,
        "body_budget_chars": 0,
        "body_output_chars": 0,
        "partition_records": [],
    }

    evidence = _replace_tag_bodies(source_html, _SCRIPT_RE, "SCRIPT", audit)

    def replace_comment(match: re.Match[str]) -> str:
        body = match.group(1)
        audit["comment_blocks_omitted"] += 1
        audit["comment_chars_omitted"] += len(body)
        return "<!--" + _marker("COMMENT", body) + "-->"

    evidence = _COMMENT_RE.sub(replace_comment, evidence)
    audit["after_nonvisual_elision_chars"] = len(evidence)

    evidence = _summarize_css_data_urls(evidence, max_attribute_chars, audit)
    evidence = _summarize_html_data_attributes(evidence, max_attribute_chars, audit)
    audit["after_data_uri_summary_chars"] = len(evidence)

    style_matches = list(_STYLE_RE.finditer(evidence))
    audit["style_blocks_total"] = len(style_matches)
    audit["style_source_chars"] = sum(len(match.group(2)) for match in style_matches)
    audit["style_retained_source_chars"] = audit["style_source_chars"]
    audit["style_output_chars"] = audit["style_source_chars"]
    audit["small_style_blocks_total"] = sum(
        len(match.group(2)) <= SMALL_STYLE_PRESERVE_CHARS
        for match in style_matches
    )
    audit["small_style_blocks_preserved_full"] = audit[
        "small_style_blocks_total"
    ]

    if len(evidence) <= max_chars:
        mode = "full_css_full_dom"
    else:
        source_before_compaction = evidence
        compacted_evidence = _compact_stylesheets(evidence, max_chars, audit)
        audit["after_css_compaction_chars"] = len(compacted_evidence)
        if len(compacted_evidence) <= max_chars:
            evidence = compacted_evidence
            mode = "css_compacted_full_dom"
        elif audit["style_blocks_total"]:
            evidence = _build_dual_segment_evidence(
                source_before_compaction,
                compacted_evidence,
                max_chars,
                audit,
            )
            mode = (
                "css_compacted_body_head_middle_tail"
                if audit["style_blocks_compacted"]
                else "body_head_middle_tail"
            )
        else:
            evidence = _body_aware_partition(compacted_evidence, max_chars, audit)
            mode = "body_head_middle_tail"

    audit.setdefault("after_css_compaction_chars", len(evidence))
    audit.update({
        "mode": mode,
        "evidence_chars": len(evidence),
        "evidence_bytes": len(evidence.encode("utf-8")),
    })
    _validate_audit(source_html, evidence, audit)
    return evidence, audit


__all__ = [
    "CSS_BUDGET_FRACTION",
    "DEFAULT_MAX_ATTRIBUTE_CHARS",
    "DEFAULT_MAX_MODEL_HTML_CHARS",
    "MAX_CSS_BUDGET_CHARS",
    "ModelHTMLTransportError",
    "SMALL_STYLE_PRESERVE_CHARS",
    "TRANSPORT_VERSION",
    "build_model_html_evidence",
]
