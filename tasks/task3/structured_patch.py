#!/usr/bin/env python3
"""Strict, source-preserving patch primitives for Task 3.

The public protocol is intentionally small::

    {"edits": [
        {"op": "set_style", "target_xpath": "/html[1]/body[1]/p[1]",
         "declarations": {"color": "#111"}},
        {"op": "move_element", "target_xpath": "...",
         "before_xpath": "..."},
    ]}

Both XPath operands of a move are preflighted in the browser as unique
same-parent elements; the source executor moves the exact target source slice
immediately before the anchor's opening tag.  Locators are supplied by the
browser as ``{"tag": "p", "tag_index": 3}``, where ``tag_index`` is
zero-based in document order among source elements with that tag.

Edits are applied to the original HTML source.  This module never serializes a
browser DOM with ``outerHTML``: style edits replace only a target opening tag,
and move edits reorder exact source slices obtained from an offset tree.
"""

from __future__ import annotations

import copy
import html
import re
from collections import defaultdict
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Mapping, Sequence


MAX_EDITS = 32
MAX_XPATH_CHARS = 256
MAX_CSS_VALUE_CHARS = 512
MAX_DECLARATIONS_PER_EDIT = 7

ALLOWED_STYLE_PROPERTIES = frozenset(
    {
        "border-radius",
        "color",
        "font-family",
        "font-size",
        "font-weight",
        "text-align",
        "transform",
    }
)

_XPATH_RE = re.compile(
    r"(?:/[a-z][a-z0-9:_-]*\[[1-9][0-9]*\])+\Z"
)
_TAG_RE = re.compile(r"[a-z][a-z0-9:_-]*\Z")
_CSS_NAME_RE = re.compile(r"-?[a-zA-Z_][a-zA-Z0-9_-]*\Z")
_FORBIDDEN_CSS_VALUE_RE = re.compile(r"[;{}\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_IMPORTANT_RE = re.compile(r"!\s*important\b", re.IGNORECASE)

_VOID_TAGS = frozenset(
    {
        "area",
        "base",
        "br",
        "col",
        "embed",
        "hr",
        "img",
        "input",
        "link",
        "meta",
        "param",
        "source",
        "track",
        "wbr",
    }
)


class StructuredPatchError(ValueError):
    """A patch, locator, or source document violates the strict contract."""


def _require_exact_keys(
    value: Mapping[str, object], expected: set[str], context: str
) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        details: list[str] = []
        if missing:
            details.append(f"missing={missing}")
        if extra:
            details.append(f"extra={extra}")
        raise StructuredPatchError(f"{context} has invalid fields ({', '.join(details)})")


def _validate_xpath(value: object, context: str) -> str:
    if not isinstance(value, str):
        raise StructuredPatchError(f"{context} must be a string")
    if not value or len(value) > MAX_XPATH_CHARS:
        raise StructuredPatchError(
            f"{context} must contain 1..{MAX_XPATH_CHARS} characters"
        )
    if not _XPATH_RE.fullmatch(value):
        raise StructuredPatchError(
            f"{context} must be an absolute, indexed element XPath"
        )
    return value


def _validate_css_value(value: object, context: str) -> str:
    if not isinstance(value, str):
        raise StructuredPatchError(f"{context} must be a string")
    normalized = value.strip()
    if not normalized or len(normalized) > MAX_CSS_VALUE_CHARS:
        raise StructuredPatchError(
            f"{context} must contain 1..{MAX_CSS_VALUE_CHARS} characters"
        )
    if _FORBIDDEN_CSS_VALUE_RE.search(normalized):
        raise StructuredPatchError(f"{context} contains a forbidden CSS delimiter")
    if _IMPORTANT_RE.search(normalized):
        raise StructuredPatchError(
            f"{context} must not include !important; the executor owns priority"
        )
    return normalized


def validate_patch(payload: object) -> list[dict]:
    """Validate and normalize a strict Task 3 patch.

    The returned list is detached from ``payload``.  Unknown fields, unsafe
    XPath syntax, repeated targets, unsupported CSS properties, and a move
    whose target and anchor XPath are identical are rejected.
    """

    if not isinstance(payload, Mapping):
        raise StructuredPatchError("patch must be an object")
    _require_exact_keys(payload, {"edits"}, "patch")
    raw_edits = payload["edits"]
    if not isinstance(raw_edits, list):
        raise StructuredPatchError("patch.edits must be an array")
    if len(raw_edits) > MAX_EDITS:
        raise StructuredPatchError(f"patch.edits must contain 0..{MAX_EDITS} edits")

    normalized: list[dict] = []
    seen_targets: set[str] = set()
    for index, raw_edit in enumerate(raw_edits):
        context = f"patch.edits[{index}]"
        if not isinstance(raw_edit, Mapping):
            raise StructuredPatchError(f"{context} must be an object")
        op = raw_edit.get("op")
        if op == "set_style":
            _require_exact_keys(
                raw_edit, {"op", "target_xpath", "declarations"}, context
            )
            target_xpath = _validate_xpath(
                raw_edit["target_xpath"], f"{context}.target_xpath"
            )
            declarations = raw_edit["declarations"]
            if not isinstance(declarations, Mapping):
                raise StructuredPatchError(f"{context}.declarations must be an object")
            if not 1 <= len(declarations) <= MAX_DECLARATIONS_PER_EDIT:
                raise StructuredPatchError(
                    f"{context}.declarations must contain "
                    f"1..{MAX_DECLARATIONS_PER_EDIT} properties"
                )
            clean_declarations: dict[str, str] = {}
            for prop, value in declarations.items():
                if not isinstance(prop, str) or prop not in ALLOWED_STYLE_PROPERTIES:
                    raise StructuredPatchError(
                        f"{context}.declarations has unsupported property {prop!r}"
                    )
                clean_declarations[prop] = _validate_css_value(
                    value, f"{context}.declarations[{prop!r}]"
                )
            edit = {
                "op": "set_style",
                "target_xpath": target_xpath,
                "declarations": clean_declarations,
            }
        elif op == "move_element":
            _require_exact_keys(
                raw_edit,
                {"op", "target_xpath", "before_xpath"},
                context,
            )
            target_xpath = _validate_xpath(
                raw_edit["target_xpath"], f"{context}.target_xpath"
            )
            before_xpath = _validate_xpath(
                raw_edit["before_xpath"], f"{context}.before_xpath"
            )
            if target_xpath == before_xpath:
                raise StructuredPatchError(
                    f"{context}.before_xpath must differ from target_xpath"
                )
            edit = {
                "op": "move_element",
                "target_xpath": target_xpath,
                "before_xpath": before_xpath,
            }
        else:
            raise StructuredPatchError(f"{context}.op is unsupported: {op!r}")

        if target_xpath in seen_targets:
            raise StructuredPatchError(
                f"target_xpath must be unique within a patch: {target_xpath}"
            )
        seen_targets.add(target_xpath)
        normalized.append(edit)

    return copy.deepcopy(normalized)


@dataclass(eq=False)
class _SourceElement:
    tag: str
    start: int
    start_tag_end: int
    tag_index: int
    parent: "_SourceElement | None"
    children: list["_SourceElement"] = field(default_factory=list)
    end_tag_start: int | None = None
    end: int | None = None
    xpath: str | None = None


class _OffsetTreeParser(HTMLParser):
    def __init__(self, raw_html: str):
        super().__init__(convert_charrefs=False)
        self.raw_html = raw_html
        self.line_starts = [0]
        self.line_starts.extend(
            match.end() for match in re.finditer(r"\n", raw_html)
        )
        self.roots: list[_SourceElement] = []
        self.stack: list[_SourceElement] = []
        self.elements: list[_SourceElement] = []
        self.tag_counts: defaultdict[str, int] = defaultdict(int)

    def _offset(self) -> int:
        line, column = self.getpos()
        try:
            return self.line_starts[line - 1] + column
        except IndexError as error:  # pragma: no cover - HTMLParser invariant
            raise StructuredPatchError("HTML parser returned an invalid source position") from error

    def _add_element(self, tag: str, self_closing: bool) -> None:
        tag = tag.lower()
        start = self._offset()
        start_text = self.get_starttag_text()
        if start_text is None:  # pragma: no cover - HTMLParser invariant
            raise StructuredPatchError("HTML parser lost an opening tag")
        start_tag_end = start + len(start_text)
        ordinal = self.tag_counts[tag]
        self.tag_counts[tag] += 1
        parent = self.stack[-1] if self.stack else None
        node = _SourceElement(tag, start, start_tag_end, ordinal, parent)
        if parent is None:
            self.roots.append(node)
        else:
            parent.children.append(node)
        self.elements.append(node)
        if self_closing or tag in _VOID_TAGS:
            # A void element ends at its opening-tag boundary.  Treating it as
            # unclosed would make a move consume following siblings.
            node.end_tag_start = start_tag_end
            node.end = start_tag_end
        else:
            self.stack.append(node)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        self._add_element(tag, self_closing=False)

    def handle_startendtag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        del attrs
        self._add_element(tag, self_closing=True)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        match_index: int | None = None
        for index in range(len(self.stack) - 1, -1, -1):
            if self.stack[index].tag == tag:
                match_index = index
                break
        if match_index is None:
            return
        start = self._offset()
        close = self.raw_html.find(">", start)
        if close < 0:
            raise StructuredPatchError(f"unterminated closing tag at offset {start}")
        end = close + 1
        for dangling in self.stack[match_index + 1 :]:
            if dangling.end is None:
                dangling.end_tag_start = start
                dangling.end = start
        node = self.stack[match_index]
        node.end_tag_start = start
        node.end = end
        del self.stack[match_index:]

    def finish(self) -> None:
        self.feed(self.raw_html)
        self.close()
        for node in self.stack:
            if node.end is None:
                node.end_tag_start = len(self.raw_html)
                node.end = len(self.raw_html)
        self.stack.clear()


def _assign_xpaths(nodes: Sequence[_SourceElement], parent_xpath: str = "") -> None:
    counts: defaultdict[str, int] = defaultdict(int)
    for node in nodes:
        counts[node.tag] += 1
        node.xpath = f"{parent_xpath}/{node.tag}[{counts[node.tag]}]"
        _assign_xpaths(node.children, node.xpath)


def _parse_source(raw_html: str) -> tuple[list[_SourceElement], dict[str, _SourceElement]]:
    if not isinstance(raw_html, str):
        raise StructuredPatchError("raw_html must be a string")
    parser = _OffsetTreeParser(raw_html)
    try:
        parser.finish()
    except StructuredPatchError:
        raise
    except Exception as error:
        raise StructuredPatchError(f"could not parse HTML source: {error}") from error
    _assign_xpaths(parser.roots)
    by_xpath: dict[str, _SourceElement] = {}
    for node in parser.elements:
        assert node.xpath is not None
        if node.xpath in by_xpath:  # pragma: no cover - construction invariant
            raise StructuredPatchError(f"source XPath is not unique: {node.xpath}")
        by_xpath[node.xpath] = node
    return parser.elements, by_xpath


def source_element_inventory(raw_html: str) -> list[dict]:
    """Return a detached source-tree locator inventory for safe DOM handoff.

    Browser HTML parsing can insert or reorder nodes (most notably implicit
    ``tbody`` elements), so a browser-global tag ordinal is not necessarily a
    source-global ordinal.  This read-only inventory exposes the exact ordinal
    and parent convention used by :func:`apply_source_patch`, together with
    opening-tag byte-character spans that callers may use to instrument a
    temporary rendering copy.  It never mutates ``raw_html``.
    """

    elements, _ = _parse_source(raw_html)
    output: list[dict] = []
    for node in elements:
        assert node.xpath is not None
        parent = node.parent
        output.append(
            {
                "source_xpath": node.xpath,
                "tag": node.tag,
                "tag_index": node.tag_index,
                "parent_locator": (
                    None
                    if parent is None
                    else {"tag": parent.tag, "tag_index": parent.tag_index}
                ),
                "opening_tag_start": node.start,
                "opening_tag_end": node.start_tag_end,
            }
        )
    return output


@dataclass(frozen=True)
class _AttributeSpan:
    name: str
    start: int
    end: int
    value_start: int | None
    value_end: int | None


def _scan_attributes(start_tag: str) -> list[_AttributeSpan]:
    if not start_tag.startswith("<"):
        raise StructuredPatchError("opening-tag slice does not start with '<'")
    length = len(start_tag)
    cursor = 1
    while cursor < length and start_tag[cursor].isspace():
        cursor += 1
    while cursor < length and not start_tag[cursor].isspace() and start_tag[cursor] not in "/>":
        cursor += 1

    spans: list[_AttributeSpan] = []
    while cursor < length:
        while cursor < length and start_tag[cursor].isspace():
            cursor += 1
        if cursor >= length or start_tag[cursor] == ">" or start_tag.startswith("/>", cursor):
            break
        attr_start = cursor
        while (
            cursor < length
            and not start_tag[cursor].isspace()
            and start_tag[cursor] not in "=/>"
        ):
            cursor += 1
        if cursor == attr_start:
            raise StructuredPatchError("could not tokenize an opening-tag attribute")
        name = start_tag[attr_start:cursor]
        while cursor < length and start_tag[cursor].isspace():
            cursor += 1
        value_start: int | None = None
        value_end: int | None = None
        if cursor < length and start_tag[cursor] == "=":
            cursor += 1
            while cursor < length and start_tag[cursor].isspace():
                cursor += 1
            if cursor >= length:
                raise StructuredPatchError(f"attribute {name!r} has no value")
            if start_tag[cursor] in {'"', "'"}:
                quote = start_tag[cursor]
                cursor += 1
                value_start = cursor
                close = start_tag.find(quote, cursor)
                if close < 0:
                    raise StructuredPatchError(f"attribute {name!r} is unterminated")
                value_end = close
                cursor = close + 1
            else:
                value_start = cursor
                while (
                    cursor < length
                    and not start_tag[cursor].isspace()
                    and start_tag[cursor] not in ">"
                    and not (
                        start_tag[cursor] == "/"
                        and cursor + 1 < length
                        and start_tag[cursor + 1] == ">"
                    )
                ):
                    cursor += 1
                value_end = cursor
        spans.append(_AttributeSpan(name, attr_start, cursor, value_start, value_end))
    return spans


def _split_top_level(value: str, delimiter: str) -> list[str]:
    pieces: list[str] = []
    start = 0
    quote: str | None = None
    escaped = False
    comment = False
    nesting = 0
    index = 0
    while index < len(value):
        char = value[index]
        following = value[index + 1] if index + 1 < len(value) else ""
        if comment:
            if char == "*" and following == "/":
                comment = False
                index += 2
                continue
        elif quote is not None:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
        elif char == "/" and following == "*":
            comment = True
            index += 2
            continue
        elif char in {'"', "'"}:
            quote = char
        elif char in "([":
            nesting += 1
        elif char in ")]" and nesting:
            nesting -= 1
        elif char == delimiter and nesting == 0:
            pieces.append(value[start:index])
            start = index + 1
        index += 1
    pieces.append(value[start:])
    return pieces


def _declaration_name(fragment: str) -> str | None:
    parts = _split_top_level(fragment, ":")
    if len(parts) < 2:
        return None
    name = parts[0].strip()
    if not _CSS_NAME_RE.fullmatch(name):
        return None
    return name.lower()


def _merge_style(existing: str, declarations: Mapping[str, str]) -> str:
    fragments = [part.strip() for part in _split_top_level(existing, ";") if part.strip()]
    for prop, value in declarations.items():
        replacement = f"{prop}: {value} !important"
        matches = [
            index
            for index, fragment in enumerate(fragments)
            if _declaration_name(fragment) == prop
        ]
        if matches:
            fragments[matches[0]] = replacement
            for duplicate in reversed(matches[1:]):
                del fragments[duplicate]
        else:
            fragments.append(replacement)
    return "; ".join(fragments)


def _rewrite_opening_tag(
    opening_tag: str, declarations: Mapping[str, str]
) -> tuple[str, str | None, str]:
    attributes = _scan_attributes(opening_tag)
    style_attributes = [attr for attr in attributes if attr.name.lower() == "style"]
    if len(style_attributes) > 1:
        raise StructuredPatchError("target opening tag has duplicate style attributes")
    old_style: str | None = None
    if style_attributes:
        attribute = style_attributes[0]
        if attribute.value_start is None or attribute.value_end is None:
            old_style = ""
        else:
            old_style = html.unescape(
                opening_tag[attribute.value_start : attribute.value_end]
            )
        new_style = _merge_style(old_style, declarations)
        encoded = html.escape(new_style, quote=True)
        replacement = f'style="{encoded}"'
        rewritten = (
            opening_tag[: attribute.start]
            + replacement
            + opening_tag[attribute.end :]
        )
        return rewritten, old_style, new_style

    new_style = _merge_style("", declarations)
    encoded = html.escape(new_style, quote=True)
    insertion = len(opening_tag) - 1
    probe = insertion - 1
    while probe >= 0 and opening_tag[probe].isspace():
        probe -= 1
    if probe >= 0 and opening_tag[probe] == "/":
        insertion = probe
    separator = "" if insertion and opening_tag[insertion - 1].isspace() else " "
    rewritten = (
        opening_tag[:insertion]
        + separator
        + f'style="{encoded}"'
        + opening_tag[insertion:]
    )
    return rewritten, None, new_style


def _validated_locator(
    locator_by_xpath: Mapping[str, object], xpath: str
) -> tuple[str, int]:
    if xpath not in locator_by_xpath:
        raise StructuredPatchError(f"missing browser locator for {xpath}")
    locator = locator_by_xpath[xpath]
    if not isinstance(locator, Mapping):
        raise StructuredPatchError(f"locator for {xpath} must be an object")
    _require_exact_keys(locator, {"tag", "tag_index"}, f"locator[{xpath!r}]")
    tag = locator["tag"]
    tag_index = locator["tag_index"]
    if not isinstance(tag, str) or not _TAG_RE.fullmatch(tag):
        raise StructuredPatchError(f"locator for {xpath} has an invalid lowercase tag")
    if isinstance(tag_index, bool) or not isinstance(tag_index, int) or tag_index < 0:
        raise StructuredPatchError(
            f"locator for {xpath} has an invalid zero-based tag_index"
        )
    return tag, tag_index


def set_style_by_ordinals(
    raw_html: str,
    edits: Sequence[Mapping[str, object]],
    locator_by_xpath: Mapping[str, object],
) -> tuple[str, list[dict]]:
    """Apply validated ``set_style`` edits using browser-resolved ordinals."""

    validated = validate_patch({"edits": list(edits)})
    if any(edit["op"] != "set_style" for edit in validated):
        raise StructuredPatchError("set_style_by_ordinals accepts only set_style edits")
    if not isinstance(locator_by_xpath, Mapping):
        raise StructuredPatchError("locator_by_xpath must be an object")
    elements, _ = _parse_source(raw_html)
    by_ordinal = {(node.tag, node.tag_index): node for node in elements}

    replacements: list[tuple[int, int, str, dict]] = []
    used_locators: set[tuple[str, int]] = set()
    for edit in validated:
        xpath = edit["target_xpath"]
        tag, tag_index = _validated_locator(locator_by_xpath, xpath)
        key = (tag, tag_index)
        if key in used_locators:
            raise StructuredPatchError(
                f"browser locators are not unique for style target {xpath}"
            )
        used_locators.add(key)
        node = by_ordinal.get(key)
        if node is None:
            raise StructuredPatchError(
                f"browser locator {tag}[{tag_index}] for {xpath} has no source element"
            )
        opening_tag = raw_html[node.start : node.start_tag_end]
        rewritten, old_style, new_style = _rewrite_opening_tag(
            opening_tag, edit["declarations"]
        )
        log = {
            "op": "set_style",
            "target_xpath": xpath,
            "tag": tag,
            "tag_index": tag_index,
            "source_start": node.start,
            "old_style": old_style,
            "new_style": new_style,
            "changed": rewritten != opening_tag,
        }
        replacements.append((node.start, node.start_tag_end, rewritten, log))

    new_html = raw_html
    for start, end, replacement, _ in sorted(replacements, reverse=True):
        new_html = new_html[:start] + replacement + new_html[end:]
    logs = [entry[3] for entry in replacements]
    return new_html, logs


def _ranges_overlap(left: _SourceElement, right: _SourceElement) -> bool:
    assert left.end is not None and right.end is not None
    return left.start < right.end and right.start < left.end


def _move_elements_by_ordinals(
    raw_html: str,
    edits: Sequence[Mapping[str, object]],
    locator_by_xpath: Mapping[str, object],
) -> tuple[str, list[dict]]:
    if not edits:
        return raw_html, []
    elements, _ = _parse_source(raw_html)
    by_ordinal = {(node.tag, node.tag_index): node for node in elements}
    records: list[tuple[dict, _SourceElement, _SourceElement, tuple[str, int], tuple[str, int]]] = []
    target_keys: set[tuple[str, int]] = set()
    for edit in edits:
        target_xpath = edit["target_xpath"]
        before_xpath = edit["before_xpath"]
        target_key = _validated_locator(locator_by_xpath, target_xpath)
        before_key = _validated_locator(locator_by_xpath, before_xpath)
        if target_key == before_key:
            raise StructuredPatchError(
                f"move target and before anchor resolve to the same source element: {target_xpath}"
            )
        if target_key in target_keys:
            raise StructuredPatchError(
                f"browser locators are not unique for move target {target_xpath}"
            )
        target_keys.add(target_key)
        target = by_ordinal.get(target_key)
        before = by_ordinal.get(before_key)
        if target is None:
            raise StructuredPatchError(
                f"browser locator {target_key[0]}[{target_key[1]}] for "
                f"{target_xpath} has no source element"
            )
        if before is None:
            raise StructuredPatchError(
                f"browser locator {before_key[0]}[{before_key[1]}] for "
                f"{before_xpath} has no source element"
            )
        if target.parent is not before.parent:
            raise StructuredPatchError(
                "move target and before anchor must be source siblings"
            )
        if target.end is None or before.end is None:
            raise StructuredPatchError("move locator resolved to an unbounded source element")
        records.append((edit, target, before, target_key, before_key))

    # A target chunk must be independent from every target and anchor chunk.
    # In particular, moving an ancestor around its descendant would duplicate
    # or discard source.  A target used as another move's anchor is also
    # ambiguous in a simultaneous bounded patch and is rejected.
    for index, (_, target, _, _, _) in enumerate(records):
        for other_index, (_, other_target, other_before, _, _) in enumerate(records):
            if index != other_index and _ranges_overlap(target, other_target):
                raise StructuredPatchError("move target source spans overlap or nest")
            if _ranges_overlap(target, other_before):
                raise StructuredPatchError("move target and before-anchor spans overlap or nest")

    removals = {target.start: target.end for _, target, _, _, _ in records}
    insertions: defaultdict[int, list[str]] = defaultdict(list)
    for _, target, before, _, _ in records:
        assert target.end is not None
        insertions[before.start].append(raw_html[target.start : target.end])

    output: list[str] = []
    cursor = 0
    for position in sorted(set(removals) | set(insertions)):
        if position < cursor:  # pragma: no cover - guarded by overlap checks
            raise StructuredPatchError("move source events overlap")
        output.append(raw_html[cursor:position])
        output.extend(insertions.get(position, []))
        cursor = removals.get(position, position)
    output.append(raw_html[cursor:])
    new_html = "".join(output)

    logs: list[dict] = []
    for edit, target, before, target_key, before_key in records:
        assert target.end is not None
        logs.append(
            {
                "op": "move_element",
                "target_xpath": edit["target_xpath"],
                "before_xpath": edit["before_xpath"],
                "target_tag": target_key[0],
                "target_tag_index": target_key[1],
                "before_tag": before_key[0],
                "before_tag_index": before_key[1],
                "source_start": target.start,
                "source_end": target.end,
                "before_source_start": before.start,
                "changed": not (target.end == before.start),
            }
        )
    return new_html, logs


def apply_source_patch(
    raw_html: str,
    payload: object,
    locator_by_xpath: Mapping[str, object],
) -> tuple[str, list[dict]]:
    """Validate and apply a complete patch without DOM serialization.

    Style edits are deliberately applied before moves, so the browser's
    pre-patch global tag ordinals remain stable.  Move operations then use an
    independently parsed source-offset tree and strict same-parent semantics.
    """

    edits = validate_patch(payload)
    style_edits = [edit for edit in edits if edit["op"] == "set_style"]
    move_edits = [edit for edit in edits if edit["op"] == "move_element"]
    current = raw_html
    logs_by_target: dict[str, dict] = {}
    if style_edits:
        current, style_logs = set_style_by_ordinals(
            current, style_edits, locator_by_xpath
        )
        logs_by_target.update((entry["target_xpath"], entry) for entry in style_logs)
    if move_edits:
        current, move_logs = _move_elements_by_ordinals(
            current, move_edits, locator_by_xpath
        )
        logs_by_target.update((entry["target_xpath"], entry) for entry in move_logs)
    return current, [logs_by_target[edit["target_xpath"]] for edit in edits]


__all__ = [
    "ALLOWED_STYLE_PROPERTIES",
    "MAX_EDITS",
    "MAX_XPATH_CHARS",
    "StructuredPatchError",
    "apply_source_patch",
    "set_style_by_ordinals",
    "source_element_inventory",
    "validate_patch",
]
