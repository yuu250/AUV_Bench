#!/usr/bin/env python3
"""Content-preservation scoring for bounded Task 3 patches.

The historical Task 3 gate compared rendered ``innerText`` in document order.
That makes a correct ``move_element`` repair look destructive: the words are
unchanged, but their line order necessarily changes.  This module separates
two questions that the old similarity score conflated:

* did the page keep the same content-bearing elements, text, links, images,
  form semantics, and accessibility metadata; and
* is the complete fixed source exactly the result of the authorized bounded
  patch?

The first check is an order-insensitive multiset.  The second replays the
strict source patch, so an unlisted reorder cannot hide behind the multiset.
Together they allow an authorized sibling move without weakening protection
against arbitrary HTML changes.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Mapping

try:
    from .structured_patch import apply_source_patch
except ImportError:  # direct execution/import from the module directory
    from structured_patch import apply_source_patch


SCHEMA_VERSION = "task3-content-preservation-v1"

# ``class`` and ``style`` are deliberately absent: visual repair is allowed to
# change style declarations.  Identity/content/navigation/accessibility fields
# remain protected.  All aria-* attributes are included dynamically below.
PROTECTED_ATTRIBUTES = frozenset(
    {
        "id",
        "href",
        "xlink:href",
        "src",
        "srcset",
        "data-src",
        "data-srcset",
        "data-original",
        "poster",
        "alt",
        "title",
        "role",
        "name",
        "type",
        "value",
        "placeholder",
        "for",
        "action",
        "method",
        "target",
        "rel",
        "download",
        "cite",
        "checked",
        "selected",
        "disabled",
        "readonly",
        "required",
        "contenteditable",
        "tabindex",
        "d",
        "points",
        "viewbox",
    }
)

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


def _normalize_text(value: str) -> str:
    return " ".join(value.split())


@dataclass
class _ElementProjection:
    tag: str
    attributes: tuple[tuple[str, str | None], ...]
    direct_text_parts: list[str] = field(default_factory=list)


class _ProjectionParser(HTMLParser):
    """Build order-independent, content-bearing element records.

    Text is attached to its direct parent rather than stored as a global bag.
    Moving a complete element therefore preserves its record, while moving or
    rewriting text between unlike elements is detected.  Exact authorized
    replay remains the final guard for otherwise indistinguishable duplicates.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: list[_ElementProjection] = []
        self.records: Counter[str] = Counter()
        self.document_text: list[str] = []

    @staticmethod
    def _protected_attributes(
        attrs: list[tuple[str, str | None]],
    ) -> tuple[tuple[str, str | None], ...]:
        selected = []
        for raw_name, value in attrs:
            name = raw_name.lower()
            if name in PROTECTED_ATTRIBUTES or name.startswith("aria-"):
                selected.append((name, value))
        return tuple(sorted(selected))

    def _finish(self, node: _ElementProjection) -> None:
        direct_text = ""
        # Style rules are visual implementation, not page content.  Script and
        # JSON-script payloads stay protected because changing them is never an
        # authorized Task 3 repair.
        if node.tag != "style":
            direct_text = _normalize_text("".join(node.direct_text_parts))
        record = {
            "kind": "element",
            "tag": node.tag,
            "attributes": list(node.attributes),
            "direct_text": direct_text,
        }
        self.records[
            json.dumps(record, ensure_ascii=False, sort_keys=True)
        ] += 1

    def _start(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
        *,
        self_closing: bool,
    ) -> None:
        node = _ElementProjection(
            tag=tag.lower(),
            attributes=self._protected_attributes(attrs),
        )
        if self_closing or node.tag in _VOID_TAGS:
            self._finish(node)
        else:
            self.stack.append(node)

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        self._start(tag, attrs, self_closing=False)

    def handle_startendtag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        self._start(tag, attrs, self_closing=True)

    def handle_endtag(self, tag: str) -> None:
        lowered = tag.lower()
        match_index: int | None = None
        for index in range(len(self.stack) - 1, -1, -1):
            if self.stack[index].tag == lowered:
                match_index = index
                break
        if match_index is None:
            return
        # Tolerate browser-source HTML with implicitly closed descendants while
        # still emitting a deterministic record for every parsed element.
        for node in reversed(self.stack[match_index:]):
            self._finish(node)
        del self.stack[match_index:]

    def handle_data(self, data: str) -> None:
        if self.stack:
            self.stack[-1].direct_text_parts.append(data)
        else:
            self.document_text.append(data)

    def finish(self, raw_html: str) -> Counter[str]:
        self.feed(raw_html)
        self.close()
        for node in reversed(self.stack):
            self._finish(node)
        self.stack.clear()
        document_text = _normalize_text("".join(self.document_text))
        if document_text:
            record = {"kind": "document_text", "text": document_text}
            self.records[
                json.dumps(record, ensure_ascii=False, sort_keys=True)
            ] += 1
        return self.records


def content_projection(raw_html: str) -> Counter[str]:
    """Return the protected, order-insensitive source-content multiset."""

    if not isinstance(raw_html, str):
        raise TypeError("raw_html must be a string")
    return _ProjectionParser().finish(raw_html)


def _counter_delta(left: Counter[str], right: Counter[str]) -> list[dict]:
    output = []
    for encoded, count in sorted((left - right).items()):
        output.append({"record": json.loads(encoded), "count": count})
    return output


def evaluate_content_preservation(
    before_html: str,
    after_html: str,
    *,
    authorized_patch: object | None = None,
    locator_by_xpath: Mapping[str, object] | None = None,
) -> dict:
    """Score content preservation and exact bounded-patch authorization.

    ``authorized_patch=None`` means no source change is authorized.  Otherwise
    the function deterministically replays ``authorized_patch`` through the
    strict source executor.  A move is therefore accepted only when it is one
    of the declared moves and the resulting source is byte-for-byte equal to
    ``after_html``.

    Invalid patch/locator data is returned as a failed authorization result so
    a formal evaluation can record the failure instead of crashing midway.
    """

    if not isinstance(before_html, str) or not isinstance(after_html, str):
        raise TypeError("before_html and after_html must be strings")

    before_projection = content_projection(before_html)
    after_projection = content_projection(after_html)
    removed = _counter_delta(before_projection, after_projection)
    added = _counter_delta(after_projection, before_projection)
    content_equal = not removed and not added

    replay_error: str | None = None
    replay_logs: list[dict] = []
    if authorized_patch is None:
        replayed_html = before_html
    else:
        try:
            replayed_html, replay_logs = apply_source_patch(
                before_html,
                authorized_patch,
                locator_by_xpath or {},
            )
        except Exception as error:  # strict scorer result, not run abortion
            replayed_html = None
            replay_error = f"{type(error).__name__}: {error}"

    authorization_equal = replayed_html is not None and replayed_html == after_html
    passed = content_equal and authorization_equal
    return {
        "schema_version": SCHEMA_VERSION,
        "passed": passed,
        "content_equal": content_equal,
        "authorization_equal": authorization_equal,
        "authorization_error": replay_error,
        "n_content_records_before": sum(before_projection.values()),
        "n_content_records_after": sum(after_projection.values()),
        "n_authorized_edits": len(replay_logs),
        "removed_records": removed,
        "added_records": added,
    }


__all__ = [
    "PROTECTED_ATTRIBUTES",
    "SCHEMA_VERSION",
    "content_projection",
    "evaluate_content_preservation",
]
