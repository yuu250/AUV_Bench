#!/usr/bin/env python3
"""Output-policy audit for Task 4 generations.

``validate_topk_output`` enforces that a generated standalone HTML uses only
the visuals exposed in Top-K; ``inspect_generation`` verifies one sample
directory holds a complete, protocol-matching, violation-free generation.
Identity is established by byte counts and direct comparisons only.
"""

from __future__ import annotations

import html as html_lib
import re
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from tasks.task4.cohort import read_json
from tasks.task4.prompts import INPUT_PROTOCOL, OUTPUT_AUDIT_PROTOCOL


def validate_topk_output(html: str, allowed_assets: list[str]) -> tuple[list[str], list[str]]:
    """Enforce that generated standalone HTML uses only the visuals exposed in Top-K."""

    allowed = set(allowed_assets)
    used, violations = set(), []

    def check_ref(raw, source):
        ref = html_lib.unescape(raw.strip()).strip("\"'")
        if not ref or ref.startswith("#"):
            return
        low = ref.lower()
        if low.startswith("data:"):
            if source in {"src", "poster", "srcset"}:
                violations.append(f"inline_data_media:{ref[:40]}")
            return
        if low.startswith(("http://", "https://", "//")):
            violations.append(f"external_resource:{ref[:120]}")
            return
        clean = ref.split("?", 1)[0].split("#", 1)[0]
        if clean.startswith("assets/"):
            if ".." in Path(clean).parts:
                violations.append(f"unsafe_asset_path:{clean}")
            elif clean not in allowed:
                violations.append(f"outside_topk:{clean}")
            else:
                used.add(clean)
            return
        violations.append(f"non_asset_resource:{ref[:120]}")

    def check_srcset(value):
        if value.strip().lower().startswith("data:"):
            check_ref(value, "srcset")
        else:
            for candidate in value.split(","):
                check_ref(candidate.strip().split()[0] if candidate.strip() else "", "srcset")

    def check_css(value):
        for match in re.finditer(r"url\(\s*([\"']?)(.*?)\1\s*\)", value, re.I | re.S):
            check_ref(match.group(2), "css_url")
        for match in re.finditer(r"@import\s+(?:url\()?\s*([\"'])(.*?)\1", value, re.I | re.S):
            check_ref(match.group(2), "css_import")

    class ResourceParser(HTMLParser):
        def __init__(self):
            super().__init__(convert_charrefs=True)
            self.in_style = 0

        def handle_starttag(self, tag, attrs):
            values = {key.lower(): value or "" for key, value in attrs}
            for name in ("src", "poster"):
                if name in values:
                    check_ref(values[name], name)
            if "srcset" in values:
                check_srcset(values["srcset"])
            if tag.lower() == "link" and "href" in values:
                check_ref(values["href"], "link_href")
            if tag.lower() == "image":
                for name in ("href", "xlink:href"):
                    if name in values:
                        check_ref(values[name], "svg_image_href")
            if "style" in values:
                check_css(values["style"])
            if tag.lower() == "style":
                self.in_style += 1

        def handle_startendtag(self, tag, attrs):
            self.handle_starttag(tag, attrs)
            if tag.lower() == "style" and self.in_style:
                self.in_style -= 1

        def handle_endtag(self, tag):
            if tag.lower() == "style" and self.in_style:
                self.in_style -= 1

        def handle_data(self, data):
            if self.in_style:
                check_css(data)

    try:
        ResourceParser().feed(html)
    except Exception as exc:
        violations.append(f"html_resource_parse_error:{type(exc).__name__}")

    # Dynamic card renderers commonly keep valid asset paths in JavaScript arrays.
    # Inspect quoted asset literals without mistaking template expressions for URLs.
    for match in re.finditer(r"([\"'])(assets/[^\"']+)\1", html, re.I):
        check_ref(match.group(2), "asset_literal")
    for match in re.finditer(
            r"([\"'])(https?://[^\"']+\.(?:png|jpe?g|gif|webp|avif|svg)(?:\?[^\"']*)?)\1",
            html, re.I):
        check_ref(match.group(2), "external_image_literal")
    source_markers = re.findall(
        r"\[(?:h[1-6]|p|li|a|button|figcaption|blockquote|th|td|label|summary)\]",
        html, re.I,
    )
    if source_markers:
        violations.append("rendered_source_tag_markers:" + ",".join(sorted(set(source_markers))[:8]))
    if allowed and not used:
        violations.append("no_selected_asset_used")
    return sorted(used), sorted(set(violations))


def inspect_generation(sample_dir: Path) -> tuple[dict[str, Any], str | None]:
    """Verify one generated sample directory; return (fingerprint, error)."""

    sample_dir = Path(sample_dir)
    answer = sample_dir / "answer.html"
    manifest_path = sample_dir / "input_manifest.json"
    audit_path = sample_dir / "output_audit.json"
    fingerprint: dict[str, Any] = {
        "answer_exists": answer.is_file(),
        "input_manifest_exists": manifest_path.is_file(),
        "output_audit_exists": audit_path.is_file(),
    }
    for name, path in (
        ("answer_bytes", answer),
        ("input_manifest_bytes", manifest_path),
        ("output_audit_bytes", audit_path),
    ):
        if path.is_file():
            fingerprint[name] = path.stat().st_size
    if not answer.is_file():
        return fingerprint, "missing answer.html"
    if not manifest_path.is_file():
        return fingerprint, "missing input_manifest.json"
    if not audit_path.is_file():
        return fingerprint, "missing output_audit.json"
    try:
        manifest = read_json(manifest_path)
        audit = read_json(audit_path)
    except Exception as exc:
        return fingerprint, f"invalid generation metadata: {type(exc).__name__}: {exc}"
    if manifest.get("protocol_version") != INPUT_PROTOCOL:
        return fingerprint, "input manifest protocol mismatch"
    if audit.get("protocol_version") != OUTPUT_AUDIT_PROTOCOL:
        return fingerprint, "output audit protocol mismatch"
    if audit.get("violations") != []:
        return fingerprint, "output audit contains violations"
    asset_mapping = manifest.get("asset_mapping")
    allowed_assets = manifest.get("allowed_asset_paths")
    if not isinstance(asset_mapping, list) or not isinstance(allowed_assets, list):
        return fingerprint, "input manifest asset lists must be arrays"
    allowed = {path for path in allowed_assets if isinstance(path, str)}
    if len(allowed) != len(allowed_assets):
        return fingerprint, "input manifest contains an invalid allowed asset path"
    for item in asset_mapping:
        alias = item.get("alias_src") if isinstance(item, dict) else None
        alias_path = Path(alias) if isinstance(alias, str) else None
        if (
            alias_path is None
            or alias_path.is_absolute()
            or not alias_path.parts
            or alias_path.parts[0] != "assets"
            or ".." in alias_path.parts
            or alias not in allowed
        ):
            return fingerprint, "input manifest contains an invalid asset mapping"
        if not (sample_dir / alias_path).is_file():
            return fingerprint, f"missing allowed asset: {alias}"
    return fingerprint, None


__all__ = [
    "inspect_generation",
    "validate_topk_output",
]
