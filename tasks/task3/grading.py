#!/usr/bin/env python3
"""Task 3 orchestration: request materialization and F/C/P/S grading.

One sample flows through:

1. ``prepare_request`` — build the frozen multimodal request from the degraded
   page: overview + fixed-canvas slices of the local reference render, the
   five-field Task 2 handoff JSON in the prompt, a page-wide target catalog,
   and bounded HTML evidence.  The catalog and local model-input geometry come
   from the prebuilt reference cache so every model run issues byte-identical
   requests.
2. ``run_one_model_sample`` — call the provider through ``src.api`` with the
   strict JSON-schema response format, parse and apply the patch through the
   mechanical protocol layers, then grade.
3. ``grade_applied_sample`` — render the fixed page under the authoritative
   local contract and compute Repair Pass = fix_pass AND collateral_pass AND
   content_preservation_pass AND visual_scope_pass over the frozen reference
   baseline/degraded element indexes.

Terminal records are appended to a result JSONL (fsync per record) so a run
resumes without changing its fixed denominator.  No content is hashed
anywhere: identity and staleness are established by direct structural
comparison, and payload sizes are recorded as character/byte counts.
"""

from __future__ import annotations

import asyncio
import base64
import copy
import io
import json
import math
import os
import re
import tempfile
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from PIL import Image, __version__ as PILLOW_VERSION

from src.api.client import InvalidResponseError, is_retryable_api_error
from tasks.task3 import advisory_protocol
from tasks.task3 import element_index
from tasks.task3 import reference_cache
from tasks.task3.content_preservation import evaluate_content_preservation
from tasks.task3.html_evidence import (
    DEFAULT_MAX_ATTRIBUTE_CHARS,
    TRANSPORT_VERSION as MODEL_HTML_TRANSPORT_VERSION,
    build_model_html_evidence,
)
from tasks.task3.lesion_scorer import grade_page
from tasks.task3.fontenv import resolve_render_fonts_dir
from tasks.task3.render_environment import render_environment_provenance
from tasks.task3.render_gate import (
    MAX_REQUEST_FAILURE_PAIR_ATTEMPTS,
    SourceRenderGateError,
    build_source_render_gate_audit,
    png_bytes_audit,
    png_pixels_equal,
)
from tasks.task3.render_transport import (
    CHROMIUM_RENDER_ARGS,
    CHROMIUM_RENDER_POLICY_VERSION,
    DEFAULT_RENDER_MOTION_FREEZE_TIMEOUT_MS,
    DEFAULT_RENDER_READY_TIMEOUT_MS,
    FONT_CACHE_WARM_RELOAD_POLICY,
    HTML_NAVIGATION_TRANSPORT_VERSION,
    RENDER_MOTION_FREEZE_CONTRACT_VERSION,
    RENDER_READY_CONTRACT_VERSION,
    SMIL_CURRENT_TIME_TOLERANCE_SECONDS,
    Utf8HtmlTransportError,
    freeze_render_motion,
    goto_utf8_html,
    require_same_browser_character_set,
    wait_for_render_ready,
)
from tasks.task3.structured_patch import apply_source_patch
from tasks.task3.target_ids import (
    COLLECT_TARGETS_JS,
    build_pagewide_target_catalog,
    build_source_marker_transport,
    model_target_view,
    resolve_model_patch,
)
from tasks.task3.visual_scope import (
    apply_baseline_restoration_fast_path,
    frozen_gt_permitted_regions,
    score_visual_scope,
    slice_offsets,
)


Image.MAX_IMAGE_PIXELS = None

EVAL_VERSION = "task3-repair-pass-v3"
RESULT_SCHEMA_VERSION = "task3-repair-pass-result-v2"
SUMMARY_SCHEMA_VERSION = "task3-repair-pass-summary-v2"
MESSAGE_FORMAT_VERSION = "task3-repair-pass-degraded-page-multimodal-v1"
GEOMETRY_AUDIT_VERSION = "task3-repair-pass-degraded-geometry-v1"
OVERLAY_POLICY_VERSION = "task3-repair-pass-advisory-overlay-only-v1"
GRADING_RENDER_POLICY_VERSION = (
    "task3-repair-pass-local-authoritative-required-fonts-v3"
)
ASSET_CLOSURE_VERSION = "task3-repair-pass-degraded-html-relative-assets-v2"
LOCAL_MODEL_INPUT_POLICY_VERSION = (
    "task3-repair-pass-local-degraded-model-input-open-v1"
)
GT_SCHEMA_VERSION = "design-bench-unified-gt-v3"
EXPECTED_FORMAL_COHORT = 660
MODEL_MAX_TOKENS = 8_000
MAX_RESPONSE_CHARS = advisory_protocol.MAX_RESPONSE_CHARS
SETTLE_MS = 800
GRADING_SETTLE_MS = 800
GRADING_VIEWPORT_HEIGHT_CSS = 900
RENDER_READY_TIMEOUT_MS = DEFAULT_RENDER_READY_TIMEOUT_MS
MAX_WORKERS = 16

SLICE_H = 1800
OVERLAP = 180
THUMB_H = 1500
MAX_IMG_BYTES = 9 * 1024 * 1024  # one image above this many bytes becomes JPEG

METRIC_MAIN = "repair_pass"
METRIC_DISPLAY_NAMES = {
    METRIC_MAIN: "Repair Pass",
}
METRIC_COMPONENTS = (
    "fix_pass",
    "collateral_pass",
    "content_preservation_pass",
    "visual_scope_pass",
)


class Task3Error(RuntimeError):
    """A frozen input, request, response, or scoring gate is invalid."""


class GradingRenderGateError(Task3Error):
    """A grading render failed its deterministic local-render contract."""

    def __init__(self, message: str, grading_render: Mapping[str, object]):
        super().__init__(message)
        self.grading_render = copy.deepcopy(dict(grading_render))


@dataclass(frozen=True)
class RequestInput:
    """The complete request-side view; it deliberately has no scoring fields."""

    sample_id: str
    task2_handoff: dict | None
    geometry: dict
    selected_version: int
    selected_source: dict
    task2_provenance: dict
    viewport: tuple[int, int]
    dpr: float


@dataclass
class PreparedRequest:
    """One in-memory request.  At most ``--workers`` exist during execution."""

    request: RequestInput
    local_model_input: dict
    source_html: str
    model_html_evidence: str
    model_html_audit: dict
    target_catalog: dict
    catalog_navigation_audit: dict
    target_view: list[dict]
    model_schema: dict
    response_format: dict
    prompt: str
    messages: list[dict]
    message_audit: dict


def atomic_write_text(path: Path, content: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def atomic_write_json(path: Path, value: Mapping[str, object]) -> Path:
    return atomic_write_text(
        path,
        json.dumps(dict(value), ensure_ascii=False, sort_keys=True, indent=2) + "\n",
    )


def b64_image(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    data = buf.getvalue()
    mime = "image/png"
    if len(data) > MAX_IMG_BYTES:
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=90)
        data = buf.getvalue()
        mime = "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(data).decode()


def fixed_canvas_slice(image: Image.Image, top: int, *, canvas_width: int) -> Image.Image:
    if image.width != canvas_width:
        raise Task3Error(
            f"selected screenshot width {image.width} != canvas {canvas_width}"
        )
    canvas = Image.new("RGB", (canvas_width, SLICE_H), "white")
    if top < image.height:
        visible = image.crop(
            (0, top, canvas_width, min(top + SLICE_H, image.height))
        ).convert("RGB")
        canvas.paste(visible, (0, 0))
    return canvas


def symmetric_overview(image: Image.Image, geometry: Mapping[str, object]) -> Image.Image:
    width = int(geometry["canvas_width"])
    common_height = int(geometry["common_height"])
    if image.width != width or image.height > common_height:
        raise Task3Error("selected screenshot violates the frozen overview geometry")
    scale = THUMB_H / common_height
    overview_width = max(1, round(width * scale))
    scaled_width = max(1, round(image.width * scale))
    scaled_height = max(1, round(image.height * scale))
    resampling = getattr(Image, "Resampling", Image).LANCZOS
    scaled = image.convert("RGB").resize((scaled_width, scaled_height), resampling)
    canvas = Image.new("RGB", (overview_width, THUMB_H), "white")
    canvas.paste(scaled, (0, 0))
    return canvas


def _image_audit(data_url: str, label: str) -> dict:
    prefix = data_url.partition(",")[0]
    return {
        "label": label,
        "data_url_chars": len(data_url),
        "data_url_mime_prefix": prefix,
    }


def build_request_messages(
    request: RequestInput,
    *,
    prompt: str,
    model_html_evidence: str,
    model_html_audit: Mapping[str, object],
) -> tuple[list[dict], dict]:
    """Build selected-page evidence at the validated model-input geometry."""

    source = request.selected_source
    screenshot_path = Path(source["screenshot_path"])
    with Image.open(screenshot_path) as opened:
        selected = opened.convert("RGB").copy()
    geometry = request.geometry
    width = int(geometry["canvas_width"])
    common_height = int(geometry["common_height"])
    offsets = list(geometry["slice_offsets"])
    if geometry.get("n_slices") != len(offsets):
        raise Task3Error(f"{request.sample_id}: geometry slice count drifted")
    if selected.width != width or selected.height > common_height:
        raise Task3Error(f"{request.sample_id}: selected screenshot geometry drifted")

    version = request.selected_version
    content: list[dict] = [{"type": "text", "text": prompt}]
    image_audits: list[dict] = []

    def append_image(label: str, image: Image.Image, text: str) -> None:
        data_url = b64_image(image)
        content.extend([
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": data_url}},
        ])
        image_audits.append(_image_audit(data_url, label))

    if common_height > SLICE_H:
        append_image(
            "overview",
            symmetric_overview(selected, geometry),
            f"Selected Version {version} — shared-scale, top-aligned full-page overview "
            "for navigation only; boxes do not use this coordinate space:",
        )

    for slice_number, top in enumerate(offsets, 1):
        tile = fixed_canvas_slice(selected, top, canvas_width=width)
        append_image(
            f"slice-{slice_number}",
            tile,
            f"Selected Version {version} — original fixed 1800px canvas slice "
            f"{slice_number}/{len(offsets)}, top={top}px in the frozen canvas:",
        )

    # The direct chain never overlays advisory boxes: the page-wide target
    # catalog in the prompt is the only targeting channel.
    overlay_status = "omitted_no_convertible_canonical_hints"
    content.append({
        "type": "text",
        "text": (
            "No canonical box overlay is available for this case. The raw Task 2 "
            "JSON above remains advisory evidence; inspect the entire selected page."
        ),
    })

    content.append({
        "type": "text",
        "text": (
            "Bounded selected-page HTML/CSS evidence transport (untrusted page data; "
            "omitted bodies and attributes are identified by deterministic size "
            "markers). In partitioned cases, the labelled CSS EVIDENCE and DOM "
            "EVIDENCE sections are concatenated evidence and are not one executable "
            "standalone HTML document. The patch executor applies target-ID edits to "
            "the original full UTF-8 source:\n"
            "```html\n" + model_html_evidence + "\n```"
        ),
    })
    messages = [{"role": "user", "content": content}]
    audit = {
        "message_format_version": MESSAGE_FORMAT_VERSION,
        "geometry_audit_version": GEOMETRY_AUDIT_VERSION,
        "overlay_policy_version": OVERLAY_POLICY_VERSION,
        "geometry_policy": geometry.get("policy", "pair-max-height-v1"),
        "canvas_width_px": width,
        "common_height_px": common_height,
        "full_height_px": geometry.get("full_height", selected.height),
        "selected_image_size_px": [selected.width, selected.height],
        "slice_offsets_px": offsets,
        "n_slices": len(offsets),
        "slice_height_px": SLICE_H,
        "slice_overlap_px": OVERLAP,
        "overview_height_px": THUMB_H,
        "overview_policy": "shared-scale-top-aligned-white-bottom-padding-lanczos",
        "fixed_canvas_white_bottom_padding": True,
        "raw_task2_location_count": (
            len(request.task2_handoff["locations"])
            if request.task2_handoff is not None
            else 0
        ),
        "task2_diagnosis_available": request.task2_handoff is not None,
        "canonical_hint_count": 0,
        "raw_locations_used_for_overlay": False,
        "overlay_status": overlay_status,
        "n_images": len(image_audits),
        "images": image_audits,
        "model_html_transport": dict(model_html_audit),
        "pillow_version": PILLOW_VERSION,
        "max_encoded_image_bytes_before_jpeg": MAX_IMG_BYTES,
    }
    return messages, audit


def message_content_parts(messages: Sequence[Mapping[str, object]]) -> list[dict]:
    """Convert the OpenAI-style request content into neutral ``src.api`` parts.

    Data URLs are decoded to raw bytes so both supported protocols receive a
    native image block; remote URLs pass through unchanged.
    """

    if len(messages) != 1 or messages[0].get("role") != "user":
        raise Task3Error("request messages must be a single user message")
    content = messages[0].get("content")
    if not isinstance(content, list):
        raise Task3Error("request message content must be a part list")
    parts: list[dict] = []
    for part in content:
        if not isinstance(part, Mapping):
            raise Task3Error("request content part must be an object")
        if part.get("type") == "text":
            parts.append({"type": "text", "text": part["text"]})
            continue
        if part.get("type") == "image_url":
            url = part.get("image_url", {}).get("url")
            if not isinstance(url, str) or not url:
                raise Task3Error("request image part lacks a url")
            if url.startswith("data:"):
                header, _, encoded = url.partition(",")
                mime = header[len("data:"):].split(";")[0] or "image/png"
                try:
                    data = base64.b64decode(encoded, validate=True)
                except Exception as error:  # noqa: BLE001
                    raise Task3Error(f"request image data URL is invalid: {error}") from error
                parts.append({"type": "image", "data": data, "media_type": mime})
            else:
                parts.append({"type": "image", "url": url})
            continue
        raise Task3Error(f"unsupported request content part type: {part.get('type')!r}")
    return parts


def selected_html_asset_closure(html_path: Path) -> dict:
    """Canonical identity of every regular file under the page's assets/ tree."""

    html_path = Path(html_path)
    assets_dir = html_path.parent / "assets"
    records = []
    if assets_dir.is_dir():
        files = sorted(
            (path for path in assets_dir.rglob("*") if path.is_file()),
            key=lambda path: path.relative_to(html_path.parent).as_posix(),
        )
        for path in files:
            records.append({
                "path": path.relative_to(html_path.parent).as_posix(),
                "bytes": path.stat().st_size,
            })
    return {
        "version": ASSET_CLOSURE_VERSION,
        "root": "assets",
        "n_files": len(records),
        "n_bytes": sum(record["bytes"] for record in records),
        "records": records,
    }


def local_model_input_from_reference(
    request: RequestInput,
    reference: Mapping[str, object],
) -> dict:
    """Derive the frozen model-input geometry from the local degraded render."""

    try:
        audit = reference["bundle"]["roles"]["degraded"]["grading_render"]
        size = list(audit["screenshot"]["size_px"])
        screenshot_record = reference["manifest_record"]["screenshots"]["degraded"]
        screenshot_path = Path(reference["screenshot_paths"]["degraded"])
    except (KeyError, TypeError) as error:
        raise Task3Error(
            f"{request.sample_id}: validated degraded reference is incomplete"
        ) from error
    if (
        len(size) != 2
        or type(size[0]) is not int
        or type(size[1]) is not int
        or size[0] <= 0
        or size[1] <= 0
    ):
        raise Task3Error(f"{request.sample_id}: local degraded dimensions are invalid")
    offsets = slice_offsets(size[1], slice_height_px=SLICE_H, slice_overlap_px=OVERLAP)
    return {
        "policy_version": LOCAL_MODEL_INPUT_POLICY_VERSION,
        "screenshot": {
            "path": screenshot_path.resolve().as_posix(),
            "bytes": int(screenshot_record["bytes"]),
        },
        "geometry": {
            "policy": LOCAL_MODEL_INPUT_POLICY_VERSION,
            "canvas_width": size[0],
            "common_height": size[1],
            "full_height": size[1],
            "slice_offsets": offsets,
            "n_slices": len(offsets),
        },
        "viewport": [request.viewport[0], request.viewport[1]],
        "dpr": request.dpr,
    }


def request_materials_from_reference(reference: Mapping[str, object]) -> dict:
    """Return the frozen request section of a validated reference bundle.

    Paths inside the bundle are repository-relative; the returned copy points
    the local model-input screenshot at this checkout's absolute location.
    """

    bundle = reference.get("bundle")
    request_section = bundle.get("request") if isinstance(bundle, Mapping) else None
    if not isinstance(request_section, Mapping) or set(request_section) != {
        "target_catalog",
        "catalog_navigation_audit",
        "local_model_input",
    }:
        raise Task3Error("reference bundle lacks frozen request materials")
    materials = copy.deepcopy(dict(request_section))
    screenshot_path = Path(
        reference["screenshot_paths"]["degraded"]
    ).resolve(strict=True).as_posix()
    materials["local_model_input"]["screenshot"]["path"] = screenshot_path
    materials["local_screenshot_path"] = screenshot_path
    return materials


async def extract_target_catalog(
    browser, request: RequestInput, source_html: str
) -> tuple[dict, dict]:
    """Extract the GT-blind page-wide catalog from the selected source."""

    html_path = Path(request.selected_source["html_path"])
    marker_transport = build_source_marker_transport(source_html)

    async def render_variant(source_override: str | None) -> dict:
        context = await browser.new_context(
            java_script_enabled=False,
            viewport={"width": request.viewport[0], "height": request.viewport[1]},
            device_scale_factor=request.dpr,
        )
        try:
            page = await context.new_page()
            request_failures: list[str] = []
            page.on(
                "requestfailed",
                lambda failed: request_failures.append(failed.url[:1000]),
            )

            async def abort_external(route):
                await route.abort()

            await page.route(re.compile(r"^https?://"), abort_external)
            navigation = await goto_utf8_html(
                page,
                html_path,
                source_override=source_override,
                wait_until="load",
                optional_font_mode="required-local",
            )
            load = await wait_for_render_ready(
                page,
                settle_ms=SETTLE_MS,
                timeout_ms=RENDER_READY_TIMEOUT_MS,
            )
            motion = await freeze_render_motion(
                page,
                timeout_ms=DEFAULT_RENDER_MOTION_FREEZE_TIMEOUT_MS,
            )
            extraction = await page.evaluate(COLLECT_TARGETS_JS)
            scroll = await page.evaluate(
                """
                () => ({
                  width: Math.max(document.documentElement.scrollWidth,
                                  document.body ? document.body.scrollWidth : 0),
                  height: Math.max(document.documentElement.scrollHeight,
                                   document.body ? document.body.scrollHeight : 0),
                })
                """
            )
            png_payload = await page.screenshot(
                full_page=True, scale="device", timeout=120_000
            )
            png_audit = png_bytes_audit(png_payload)
            return {
                "navigation": navigation,
                "load": load,
                "motion": motion,
                "extraction": extraction,
                "scroll": scroll,
                "png_audit": png_audit,
                "png_payload": png_payload,
                "request_failures": request_failures,
            }
        finally:
            await context.close()

    for attempt in range(1, MAX_REQUEST_FAILURE_PAIR_ATTEMPTS + 1):
        original = await render_variant(None)
        marked = await render_variant(marker_transport["source_override"])
        rgb_exact = png_pixels_equal(original["png_payload"], marked["png_payload"])
        try:
            source_render_gate = build_source_render_gate_audit(
                original_extraction=original["extraction"],
                marker_extraction=marked["extraction"],
                original_scroll=original["scroll"],
                marker_scroll=marked["scroll"],
                original_png_audit=original["png_audit"],
                marker_png_audit=marked["png_audit"],
                original_load_audit=original["load"],
                marker_load_audit=marked["load"],
                original_motion_audit=original["motion"],
                marker_motion_audit=marked["motion"],
                original_navigation_audit=original["navigation"],
                marker_navigation_audit=marked["navigation"],
                original_request_failures=original["request_failures"],
                marker_request_failures=marked["request_failures"],
                browser_version=browser.version,
                viewport_width_css=request.viewport[0],
                dpr=request.dpr,
                rgb_exact=rgb_exact,
                pair_attempt_count=attempt,
                request_failure_mismatch_retry_count=attempt - 1,
            )
            break
        except SourceRenderGateError as error:
            if (
                "request-failure multiset" not in str(error)
                or attempt == MAX_REQUEST_FAILURE_PAIR_ATTEMPTS
            ):
                raise
    navigation_audit = copy.deepcopy(source_render_gate["navigation"])
    navigation_audit["font_mode"] = copy.deepcopy(source_render_gate["font_mode"])
    return (
        build_pagewide_target_catalog(
            marked["extraction"],
            [],
            source_marker_map=marker_transport["source_marker_map"],
            source_marker_audit=marker_transport["instrumentation_audit"],
            source_render_gate_audit=source_render_gate,
        ),
        navigation_audit,
    )


def prepare_request(
    request: RequestInput,
    *,
    provider: str,
    max_model_html_chars: int,
    request_materials: Mapping[str, object],
    union_keyword: str = "oneOf",
) -> PreparedRequest:
    """Materialize exactly one API request without any evaluator-only input."""

    if max_model_html_chars < 2_000:
        raise Task3Error("max_model_html_chars must be at least 2000")
    if union_keyword not in ("oneOf", "anyOf"):
        raise Task3Error(f"union_keyword must be oneOf or anyOf, got {union_keyword!r}")
    html_path = Path(request.selected_source["html_path"])
    source_bytes = html_path.read_bytes()
    try:
        source_html = source_bytes.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise Task3Error(
            f"{request.sample_id}: selected HTML is not strict UTF-8"
        ) from error
    model_html_evidence, model_html_audit = build_model_html_evidence(
        source_html,
        max_chars=max_model_html_chars,
        max_attribute_chars=DEFAULT_MAX_ATTRIBUTE_CHARS,
    )
    try:
        target_catalog = copy.deepcopy(request_materials["target_catalog"])
        catalog_navigation_audit = copy.deepcopy(
            request_materials["catalog_navigation_audit"]
        )
        local_model_input = copy.deepcopy(request_materials["local_model_input"])
        local_screenshot_path = Path(
            str(request_materials["local_screenshot_path"])
        ).resolve(strict=True)
        local_geometry = copy.deepcopy(local_model_input["geometry"])
    except (KeyError, TypeError, OSError) as error:
        raise Task3Error("invalid frozen request material") from error
    if (
        local_model_input.get("viewport") != list(request.viewport)
        or local_model_input.get("dpr") != request.dpr
    ):
        raise Task3Error("cached local model render context drifted")
    local_source = copy.deepcopy(request.selected_source)
    local_source["screenshot"] = local_model_input["screenshot"]["path"]
    local_source["screenshot_path"] = local_screenshot_path.as_posix()
    model_request = replace(
        request,
        geometry=local_geometry,
        selected_source=local_source,
    )
    target_view = model_target_view(target_catalog)
    protocol_kwargs = {
        "n_slices": int(model_request.geometry["n_slices"]),
        "canvas_width": int(model_request.geometry["canvas_width"]),
    }
    model_schema = advisory_protocol.advisory_patch_schema(
        model_request.task2_handoff,
        target_catalog,
        union_keyword=union_keyword,
        **protocol_kwargs,
    )
    response_format = advisory_protocol.strict_json_schema_response_format(
        model_request.task2_handoff,
        target_catalog,
        provider=provider,
        union_keyword=union_keyword,
        **protocol_kwargs,
    )
    if response_format["json_schema"]["schema"] != model_schema:
        raise AssertionError("provider response schema differs from local schema")
    prompt = advisory_protocol.build_structured_prompt(
        model_request.task2_handoff,
        target_catalog,
        **protocol_kwargs,
    )
    messages, message_audit = build_request_messages(
        model_request,
        prompt=prompt,
        model_html_evidence=model_html_evidence,
        model_html_audit=model_html_audit,
    )
    return PreparedRequest(
        request=model_request,
        local_model_input=local_model_input,
        source_html=source_html,
        model_html_evidence=model_html_evidence,
        model_html_audit=model_html_audit,
        target_catalog=target_catalog,
        catalog_navigation_audit=catalog_navigation_audit,
        target_view=target_view,
        model_schema=model_schema,
        response_format=response_format,
        prompt=prompt,
        messages=messages,
        message_audit=message_audit,
    )


CSS_SUPPORTS_JS = """
(pairs) => pairs.map(([property, value]) => ({
  property,
  value,
  supported: CSS.supports(property, value),
}))
"""


def css_pairs(resolved_patch: Mapping[str, object]) -> list[list[str]]:
    """Return unique property/value pairs for the browser CSS parser gate."""

    edits = resolved_patch.get("edits")
    if not isinstance(edits, list):
        raise Task3Error("resolved patch lacks an edits array")
    pairs: set[tuple[str, str]] = set()
    for edit in edits:
        if not isinstance(edit, Mapping):
            raise Task3Error("resolved patch contains a non-object edit")
        if edit.get("op") != "set_style":
            continue
        declarations = edit.get("declarations")
        if not isinstance(declarations, Mapping):
            raise Task3Error("set_style declarations are invalid")
        for prop, value in declarations.items():
            if not isinstance(prop, str) or not isinstance(value, str):
                raise Task3Error("CSS declaration must be string:string")
            pairs.add((prop, value))
    return [list(value) for value in sorted(pairs)]


def require_css_support(results: object) -> list[dict]:
    """Reject a model value unless Chromium's CSS parser accepts every pair."""

    if not isinstance(results, list):
        raise Task3Error("CSS.supports returned a non-array result")
    failures = [
        result
        for result in results
        if not isinstance(result, Mapping) or result.get("supported") is not True
    ]
    if failures:
        raise Task3Error(
            "CSS.supports rejected declarations: "
            + json.dumps(failures, ensure_ascii=False, sort_keys=True)[:1000]
        )
    return [dict(result) for result in results]


async def validate_css_in_browser(
    browser,
    selected_html_path: Path,
    resolved_patch: Mapping[str, object],
    *,
    css_width: int,
    dpr: float,
) -> list[dict]:
    pairs = css_pairs(resolved_patch)
    if not pairs:
        return []
    context = await browser.new_context(
        java_script_enabled=False,
        viewport={"width": css_width, "height": 900},
        device_scale_factor=dpr,
    )
    try:
        page = await context.new_page()

        async def abort_external(route):
            await route.abort()

        await page.route(re.compile(r"^https?://"), abort_external)
        await goto_utf8_html(page, selected_html_path)
        return require_css_support(await page.evaluate(CSS_SUPPORTS_JS, pairs))
    finally:
        await context.close()


async def process_raw_response(
    browser,
    prepared: PreparedRequest,
    raw_response: str,
) -> dict:
    """Parse, independently resolve, CSS-check, and apply to original HTML."""

    request = prepared.request
    protocol_kwargs = {
        "n_slices": request.geometry["n_slices"],
        "canvas_width": request.geometry["canvas_width"],
    }
    parsed = advisory_protocol.parse_structured_response(
        raw_response,
        request.task2_handoff,
        prepared.target_catalog,
        **protocol_kwargs,
    )
    source_patch, locator_by_xpath, resolution_log = resolve_model_patch(
        parsed["model_patch"], prepared.target_catalog
    )
    if (
        source_patch != parsed["source_patch"]
        or locator_by_xpath != parsed["locator_by_xpath"]
        or resolution_log != parsed["resolution_log"]
    ):
        raise Task3Error("protocol parser and target resolver disagree")
    css_support = await validate_css_in_browser(
        browser,
        Path(request.selected_source["html_path"]),
        source_patch,
        css_width=request.viewport[0],
        dpr=request.dpr,
    )
    fixed_html, source_edit_log = apply_source_patch(
        prepared.source_html,
        source_patch,
        locator_by_xpath,
    )
    return {
        "protocol": parsed,
        "model_patch": parsed["model_patch"],
        "resolved_patch": source_patch,
        "locator_by_xpath": locator_by_xpath,
        "resolution_log": resolution_log,
        "css_support": css_support,
        "source_edit_log": source_edit_log,
        "fixed_html": fixed_html,
        "n_edits": len(source_patch["edits"]),
    }


def provider_output_failure(raw_response: object) -> str | None:
    """Classify a completed provider response before the local JSON parser.

    The ``src.api`` client is called with ``require_stop=True``, so a
    non-standard finish reason or non-string content has already been rejected
    as ``InvalidResponseError`` upstream of this gate.
    """

    if not isinstance(raw_response, str):
        return "provider_content_missing_or_non_string"
    if len(raw_response) > MAX_RESPONSE_CHARS:
        return f"provider_content_exceeds_{MAX_RESPONSE_CHARS}_chars"
    return None


_RENDER_ENVIRONMENT_PROVENANCE: dict | None = None


def _validated_render_environment() -> dict[str, object]:
    global _RENDER_ENVIRONMENT_PROVENANCE
    if _RENDER_ENVIRONMENT_PROVENANCE is None:
        _RENDER_ENVIRONMENT_PROVENANCE = render_environment_provenance(
            render_fonts_dir=resolve_render_fonts_dir()
        )
    return copy.deepcopy(_RENDER_ENVIRONMENT_PROVENANCE)


def grading_render_policy() -> dict:
    """Return the authoritative local grading-render contract."""

    return {
        "version": GRADING_RENDER_POLICY_VERSION,
        "render_environment": _validated_render_environment(),
        "java_script_enabled": False,
        "navigation_wait_until": "load",
        "external_http_requests": "aborted",
        "optional_font_mode": "required-local",
        "native_optional_font_mode": "rejected",
        "font_cache_warm_reload_policy": FONT_CACHE_WARM_RELOAD_POLICY,
        "chromium_render_policy_version": CHROMIUM_RENDER_POLICY_VERSION,
        "chromium_launch_args": list(CHROMIUM_RENDER_ARGS),
        "fonts_ready_gate": "document.fonts.ready_or_unsupported",
        "images_ready_gate": "all_document_images_complete_load_or_error",
        "render_ready_contract_version": RENDER_READY_CONTRACT_VERSION,
        "render_ready_timeout_ms": RENDER_READY_TIMEOUT_MS,
        "post_ready_settle_ms": GRADING_SETTLE_MS,
        "render_motion_freeze_contract_version": (
            RENDER_MOTION_FREEZE_CONTRACT_VERSION
        ),
        "render_motion_freeze_timeout_ms": (
            DEFAULT_RENDER_MOTION_FREEZE_TIMEOUT_MS
        ),
        "render_motion_freeze_order": (
            "after_render_ready_settle_before_extract_and_screenshot"
        ),
        "render_motion_freeze_scope": "web_animations_and_svg_smil",
        "render_motion_smil_current_time_tolerance_seconds": (
            SMIL_CURRENT_TIME_TOLERANCE_SECONDS
        ),
        "render_motion_freeze_gate": (
            "all_web_animations_and_smil_roots_paused_at_time_zero_after_two_layout_flushes"
        ),
        "viewport_height_css": GRADING_VIEWPORT_HEIGHT_CSS,
        "full_page_screenshot": True,
        "screenshot_scale": "device",
        "baseline_degraded_authority": "hermetic_local_render",
        "full_page_physical_width_formula": (
            "round(max(viewport_width_css,scroll_width_css)*dpr)"
        ),
        "local_screenshot_decode_integrity": "hard_failure",
        "physical_width_formula_mismatch": "hard_failure",
        "local_render_gate_precedes_scoring": True,
    }


def _rendered_screenshot_audit(path: Path) -> dict:
    try:
        with Image.open(path) as image:
            image.load()
            if image.format != "PNG" or image.width <= 0 or image.height <= 0:
                raise ValueError("screenshot is not a non-empty PNG")
            mode = image.mode
            size = list(image.size)
        png_bytes = path.stat().st_size
        if png_bytes <= 0:
            raise ValueError("screenshot is empty")
    except Exception as error:  # noqa: BLE001
        raise Task3Error(f"cannot audit grading screenshot {path}: {error}") from error
    return {
        "path": path.resolve().as_posix(),
        "png_bytes": png_bytes,
        "format": "PNG",
        "mode": mode,
        "size_px": size,
    }


def expected_full_page_physical_width(
    css_width: int | float,
    scroll_width_css: int | float,
    dpr: int | float,
) -> int:
    """Mirror Playwright full-page width for a viewport with horizontal overflow."""

    values = {
        "css_width": css_width,
        "scroll_width_css": scroll_width_css,
        "dpr": dpr,
    }
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0
        for value in values.values()
    ):
        raise GradingRenderGateError(
            f"invalid full-page physical-width inputs: {values}",
            {"policy": grading_render_policy(), "passed": False},
        )
    return round(
        max(float(css_width), float(scroll_width_css)) * float(dpr)
    )


async def grading_render_scan_capture(
    browser,
    html_path: Path,
    *,
    css_width: int,
    dpr: float,
    screenshot_path: Path,
    role: str,
) -> tuple[dict, dict]:
    """Render one grading artifact under the authoritative local contract."""

    if role not in {"baseline", "degraded", "fixed"}:
        raise Task3Error(f"invalid grading render role {role!r}")
    policy = grading_render_policy()
    context = await browser.new_context(
        java_script_enabled=False,
        viewport={"width": css_width, "height": GRADING_VIEWPORT_HEIGHT_CSS},
        device_scale_factor=dpr,
    )
    try:
        page = await context.new_page()

        async def abort_external(route):
            await route.abort()

        await page.route(re.compile(r"^https?://"), abort_external)
        page.on("dialog", lambda dialog: asyncio.ensure_future(dialog.dismiss()))
        navigation_audit = await goto_utf8_html(
            page,
            html_path,
            wait_until="load",
            optional_font_mode="required-local",
        )
        load_audit = await wait_for_render_ready(
            page,
            settle_ms=GRADING_SETTLE_MS,
            timeout_ms=RENDER_READY_TIMEOUT_MS,
        )
        motion_audit = await freeze_render_motion(
            page,
            timeout_ms=DEFAULT_RENDER_MOTION_FREEZE_TIMEOUT_MS,
        )
        data = await page.evaluate(element_index.COLLECT_JS, element_index.STYLE_PROPS)
        full_page_geometry = await page.evaluate(
            """
            () => ({
              width: Math.max(
                document.documentElement.scrollWidth,
                document.body ? document.body.scrollWidth : 0
              ),
              height: Math.max(
                document.documentElement.scrollHeight,
                document.body ? document.body.scrollHeight : 0
              ),
            })
            """
        )
        screenshot_path.parent.mkdir(parents=True, exist_ok=True)
        await page.screenshot(
            path=str(screenshot_path), full_page=True, scale="device"
        )
        screenshot_audit = _rendered_screenshot_audit(screenshot_path)
        scroll_width_css = full_page_geometry.get("width")
        scroll_height_css = full_page_geometry.get("height")
        if (
            isinstance(scroll_width_css, bool)
            or not isinstance(scroll_width_css, (int, float))
            or not math.isfinite(float(scroll_width_css))
            or scroll_width_css <= 0
            or isinstance(scroll_height_css, bool)
            or not isinstance(scroll_height_css, (int, float))
            or not math.isfinite(float(scroll_height_css))
            or scroll_height_css <= 0
        ):
            raise GradingRenderGateError(
                "grading page reported invalid full-page scroll geometry",
                {
                    "policy": grading_render_policy(),
                    "role": role,
                    "css_width": css_width,
                    "dpr": dpr,
                    "scroll_width_css": scroll_width_css,
                    "scroll_height_css": scroll_height_css,
                    "screenshot": screenshot_audit,
                    "passed": False,
                },
            )
        expected_physical_width = expected_full_page_physical_width(
            css_width, scroll_width_css, dpr
        )
        if screenshot_audit["size_px"][0] != expected_physical_width:
            raise GradingRenderGateError(
                "grading screenshot width disagrees with full-page viewport/scroll/DPR formula",
                {
                    "policy": grading_render_policy(),
                    "role": role,
                    "css_width": css_width,
                    "dpr": dpr,
                    "scroll_width_css": scroll_width_css,
                    "expected_physical_width": expected_physical_width,
                    "screenshot": screenshot_audit,
                    "passed": False,
                },
            )
        element_index_payload = {
            "id": html_path.parent.name,
            "viewport": [css_width, GRADING_VIEWPORT_HEIGHT_CSS],
            "dpr": dpr,
            "scrollW": scroll_width_css,
            "scrollH": scroll_height_css,
            "overflowX": scroll_width_css > css_width + 1,
            "n_elements": data["n_elements"],
            "peer_groups": element_index.detect_peer_groups(data["elements"]),
            "elements": data["elements"],
        }
        audit = {
            "policy_version": GRADING_RENDER_POLICY_VERSION,
            "render_environment": copy.deepcopy(policy["render_environment"]),
            "role": role,
            "html_path": html_path.resolve().as_posix(),
            "html_bytes": html_path.stat().st_size,
            "browser_version": browser.version,
            "css_width": css_width,
            "dpr": dpr,
            "scroll_width_css": scroll_width_css,
            "scroll_height_css": scroll_height_css,
            "document_element_scroll_width_css": data["scrollW"],
            "document_element_scroll_height_css": data["scrollH"],
            "expected_physical_width": expected_physical_width,
            "navigation_audit": copy.deepcopy(navigation_audit),
            "load_audit": copy.deepcopy(load_audit),
            "motion_freeze_audit": copy.deepcopy(motion_audit),
            "render_contract": {
                "java_script_enabled": False,
                "external_http_requests": "aborted",
                "navigation_wait_until": navigation_audit["wait_until"],
                "html_navigation_transport_version": navigation_audit[
                    "transport_version"
                ],
                "optional_font_mode": "required-local",
                "font_cache_warm_reload_policy": FONT_CACHE_WARM_RELOAD_POLICY,
                "render_ready_contract_version": load_audit[
                    "contract_version"
                ],
                "render_ready_timeout_ms": load_audit[
                    "readiness_timeout_ms"
                ],
                "post_ready_settle_ms": load_audit["settle_ms"],
                "render_motion_freeze_contract_version": (
                    motion_audit["contract_version"]
                ),
                "render_motion_freeze_timeout_ms": motion_audit["timeout_ms"],
                "render_motion_freeze_order": (
                    "after_render_ready_settle_before_extract_and_screenshot"
                ),
                "render_motion_freeze_scope": (
                    "web_animations_and_svg_smil"
                ),
                "render_motion_smil_current_time_tolerance_seconds": (
                    motion_audit["smil_current_time_tolerance_seconds"]
                ),
                "render_motion_freeze_two_layout_flushes": motion_audit[
                    "two_style_layout_flushes_completed"
                ],
                "viewport_height_css": GRADING_VIEWPORT_HEIGHT_CSS,
                "full_page_screenshot": True,
                "screenshot_scale": "device",
            },
            "screenshot": screenshot_audit,
            "font_mode": {
                "policy": "required-local-only-v2",
                "mode": "required-local",
                "native_fallback_allowed": False,
            },
        }
        return element_index_payload, audit
    finally:
        await context.close()


def require_local_grading_renders(
    grading_render: Mapping[str, object],
) -> dict[str, object]:
    """Validate authoritative baseline/degraded local render artifacts."""

    policy = grading_render_policy()
    validated = {}
    try:
        for role in ("baseline", "degraded"):
            role_audit = grading_render[role]
            if (
                not isinstance(role_audit, Mapping)
                or role_audit.get("policy_version") != GRADING_RENDER_POLICY_VERSION
                or role_audit.get("role") != role
                or role_audit.get("render_environment") != policy["render_environment"]
                or role_audit.get("font_mode")
                != {
                    "policy": "required-local-only-v2",
                    "mode": "required-local",
                    "native_fallback_allowed": False,
                }
            ):
                raise ValueError(f"{role} local render contract is invalid")
            navigation = role_audit.get("navigation_audit")
            if (
                not isinstance(navigation, Mapping)
                or navigation.get("optional_font_mode") != "required-local"
            ):
                raise ValueError(f"{role} did not use required-local fonts")
            expected_width = expected_full_page_physical_width(
                role_audit["css_width"],
                role_audit["scroll_width_css"],
                role_audit["dpr"],
            )
            if role_audit.get("expected_physical_width") != expected_width:
                raise ValueError(f"{role} expected width audit is invalid")
            screenshot = role_audit.get("screenshot")
            expected_screenshot_fields = {
                "path",
                "png_bytes",
                "format",
                "mode",
                "size_px",
            }
            if (
                not isinstance(screenshot, Mapping)
                or set(screenshot) != expected_screenshot_fields
            ):
                raise ValueError(f"{role} screenshot audit is missing or invalid")
            current = _rendered_screenshot_audit(Path(str(screenshot.get("path", ""))))
            if current["size_px"][0] != expected_width:
                raise ValueError(f"{role} local screenshot width mismatch")
            if current["size_px"] != screenshot.get("size_px"):
                raise ValueError(f"{role} local screenshot dimensions drifted")
            validated[role] = {
                "png_bytes": current["png_bytes"],
                "size_px": current["size_px"],
                "expected_physical_width": expected_width,
            }
    except (KeyError, TypeError, ValueError, OSError, Task3Error) as error:
        raise GradingRenderGateError(
            f"authoritative local grading render failed: {error}",
            grading_render,
        ) from error
    return {
        "required_roles": ["baseline", "degraded"],
        "authority": "hermetic_local_render",
        "validated": validated,
        "all_passed": True,
    }


def require_grading_browser_character_set(
    catalog_navigation_audit: Mapping[str, object],
    grading_render: Mapping[str, object],
) -> dict:
    """Hard-check that catalog, baseline, degraded, and fixed decode alike."""

    try:
        audits = {
            "catalog_original": catalog_navigation_audit["original"],
            "catalog_marker": catalog_navigation_audit["marker"],
            "baseline": grading_render["baseline"]["navigation_audit"],
            "degraded": grading_render["degraded"]["navigation_audit"],
            "fixed": grading_render["fixed"]["navigation_audit"],
        }
        return require_same_browser_character_set(
            audits,
            required_labels=(
                "catalog_original",
                "catalog_marker",
                "baseline",
                "degraded",
                "fixed",
            ),
        )
    except (KeyError, TypeError, Utf8HtmlTransportError) as error:
        raise GradingRenderGateError(
            f"grading browser character-set gate failed: {error}",
            grading_render,
        ) from error


def require_same_grading_render_contract(
    grading_render: Mapping[str, object],
) -> dict:
    """Prove all three role renders used one identical stability contract."""

    try:
        contracts = {
            role: grading_render[role]["render_contract"]
            for role in ("baseline", "degraded", "fixed")
        }
    except (KeyError, TypeError) as error:
        raise GradingRenderGateError(
            f"grading role render contract is incomplete: {error}",
            grading_render,
        ) from error
    if contracts["baseline"] != contracts["degraded"] or contracts["baseline"] != contracts["fixed"]:
        raise GradingRenderGateError(
            "baseline/degraded/fixed grading render contracts differ",
            grading_render,
        )
    return {
        "required_roles": ["baseline", "degraded", "fixed"],
        "all_equal": True,
    }


def ensure_relative_assets_link(run_dir: Path, assets_dir: Path) -> Path:
    """Create or validate a run-local, workspace-portable assets symlink."""

    run_dir.mkdir(parents=True, exist_ok=True)
    assets_dir = assets_dir.resolve(strict=True)
    link = run_dir / "assets"
    relative_target = os.path.relpath(assets_dir, start=run_dir.resolve())
    if link.is_symlink():
        raw_target = os.readlink(link)
        if os.path.isabs(raw_target) or link.resolve(strict=True) != assets_dir:
            raise Task3Error(f"run assets link is not portable: {link}")
        return link
    if link.exists():
        raise Task3Error(f"run assets entry is not a symlink: {link}")

    tmp = run_dir / ".assets-portable.tmp"
    if tmp.exists() or tmp.is_symlink():
        raise Task3Error(f"refusing to overwrite stale temporary link: {tmp}")
    try:
        tmp.symlink_to(relative_target)
        if not tmp.exists() or tmp.resolve(strict=True) != assets_dir:
            raise Task3Error(f"new run assets link is invalid: {link}")
        os.replace(tmp, link)
    finally:
        if tmp.exists() or tmp.is_symlink():
            tmp.unlink()
    return link


def validate_scoring_gt_row(
    row: Mapping[str, object],
    sample_id: str,
    *,
    schema_version: str | None = GT_SCHEMA_VERSION,
) -> None:
    """Validate fields consumed by lesion and clean-page safety scoring."""

    if row.get("sample_id") != sample_id:
        raise Task3Error(f"{sample_id}: scoring GT row has mismatched sample_id")
    if schema_version is not None and row.get("schema_version") != schema_version:
        raise Task3Error(f"{sample_id}: scoring GT schema mismatch")
    context = row.get("baseline_context")
    if not isinstance(context, Mapping):
        raise Task3Error(f"{sample_id}: scoring GT lacks baseline_context")
    if not isinstance(context.get("peer_groups"), list):
        raise Task3Error(f"{sample_id}: scoring GT baseline_context lacks peer_groups")
    if not isinstance(context.get("image_natural_by_xpath"), Mapping):
        raise Task3Error(
            f"{sample_id}: scoring GT baseline_context lacks image natural sizes"
        )
    defects = row.get("defects")
    if not isinstance(defects, list) or not defects:
        raise Task3Error(f"{sample_id}: scoring GT defects must be non-empty")
    for key in ("safety_gate_rule_ids", "safety_report_only_rule_ids"):
        values = row.get(key)
        if not isinstance(values, list) or any(
            not isinstance(value, str) or not value for value in values
        ):
            raise Task3Error(f"{sample_id}: scoring GT {key} must be a string array")


def frozen_scoring_gt(item) -> dict:
    validate_scoring_gt_row(item.scoring_gt, item.sample_id)
    return item.scoring_gt


def final_metric_fields(
    item,
    *,
    lesion_score: Mapping[str, object],
    content_preservation: Mapping[str, object],
    visual_scope: Mapping[str, object],
) -> dict:
    """Compute Repair Pass from F, C, P, and S only."""

    gates = item.task2_metric_gates
    fix_pass = lesion_score.get("fix_pass") is True
    collateral_pass = lesion_score.get("collateral_pass") is True
    content_preservation_pass = content_preservation.get("passed") is True
    visual_scope_pass = visual_scope.get("visual_scope_pass") is True
    repair_pass = bool(
        fix_pass
        and collateral_pass
        and content_preservation_pass
        and visual_scope_pass
    )
    return {
        "task2_det_correct": gates["det_correct"] is True,
        "task2_loc_exact_iou50": gates["loc_exact_iou50"] is True,
        "task2_loc_iou50_tp_diagnostic": gates["loc_iou50_tp"],
        "task2_loc_iou50_fp_diagnostic": gates["loc_iou50_fp"],
        "task2_loc_iou50_fn_diagnostic": gates["loc_iou50_fn"],
        "fix_pass": fix_pass,
        "collateral_pass": collateral_pass,
        "content_preservation_pass": content_preservation_pass,
        "visual_scope_pass": visual_scope_pass,
        METRIC_MAIN: repair_pass,
        "all_pass": repair_pass,
    }


async def grade_applied_sample(
    browser,
    item,
    prepared: PreparedRequest,
    applied: Mapping[str, object],
    reference: Mapping[str, object],
    *,
    fixed_path: Path,
    run_dir: Path,
) -> dict:
    if item.request is None:
        raise Task3Error("cannot grade an upstream no-call sample")
    request = item.request
    manifest = item.task3_record
    scoring_gt = frozen_scoring_gt(item)
    degraded_path = Path(manifest["artifact_paths"]["degraded_html"])
    assets_dir = degraded_path.parent / "assets"
    ensure_relative_assets_link(run_dir, assets_dir)
    fixed_screenshot = run_dir / "fixed_fullpage.png"
    selected_kind = item.selection["kind"]
    if selected_kind != "degraded":
        raise Task3Error("Repair Pass grading requires the actual degraded page")
    selected_role = "degraded"
    try:
        reference_bundle = reference["bundle"]
        reference_roles = reference_bundle["roles"]
        reference_screenshots = reference["screenshot_paths"]
        baseline_ei = copy.deepcopy(reference_roles["baseline"]["element_index"])
        degraded_ei = copy.deepcopy(reference_roles["degraded"]["element_index"])
        selected_before_screenshot = Path(reference_screenshots[selected_role])
        baseline_screenshot = Path(reference_screenshots["baseline"])
    except (KeyError, TypeError) as error:
        raise Task3Error("invalid model-independent grading reference") from error
    grading_render: dict[str, object] = {
        "policy": grading_render_policy(),
        "selected_kind": selected_kind,
        "reference_cache": {
            "policy_version": reference_cache.POLICY_VERSION,
            "bundle": copy.deepcopy(reference["manifest_record"]["bundle"]),
            "screenshots": copy.deepcopy(
                reference["manifest_record"]["screenshots"]
            ),
            "baseline_degraded_rendered_during_model_run": False,
            "fixed_rendered_during_model_run": True,
        },
        "baseline": copy.deepcopy(
            reference_roles["baseline"]["grading_render"]
        ),
        "degraded": copy.deepcopy(
            reference_roles["degraded"]["grading_render"]
        ),
    }
    try:
        grading_render["local_render_gate"] = require_local_grading_renders(
            grading_render
        )
        selected_font_mode = grading_render[selected_role]["font_mode"]["mode"]
        catalog_font_mode = prepared.catalog_navigation_audit.get("font_mode")
        if (
            not isinstance(catalog_font_mode, Mapping)
            or catalog_font_mode.get("mode") != selected_font_mode
            or selected_font_mode != "required-local"
        ):
            raise Task3Error("selected catalog/grading font modes differ")
        grading_render["selected_font_mode_gate"] = {
            "selected_role": selected_role,
            "catalog_mode": catalog_font_mode["mode"],
            "grading_mode": selected_font_mode,
            "all_equal": True,
        }
        fixed_ei, grading_render["fixed"] = await grading_render_scan_capture(
            browser,
            fixed_path,
            css_width=request.viewport[0],
            dpr=request.dpr,
            screenshot_path=fixed_screenshot,
            role="fixed",
        )
        grading_render["same_contract_gate"] = (
            require_same_grading_render_contract(grading_render)
        )
        # The catalog extraction and all three grading renders must have used
        # the same browser-decoded document character set. This prevents a
        # repair from being scored against a differently decoded DOM.
        grading_render["charset_gate"] = require_grading_browser_character_set(
            prepared.catalog_navigation_audit, grading_render
        )
    except GradingRenderGateError as error:
        if set(error.grading_render).intersection({"baseline", "degraded", "fixed"}):
            raise
        grading_render["gate_failure"] = copy.deepcopy(error.grading_render)
        raise GradingRenderGateError(str(error), grading_render) from error
    except Exception as error:  # noqa: BLE001
        grading_render["gate_failure"] = {
            "type": type(error).__name__,
            "message": str(error)[:2000],
        }
        raise GradingRenderGateError(
            f"grading render contract failed: {type(error).__name__}: {error}",
            grading_render,
        ) from error
    lesion_score = grade_page(baseline_ei, degraded_ei, fixed_ei, scoring_gt)
    content = evaluate_content_preservation(
        prepared.source_html,
        str(applied["fixed_html"]),
        authorized_patch=applied["resolved_patch"],
        locator_by_xpath=applied["locator_by_xpath"],
    )
    permitted_regions = frozen_gt_permitted_regions(scoring_gt)
    scope = score_visual_scope(
        selected_before_screenshot,
        fixed_screenshot,
        permitted_regions,
        dpr=request.dpr,
    )
    scope = apply_baseline_restoration_fast_path(
        scope,
        fixed_screenshot,
        baseline_screenshot,
    )
    metrics = final_metric_fields(
        item,
        lesion_score=lesion_score,
        content_preservation=content,
        visual_scope=scope,
    )
    result = {
        **lesion_score,
        "grading_render": grading_render,
        "content_preservation": content,
        "visual_scope": scope,
        **metrics,
    }
    if result["all_pass"] != result[METRIC_MAIN]:
        raise AssertionError("all_pass must equal Repair Pass")
    atomic_write_json(run_dir / "content_preservation.json", content)
    atomic_write_json(run_dir / "visual_scope.json", scope)
    atomic_write_json(run_dir / "lesion_score.json", lesion_score)
    atomic_write_json(run_dir / "grading_render.json", grading_render)
    return result


def failure_record(
    item,
    reason: str,
    *,
    status: str,
    api_attempted: bool = False,
) -> dict:
    request = item.request
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "sample_id": item.sample_id,
        "eval_version": EVAL_VERSION,
        "status": status,
        "error": reason[:2000],
        "api_attempted": api_attempted,
        "api_attempted_now": False,
        "call_eligible": item.call_eligible,
        "upstream_failure_reason": item.upstream_failure_reason,
        "task2_handoff": (
            copy.deepcopy(request.task2_handoff) if request is not None else None
        ),
        "selection": copy.deepcopy(item.selection),
        "n_lesions": int(item.task3_record["n_defects"]),
        "n_lesions_fixed": 0,
        "task2_det_correct": item.task2_metric_gates["det_correct"],
        "task2_loc_exact_iou50": item.task2_metric_gates["loc_exact_iou50"],
        "fix_pass": False,
        "collateral_pass": False,
        "content_preservation_pass": False,
        "visual_scope_pass": False,
        METRIC_MAIN: False,
        "all_pass": False,
    }


async def run_one_model_sample(
    browser,
    item,
    prepared: PreparedRequest,
    reference: Mapping[str, object],
    *,
    client,
    model: str,
    run_dir: Path,
    retries: int,
) -> dict:
    """Call the provider once (with transport retries), then grade the repair."""

    if retries < 1:
        raise Task3Error("retries must be positive")
    started = time.time()
    attempts = 0
    raw_response = None
    last_error: Exception | None = None
    for attempt in range(retries):
        attempts = attempt + 1
        try:
            raw_response = await client.complete(
                model=model,
                parts=message_content_parts(prepared.messages),
                temperature=0,
                max_tokens=MODEL_MAX_TOKENS,
                disable_thinking=model.lower().startswith("qwen"),
                response_format=prepared.response_format,
                require_stop=True,
            )
            break
        except Exception as error:  # noqa: BLE001
            last_error = error
            if not is_retryable_api_error(error) or attempt + 1 >= retries:
                break
            await asyncio.sleep(5 * (attempt + 1))
    if raw_response is None:
        if isinstance(last_error, InvalidResponseError):
            record = failure_record(
                item,
                f"provider_output:invalid_response:{last_error}",
                status="provider_output_failure",
                api_attempted=True,
            )
        else:
            record = failure_record(
                item,
                f"api_transport:{type(last_error).__name__}:{last_error}",
                status="api_transport_failure",
                api_attempted=True,
            )
        record.update({
            "api_attempted_now": True,
            "api_latency_s": round(time.time() - started, 3),
            "api_attempts": attempts,
        })
        return record
    api_latency_s = round(time.time() - started, 3)

    provider_failure = provider_output_failure(raw_response)
    if provider_failure is not None:
        record = failure_record(
            item,
            f"provider_output:{provider_failure}",
            status="provider_output_failure",
            api_attempted=True,
        )
        record.update({
            "api_attempted_now": True,
            "api_latency_s": api_latency_s,
            "api_attempts": attempts,
            "raw_response": raw_response,
        })
        return record

    try:
        applied = await process_raw_response(browser, prepared, raw_response)
    except Exception as error:  # noqa: BLE001
        record = failure_record(
            item,
            f"local_protocol:{type(error).__name__}:{error}",
            status="local_protocol_failure",
            api_attempted=True,
        )
        record.update({
            "api_attempted_now": True,
            "api_latency_s": api_latency_s,
            "api_attempts": attempts,
            "raw_response": raw_response,
        })
        return record

    fixed_path = run_dir / "fixed.html"
    atomic_write_text(fixed_path, str(applied["fixed_html"]))

    try:
        grade = await grade_applied_sample(
            browser,
            item,
            prepared,
            applied,
            reference,
            fixed_path=fixed_path,
            run_dir=run_dir,
        )
    except Exception as error:  # noqa: BLE001
        record = failure_record(
            item,
            f"grade:{type(error).__name__}:{error}",
            status="grade_failure",
            api_attempted=True,
        )
        record.update({
            "api_attempted_now": True,
            "api_latency_s": api_latency_s,
            "api_attempts": attempts,
            "raw_response": raw_response,
            "model_patch": applied["model_patch"],
            "resolved_patch": applied["resolved_patch"],
        })
        if isinstance(error, GradingRenderGateError):
            record["grading_render"] = copy.deepcopy(error.grading_render)
        return record

    request = prepared.request
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "sample_id": item.sample_id,
        "eval_version": EVAL_VERSION,
        "status": "complete",
        "api_attempted": True,
        "api_attempted_now": True,
        "api_latency_s": api_latency_s,
        "api_attempts": attempts,
        "call_eligible": True,
        "upstream_failure_reason": None,
        "task2_handoff": copy.deepcopy(request.task2_handoff),
        "input_geometry": copy.deepcopy(request.geometry),
        "selection": copy.deepcopy(item.selection),
        "model_html_transport": copy.deepcopy(prepared.model_html_audit),
        "message_audit": copy.deepcopy(prepared.message_audit),
        "raw_response": raw_response,
        "model_patch": applied["model_patch"],
        "resolved_patch": applied["resolved_patch"],
        "resolution_log": applied["resolution_log"],
        "css_support": applied["css_support"],
        "source_edit_log": applied["source_edit_log"],
        "n_edits": applied["n_edits"],
        **grade,
    }


def load_result_jsonl(
    path: Path,
    *,
    selected_ids: Sequence[str],
) -> dict[str, dict]:
    if not path.exists():
        return {}
    selected = set(selected_ids)
    records: dict[str, dict] = {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.endswith("\n"):
                    raise Task3Error(
                        f"result JSONL line {line_number} is not newline-terminated"
                    )
                if not line.strip():
                    raise Task3Error(f"result JSONL line {line_number} is blank")
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise Task3Error(
                        f"invalid result JSONL line {line_number}: {error}"
                    ) from error
                if not isinstance(record, dict):
                    raise Task3Error("result JSONL record is not an object")
                sample_id = record.get("sample_id")
                if not isinstance(sample_id, str) or sample_id not in selected:
                    raise Task3Error(
                        f"result JSONL contains out-of-selection sample {sample_id!r}"
                    )
                if sample_id in records:
                    raise Task3Error(
                        f"result JSONL contains duplicate sample {sample_id}"
                    )
                if (
                    record.get("schema_version") != RESULT_SCHEMA_VERSION
                    or record.get("eval_version") != EVAL_VERSION
                ):
                    raise Task3Error(f"{sample_id}: result protocol version mismatch")
                for component in METRIC_COMPONENTS + (METRIC_MAIN,):
                    if type(record.get(component)) is not bool:
                        raise Task3Error(
                            f"{sample_id}: Repair Pass field {component!r} missing"
                        )
                expected_repair_pass = all(
                    record[component] for component in METRIC_COMPONENTS
                )
                if record.get(METRIC_MAIN) is not expected_repair_pass:
                    raise Task3Error(f"{sample_id}: Repair Pass formula mismatch")
                if record.get("all_pass") != record.get(METRIC_MAIN):
                    raise Task3Error(f"{sample_id}: all_pass/main mismatch")
                records[sample_id] = record
    except UnicodeDecodeError as error:
        raise Task3Error(f"result JSONL is not UTF-8: {error}") from error
    return records


def append_result_record(path: Path, record: dict) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        record, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ) + "\n"
    with path.open("a", encoding="utf-8", newline="") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return record


def summarize_records(records: Sequence[Mapping[str, object]]) -> dict:
    n = len(records)
    if any(record.get("all_pass") != record.get(METRIC_MAIN) for record in records):
        raise Task3Error("all_pass/main metric mismatch in output records")
    counts = {
        component: sum(record.get(component) is True for record in records)
        for component in METRIC_COMPONENTS
    }
    repair_pass = sum(record.get(METRIC_MAIN) is True for record in records)
    statuses = Counter(str(record.get("status")) for record in records)
    summary = {
        "n": n,
        "fixed_denominator_includes_all_failures": True,
        "status_counts": dict(sorted(statuses.items())),
        "n_complete": statuses.get("complete", 0),
        "n_api_attempted_ever": sum(
            record.get("api_attempted") is True for record in records
        ),
        "n_api_attempted_this_invocation": sum(
            record.get("api_attempted_now") is True for record in records
        ),
        "task2_diagnostics": {
            "det_correct_count": sum(
                record.get("task2_det_correct") is True for record in records
            ),
            "loc_exact_iou50_count": sum(
                record.get("task2_loc_exact_iou50") is True for record in records
            ),
        },
        "repair_pass_count": repair_pass,
        "repair_pass_rate": repair_pass / n if n else 0.0,
        "metric_display_names": METRIC_DISPLAY_NAMES,
    }
    for component, count in counts.items():
        summary[f"{component}_count"] = count
        summary[f"{component}_rate"] = count / n if n else 0.0
    return summary


def expected_summary(
    *,
    out_path: Path,
    records: Sequence[Mapping[str, object]],
) -> dict:
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "eval_version": EVAL_VERSION,
        "result_path": Path(out_path).resolve().as_posix(),
        **summarize_records(records),
    }
