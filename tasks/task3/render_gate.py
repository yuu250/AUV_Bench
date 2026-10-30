"""Strict required-local render equivalence gate for source markers.

The gate compares fresh original and marker-instrumented renders under the
same hermetic local font environment. Collector projection, scroll geometry,
browser charset, readiness, motion freeze, request failures, and full-page PNG
geometry are hard gates. RGB equality is diagnostic because pages can contain
nondeterministic raster content outside the catalog projection.

All comparisons are direct (structures, byte lengths, pixel bands); no digests
are computed anywhere.
"""

from __future__ import annotations

import io
import json
import math
from pathlib import Path
from typing import BinaryIO, Mapping

from PIL import Image

from tasks.task3.render_transport import (
    DEFAULT_RENDER_READY_TIMEOUT_MS,
    DEFAULT_RENDER_SETTLE_MS,
    HTML_NAVIGATION_TRANSPORT_VERSION,
    LOCAL_CSS_FONT_DISPLAY_POLICY_VERSION,
    RENDER_READY_CONTRACT_VERSION,
    RENDER_MOTION_FREEZE_CONTRACT_VERSION,
    require_same_browser_character_set,
    validate_render_motion_freeze_audit,
)


SOURCE_RENDER_GATE_VERSION = "task3-source-marker-required-local-equivalence-v10"
MAX_REQUEST_FAILURE_PAIR_ATTEMPTS = 4
PNG_AUDIT_VERSION = "pil-decode-size-v1"
PNG_RGB_BAND_HEIGHT = 512


class SourceRenderGateError(RuntimeError):
    """Temporary source instrumentation changed the current page render model."""


def _rgb_bands(image: Image.Image):
    """Yield RGB pixel bytes in bands without one full-page buffer."""

    width, height = image.size
    for top in range(0, height, PNG_RGB_BAND_HEIGHT):
        bottom = min(height, top + PNG_RGB_BAND_HEIGHT)
        band = image.crop((0, top, width, bottom))
        try:
            converted = band if band.mode == "RGB" else band.convert("RGB")
            try:
                yield converted.tobytes()
            finally:
                if converted is not band:
                    converted.close()
        finally:
            band.close()


def png_pixels_equal(original: bytes, marker: bytes) -> bool:
    """Direct banded RGB comparison of two PNG payloads (diagnostic only)."""

    images = []
    try:
        for payload in (original, marker):
            image = Image.open(io.BytesIO(payload))
            image.load()
            images.append(image)
        if images[0].size != images[1].size:
            return False
        return all(
            original_band == marker_band
            for original_band, marker_band in zip(
                _rgb_bands(images[0]), _rgb_bands(images[1]), strict=True
            )
        )
    finally:
        for image in images:
            image.close()


def _decoded_png_audit(source: BinaryIO) -> dict:
    try:
        with Image.open(source) as image:
            image.load()
            width, height = image.size
            if width <= 0 or height <= 0:
                raise SourceRenderGateError("PNG has an empty image size")
    except SourceRenderGateError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise SourceRenderGateError(f"cannot decode PNG: {exc}") from exc
    return {
        "audit_version": PNG_AUDIT_VERSION,
        "size": [width, height],
    }


def png_bytes_audit(payload: bytes) -> dict:
    """Return the decode-and-size audit for screenshot bytes."""

    if not isinstance(payload, bytes) or not payload:
        raise SourceRenderGateError("current PNG payload must be non-empty bytes")
    return _decoded_png_audit(io.BytesIO(payload))


def png_file_audit(path: str | Path) -> dict:
    """Audit a PNG file without reading its encoded bytes into one buffer."""

    png_path = Path(path)
    try:
        with png_path.open("rb") as handle:
            return _decoded_png_audit(handle)
    except SourceRenderGateError:
        raise
    except OSError as exc:
        raise SourceRenderGateError(f"cannot read PNG {png_path}: {exc}") from exc


def _validate_png_audit(value: object, name: str) -> dict:
    expected = {
        "audit_version",
        "size",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise SourceRenderGateError(f"{name} PNG audit has invalid fields")
    size = value["size"]
    if (
        value["audit_version"] != PNG_AUDIT_VERSION
        or not isinstance(size, list)
        or len(size) != 2
        or any(
            isinstance(item, bool) or not isinstance(item, int) or item <= 0
            for item in size
        )
    ):
        raise SourceRenderGateError(f"{name} PNG audit is invalid")
    return dict(value)


def catalog_projection(extraction: object) -> list[dict]:
    """Remove only the temporary marker value from full collector output."""

    if not isinstance(extraction, Mapping) or set(extraction) != {"elements"}:
        raise SourceRenderGateError("browser extraction must contain exactly elements")
    values = extraction["elements"]
    if not isinstance(values, list):
        raise SourceRenderGateError("browser extraction elements must be an array")
    projection: list[dict] = []
    for index, row in enumerate(values):
        if not isinstance(row, Mapping) or "source_marker" not in row:
            raise SourceRenderGateError(
                f"browser extraction element {index} lacks source_marker"
            )
        clean = dict(row)
        clean.pop("source_marker")
        projection.append(clean)
    return projection


def _scroll(value: object, name: str) -> dict:
    if (
        not isinstance(value, Mapping)
        or set(value) != {"width", "height"}
        or any(
            isinstance(value[key], bool)
            or not isinstance(value[key], (int, float))
            or not math.isfinite(float(value[key]))
            or not value[key] > 0
            for key in ("width", "height")
        )
    ):
        raise SourceRenderGateError(f"{name} scroll size is invalid")
    return {"width": value["width"], "height": value["height"]}


def _request_failure_audit(urls: object, name: str) -> dict:
    if not isinstance(urls, list) or any(not isinstance(url, str) for url in urls):
        raise SourceRenderGateError(f"{name} request failures are invalid")
    clean = sorted(urls)
    return {
        "count": len(clean),
        "urls": clean,
    }


_LOAD_KEYS = {
    "fonts_status",
    "n_images",
    "n_images_complete",
    "n_images_natural_nonzero",
    "contract_version",
    "wait_until",
    "settle_ms",
    "readiness_timeout_ms",
    "readiness_timed_out",
}


def _validate_load_audit(value: object, name: str) -> dict:
    if not isinstance(value, Mapping) or set(value) != _LOAD_KEYS:
        raise SourceRenderGateError(f"{name} load audit has invalid fields")
    counts = [
        value["n_images"],
        value["n_images_complete"],
        value["n_images_natural_nonzero"],
    ]
    if (
        value["contract_version"] != RENDER_READY_CONTRACT_VERSION
        or value["wait_until"] != "load"
        or value["settle_ms"] != DEFAULT_RENDER_SETTLE_MS
        or value["readiness_timeout_ms"] != DEFAULT_RENDER_READY_TIMEOUT_MS
        or value["readiness_timed_out"] is not False
        or value["fonts_status"] not in {"loaded", "unsupported"}
        or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0
            for item in counts
        )
        or value["n_images_complete"] != value["n_images"]
        or value["n_images_natural_nonzero"] > value["n_images_complete"]
    ):
        raise SourceRenderGateError(f"{name} load audit is invalid")
    return dict(value)


_NAVIGATION_KEYS = {
    "transport_version",
    "url",
    "source_path",
    "source_bytes",
    "offline_source_encoding",
    "offline_source_utf8_valid",
    "source_override_used",
    "served_bytes",
    "served_payload_utf8_valid",
    "meta_charset_scan_bytes",
    "source_declared_charset",
    "served_declared_charset",
    "charset_response_policy",
    "response_headers_supplied",
    "response_content_type",
    "response_charset",
    "browser_character_set",
    "browser_charset_forced_by_transport",
    "optional_font_mode",
    "local_css_font_display_audit",
    "wait_until",
    "navigation_timeout_ms",
}


def _validate_navigation_audit(
    value: object, name: str, *, source_override_used: bool
) -> dict:
    if not isinstance(value, Mapping) or set(value) != _NAVIGATION_KEYS:
        raise SourceRenderGateError(f"{name} navigation audit has invalid fields")
    nullable_strings = (
        "source_declared_charset",
        "served_declared_charset",
        "response_charset",
    )
    if (
        value["transport_version"] != HTML_NAVIGATION_TRANSPORT_VERSION
        or not isinstance(value["url"], str)
        or not value["url"].startswith("file://")
        or not isinstance(value["source_path"], str)
        or not value["source_path"]
        or any(
            isinstance(value[key], bool)
            or not isinstance(value[key], int)
            or value[key] < 0
            for key in ("source_bytes", "served_bytes")
        )
        or value["offline_source_encoding"] != "utf-8-strict"
        or value["offline_source_utf8_valid"] is not True
        or value["served_payload_utf8_valid"] is not True
        or value["source_override_used"] is not source_override_used
        or value["meta_charset_scan_bytes"] != 1024
        or any(
            value[key] is not None and not isinstance(value[key], str)
            for key in nullable_strings
        )
        or value["source_declared_charset"] != value["served_declared_charset"]
        or value["charset_response_policy"] not in {
            "page_declared_non_utf8_preserved_no_response_charset_override",
            "page_declared_utf8_preserved_no_response_charset_override",
            "no_page_charset_chromium_sniffed_or_default_no_response_charset_override",
        }
        or value["response_headers_supplied"] is not True
        or value["response_content_type"] != "text/html"
        or value["response_charset"] is not None
        or not isinstance(value["browser_character_set"], str)
        or not value["browser_character_set"].strip()
        or not isinstance(value["browser_charset_forced_by_transport"], bool)
        or value["optional_font_mode"] != "required-local"
        or not isinstance(value["local_css_font_display_audit"], Mapping)
        or value["local_css_font_display_audit"].get("policy_version")
        != LOCAL_CSS_FONT_DISPLAY_POLICY_VERSION
        or value["local_css_font_display_audit"].get("mode")
        != "required-local"
        or value["local_css_font_display_audit"].get("rewrite_enabled") is not True
        or value["local_css_font_display_audit"].get("passed") is not True
        or value["wait_until"] != "load"
        or value["navigation_timeout_ms"] != 20_000
    ):
        raise SourceRenderGateError(f"{name} navigation audit is invalid")
    if value["browser_charset_forced_by_transport"] is not False:
        raise SourceRenderGateError(f"{name} navigation charset policy is inconsistent")
    return dict(value)


def _required_local_font_mode() -> dict:
    return {
        "policy": "required-local-only-v2",
        "mode": "required-local",
        "native_fallback_allowed": False,
    }


def _validate_required_local_font_mode(value: object) -> dict:
    expected = _required_local_font_mode()
    if value != expected:
        raise SourceRenderGateError("source render font mode is not required-local")
    return dict(expected)


def build_source_render_gate_audit(
    *,
    original_extraction: object,
    marker_extraction: object,
    original_scroll: object,
    marker_scroll: object,
    original_png_audit: Mapping[str, object],
    marker_png_audit: Mapping[str, object],
    original_load_audit: Mapping[str, object],
    marker_load_audit: Mapping[str, object],
    original_motion_audit: Mapping[str, object],
    marker_motion_audit: Mapping[str, object],
    original_navigation_audit: Mapping[str, object],
    marker_navigation_audit: Mapping[str, object],
    original_request_failures: list[str],
    marker_request_failures: list[str],
    browser_version: str,
    viewport_width_css: int,
    dpr: int | float,
    rgb_exact: bool,
    pair_attempt_count: int = 1,
    request_failure_mismatch_retry_count: int = 0,
) -> dict:
    """Build a fail-closed current original/marker equivalence audit."""

    if not isinstance(rgb_exact, bool):
        raise SourceRenderGateError("rgb_exact must be a boolean")
    if not isinstance(browser_version, str) or not browser_version:
        raise SourceRenderGateError("browser_version is invalid")
    if (
        isinstance(viewport_width_css, bool)
        or not isinstance(viewport_width_css, int)
        or viewport_width_css <= 0
        or isinstance(dpr, bool)
        or not isinstance(dpr, (int, float))
        or not math.isfinite(float(dpr))
        or not dpr > 0
        or isinstance(pair_attempt_count, bool)
        or not isinstance(pair_attempt_count, int)
        or pair_attempt_count < 1
        or isinstance(request_failure_mismatch_retry_count, bool)
        or not isinstance(request_failure_mismatch_retry_count, int)
        or request_failure_mismatch_retry_count < 0
        or request_failure_mismatch_retry_count >= pair_attempt_count
    ):
        raise SourceRenderGateError("viewport/DPR or pair-attempt audit is invalid")
    original_projection = catalog_projection(original_extraction)
    marker_projection = catalog_projection(marker_extraction)
    if original_projection != marker_projection:
        raise SourceRenderGateError(
            "temporary source markers changed DOM/bbox/computed-style projection: "
            f"original {len(original_projection)} elements vs "
            f"marker {len(marker_projection)} elements"
        )
    clean_original_scroll = _scroll(original_scroll, "original")
    clean_marker_scroll = _scroll(marker_scroll, "marker")
    if clean_original_scroll != clean_marker_scroll:
        raise SourceRenderGateError(
            "temporary source markers changed scroll size: "
            f"original={clean_original_scroll}, marker={clean_marker_scroll}"
        )
    expected_width = round(
        max(viewport_width_css, float(clean_original_scroll["width"]))
        * float(dpr)
    )
    clean_original_png = _validate_png_audit(original_png_audit, "original current")
    clean_marker_png = _validate_png_audit(marker_png_audit, "marker current")
    if clean_original_png["size"] != clean_marker_png["size"]:
        raise SourceRenderGateError(
            "current original/marker full-page PNG sizes differ: "
            f"original={clean_original_png['size']}, marker={clean_marker_png['size']}"
        )
    if clean_original_png["size"][0] != expected_width:
        raise SourceRenderGateError(
            "current full-page PNG width disagrees with scroll geometry/DPR: "
            f"current={clean_original_png['size'][0]}, expected={expected_width}"
        )
    clean_original_load = _validate_load_audit(original_load_audit, "original")
    clean_marker_load = _validate_load_audit(marker_load_audit, "marker")
    if clean_original_load != clean_marker_load:
        raise SourceRenderGateError("temporary source markers changed render-ready audit")
    try:
        clean_original_motion = validate_render_motion_freeze_audit(
            original_motion_audit
        )
        clean_marker_motion = validate_render_motion_freeze_audit(
            marker_motion_audit
        )
    except Exception as exc:  # noqa: BLE001
        raise SourceRenderGateError("source render motion freeze audit is invalid") from exc
    if clean_original_motion != clean_marker_motion:
        raise SourceRenderGateError(
            "temporary source markers changed the active animation multiset"
        )
    clean_original_navigation = _validate_navigation_audit(
        original_navigation_audit, "original", source_override_used=False
    )
    clean_marker_navigation = _validate_navigation_audit(
        marker_navigation_audit, "marker", source_override_used=True
    )
    navigation_identity_keys = (
        "url",
        "source_path",
        "source_bytes",
        "offline_source_encoding",
        "source_declared_charset",
        "charset_response_policy",
        "response_content_type",
        "response_charset",
        "browser_charset_forced_by_transport",
        "optional_font_mode",
        "local_css_font_display_audit",
    )
    if any(
        clean_original_navigation[key] != clean_marker_navigation[key]
        for key in navigation_identity_keys
    ):
        raise SourceRenderGateError("original/marker navigation source or charset policy differs")
    charset_gate = require_same_browser_character_set(
        {"original": clean_original_navigation, "marker": clean_marker_navigation},
        required_labels=("original", "marker"),
    )
    original_failures = _request_failure_audit(
        original_request_failures, "original"
    )
    marker_failures = _request_failure_audit(marker_request_failures, "marker")
    if original_failures != marker_failures:
        raise SourceRenderGateError(
            "temporary source markers changed the request-failure multiset: "
            f"original={original_failures}, marker={marker_failures}"
        )
    font_mode = _validate_required_local_font_mode(_required_local_font_mode())
    return {
        "version": SOURCE_RENDER_GATE_VERSION,
        "browser_version": browser_version,
        "viewport_width_css": viewport_width_css,
        "dpr": float(dpr),
        "expected_full_page_physical_width": expected_width,
        "pair_attempt_count": pair_attempt_count,
        "request_failure_mismatch_retry_count": (
            request_failure_mismatch_retry_count
        ),
        "projection_exact": True,
        "scroll_exact": True,
        "original_scroll_css": clean_original_scroll,
        "marker_scroll_css": clean_marker_scroll,
        "load_audit_exact": True,
        "original_load_audit": clean_original_load,
        "marker_load_audit": clean_marker_load,
        "motion_freeze": {
            "contract_version": RENDER_MOTION_FREEZE_CONTRACT_VERSION,
            "exact": True,
            "original": clean_original_motion,
            "marker": clean_marker_motion,
        },
        "navigation": {
            "original": clean_original_navigation,
            "marker": clean_marker_navigation,
            "charset_gate": charset_gate,
        },
        "font_mode": font_mode,
        "request_failures": {
            "exact": True,
            "original": original_failures,
            "marker": marker_failures,
        },
        "local_screenshots": {
            "original": clean_original_png,
            "marker": clean_marker_png,
            "size_exact": True,
            "physical_width_exact": True,
            "rgb_exact": rgb_exact,
            "rgb_exact_is_diagnostic_only": True,
        },
    }


def validate_source_render_gate_audit(value: object) -> dict:
    """Strictly validate a serialized gate result, including nested shapes."""

    expected = {
        "version",
        "browser_version",
        "viewport_width_css",
        "dpr",
        "expected_full_page_physical_width",
        "pair_attempt_count",
        "request_failure_mismatch_retry_count",
        "projection_exact",
        "scroll_exact",
        "original_scroll_css",
        "marker_scroll_css",
        "load_audit_exact",
        "original_load_audit",
        "marker_load_audit",
        "motion_freeze",
        "navigation",
        "font_mode",
        "request_failures",
        "local_screenshots",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise SourceRenderGateError("source render gate audit has invalid fields")
    if (
        value["version"] != SOURCE_RENDER_GATE_VERSION
        or not isinstance(value["browser_version"], str)
        or not value["browser_version"]
        or value["projection_exact"] is not True
        or value["scroll_exact"] is not True
        or value["load_audit_exact"] is not True
        or isinstance(value["pair_attempt_count"], bool)
        or not isinstance(value["pair_attempt_count"], int)
        or value["pair_attempt_count"] < 1
        or isinstance(value["request_failure_mismatch_retry_count"], bool)
        or not isinstance(value["request_failure_mismatch_retry_count"], int)
        or value["request_failure_mismatch_retry_count"] < 0
        or value["request_failure_mismatch_retry_count"]
        >= value["pair_attempt_count"]
    ):
        raise SourceRenderGateError("source render equivalence gate did not pass")
    original_scroll = _scroll(value["original_scroll_css"], "original")
    marker_scroll = _scroll(value["marker_scroll_css"], "marker")
    if original_scroll != marker_scroll:
        raise SourceRenderGateError("source render scroll geometry differs")
    viewport = value["viewport_width_css"]
    dpr = value["dpr"]
    expected_width = value["expected_full_page_physical_width"]
    if (
        isinstance(viewport, bool)
        or not isinstance(viewport, int)
        or viewport <= 0
        or isinstance(dpr, bool)
        or not isinstance(dpr, (int, float))
        or not math.isfinite(float(dpr))
        or dpr <= 0
        or expected_width
        != round(max(viewport, float(original_scroll["width"])) * float(dpr))
    ):
        raise SourceRenderGateError("source render gate viewport/DPR is invalid")
    original_load = _validate_load_audit(value["original_load_audit"], "original")
    marker_load = _validate_load_audit(value["marker_load_audit"], "marker")
    if original_load != marker_load:
        raise SourceRenderGateError("source render load audits differ")
    motion = value["motion_freeze"]
    if (
        not isinstance(motion, Mapping)
        or set(motion) != {"contract_version", "exact", "original", "marker"}
        or motion["contract_version"] != RENDER_MOTION_FREEZE_CONTRACT_VERSION
        or motion["exact"] is not True
    ):
        raise SourceRenderGateError("source render motion freeze gate is invalid")
    try:
        original_motion = validate_render_motion_freeze_audit(motion["original"])
        marker_motion = validate_render_motion_freeze_audit(motion["marker"])
    except Exception as exc:  # noqa: BLE001
        raise SourceRenderGateError("source render motion freeze audit is invalid") from exc
    if original_motion != marker_motion:
        raise SourceRenderGateError("source render motion freeze audits differ")
    navigation = value["navigation"]
    if not isinstance(navigation, Mapping) or set(navigation) != {
        "original",
        "marker",
        "charset_gate",
    }:
        raise SourceRenderGateError("source render navigation audit is invalid")
    original_navigation = _validate_navigation_audit(
        navigation["original"], "original", source_override_used=False
    )
    marker_navigation = _validate_navigation_audit(
        navigation["marker"], "marker", source_override_used=True
    )
    identity_keys = (
        "url",
        "source_path",
        "source_bytes",
        "offline_source_encoding",
        "source_declared_charset",
        "charset_response_policy",
        "response_content_type",
        "response_charset",
        "browser_charset_forced_by_transport",
        "optional_font_mode",
        "local_css_font_display_audit",
    )
    if any(
        original_navigation[key] != marker_navigation[key]
        for key in identity_keys
    ):
        raise SourceRenderGateError("source render navigation identity differs")
    try:
        expected_charset_gate = require_same_browser_character_set(
            {"original": original_navigation, "marker": marker_navigation},
            required_labels=("original", "marker"),
        )
    except Exception as exc:  # noqa: BLE001
        raise SourceRenderGateError("source render charset gate failed") from exc
    if navigation["charset_gate"] != expected_charset_gate:
        raise SourceRenderGateError("source render charset audit is inconsistent")
    _validate_required_local_font_mode(value["font_mode"])
    failures = value["request_failures"]
    if (
        not isinstance(failures, Mapping)
        or set(failures) != {"exact", "original", "marker"}
        or failures["exact"] is not True
    ):
        raise SourceRenderGateError("source render request-failure audit is invalid")
    clean_failure_audits = []
    for label in ("original", "marker"):
        audit = failures[label]
        if (
            not isinstance(audit, Mapping)
            or set(audit) != {"count", "urls"}
            or isinstance(audit["count"], bool)
            or not isinstance(audit["count"], int)
            or audit["count"] < 0
            or not isinstance(audit["urls"], list)
            or any(not isinstance(url, str) for url in audit["urls"])
            or audit["urls"] != sorted(audit["urls"])
            or audit["count"] != len(audit["urls"])
        ):
            raise SourceRenderGateError(
                f"source render {label} request failures are invalid"
            )
        clean_failure_audits.append(dict(audit))
    if clean_failure_audits[0] != clean_failure_audits[1]:
        raise SourceRenderGateError("source render request failures differ")
    screenshots = value["local_screenshots"]
    expected_screenshot_fields = {
        "original",
        "marker",
        "size_exact",
        "physical_width_exact",
        "rgb_exact",
        "rgb_exact_is_diagnostic_only",
    }
    if (
        not isinstance(screenshots, Mapping)
        or set(screenshots) != expected_screenshot_fields
        or screenshots["size_exact"] is not True
        or screenshots["physical_width_exact"] is not True
        or screenshots["rgb_exact_is_diagnostic_only"] is not True
        or not isinstance(screenshots["rgb_exact"], bool)
    ):
        raise SourceRenderGateError("source render local screenshots are invalid")
    original_png = _validate_png_audit(
        screenshots["original"], "original current"
    )
    marker_png = _validate_png_audit(screenshots["marker"], "marker current")
    if (
        original_png["size"] != marker_png["size"]
        or original_png["size"][0] != expected_width
    ):
        raise SourceRenderGateError(
            "source render local screenshot audit is inconsistent"
        )
    try:
        clean = json.loads(json.dumps(value, ensure_ascii=False))
    except (TypeError, ValueError) as exc:
        raise SourceRenderGateError("source render gate is not JSON-safe") from exc
    return clean


__all__ = [
    "MAX_REQUEST_FAILURE_PAIR_ATTEMPTS",
    "PNG_AUDIT_VERSION",
    "PNG_RGB_BAND_HEIGHT",
    "SOURCE_RENDER_GATE_VERSION",
    "SourceRenderGateError",
    "build_source_render_gate_audit",
    "catalog_projection",
    "png_bytes_audit",
    "png_file_audit",
    "validate_source_render_gate_audit",
]
