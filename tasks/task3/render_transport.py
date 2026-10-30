"""Byte-preserving local HTML navigation plus strict source-file validation.

Task 3 edits source as UTF-8 text, so every input and override must still be
strictly UTF-8 decodable/encodable offline.  Browser rendering is a different
contract: benchmark pages may deliberately contain another HTML charset
declaration, and the frozen screenshots were produced with Chromium honoring
that declaration. The main-document route therefore sends a MIME-only
``Content-Type: text/html`` header and omits any response charset override:
Chromium remains free to honor a page declaration or
apply the same sniff/default behavior as the frozen local document.  Its actual
``document.characterSet`` is always
recorded instead of inferred.

The original ``file://`` URL remains unchanged so relative assets keep their
benchmark resolution semantics.  ``source_override`` is still in-memory and
never mutates the source file.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Mapping
from html.parser import HTMLParser
from pathlib import Path
from typing import Literal
from urllib.parse import unquote, urlparse

from playwright.async_api import Page


HTML_NAVIGATION_TRANSPORT_VERSION = (
    "task3-local-byte-preserving-page-charset-explicit-optional-font-mode-v5"
)
FONT_CACHE_WARM_RELOAD_POLICY = (
    "explicit-native-or-required-local-fontface-terminal-exact-reload-v3"
)
LOCAL_CSS_FONT_DISPLAY_POLICY_VERSION = (
    "local-css-font-display-explicit-native-or-optional-to-block-in-memory-v2"
)
OPTIONAL_FONT_MODES = frozenset({"native", "required-local"})
CHROMIUM_RENDER_ARGS = (
    "--font-render-hinting=none",
    "--disable-lcd-text",
)
CHROMIUM_RENDER_POLICY_VERSION = "task3-chromium-pinned-font-rasterization-v1"
_UTF8_LABELS = frozenset({"utf-8", "utf8", "unicode-1-1-utf-8"})
_META_CHARSET_SCAN_BYTES = 1_024
DEFAULT_RENDER_SETTLE_MS = 800
DEFAULT_RENDER_READY_TIMEOUT_MS = 20_000
RENDER_READY_CONTRACT_VERSION = (
    "utf8-local-force-fontfaces-images-deadline-v2"
)
DEFAULT_RENDER_MOTION_FREEZE_TIMEOUT_MS = 5_000
RENDER_MOTION_FREEZE_CONTRACT_VERSION = (
    "web-animations-smil-pause-zero-two-layout-flush-verify-v2"
)
SMIL_CURRENT_TIME_TOLERANCE_SECONDS = 1e-6
_FONT_DISPLAY_OPTIONAL_PATTERN = re.compile(
    rb"font-display\s*:\s*optional",
    flags=re.IGNORECASE,
)


class Utf8HtmlTransportError(RuntimeError):
    """Raised when a local HTML document cannot be transported as UTF-8."""


class _FirstMetaCharsetParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self.charset: str | None = None

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        if self.charset is not None or tag.lower() != "meta":
            return
        values = {
            key.lower(): value.strip() if isinstance(value, str) else ""
            for key, value in attrs
        }
        direct = values.get("charset", "").strip(" \t\r\n\f\"'")
        if direct:
            self.charset = direct
            return
        if values.get("http-equiv", "").lower() != "content-type":
            return
        match = re.search(
            r"(?i)(?:^|;)\s*charset\s*=\s*([^;\s]+)",
            values.get("content", ""),
        )
        if match:
            self.charset = match.group(1).strip(" \t\r\n\f\"'")


def _declared_html_charset(payload: bytes) -> str | None:
    """Return the first browser-relevant early meta charset, if present."""

    parser = _FirstMetaCharsetParser()
    parser.feed(
        payload[:_META_CHARSET_SCAN_BYTES].decode("utf-8", errors="ignore")
    )
    return parser.charset


def _charset_response_policy(declared: str | None) -> tuple[dict, str]:
    if declared is not None and declared.strip().lower() not in _UTF8_LABELS:
        return (
            {"Content-Type": "text/html"},
            "page_declared_non_utf8_preserved_no_response_charset_override",
        )
    if declared is not None:
        return (
            {"Content-Type": "text/html"},
            "page_declared_utf8_preserved_no_response_charset_override",
        )
    return (
        {"Content-Type": "text/html"},
        "no_page_charset_chromium_sniffed_or_default_no_response_charset_override",
    )


def write_utf8_html(html_path: Path, source: str) -> None:
    """Write HTML as strict UTF-8 without platform newline translation."""

    path = Path(html_path)
    if not isinstance(source, str):
        raise TypeError("HTML source must be a string")
    try:
        path.write_text(
            source,
            encoding="utf-8",
            errors="strict",
            newline="",
        )
    except (OSError, UnicodeEncodeError) as error:
        raise Utf8HtmlTransportError(
            f"cannot write HTML file as UTF-8: {path}: {error}"
        ) from error


async def goto_utf8_html(
    page: Page,
    html_path: Path,
    *,
    source_override: str | None = None,
    wait_until: Literal["commit", "domcontentloaded", "load", "networkidle"] = (
        "domcontentloaded"
    ),
    timeout: float = 20_000,
    optional_font_mode: Literal["native", "required-local"] = "required-local",
) -> dict:
    """Navigate with exact bytes and return the observed charset/provenance audit.

    Only the exact main-document URL is intercepted.  Subresources continue
    to load from the page's unchanged ``file://`` base URL.  ``source_override``
    may supply an in-memory UTF-8 rendering copy while retaining that original
    URL; the on-disk file is still validated and is never modified.  No
    response charset overrides a non-UTF-8 page declaration: Chromium must
    apply the page's declaration exactly as it does for the frozen document.
    """

    if optional_font_mode not in OPTIONAL_FONT_MODES:
        raise ValueError(
            "optional_font_mode must be 'native' or 'required-local'"
        )
    path = Path(html_path).resolve()
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise Utf8HtmlTransportError(
            f"cannot read HTML file {path}: {error}"
        ) from error
    source_payload = payload
    source_bytes = len(payload)
    try:
        payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise Utf8HtmlTransportError(
            f"HTML file is not valid UTF-8: {path}: byte {error.start}"
        ) from error

    if source_override is not None:
        if not isinstance(source_override, str):
            raise TypeError("source_override must be a string or None")
        try:
            payload = source_override.encode("utf-8", errors="strict")
        except UnicodeEncodeError as error:
            raise Utf8HtmlTransportError(
                f"source_override is not encodable as UTF-8: {error}"
            ) from error
    source_declared_charset = _declared_html_charset(source_payload)
    served_declared_charset = _declared_html_charset(payload)
    if served_declared_charset != source_declared_charset:
        raise Utf8HtmlTransportError(
            "source_override changed the page's early meta charset declaration"
        )
    response_headers, charset_policy = _charset_response_policy(
        served_declared_charset
    )

    url = path.as_uri()
    css_route_pattern = re.compile(r"^file://.*\.css(?:[?#].*)?$", re.IGNORECASE)
    css_records: dict[str, dict] = {}
    css_seen: dict[str, tuple[bytes, bytes]] = {}
    missing_css_requests: dict[str, int] = {}
    css_route_errors: list[str] = []

    async def fulfill_local_css_with_required_fonts(route) -> None:
        """Remove the local-font timing race without changing source assets.

        ``font-display: optional`` permanently chooses fallback metrics when a
        face misses its very short first-layout window, even if FontFace.load()
        succeeds later.  Rewriting only that descriptor in the in-memory CSS
        response makes Chromium wait for the already-local face.  The CSS file
        on disk, its URL, and every other declaration remain unchanged.
        """

        request_url = route.request.url
        parsed = urlparse(request_url)
        if parsed.scheme.lower() != "file":
            await route.fallback()
            return
        css_path = Path(unquote(parsed.path)).resolve()
        if not css_path.is_file():
            # Preserve Chromium's native missing-subresource behavior. Some
            # frozen pages contain root-relative CSS links that were already
            # absent when the benchmark screenshot was captured.
            missing_css_requests[request_url] = (
                missing_css_requests.get(request_url, 0) + 1
            )
            await route.fallback()
            return
        try:
            source_css = css_path.read_bytes()
            served_css, replacements = _FONT_DISPLAY_OPTIONAL_PATTERN.subn(
                b"font-display:block",
                source_css,
            )
            key = css_path.as_posix()
            if css_seen.setdefault(key, (source_css, served_css)) != (
                source_css,
                served_css,
            ):
                raise Utf8HtmlTransportError(
                    f"local CSS changed during one navigation: {css_path}"
                )
            record = css_records.setdefault(
                key,
                {
                    "path": key,
                    "source_bytes": len(source_css),
                    "served_bytes": len(served_css),
                    "request_count": 0,
                    "rewritten_request_count": 0,
                    "font_display_optional_replacements": 0,
                },
            )
            record["request_count"] += 1
            if replacements:
                record["rewritten_request_count"] += 1
            record["font_display_optional_replacements"] += replacements
            await route.fulfill(
                status=200,
                body=served_css,
                headers={"Content-Type": "text/css"},
            )
        except Exception as error:  # noqa: BLE001
            css_route_errors.append(f"{request_url}: {type(error).__name__}: {error}")
            await route.abort()

    async def fulfill_main_document(route) -> None:
        # Send only the HTML MIME type and no response charset so Chromium
        # honors a page meta declaration or applies its own sniff/default.
        # The MIME is explicit because a headerless fulfilled file:// response
        # can be nondeterministically treated as text/plain.
        kwargs = {"status": 200, "body": payload}
        kwargs["headers"] = response_headers
        await route.fulfill(**kwargs)

    async def navigate_once() -> str:
        await page.route(url, fulfill_main_document, times=1)
        try:
            await page.goto(url, wait_until=wait_until, timeout=timeout)
        finally:
            # Also removes the handler if navigation failed before the
            # one-shot route was consumed.
            await page.unroute(url, fulfill_main_document)
        observed = await page.evaluate("document.characterSet")
        if not isinstance(observed, str) or not observed.strip():
            raise Utf8HtmlTransportError(
                f"browser did not report a document character set: {observed!r}"
            )
        return observed

    css_route_installed = optional_font_mode == "required-local"
    if css_route_installed:
        await page.route(css_route_pattern, fulfill_local_css_with_required_fonts)
    try:
        # Warm every declared face inside the same isolated browser context,
        # then navigate the exact same payload again. The local CSS transport
        # above changes only `font-display: optional` to `block` before the
        # first layout, so a successfully loaded local face cannot be silently
        # replaced with fallback metrics by Chromium's optional-font window.
        warm_character_set = await navigate_once()
        try:
            warm_fonts = await asyncio.wait_for(
                page.evaluate(
                    """
                    async () => {
                      if (!document.fonts) return {status: 'unsupported'};
                      const faces = Array.from(document.fonts);
                      await Promise.all(faces.map(async (face) => {
                        try { await face.load(); } catch (_) { /* terminal error */ }
                      }));
                      await document.fonts.ready;
                      return {status: 'ready'};
                    }
                    """
                ),
                timeout=(float(timeout) / 1000) + 1.0,
            )
        except asyncio.TimeoutError as error:
            raise Utf8HtmlTransportError(
                f"font-cache warmup exceeded outer deadline {timeout}ms"
            ) from error
        if warm_fonts not in ({"status": "ready"}, {"status": "unsupported"}):
            raise Utf8HtmlTransportError("font-cache warmup returned invalid state")

        character_set = await navigate_once()
    finally:
        if css_route_installed:
            await page.unroute(
                css_route_pattern,
                fulfill_local_css_with_required_fonts,
            )
    if css_route_errors:
        raise Utf8HtmlTransportError(
            "local CSS font-display transport failed: "
            + "; ".join(css_route_errors[:3])
        )
    if not isinstance(character_set, str) or not character_set.strip():
        raise Utf8HtmlTransportError(
            f"browser did not report a document character set: {character_set!r}"
        )
    if character_set.strip().upper() != warm_character_set.strip().upper():
        raise Utf8HtmlTransportError(
            "browser character set changed across font-cache warm reload"
        )
    return {
        "transport_version": HTML_NAVIGATION_TRANSPORT_VERSION,
        "url": url,
        "source_path": path.as_posix(),
        "source_bytes": source_bytes,
        "offline_source_encoding": "utf-8-strict",
        "offline_source_utf8_valid": True,
        "source_override_used": source_override is not None,
        "served_bytes": len(payload),
        "served_payload_utf8_valid": True,
        "meta_charset_scan_bytes": _META_CHARSET_SCAN_BYTES,
        "source_declared_charset": source_declared_charset,
        "served_declared_charset": served_declared_charset,
        "charset_response_policy": charset_policy,
        "response_headers_supplied": True,
        "response_content_type": response_headers["Content-Type"],
        "response_charset": None,
        "browser_character_set": character_set,
        "browser_charset_forced_by_transport": False,
        "optional_font_mode": optional_font_mode,
        "local_css_font_display_audit": {
            "policy_version": LOCAL_CSS_FONT_DISPLAY_POLICY_VERSION,
            "mode": optional_font_mode,
            "rewrite_enabled": css_route_installed,
            "source_assets_mutated": False,
            "response_content_type": "text/css",
            "rewrite_from": "font-display:optional",
            "rewrite_to": "font-display:block",
            "n_unique_local_css_files": len(css_records),
            "n_local_css_requests": sum(
                record["request_count"] for record in css_records.values()
            ),
            "n_rewritten_local_css_requests": sum(
                record["rewritten_request_count"]
                for record in css_records.values()
            ),
            "n_font_display_optional_replacements": sum(
                record["font_display_optional_replacements"]
                for record in css_records.values()
            ),
            "n_missing_local_css_requests_preserved": sum(
                missing_css_requests.values()
            ),
            "missing_local_css_requests": [
                {"url": url, "request_count": missing_css_requests[url]}
                for url in sorted(missing_css_requests)
            ],
            "files": [css_records[key] for key in sorted(css_records)],
            "passed": True,
        },
        "wait_until": wait_until,
        "navigation_timeout_ms": timeout,
    }


def require_same_browser_character_set(
    audits: Mapping[str, Mapping[str, object]],
    *,
    required_labels: tuple[str, ...],
) -> dict:
    """Fail closed unless named byte-preserving navigations decoded identically."""

    if not isinstance(audits, Mapping) or not required_labels:
        raise Utf8HtmlTransportError("character-set comparison inputs are invalid")
    observed: dict[str, str] = {}
    for label in required_labels:
        audit = audits.get(label)
        if (
            not isinstance(audit, Mapping)
            or audit.get("transport_version") != HTML_NAVIGATION_TRANSPORT_VERSION
            or audit.get("offline_source_utf8_valid") is not True
            or not isinstance(audit.get("browser_character_set"), str)
            or not audit["browser_character_set"].strip()
        ):
            raise Utf8HtmlTransportError(
                f"{label}: incomplete byte-preserving navigation audit"
            )
        observed[label] = audit["browser_character_set"]
    normalized = {value.strip().upper() for value in observed.values()}
    if len(normalized) != 1:
        raise Utf8HtmlTransportError(
            "browser character-set mismatch across render stages: "
            + ", ".join(f"{label}={value}" for label, value in observed.items())
        )
    return {
        "transport_version": HTML_NAVIGATION_TRANSPORT_VERSION,
        "required_labels": list(required_labels),
        "browser_character_sets": observed,
        "normalized_browser_character_set": next(iter(normalized)),
        "all_equal": True,
    }


async def wait_for_render_ready(
    page: Page,
    *,
    settle_ms: int = DEFAULT_RENDER_SETTLE_MS,
    timeout_ms: int = DEFAULT_RENDER_READY_TIMEOUT_MS,
) -> dict:
    """Apply the shared deterministic local-page load contract.

    Callers navigate with ``wait_until='load'`` first.  This helper then waits
    for the Font Loading API, every image's terminal load/error state, and the
    fixed post-load settle interval used by the frozen benchmark renderer.
    """

    if isinstance(settle_ms, bool) or not isinstance(settle_ms, int) or settle_ms < 0:
        raise ValueError("settle_ms must be a non-negative integer")
    if (
        isinstance(timeout_ms, bool)
        or not isinstance(timeout_ms, int)
        or timeout_ms <= 0
    ):
        raise ValueError("timeout_ms must be a positive integer")
    try:
        readiness = await asyncio.wait_for(
            page.evaluate(
                """
                async ({timeoutMs}) => {
                  let timer;
                  const ready = (async () => {
                    if (document.fonts && document.fonts.ready) {
                      // The navigation helper has already warmed the context's
                      // font cache. Calling FontFace.load() again here closes
                      // any faces introduced after load and leaves every face
                      // in a terminal loaded/error state before measurement.
                      const faces = Array.from(document.fonts);
                      await Promise.all(faces.map(async (face) => {
                        try { await face.load(); } catch (_) { /* terminal error */ }
                      }));
                      await document.fonts.ready;
                    }
                    await Promise.all(Array.from(document.images).map((image) => {
                      if (image.complete) return Promise.resolve();
                      return new Promise((resolve) => {
                        image.addEventListener('load', resolve, {once: true});
                        image.addEventListener('error', resolve, {once: true});
                      });
                    }));
                    return {status: 'ready'};
                  })();
                  const deadline = new Promise((resolve) => {
                    timer = setTimeout(
                      () => resolve({status: 'timeout'}), timeoutMs
                    );
                  });
                  try {
                    return await Promise.race([ready, deadline]);
                  } finally {
                    clearTimeout(timer);
                  }
                }
                """,
                {"timeoutMs": timeout_ms},
            ),
            timeout=(timeout_ms / 1000) + 1.0,
        )
    except asyncio.TimeoutError as error:
        raise Utf8HtmlTransportError(
            f"render readiness exceeded outer deadline {timeout_ms}ms"
        ) from error
    except Exception as error:  # noqa: BLE001
        raise Utf8HtmlTransportError(
            f"render readiness evaluation failed before {timeout_ms}ms: {error}"
        ) from error
    if readiness != {"status": "ready"}:
        raise Utf8HtmlTransportError(
            f"render readiness timed out after {timeout_ms}ms"
        )
    if settle_ms:
        await page.wait_for_timeout(settle_ms)
    audit = await page.evaluate(
        """
        () => {
          const images = Array.from(document.images);
          return {
            fonts_status: document.fonts ? document.fonts.status : 'unsupported',
            n_images: images.length,
            n_images_complete: images.filter((image) => image.complete).length,
            n_images_natural_nonzero: images.filter(
                (image) => image.complete && image.naturalWidth > 0 && image.naturalHeight > 0
            ).length,
          };
        }
        """
    )
    if (
        not isinstance(audit, dict)
        or audit.get("fonts_status") not in {"loaded", "unsupported"}
        or audit.get("n_images_complete") != audit.get("n_images")
    ):
        raise Utf8HtmlTransportError(f"page did not reach render-ready state: {audit!r}")
    return {
        **audit,
        "contract_version": RENDER_READY_CONTRACT_VERSION,
        "wait_until": "load",
        "settle_ms": settle_ms,
        "readiness_timeout_ms": timeout_ms,
        "readiness_timed_out": False,
    }


_MOTION_DESCRIPTOR_KEYS = {
    "kind",
    "target_xpath",
    "target_tag",
    "pseudo_element",
    "animation_name",
    "transition_property",
    "playback_rate",
    "timing",
}
_MOTION_TIMING_KEYS = {
    "delay",
    "direction",
    "duration",
    "easing",
    "end_delay",
    "fill",
    "iteration_start",
    "iterations",
}
_SMIL_DESCRIPTOR_KEYS = {
    "kind",
    "element_xpath",
    "target_xpath",
    "owner_svg_xpath",
    "semantic_attributes",
}
_SMIL_SEMANTIC_ATTRIBUTE_KEYS = {
    "attributeName",
    "attributeType",
    "begin",
    "dur",
    "end",
    "min",
    "max",
    "restart",
    "repeatCount",
    "repeatDur",
    "fill",
    "calcMode",
    "values",
    "keyTimes",
    "keySplines",
    "from",
    "to",
    "by",
    "additive",
    "accumulate",
    "path",
    "keyPoints",
    "rotate",
    "type",
    "href",
}


def _validate_motion_scalar(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and value == value
    ) or (
        isinstance(value, str)
        and value in {"Infinity", "-Infinity", "NaN", "auto"}
    )


def validate_render_motion_freeze_audit(value: object) -> dict:
    """Validate the closed Web Animations plus SVG SMIL freeze audit."""

    expected = {
        "contract_version",
        "timeout_ms",
        "freeze_timed_out",
        "requested_current_time_ms",
        "current_time_tolerance_ms",
        "smil_current_time_tolerance_seconds",
        "two_style_layout_flushes_completed",
        "style_layout_flushes",
        "n_animations",
        "n_css_animations",
        "n_css_transitions",
        "n_generic_animations",
        "n_smil_animations",
        "n_smil_svg_roots",
        "descriptors",
        "smil_descriptors",
        "initial_final_descriptors_exact",
        "initial_final_smil_descriptors_exact",
        "n_pause_exceptions",
        "n_seek_exceptions",
        "n_smil_pause_exceptions",
        "n_smil_seek_exceptions",
        "all_pause_calls_succeeded",
        "all_seek_calls_succeeded",
        "all_smil_pause_calls_succeeded",
        "all_smil_seek_calls_succeeded",
        "n_verified_paused",
        "n_verified_current_time_zero",
        "smil_root_states",
        "n_smil_roots_verified_paused",
        "n_smil_roots_verified_current_time_zero",
        "all_play_states_paused",
        "all_current_times_zero",
        "all_smil_roots_paused",
        "all_smil_root_current_times_zero",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise Utf8HtmlTransportError("render motion freeze audit has invalid fields")
    counts = (
        "n_animations",
        "n_css_animations",
        "n_css_transitions",
        "n_generic_animations",
        "n_smil_animations",
        "n_smil_svg_roots",
        "n_pause_exceptions",
        "n_seek_exceptions",
        "n_smil_pause_exceptions",
        "n_smil_seek_exceptions",
        "n_verified_paused",
        "n_verified_current_time_zero",
        "n_smil_roots_verified_paused",
        "n_smil_roots_verified_current_time_zero",
    )
    if (
        value["contract_version"] != RENDER_MOTION_FREEZE_CONTRACT_VERSION
        or isinstance(value["timeout_ms"], bool)
        or not isinstance(value["timeout_ms"], int)
        or value["timeout_ms"] <= 0
        or value["freeze_timed_out"] is not False
        or value["requested_current_time_ms"] != 0
        or value["current_time_tolerance_ms"] != 0
        or value["smil_current_time_tolerance_seconds"]
        != SMIL_CURRENT_TIME_TOLERANCE_SECONDS
        or value["two_style_layout_flushes_completed"] is not True
        or any(
            isinstance(value[key], bool)
            or not isinstance(value[key], int)
            or value[key] < 0
            for key in counts
        )
        or value["n_css_animations"]
        + value["n_css_transitions"]
        + value["n_generic_animations"]
        != value["n_animations"]
        or value["n_pause_exceptions"] != 0
        or value["n_seek_exceptions"] != 0
        or value["n_smil_pause_exceptions"] != 0
        or value["n_smil_seek_exceptions"] != 0
        or value["all_pause_calls_succeeded"] is not True
        or value["all_seek_calls_succeeded"] is not True
        or value["all_smil_pause_calls_succeeded"] is not True
        or value["all_smil_seek_calls_succeeded"] is not True
        or value["n_verified_paused"] != value["n_animations"]
        or value["n_verified_current_time_zero"] != value["n_animations"]
        or value["n_smil_roots_verified_paused"] != value["n_smil_svg_roots"]
        or value["n_smil_roots_verified_current_time_zero"]
        != value["n_smil_svg_roots"]
        or value["all_play_states_paused"] is not True
        or value["all_current_times_zero"] is not True
        or value["all_smil_roots_paused"] is not True
        or value["all_smil_root_current_times_zero"] is not True
        or value["initial_final_descriptors_exact"] is not True
        or value["initial_final_smil_descriptors_exact"] is not True
        or not isinstance(value["descriptors"], list)
        or len(value["descriptors"]) != value["n_animations"]
        or not isinstance(value["smil_descriptors"], list)
        or len(value["smil_descriptors"]) != value["n_smil_animations"]
    ):
        raise Utf8HtmlTransportError("render motion freeze audit did not pass")
    flushes = value["style_layout_flushes"]
    if not isinstance(flushes, list) or len(flushes) != 2:
        raise Utf8HtmlTransportError("render motion style/layout flush audit is invalid")
    for turn, flush in enumerate(flushes, start=1):
        if (
            not isinstance(flush, Mapping)
            or set(flush)
            != {
                "turn",
                "n_animations_observed",
                "n_animation_targets_flushed",
                "n_smil_animations_observed",
                "n_smil_animation_targets_flushed",
                "document_rect_read",
                "document_offset_height_read",
                "animation_target_styles_read",
            }
            or flush["turn"] != turn
            or flush["n_animations_observed"] != value["n_animations"]
            or flush["n_smil_animations_observed"]
            != value["n_smil_animations"]
            or isinstance(flush["n_animation_targets_flushed"], bool)
            or not isinstance(flush["n_animation_targets_flushed"], int)
            or not 0
            <= flush["n_animation_targets_flushed"]
            <= value["n_animations"]
            or isinstance(flush["n_smil_animation_targets_flushed"], bool)
            or not isinstance(flush["n_smil_animation_targets_flushed"], int)
            or not 0
            <= flush["n_smil_animation_targets_flushed"]
            <= value["n_smil_animations"]
            or flush["document_rect_read"] is not True
            or flush["document_offset_height_read"] is not True
            or flush["animation_target_styles_read"] is not True
        ):
            raise Utf8HtmlTransportError(
                "render motion style/layout flush audit did not pass"
            )
    descriptors: list[dict] = []
    for index, descriptor in enumerate(value["descriptors"]):
        if not isinstance(descriptor, Mapping) or set(descriptor) != _MOTION_DESCRIPTOR_KEYS:
            raise Utf8HtmlTransportError(
                f"render motion descriptor {index} has invalid fields"
            )
        timing = descriptor["timing"]
        if (
            not isinstance(descriptor["kind"], str)
            or not descriptor["kind"]
            or not isinstance(descriptor["target_xpath"], str)
            or not isinstance(descriptor["target_tag"], str)
            or not isinstance(descriptor["pseudo_element"], str)
            or not isinstance(descriptor["animation_name"], str)
            or not isinstance(descriptor["transition_property"], str)
            or not _validate_motion_scalar(descriptor["playback_rate"])
            or not isinstance(timing, Mapping)
            or set(timing) != _MOTION_TIMING_KEYS
            or not all(
                isinstance(timing[key], str)
                for key in ("direction", "easing", "fill")
            )
            or not all(
                _validate_motion_scalar(timing[key])
                for key in (
                    "delay",
                    "duration",
                    "end_delay",
                    "iteration_start",
                    "iterations",
                )
            )
        ):
            raise Utf8HtmlTransportError(
                f"render motion descriptor {index} is invalid"
            )
        descriptors.append(json.loads(json.dumps(descriptor, ensure_ascii=False)))
    if descriptors != sorted(
        descriptors,
        key=lambda item: json.dumps(
            item, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ),
    ):
        raise Utf8HtmlTransportError("render motion descriptors are not canonical")
    smil_descriptors: list[dict] = []
    for index, descriptor in enumerate(value["smil_descriptors"]):
        if (
            not isinstance(descriptor, Mapping)
            or set(descriptor) != _SMIL_DESCRIPTOR_KEYS
            or descriptor["kind"]
            not in {"animate", "animatemotion", "animatetransform", "set"}
            or any(
                not isinstance(descriptor[key], str) or not descriptor[key]
                for key in (
                    "element_xpath",
                    "target_xpath",
                    "owner_svg_xpath",
                )
            )
            or not isinstance(descriptor["semantic_attributes"], Mapping)
            or set(descriptor["semantic_attributes"])
            != _SMIL_SEMANTIC_ATTRIBUTE_KEYS
            or any(
                not isinstance(item, str)
                for item in descriptor["semantic_attributes"].values()
            )
        ):
            raise Utf8HtmlTransportError(
                f"render SMIL descriptor {index} is invalid"
            )
        smil_descriptors.append(
            json.loads(json.dumps(descriptor, ensure_ascii=False))
        )
    if smil_descriptors != sorted(
        smil_descriptors,
        key=lambda item: json.dumps(
            item, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ),
    ):
        raise Utf8HtmlTransportError("render SMIL descriptors are not canonical")
    root_states = value["smil_root_states"]
    if (
        not isinstance(root_states, list)
        or len(root_states) != value["n_smil_svg_roots"]
    ):
        raise Utf8HtmlTransportError("render SMIL root states are invalid")
    for index, state in enumerate(root_states):
        if (
            not isinstance(state, Mapping)
            or set(state) != {"root_xpath", "paused", "current_time_seconds"}
            or not isinstance(state["root_xpath"], str)
            or not state["root_xpath"]
            or state["paused"] is not True
            or isinstance(state["current_time_seconds"], bool)
            or not isinstance(state["current_time_seconds"], (int, float))
            or state["current_time_seconds"] != 0
        ):
            raise Utf8HtmlTransportError(
                f"render SMIL root state {index} is invalid"
            )
    if list(root_states) != sorted(root_states, key=lambda state: state["root_xpath"]):
        raise Utf8HtmlTransportError("render SMIL root states are not canonical")
    if {
        descriptor["owner_svg_xpath"] for descriptor in smil_descriptors
    } != {state["root_xpath"] for state in root_states}:
        raise Utf8HtmlTransportError(
            "render SMIL descriptors and root states disagree"
        )
    return json.loads(json.dumps(value, ensure_ascii=False))


async def freeze_render_motion(
    page: Page,
    *,
    timeout_ms: int = DEFAULT_RENDER_MOTION_FREEZE_TIMEOUT_MS,
) -> dict:
    """Pause every Web Animation and SVG SMIL timeline at time zero.

    This is called only after :func:`wait_for_render_ready`.  It freezes CSS
    animations, CSS transitions, and script-created Web Animations through the
    Web Animations API, and separately freezes native SVG SMIL roots through
    ``pauseAnimations`` plus ``setCurrentTime(0)``. It performs two separate
    style/layout flush turns, then fails closed unless both descriptor
    multisets are unchanged and every timeline is paused at zero. Separate
    browser evaluation turns are used because benchmark contexts deliberately
    disable page JavaScript, in which case page-owned timer/rAF callbacks do
    not run.
    """

    if (
        isinstance(timeout_ms, bool)
        or not isinstance(timeout_ms, int)
        or timeout_ms <= 0
    ):
        raise ValueError("timeout_ms must be a positive integer")
    evaluation = r"""
                ({phase}) => {
                  const normalizeNumber = (value) => {
                    if (typeof value === 'number') {
                      if (Number.isFinite(value)) return value;
                      if (Number.isNaN(value)) return 'NaN';
                      return value > 0 ? 'Infinity' : '-Infinity';
                    }
                    if (value === 'auto') return 'auto';
                    return String(value);
                  };
                  const xpathOf = (element) => {
                    if (!element || element.nodeType !== Node.ELEMENT_NODE) return '';
                    const parts = [];
                    let node = element;
                    while (node && node.nodeType === Node.ELEMENT_NODE) {
                      let index = 1;
                      let sibling = node.previousElementSibling;
                      while (sibling) {
                        if (sibling.localName === node.localName) index += 1;
                        sibling = sibling.previousElementSibling;
                      }
                      parts.unshift(`${node.localName}[${index}]`);
                      node = node.parentElement;
                    }
                    return '/' + parts.join('/');
                  };
                  const descriptorOf = (animation) => {
                    const effect = animation.effect;
                    const target = effect && effect.target ? effect.target : null;
                    const timing = effect && effect.getTiming
                      ? effect.getTiming()
                      : {
                          delay: 0, direction: 'normal', duration: 'auto',
                          easing: 'linear', endDelay: 0, fill: 'auto',
                          iterationStart: 0, iterations: 1,
                        };
                    return {
                      kind: animation.constructor && animation.constructor.name
                        ? animation.constructor.name : 'Animation',
                      target_xpath: xpathOf(target),
                      target_tag: target && target.localName ? target.localName : '',
                      pseudo_element: effect && typeof effect.pseudoElement === 'string'
                        ? effect.pseudoElement : '',
                      animation_name: typeof animation.animationName === 'string'
                        ? animation.animationName : '',
                      transition_property:
                        typeof animation.transitionProperty === 'string'
                          ? animation.transitionProperty : '',
                      playback_rate: normalizeNumber(animation.playbackRate),
                      timing: {
                        delay: normalizeNumber(timing.delay),
                        direction: String(timing.direction),
                        duration: normalizeNumber(timing.duration),
                        easing: String(timing.easing),
                        end_delay: normalizeNumber(timing.endDelay),
                        fill: String(timing.fill),
                        iteration_start: normalizeNumber(timing.iterationStart),
                        iterations: normalizeNumber(timing.iterations),
                      },
                    };
                  };
                  const canonical = (value) => {
                    if (Array.isArray(value)) {
                      return '[' + value.map(canonical).join(',') + ']';
                    }
                    if (value && typeof value === 'object') {
                      return '{' + Object.keys(value).sort().map(
                        (key) => JSON.stringify(key) + ':' + canonical(value[key])
                      ).join(',') + '}';
                    }
                    return JSON.stringify(value);
                  };
                  const sortedDescriptors = (animations) => animations
                    .map(descriptorOf)
                    .sort((left, right) => {
                      const a = canonical(left);
                      const b = canonical(right);
                      return a < b ? -1 : a > b ? 1 : 0;
                    });
                  const smilTagNames = new Set([
                    'animate', 'animatemotion', 'animatetransform', 'set'
                  ]);
                  const smilAttributeNames = [
                    'attributeName', 'attributeType', 'begin', 'dur', 'end',
                    'min', 'max', 'restart', 'repeatCount', 'repeatDur', 'fill',
                    'calcMode', 'values', 'keyTimes', 'keySplines', 'from',
                    'to', 'by', 'additive', 'accumulate', 'path', 'keyPoints',
                    'rotate', 'type', 'href'
                  ];
                  const smilElements = Array.from(
                    document.querySelectorAll('svg *')
                  ).filter((element) => smilTagNames.has(
                    String(element.localName || '').toLowerCase()
                  ));
                  const smilRoots = Array.from(new Set(
                    smilElements.map((element) => element.ownerSVGElement)
                      .filter((element) => element)
                  )).sort((left, right) => {
                    const a = xpathOf(left);
                    const b = xpathOf(right);
                    return a < b ? -1 : a > b ? 1 : 0;
                  });
                  const smilDescriptorOf = (element) => {
                    const target = element.targetElement || element.parentElement;
                    const semanticAttributes = {};
                    for (const name of smilAttributeNames) {
                      let value = element.getAttribute(name);
                      if (name === 'href' && value === null) {
                        value = element.getAttributeNS(
                          'http://www.w3.org/1999/xlink', 'href'
                        );
                      }
                      semanticAttributes[name] = value === null ? '' : value;
                    }
                    return {
                      kind: String(element.localName || '').toLowerCase(),
                      element_xpath: xpathOf(element),
                      target_xpath: xpathOf(target),
                      owner_svg_xpath: xpathOf(element.ownerSVGElement),
                      semantic_attributes: semanticAttributes,
                    };
                  };
                  const sortedSmilDescriptors = () => smilElements
                    .map(smilDescriptorOf)
                    .sort((left, right) => {
                      const a = canonical(left);
                      const b = canonical(right);
                      return a < b ? -1 : a > b ? 1 : 0;
                    });
                  const animations = Array.from(
                    document.getAnimations({subtree: true})
                  );
                  if (phase === 'pause') {
                    const descriptors = sortedDescriptors(animations);
                    const smilDescriptors = sortedSmilDescriptors();
                    let pauseExceptions = 0;
                    let seekExceptions = 0;
                    let smilPauseExceptions = 0;
                    let smilSeekExceptions = 0;
                    for (const animation of animations) {
                      try { animation.pause(); } catch (_) { pauseExceptions += 1; }
                      try { animation.currentTime = 0; } catch (_) { seekExceptions += 1; }
                    }
                    for (const root of smilRoots) {
                      try {
                        if (typeof root.pauseAnimations !== 'function') {
                          throw new Error('SVG SMIL pauseAnimations unsupported');
                        }
                        root.pauseAnimations();
                      } catch (_) { smilPauseExceptions += 1; }
                      try {
                        if (typeof root.setCurrentTime !== 'function') {
                          throw new Error('SVG SMIL setCurrentTime unsupported');
                        }
                        root.setCurrentTime(0);
                      } catch (_) { smilSeekExceptions += 1; }
                    }
                    return {
                      status: 'paused', descriptors, smilDescriptors,
                      pauseExceptions, seekExceptions,
                      smilPauseExceptions, smilSeekExceptions,
                    };
                  }
                  if (phase !== 'flush_verify') {
                    throw new Error('invalid render motion freeze phase');
                  }
                  let nTargetsFlushed = 0;
                  let nSmilTargetsFlushed = 0;
                  for (const animation of animations) {
                    const effect = animation.effect;
                    const target = effect && effect.target ? effect.target : null;
                    if (!target) continue;
                    const pseudo = effect && typeof effect.pseudoElement === 'string'
                      && effect.pseudoElement ? effect.pseudoElement : null;
                    const style = getComputedStyle(target, pseudo);
                    void style.transform;
                    void style.opacity;
                    void style.width;
                    void style.height;
                    void target.getBoundingClientRect();
                    nTargetsFlushed += 1;
                  }
                  for (const element of smilElements) {
                    const target = element.targetElement || element.parentElement;
                    if (!target) continue;
                    const style = getComputedStyle(target);
                    void style.transform;
                    void style.opacity;
                    void style.width;
                    void style.height;
                    void target.getBoundingClientRect();
                    nSmilTargetsFlushed += 1;
                  }
                  void document.documentElement.getBoundingClientRect();
                  void document.documentElement.offsetHeight;
                  const states = animations.map((animation) => ({
                      paused: animation.playState === 'paused',
                      currentTimeZero: animation.currentTime === 0,
                  }));
                  const smilRootStates = smilRoots.map((root) => ({
                    rootXPath: xpathOf(root),
                    paused: typeof root.animationsPaused === 'function'
                      ? root.animationsPaused() : false,
                    currentTimeSeconds: typeof root.getCurrentTime === 'function'
                      ? root.getCurrentTime() : null,
                  }));
                  return {
                    status: 'flushed',
                    descriptors: sortedDescriptors(animations),
                    smilDescriptors: sortedSmilDescriptors(),
                    states,
                    smilRootStates,
                    nTargetsFlushed,
                    nSmilTargetsFlushed,
                    documentRectRead: true,
                    documentOffsetHeightRead: true,
                    animationTargetStylesRead: true,
                  };
                }
                """

    async def perform_freeze() -> tuple[object, object, object]:
        paused = await page.evaluate(evaluation, {"phase": "pause"})
        await page.wait_for_timeout(0)
        first_flush = await page.evaluate(evaluation, {"phase": "flush_verify"})
        await page.wait_for_timeout(0)
        second_flush = await page.evaluate(evaluation, {"phase": "flush_verify"})
        return paused, first_flush, second_flush

    try:
        paused, first_flush, second_flush = await asyncio.wait_for(
            perform_freeze(), timeout=(timeout_ms / 1000) + 1.0
        )
    except asyncio.TimeoutError as error:
        raise Utf8HtmlTransportError(
            f"render motion freeze exceeded outer deadline {timeout_ms}ms"
        ) from error
    except Exception as error:  # noqa: BLE001
        raise Utf8HtmlTransportError(
            f"render motion freeze evaluation failed before {timeout_ms}ms: {error}"
        ) from error
    if (
        not isinstance(paused, Mapping)
        or set(paused)
        != {
            "status",
            "descriptors",
            "smilDescriptors",
            "pauseExceptions",
            "seekExceptions",
            "smilPauseExceptions",
            "smilSeekExceptions",
        }
        or paused.get("status") != "paused"
    ):
        raise Utf8HtmlTransportError("render motion pause returned invalid fields")
    expected_flush = {
        "status",
        "descriptors",
        "smilDescriptors",
        "states",
        "smilRootStates",
        "nTargetsFlushed",
        "nSmilTargetsFlushed",
        "documentRectRead",
        "documentOffsetHeightRead",
        "animationTargetStylesRead",
    }
    if any(
        not isinstance(flush, Mapping)
        or set(flush) != expected_flush
        or flush.get("status") != "flushed"
        for flush in (first_flush, second_flush)
    ):
        raise Utf8HtmlTransportError("render motion flush returned invalid fields")
    initial = paused["descriptors"]
    final = second_flush["descriptors"]
    initial_smil = paused["smilDescriptors"]
    final_smil = second_flush["smilDescriptors"]
    states = second_flush["states"]
    smil_root_states_raw = second_flush["smilRootStates"]
    if (
        not isinstance(initial, list)
        or not isinstance(final, list)
        or initial != final
        or first_flush["descriptors"] != final
        or not isinstance(initial_smil, list)
        or not isinstance(final_smil, list)
        or initial_smil != final_smil
        or first_flush["smilDescriptors"] != final_smil
        or not isinstance(states, list)
        or len(states) != len(final)
        or any(
            not isinstance(state, Mapping)
            or set(state) != {"paused", "currentTimeZero"}
            for state in states
        )
    ):
        raise Utf8HtmlTransportError(
            "render motion animation multiset changed during freeze"
        )
    pause_exceptions = paused["pauseExceptions"]
    seek_exceptions = paused["seekExceptions"]
    smil_pause_exceptions = paused["smilPauseExceptions"]
    smil_seek_exceptions = paused["smilSeekExceptions"]
    n_paused = sum(state["paused"] is True for state in states)
    n_zero = sum(state["currentTimeZero"] is True for state in states)
    n_animations = len(final)
    n_smil_animations = len(final_smil)
    if not isinstance(smil_root_states_raw, list):
        raise Utf8HtmlTransportError("render SMIL root verification is invalid")
    smil_root_states = []
    for state in smil_root_states_raw:
        if (
            not isinstance(state, Mapping)
            or set(state) != {"rootXPath", "paused", "currentTimeSeconds"}
        ):
            raise Utf8HtmlTransportError("render SMIL root verification is invalid")
        observed_time = state["currentTimeSeconds"]
        canonical_time = (
            0
            if not isinstance(observed_time, bool)
            and isinstance(observed_time, (int, float))
            and abs(observed_time) <= SMIL_CURRENT_TIME_TOLERANCE_SECONDS
            else observed_time
        )
        smil_root_states.append(
            {
                "root_xpath": state["rootXPath"],
                "paused": state["paused"],
                # Canonicalize any browser roundoff admitted by the explicit
                # tolerance so original/marker audits remain exactly equal.
                "current_time_seconds": canonical_time,
            }
        )
    smil_root_states.sort(key=lambda state: str(state["root_xpath"]))
    n_smil_roots = len(smil_root_states)
    n_smil_paused = sum(state["paused"] is True for state in smil_root_states)
    n_smil_zero = sum(
        not isinstance(state["current_time_seconds"], bool)
        and isinstance(state["current_time_seconds"], (int, float))
        and abs(state["current_time_seconds"])
        <= SMIL_CURRENT_TIME_TOLERANCE_SECONDS
        for state in smil_root_states
    )
    if (
        isinstance(pause_exceptions, bool)
        or not isinstance(pause_exceptions, int)
        or pause_exceptions != 0
        or isinstance(seek_exceptions, bool)
        or not isinstance(seek_exceptions, int)
        or seek_exceptions != 0
        or isinstance(smil_pause_exceptions, bool)
        or not isinstance(smil_pause_exceptions, int)
        or smil_pause_exceptions != 0
        or isinstance(smil_seek_exceptions, bool)
        or not isinstance(smil_seek_exceptions, int)
        or smil_seek_exceptions != 0
        or n_paused != n_animations
        or n_zero != n_animations
        or n_smil_paused != n_smil_roots
        or n_smil_zero != n_smil_roots
    ):
        raise Utf8HtmlTransportError(
            "render motion freeze could not pause and seek every animation"
        )
    kinds = [descriptor.get("kind") for descriptor in final if isinstance(descriptor, Mapping)]
    audit = {
        "contract_version": RENDER_MOTION_FREEZE_CONTRACT_VERSION,
        "timeout_ms": timeout_ms,
        "freeze_timed_out": False,
        "requested_current_time_ms": 0,
        "current_time_tolerance_ms": 0,
        "smil_current_time_tolerance_seconds": (
            SMIL_CURRENT_TIME_TOLERANCE_SECONDS
        ),
        "two_style_layout_flushes_completed": True,
        "style_layout_flushes": [
            {
                "turn": turn,
                "n_animations_observed": len(flush["descriptors"]),
                "n_animation_targets_flushed": flush["nTargetsFlushed"],
                "n_smil_animations_observed": len(flush["smilDescriptors"]),
                "n_smil_animation_targets_flushed": flush[
                    "nSmilTargetsFlushed"
                ],
                "document_rect_read": flush["documentRectRead"],
                "document_offset_height_read": flush["documentOffsetHeightRead"],
                "animation_target_styles_read": flush[
                    "animationTargetStylesRead"
                ],
            }
            for turn, flush in enumerate((first_flush, second_flush), start=1)
        ],
        "n_animations": n_animations,
        "n_css_animations": kinds.count("CSSAnimation"),
        "n_css_transitions": kinds.count("CSSTransition"),
        "n_generic_animations": sum(
            kind not in {"CSSAnimation", "CSSTransition"} for kind in kinds
        ),
        "n_smil_animations": n_smil_animations,
        "n_smil_svg_roots": n_smil_roots,
        "descriptors": final,
        "smil_descriptors": final_smil,
        "initial_final_descriptors_exact": True,
        "initial_final_smil_descriptors_exact": True,
        "n_pause_exceptions": pause_exceptions,
        "n_seek_exceptions": seek_exceptions,
        "n_smil_pause_exceptions": smil_pause_exceptions,
        "n_smil_seek_exceptions": smil_seek_exceptions,
        "all_pause_calls_succeeded": True,
        "all_seek_calls_succeeded": True,
        "all_smil_pause_calls_succeeded": True,
        "all_smil_seek_calls_succeeded": True,
        "n_verified_paused": n_paused,
        "n_verified_current_time_zero": n_zero,
        "smil_root_states": smil_root_states,
        "n_smil_roots_verified_paused": n_smil_paused,
        "n_smil_roots_verified_current_time_zero": n_smil_zero,
        "all_play_states_paused": True,
        "all_current_times_zero": True,
        "all_smil_roots_paused": True,
        "all_smil_root_current_times_zero": True,
    }
    return validate_render_motion_freeze_audit(audit)


__all__ = [
    "DEFAULT_RENDER_MOTION_FREEZE_TIMEOUT_MS",
    "DEFAULT_RENDER_READY_TIMEOUT_MS",
    "DEFAULT_RENDER_SETTLE_MS",
    "FONT_CACHE_WARM_RELOAD_POLICY",
    "HTML_NAVIGATION_TRANSPORT_VERSION",
    "RENDER_READY_CONTRACT_VERSION",
    "RENDER_MOTION_FREEZE_CONTRACT_VERSION",
    "SMIL_CURRENT_TIME_TOLERANCE_SECONDS",
    "Utf8HtmlTransportError",
    "freeze_render_motion",
    "goto_utf8_html",
    "require_same_browser_character_set",
    "validate_render_motion_freeze_audit",
    "wait_for_render_ready",
    "write_utf8_html",
]
