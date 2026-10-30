#!/usr/bin/env python3
"""Evaluation image pipeline for Task 4 (shared by both judges).

A rendered ``full.png`` (3840 wide @2x) is normalized to a 1920-wide 1x RGB
image, then sent as a shared-scale overview (multi-slice pages only) followed
by fixed 2160px white-canvas slices with 180px overlap and a 12-slice
top-anchored cap.  Paired views always share one codec: if either PNG exceeds
the byte budget, both sides are re-encoded as JPEG.

The encode_* helpers return ``(payload_bytes, media_type)`` tuples for the
neutral client image parts; on the OpenAI protocol they are sent as data
URLs identical to the reference implementation's.
"""

from __future__ import annotations

import io
from pathlib import Path

from PIL import Image

from tasks.task4.cohort import EvaluationError

PREPROCESS_VERSION = "task4-pairwise-fullpage-1x-lanczos-sliced-v1"
SOURCE_WIDTH = 3840
MIN_SOURCE_HEIGHT = 4320
NORMALIZED_WIDTH = 1920
SLICE_H = 2160
OVERLAP = 180
THUMB_H = 1500
MAX_SLICES = 12
MAX_IMG_BYTES = 9 * 1024 * 1024
JPEG_QUALITY = 90

PREPROCESS_CONTRACT = {
    "version": PREPROCESS_VERSION,
    "source": {"width": SOURCE_WIDTH, "min_height": MIN_SOURCE_HEIGHT, "dpr": 2},
    "mode": "RGB",
    "resampling": "PIL.Image.Resampling.LANCZOS",
    "normalized_width": NORMALIZED_WIDTH,
    "geometry": "pair-max-height-v1",
    "overview": {
        "height": THUMB_H,
        "shared_scale": True,
        "top_align": True,
        "canvas": "white-bottom-pad",
    },
    "slice": {
        "slice_h": SLICE_H,
        "overlap": OVERLAP,
        "max_slices": MAX_SLICES,
        "canvas": "white-fixed",
        "last_slice": "bottom-aligned",
        "cap_policy": "top-anchored-truncate",
    },
    "encoding": {
        "format": "PNG",
        "jpeg_fallback_bytes": MAX_IMG_BYTES,
        "jpeg_quality": JPEG_QUALITY,
        "paired_view_same_codec": True,
    },
}


def slice_offsets(height: int) -> list[int]:
    """1x pixel height -> slice top offsets.

    A single slice when the page fits; otherwise step SLICE_H-OVERLAP with the
    last slice bottom-aligned; beyond MAX_SLICES the page is top-truncated
    (first MAX_SLICES slices, no longer bottom-aligned) while the overview
    still conveys the full-page proportions.
    """

    if height <= SLICE_H:
        return [0]
    step = SLICE_H - OVERLAP
    offsets = list(range(0, height - SLICE_H + 1, step))
    if offsets[-1] != height - SLICE_H:
        offsets.append(height - SLICE_H)
    if len(offsets) > MAX_SLICES:
        offsets = [index * step for index in range(MAX_SLICES)]
    return offsets


def fixed_canvas_slice(image: Image.Image, top: int) -> Image.Image:
    """Every slice uses the same white canvas so the last slice's pixel height
    never leaks the page's total height."""

    canvas = Image.new("RGB", (image.width, SLICE_H), "white")
    if top < image.height:
        visible = image.crop((0, top, image.width, min(top + SLICE_H, image.height)))
        canvas.paste(visible.convert("RGB"), (0, 0))
    return canvas


def overview_image(image: Image.Image, common_height: int) -> Image.Image:
    """Both sides share one scale factor, top-aligned, bottom white-padded."""

    scale = THUMB_H / common_height
    canvas_width = max(1, round(image.width * scale))
    scaled_height = max(1, round(image.height * scale))
    if scaled_height > THUMB_H:
        raise EvaluationError("overview scaling exceeded the shared canvas")
    resampling = getattr(Image, "Resampling", Image).LANCZOS
    scaled = image.convert("RGB").resize((canvas_width, scaled_height), resampling)
    canvas = Image.new("RGB", (canvas_width, THUMB_H), "white")
    canvas.paste(scaled, (0, 0))
    return canvas


def _png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=False)
    return buffer.getvalue()


def _jpeg_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=JPEG_QUALITY)
    return buffer.getvalue()


def encode_view_pair(
    image_a: Image.Image, image_b: Image.Image
) -> tuple[tuple[bytes, str], tuple[bytes, str]]:
    """One codec per paired view: if either PNG is over budget, both go JPEG."""

    png_a = _png_bytes(image_a)
    png_b = _png_bytes(image_b)
    if len(png_a) > MAX_IMG_BYTES or len(png_b) > MAX_IMG_BYTES:
        return (
            (_jpeg_bytes(image_a), "image/jpeg"),
            (_jpeg_bytes(image_b), "image/jpeg"),
        )
    return (png_a, "image/png"), (png_b, "image/png")


def encode_view(image: Image.Image) -> tuple[bytes, str]:
    """Single-sided view (pointwise): PNG unless over the byte budget."""

    png = _png_bytes(image)
    if len(png) > MAX_IMG_BYTES:
        return _jpeg_bytes(image), "image/jpeg"
    return png, "image/png"


def load_1x_image(run_dir: Path, sample_id: str) -> Image.Image:
    """Reload full.png normalized to a 1920-wide (1x) RGB image."""

    path = Path(run_dir) / sample_id / "full.png"
    try:
        with Image.open(path) as source:
            source.load()
            image = source.copy()
    except Exception as exc:
        raise EvaluationError(
            f"validated input failed to reload: {sample_id} ({type(exc).__name__})"
        ) from exc
    if image.width != SOURCE_WIDTH or image.height % 2 != 0:
        raise EvaluationError(f"validated input drifted on reload: {sample_id}")
    resampling = getattr(Image, "Resampling", Image).LANCZOS
    return image.convert("RGB").resize(
        (NORMALIZED_WIDTH, image.height // 2), resampling
    )


__all__ = [
    "JPEG_QUALITY",
    "MAX_IMG_BYTES",
    "MAX_SLICES",
    "MIN_SOURCE_HEIGHT",
    "NORMALIZED_WIDTH",
    "OVERLAP",
    "PREPROCESS_CONTRACT",
    "PREPROCESS_VERSION",
    "SLICE_H",
    "SOURCE_WIDTH",
    "THUMB_H",
    "encode_view",
    "encode_view_pair",
    "fixed_canvas_slice",
    "load_1x_image",
    "overview_image",
    "slice_offsets",
]
