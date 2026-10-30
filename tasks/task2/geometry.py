"""Pair geometry and symmetric image transport for the Task 2 v8 protocol.

Every pair is rendered onto one shared coordinate frame: both versions use
the pair's maximum page height, identical fixed-canvas slices, and (for
multi-slice pages) a shared-scale top-aligned overview. Both versions of one
corresponding view always share one codec: if either PNG exceeds
MAX_IMG_BYTES, both are sent as JPEG.

The constants below are part of the frozen experimental protocol; changing
them breaks comparability with published results.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

from PIL import Image

from src.image.slicing import slice_offsets

SLICE_H = 1800
OVERLAP = 180
THUMB_H = 1500
MAX_SLICES = 12
MAX_IMG_BYTES = 9 * 1024 * 1024
JPEG_QUALITY = 90

GEOMETRY_POLICY = "pair-max-height-v1"
OVERVIEW_POLICY = "shared-scale-top-aligned-white-bottom-padding-lanczos"

INTRO_TEXT = (
    "You will be shown two versions of the same web page. "
    "Long pages first include a shared-scale overview for navigation only. "
    "All pages are then shown in fixed-canvas horizontal slices from top to bottom; "
    "use those slices for detailed visual decisions."
)


@dataclass(frozen=True)
class PairGeometry:
    """The frozen slice geometry shared by message construction and scoring."""

    canvas_width: int
    common_height: int
    offsets: tuple[int, ...]
    policy: str

    @property
    def n_slices(self) -> int:
        return len(self.offsets)


def build_pair_geometry(deg: dict, clean: dict) -> PairGeometry:
    """Build version-order-invariant geometry without opening image bytes."""
    deg_width, deg_height = (int(value) for value in deg["image_size"])
    clean_width, clean_height = (int(value) for value in clean["image_size"])
    if deg_width <= 0 or deg_height <= 0 or clean_width <= 0 or clean_height <= 0:
        raise ValueError("pair image dimensions must be positive")
    if deg_width != clean_width:
        raise ValueError("paired screenshots must have equal widths")
    return PairGeometry(
        canvas_width=deg_width,
        common_height=max(deg_height, clean_height),
        offsets=tuple(slice_offsets(max(deg_height, clean_height), SLICE_H, OVERLAP)),
        policy=GEOMETRY_POLICY,
    )


def fixed_canvas_slice(img: Image.Image, y: int) -> Image.Image:
    """Every slice is a same-size white canvas so the last slice's pixel
    height cannot leak the total page height."""
    canvas = Image.new("RGB", (img.width, SLICE_H), "white")
    if y < img.height:
        visible = img.crop((0, y, img.width, min(y + SLICE_H, img.height))).convert("RGB")
        canvas.paste(visible, (0, 0))
    return canvas


def symmetric_overview(img: Image.Image, geometry: PairGeometry) -> Image.Image:
    """Scale both versions by one factor, top-align, and pad only at the bottom."""
    scale = THUMB_H / geometry.common_height
    overview_width = max(1, round(geometry.canvas_width * scale))
    scaled_width = max(1, round(img.width * scale))
    scaled_height = max(1, round(img.height * scale))
    if scaled_width > overview_width or scaled_height > THUMB_H:
        raise ValueError("overview scaling exceeded the shared canvas")
    resampling = getattr(Image, "Resampling", Image).LANCZOS
    scaled = img.convert("RGB").resize((scaled_width, scaled_height), resampling)
    canvas = Image.new("RGB", (overview_width, THUMB_H), "white")
    canvas.paste(scaled, (0, 0))
    return canvas


def _encoded_bytes(img: Image.Image, mime: str) -> bytes:
    buf = io.BytesIO()
    if mime == "image/png":
        img.save(buf, format="PNG")
    elif mime == "image/jpeg":
        img.convert("RGB").save(buf, format="JPEG", quality=JPEG_QUALITY)
    else:
        raise ValueError(f"unsupported image MIME: {mime}")
    return buf.getvalue()


def encode_version_image_pair(
    images: dict[int, Image.Image]
) -> dict[int, tuple[bytes, str, dict]]:
    """Use the same codec for both versions of one corresponding view.

    Returns {version: (payload_bytes, media_type, audit)} where audit carries
    canvas_size/mime/encoded_bytes for the record's input_image_audit.
    """
    if set(images) != {1, 2}:
        raise ValueError("a version image pair must contain Version 1 and Version 2")
    png = {version: _encoded_bytes(image, "image/png")
           for version, image in images.items()}
    mime = ("image/jpeg" if any(len(data) > MAX_IMG_BYTES for data in png.values())
            else "image/png")
    data_by_version = (png if mime == "image/png" else {
        version: _encoded_bytes(image, mime) for version, image in images.items()
    })
    return {
        version: (
            data_by_version[version],
            mime,
            {
                "canvas_size": [int(images[version].width), int(images[version].height)],
                "mime": mime,
                "encoded_bytes": len(data_by_version[version]),
            },
        )
        for version in (1, 2)
    }


def build_version_image_content(
    version_images: dict[int, Image.Image],
    geometry: PairGeometry,
) -> tuple[list[dict], list[dict]]:
    """Build the visual request from Version 1/2 without knowing the GT identity.

    Returns (neutral message parts, per-image audit records).
    """
    if set(version_images) != {1, 2}:
        raise ValueError("visual input must contain exactly Version 1 and Version 2")
    if any(image.width != geometry.canvas_width for image in version_images.values()):
        raise ValueError("version image width does not match pair geometry")
    if max(image.height for image in version_images.values()) != geometry.common_height:
        raise ValueError("symmetric common height does not cover both versions exactly")

    audit_records: list[dict] = []
    content = [{"type": "text", "text": INTRO_TEXT}]

    def append_image(version: int, kind: str,
                     encoded: tuple[bytes, str, dict], *,
                     slice_number: int | None = None,
                     top_px: int | None = None) -> None:
        data, mime, encoded_audit = encoded
        image_audit = dict(encoded_audit)
        image_audit.update({
            "ordinal": len(audit_records) + 1,
            "version": version,
            "kind": kind,
            "slice": slice_number,
            "top_px": top_px,
        })
        audit_records.append(image_audit)
        content.append({"type": "image", "data": data, "media_type": mime})

    if geometry.common_height > SLICE_H:
        overviews = {
            version: symmetric_overview(image, geometry)
            for version, image in version_images.items()
        }
        encoded_overviews = encode_version_image_pair(overviews)
        for version in (1, 2):
            content.append({
                "type": "text",
                "text": f"Version {version} — full-page overview:",
            })
            append_image(version, "overview", encoded_overviews[version])

    for index, top_px in enumerate(geometry.offsets, 1):
        slices = {
            version: fixed_canvas_slice(image, top_px)
            for version, image in version_images.items()
        }
        encoded_slices = encode_version_image_pair(slices)
        for version in (1, 2):
            content.append({
                "type": "text",
                "text": f"Version {version} — slice {index}/{geometry.n_slices}:",
            })
            append_image(
                version, "slice", encoded_slices[version],
                slice_number=index, top_px=top_px,
            )
    return content, audit_records
