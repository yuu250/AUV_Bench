"""Image encoding for API payloads: PNG by default, JPEG when oversized or forced."""

from __future__ import annotations

import base64
import io
from pathlib import Path

from PIL import Image

DEFAULT_MAX_IMAGE_BYTES = 9 * 1024 * 1024
DEFAULT_JPEG_QUALITY = 90


def encode_image_payload(
    img: Image.Image,
    *,
    force_jpeg: bool = False,
    max_bytes: int = DEFAULT_MAX_IMAGE_BYTES,
    jpeg_quality: int = DEFAULT_JPEG_QUALITY,
    save_dir: Path | None = None,
    save_name: str | None = None,
) -> tuple[bytes, str]:
    """Encode a PIL image for the API request.

    Returns (payload_bytes, media_type). PNG is the default; the image is
    re-encoded as JPEG when it exceeds max_bytes or force_jpeg is set.
    When save_dir/save_name are given and JPEG compression kicked in, the
    compressed image is also written to the local save_dir for inspection.
    """
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    data = buf.getvalue()
    mime = "image/png"
    if force_jpeg or len(data) > max_bytes:
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=jpeg_quality)
        data = buf.getvalue()
        mime = "image/jpeg"
        if save_dir is not None and save_name:
            save_dir.mkdir(parents=True, exist_ok=True)
            (save_dir / f"{save_name}.jpg").write_bytes(data)
    return data, mime


def to_data_url(data: bytes, media_type: str) -> str:
    return f"data:{media_type};base64," + base64.b64encode(data).decode("ascii")
