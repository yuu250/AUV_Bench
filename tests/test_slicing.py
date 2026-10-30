"""Tests for slicing offsets, message-part assembly, and image encoding."""

import io
from pathlib import Path

import pytest
from PIL import Image

from src.image.encoding import encode_image_payload, to_data_url
from src.image.slicing import n_slices, slice_offsets
from tasks.task1.run import INTRO_TEXT, build_sliced_parts

IMAGE_CFG = {
    "slice_height": 1800,
    "slice_overlap": 180,
    "overview_height": 1500,
    "max_slices": 12,
    "max_image_bytes": 9 * 1024 * 1024,
    "jpeg_quality": 90,
    "save_processed_images": False,
    "processed_dir": "unused",
}


def make_image(tmp_path: Path, width: int, height: int) -> Path:
    path = tmp_path / f"img_{width}x{height}.png"
    Image.new("RGB", (width, height), (200, 30, 30)).save(path)
    return path


@pytest.mark.parametrize(
    "height,expected",
    [
        (1000, [0]),
        (1800, [0]),
        (1801, [0, 1]),
        (4000, [0, 1620, 2200]),
        (4320, [0, 1620, 2520]),
    ],
)
def test_slice_offsets(height, expected):
    assert slice_offsets(height) == expected


@pytest.mark.parametrize("height,expected", [(19620, 12), (21239, 13)])
def test_slice_count_boundary(height, expected):
    assert n_slices(height) == expected


def test_parts_single_slice(tmp_path):
    source = make_image(tmp_path, 800, 1000)
    parts, n = build_sliced_parts(
        source, "PROMPT", IMAGE_CFG, sample_id="s1", force_jpeg=False
    )
    assert n == 1
    assert [p["type"] for p in parts] == ["text", "text", "image", "text"]
    assert parts[0]["text"] == INTRO_TEXT
    assert parts[1]["text"] == "Screenshot slice 1/1:"
    assert parts[3]["text"] == "PROMPT"
    tile = Image.open(io.BytesIO(parts[2]["data"]))
    assert tile.size == (800, 1000)
    assert parts[2]["media_type"] == "image/png"


def test_parts_multi_slice(tmp_path):
    source = make_image(tmp_path, 800, 4000)
    parts, n = build_sliced_parts(
        source, "PROMPT", IMAGE_CFG, sample_id="s2", force_jpeg=False
    )
    assert n == 3
    assert [p["type"] for p in parts] == ["text"] + ["text", "image"] * 4 + ["text"]
    assert parts[0]["text"] == INTRO_TEXT
    assert parts[1]["text"] == "Page overview (scaled down):"
    overview = Image.open(io.BytesIO(parts[2]["data"]))
    assert overview.size == (300, 1500)  # 800 * 1500 / 4000
    labels = [parts[i]["text"] for i in (3, 5, 7)]
    assert labels == [f"Screenshot slice {i}/3:" for i in (1, 2, 3)]
    for i in (4, 6, 8):
        tile = Image.open(io.BytesIO(parts[i]["data"]))
        assert tile.size == (800, 1800)
    assert parts[-1]["text"] == "PROMPT"


def test_parts_save_processed_images(tmp_path):
    source = make_image(tmp_path, 800, 4000)
    save_dir = tmp_path / "processed"
    cfg = {**IMAGE_CFG, "save_processed_images": True,
           "processed_dir": str(save_dir), "max_image_bytes": 1}
    parts, n = build_sliced_parts(
        source, "PROMPT", cfg, sample_id="s3", force_jpeg=False
    )
    assert n == 3
    assert all(p["media_type"] == "image/jpeg" for p in parts if p["type"] == "image")
    saved = sorted(p.name for p in save_dir.iterdir())
    assert saved == [
        "s3_overview.jpg",
        "s3_slice01.jpg",
        "s3_slice02.jpg",
        "s3_slice03.jpg",
    ]


def test_encode_png_default():
    img = Image.new("RGB", (64, 64), (0, 255, 0))
    data, mime = encode_image_payload(img)
    assert mime == "image/png"
    assert Image.open(io.BytesIO(data)).format == "PNG"


def test_encode_force_jpeg_saves_copy(tmp_path):
    img = Image.new("RGB", (64, 64), (0, 255, 0))
    data, mime = encode_image_payload(
        img, force_jpeg=True, save_dir=tmp_path, save_name="x"
    )
    assert mime == "image/jpeg"
    assert Image.open(io.BytesIO(data)).format == "JPEG"
    saved = tmp_path / "x.jpg"
    assert saved.exists() and saved.read_bytes() == data


def test_encode_auto_jpeg_when_over_limit():
    img = Image.new("RGB", (64, 64))
    _, mime = encode_image_payload(img, max_bytes=1)
    assert mime == "image/jpeg"


def test_to_data_url():
    url = to_data_url(b"ab", "image/png")
    assert url == "data:image/png;base64,YWI="
