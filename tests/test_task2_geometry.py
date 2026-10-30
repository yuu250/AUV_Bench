"""Pair-geometry and symmetric-transport tests for the Task 2 v8 protocol."""

import unittest

from PIL import Image

from tasks.task2 import geometry
from tasks.task2.geometry import (
    SLICE_H,
    THUMB_H,
    build_pair_geometry,
    build_version_image_content,
    encode_version_image_pair,
    fixed_canvas_slice,
    symmetric_overview,
)


def solid(width: int, height: int, color) -> Image.Image:
    return Image.new("RGB", (width, height), color)


def entry(sample_id: str, width: int, height: int) -> dict:
    return {"sample_id": sample_id, "image": f"{sample_id}.png",
            "image_size": [width, height]}


class PairGeometryTests(unittest.TestCase):
    def test_common_height_is_the_pair_maximum(self):
        geo = build_pair_geometry(entry("a", 100, 900), entry("a_clean", 100, 2000))
        self.assertEqual(geo.canvas_width, 100)
        self.assertEqual(geo.common_height, 2000)
        self.assertEqual(geo.policy, "pair-max-height-v1")
        self.assertEqual(geo.offsets, (0, 200))
        self.assertEqual(geo.n_slices, 2)

    def test_single_slice_when_within_one_canvas(self):
        geo = build_pair_geometry(entry("a", 100, 1800), entry("a_clean", 100, 300))
        self.assertEqual(geo.offsets, (0,))

    def test_mismatched_widths_and_bad_dimensions_are_rejected(self):
        with self.assertRaises(ValueError):
            build_pair_geometry(entry("a", 100, 900), entry("a_clean", 101, 900))
        with self.assertRaises(ValueError):
            build_pair_geometry(entry("a", 0, 900), entry("a_clean", 100, 900))


class FixedCanvasSliceTests(unittest.TestCase):
    def test_slice_is_full_canvas_with_white_bottom_padding(self):
        img = solid(10, 100, (255, 0, 0))
        tile = fixed_canvas_slice(img, 0)
        self.assertEqual(tile.size, (10, SLICE_H))
        self.assertEqual(tile.getpixel((5, 99)), (255, 0, 0))
        self.assertEqual(tile.getpixel((5, 100)), (255, 255, 255))
        self.assertEqual(tile.getpixel((5, SLICE_H - 1)), (255, 255, 255))

    def test_offset_beyond_image_is_all_white(self):
        tile = fixed_canvas_slice(solid(10, 100, (0, 255, 0)), 500)
        self.assertEqual(tile.getpixel((5, 0)), (255, 255, 255))

    def test_last_slice_is_bottom_aligned_crop(self):
        img = Image.new("RGB", (4, 2000))
        for y in range(2000):
            for x in range(4):
                img.putpixel((x, y), (255 if y >= 1900 else 0, 0, 0))
        tile = fixed_canvas_slice(img, 200)
        self.assertEqual(tile.size, (4, SLICE_H))
        self.assertEqual(tile.getpixel((0, 1699)), (0, 0, 0))
        self.assertEqual(tile.getpixel((0, 1700)), (255, 0, 0))
        self.assertEqual(tile.getpixel((0, 1799)), (255, 0, 0))


class SymmetricOverviewTests(unittest.TestCase):
    def test_shared_scale_top_aligned_white_bottom(self):
        geo = build_pair_geometry(entry("a", 100, 2000), entry("a_clean", 100, 1000))
        tall = symmetric_overview(solid(100, 2000, (255, 0, 0)), geo)
        short = symmetric_overview(solid(100, 1000, (0, 255, 0)), geo)
        scale = THUMB_H / 2000
        expected_width = max(1, round(100 * scale))
        self.assertEqual(tall.size, (expected_width, THUMB_H))
        self.assertEqual(short.size, (expected_width, THUMB_H))
        # The taller version fills the canvas; the shorter one is top-aligned
        # with a white bottom band.
        self.assertEqual(tall.getpixel((0, THUMB_H - 1)), (255, 0, 0))
        short_scaled_h = max(1, round(1000 * scale))
        self.assertEqual(short.getpixel((0, short_scaled_h - 1))[1], 255)
        self.assertEqual(short.getpixel((0, THUMB_H - 1)), (255, 255, 255))


class PairCodecTests(unittest.TestCase):
    def test_small_pair_stays_png(self):
        images = {1: solid(8, 8, (1, 2, 3)), 2: solid(8, 8, (4, 5, 6))}
        encoded = encode_version_image_pair(images)
        for version in (1, 2):
            data, mime, audit = encoded[version]
            self.assertEqual(mime, "image/png")
            self.assertEqual(len(data), audit["encoded_bytes"])
            self.assertEqual(audit["canvas_size"], [8, 8])
            self.assertEqual(audit["mime"], "image/png")
            self.assertNotIn("encoded_sha256", audit)

    def test_one_oversize_version_forces_both_to_jpeg(self):
        images = {1: solid(8, 8, (1, 2, 3)), 2: solid(8, 8, (4, 5, 6))}
        original = geometry.MAX_IMG_BYTES
        geometry.MAX_IMG_BYTES = 10  # force the fallback without giant fixtures
        try:
            encoded = encode_version_image_pair(images)
        finally:
            geometry.MAX_IMG_BYTES = original
        self.assertEqual(encoded[1][1], "image/jpeg")
        self.assertEqual(encoded[2][1], "image/jpeg")

    def test_requires_exactly_two_versions(self):
        with self.assertRaises(ValueError):
            encode_version_image_pair({1: solid(4, 4, (0, 0, 0))})


class VersionImageContentTests(unittest.TestCase):
    def test_multi_slice_message_order_and_audit(self):
        geo = build_pair_geometry(entry("a", 20, 1900), entry("a_clean", 20, 1900))
        images = {1: solid(20, 1900, (10, 10, 10)), 2: solid(20, 1900, (20, 20, 20))}
        parts, audit = build_version_image_content(images, geo)
        self.assertEqual(geo.n_slices, 2)

        expected_kinds = (
            ["text"]
            + ["text", "image", "text", "image"]          # V1/V2 overviews
            + ["text", "image", "text", "image"] * 2      # slices 1-2, V1 then V2
        )
        self.assertEqual([part["type"] for part in parts], list(expected_kinds))
        self.assertEqual(parts[0]["text"], geometry.INTRO_TEXT)
        self.assertEqual(parts[1]["text"], "Version 1 — full-page overview:")
        self.assertEqual(parts[3]["text"], "Version 2 — full-page overview:")
        self.assertEqual(parts[5]["text"], "Version 1 — slice 1/2:")
        self.assertEqual(parts[7]["text"], "Version 2 — slice 1/2:")
        self.assertEqual(parts[9]["text"], "Version 1 — slice 2/2:")
        self.assertEqual(parts[11]["text"], "Version 2 — slice 2/2:")

        self.assertEqual(len(audit), 6)
        self.assertEqual([a["ordinal"] for a in audit], [1, 2, 3, 4, 5, 6])
        self.assertEqual([a["kind"] for a in audit],
                         ["overview", "overview", "slice", "slice", "slice", "slice"])
        self.assertEqual([a["version"] for a in audit], [1, 2, 1, 2, 1, 2])
        self.assertEqual([a["slice"] for a in audit], [None, None, 1, 1, 2, 2])
        self.assertEqual([a["top_px"] for a in audit], [None, None, 0, 0, 100, 100])
        for record in audit:
            self.assertNotIn("encoded_sha256", record)
        image_parts = [part for part in parts if part["type"] == "image"]
        for part in image_parts:
            self.assertEqual(part["media_type"], "image/png")
            self.assertIsInstance(part["data"], bytes)

    def test_single_slice_pair_has_no_overview(self):
        geo = build_pair_geometry(entry("a", 20, 900), entry("a_clean", 20, 900))
        images = {1: solid(20, 900, (1, 1, 1)), 2: solid(20, 900, (2, 2, 2))}
        parts, audit = build_version_image_content(images, geo)
        self.assertEqual([part["type"] for part in parts],
                         ["text", "text", "image", "text", "image"])
        self.assertEqual(len(audit), 2)
        self.assertEqual([a["kind"] for a in audit], ["slice", "slice"])

    def test_version_set_and_width_are_validated(self):
        geo = build_pair_geometry(entry("a", 20, 900), entry("a_clean", 20, 900))
        with self.assertRaises(ValueError):
            build_version_image_content({1: solid(20, 900, (0, 0, 0))}, geo)
        with self.assertRaises(ValueError):
            build_version_image_content(
                {1: solid(21, 900, (0, 0, 0)), 2: solid(20, 900, (0, 0, 0))}, geo)


if __name__ == "__main__":
    unittest.main()
