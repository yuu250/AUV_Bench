"""Offline tests for the Task 4 evaluation image pipeline."""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

from src.image.slicing import slice_offsets as src_slice_offsets
from tasks.task4 import views
from tasks.task4.cohort import EvaluationError


class TestSliceOffsets(unittest.TestCase):
    def test_single_slice(self):
        self.assertEqual(views.slice_offsets(1), [0])
        self.assertEqual(views.slice_offsets(2160), [0])

    def test_two_slices(self):
        self.assertEqual(views.slice_offsets(2161), [0, 1])
        self.assertEqual(views.slice_offsets(4320), [0, 1980, 2160])

    def test_last_slice_bottom_aligned(self):
        offsets = views.slice_offsets(5000)
        self.assertEqual(offsets[-1], 5000 - 2160)
        self.assertEqual(offsets[0], 0)

    def test_twelve_slice_top_cap(self):
        height = 2160 + 1980 * 20
        offsets = views.slice_offsets(height)
        self.assertEqual(len(offsets), 12)
        self.assertEqual(offsets, [i * 1980 for i in range(12)])
        self.assertLess(offsets[-1] + 2160, height)  # truncated, not aligned

    def test_equivalent_to_src_slicing_capped(self):
        for height in (100, 2160, 2161, 3000, 4320, 10000, 50000):
            self.assertEqual(
                views.slice_offsets(height),
                src_slice_offsets(height, 2160, 180)[:12],
                height,
            )


class TestFixedCanvasSlice(unittest.TestCase):
    def test_canvas_and_padding(self):
        image = Image.new("RGB", (100, 3000), (10, 20, 30))
        canvas = views.fixed_canvas_slice(image, 2160)
        self.assertEqual(canvas.size, (100, 2160))
        self.assertEqual(canvas.getpixel((0, 0)), (10, 20, 30))
        self.assertEqual(canvas.getpixel((0, 839)), (10, 20, 30))  # last content row
        self.assertEqual(canvas.getpixel((0, 840)), (255, 255, 255))

    def test_beyond_height_is_all_white(self):
        image = Image.new("RGB", (100, 100), (0, 0, 0))
        canvas = views.fixed_canvas_slice(image, 500)
        self.assertEqual(canvas.size, (100, 2160))
        self.assertEqual(canvas.getpixel((50, 50)), (255, 255, 255))


class TestOverviewImage(unittest.TestCase):
    def test_shared_scale_top_aligned(self):
        image = Image.new("RGB", (1920, 4320), (1, 2, 3))
        overview = views.overview_image(image, 4320)
        self.assertEqual(overview.height, 1500)
        self.assertEqual(overview.getpixel((0, 0)), (1, 2, 3))
        expected_w = max(1, round(1920 * 1500 / 4320))
        self.assertEqual(overview.width, expected_w)

    def test_shorter_side_bottom_padded(self):
        image = Image.new("RGB", (1920, 2160), (9, 9, 9))
        overview = views.overview_image(image, 4320)
        self.assertEqual(overview.height, 1500)
        scaled_h = max(1, round(2160 * 1500 / 4320))
        self.assertEqual(overview.getpixel((0, scaled_h)), (255, 255, 255))

    def test_overflow_rejected(self):
        image = Image.new("RGB", (100, 5000), (0, 0, 0))
        with self.assertRaises(EvaluationError):
            views.overview_image(image, 4000)  # image taller than common height


class TestEncodeViews(unittest.TestCase):
    def test_pair_same_codec_png(self):
        image = Image.new("RGB", (50, 50), (1, 1, 1))
        (data_a, mime_a), (data_b, mime_b) = views.encode_view_pair(image, image)
        self.assertEqual((mime_a, mime_b), ("image/png", "image/png"))
        self.assertEqual(data_a[:8], b"\x89PNG\r\n\x1a\n")
        self.assertEqual(data_b[:8], b"\x89PNG\r\n\x1a\n")

    def test_pair_same_codec_jpeg_when_oversize(self):
        image = Image.new("RGB", (50, 50), (1, 1, 1))
        with mock.patch.object(views, "MAX_IMG_BYTES", 1):
            (data_a, mime_a), (data_b, mime_b) = views.encode_view_pair(image, image)
        self.assertEqual((mime_a, mime_b), ("image/jpeg", "image/jpeg"))
        self.assertEqual(data_a[:2], b"\xff\xd8")  # JPEG magic

    def test_single_view_budget(self):
        image = Image.new("RGB", (50, 50), (1, 1, 1))
        _, mime = views.encode_view(image)
        self.assertEqual(mime, "image/png")
        with mock.patch.object(views, "MAX_IMG_BYTES", 1):
            _, mime = views.encode_view(image)
        self.assertEqual(mime, "image/jpeg")


class TestLoad1xImage(unittest.TestCase):
    def setUp(self):
        self.run_dir = Path(tempfile.mkdtemp())
        self.sample = self.run_dir / "s1"
        self.sample.mkdir()

    def _write_full(self, size):
        Image.new("RGB", size, (5, 5, 5)).save(self.sample / "full.png")

    def test_normalize_to_1x(self):
        self._write_full((3840, 4320))
        image = views.load_1x_image(self.run_dir, "s1")
        self.assertEqual(image.size, (1920, 2160))
        self.assertEqual(image.mode, "RGB")

    def test_odd_height_rejected(self):
        self._write_full((3840, 4321))
        with self.assertRaises(EvaluationError):
            views.load_1x_image(self.run_dir, "s1")

    def test_wrong_width_rejected(self):
        self._write_full((1920, 4320))
        with self.assertRaises(EvaluationError):
            views.load_1x_image(self.run_dir, "s1")

    def test_missing_file_rejected(self):
        with self.assertRaises(EvaluationError):
            views.load_1x_image(self.run_dir, "missing")


if __name__ == "__main__":
    unittest.main()
