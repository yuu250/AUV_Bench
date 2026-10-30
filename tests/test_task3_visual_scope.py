"""Offline tests for the Task 3 visual-scope baseline-restoration fast path."""

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from tasks.task3 import visual_scope


def _solid(size, color):
    return Image.new("RGB", size, color)


class TestBaselineRestorationDiff(unittest.TestCase):
    def test_identical_images_pass(self):
        result = visual_scope.baseline_restoration_diff(
            _solid((100, 100), (250, 250, 250)), _solid((100, 100), (250, 250, 250))
        )
        self.assertEqual(result["baseline_diff_pixels"], 0)
        self.assertEqual(result["baseline_page_pixels"], 10000)
        self.assertEqual(result["baseline_diff_budget"], 1)
        self.assertTrue(result["baseline_fast_path_pass"])

    def test_delta_at_threshold_not_counted(self):
        baseline = _solid((100, 100), (250, 250, 250))
        fixed = _solid((100, 100), (250, 250, 250))
        fixed.putpixel((0, 0), (234, 250, 250))  # max channel delta == 16
        result = visual_scope.baseline_restoration_diff(fixed, baseline)
        self.assertEqual(result["baseline_diff_pixels"], 0)
        self.assertTrue(result["baseline_fast_path_pass"])

    def test_budget_boundary(self):
        baseline = _solid((100, 100), (250, 250, 250))
        fixed = _solid((100, 100), (250, 250, 250))
        fixed.putpixel((0, 0), (233, 250, 250))  # delta 17 > 16
        within = visual_scope.baseline_restoration_diff(fixed, baseline)
        self.assertEqual(within["baseline_diff_pixels"], 1)
        self.assertTrue(within["baseline_fast_path_pass"])  # 1 <= ceil(10000*1e-4)

        fixed.putpixel((1, 0), (233, 250, 250))
        beyond = visual_scope.baseline_restoration_diff(fixed, baseline)
        self.assertEqual(beyond["baseline_diff_pixels"], 2)
        self.assertFalse(beyond["baseline_fast_path_pass"])

    def test_size_mismatch_counts_union_only_area(self):
        result = visual_scope.baseline_restoration_diff(
            _solid((100, 100), (250, 250, 250)),
            _solid((100, 110), (250, 250, 250)),
        )
        self.assertEqual(result["baseline_diff_pixels"], 1000)
        self.assertEqual(result["baseline_page_pixels"], 11000)
        self.assertEqual(result["baseline_diff_budget"], 2)
        self.assertFalse(result["baseline_fast_path_pass"])

    def test_path_inputs(self):
        tmp = Path(tempfile.mkdtemp())
        fixed_path = tmp / "fixed.png"
        baseline_path = tmp / "baseline.png"
        _solid((50, 40), (10, 20, 30)).save(fixed_path)
        _solid((50, 40), (10, 20, 30)).save(baseline_path)
        result = visual_scope.baseline_restoration_diff(fixed_path, baseline_path)
        self.assertEqual(result["baseline_diff_pixels"], 0)
        self.assertTrue(result["baseline_fast_path_pass"])

    def test_rejects_invalid_input(self):
        with self.assertRaises(TypeError):
            visual_scope.baseline_restoration_diff(123, _solid((10, 10), (0, 0, 0)))


class TestApplyBaselineRestorationFastPath(unittest.TestCase):
    def test_flips_false_to_true(self):
        scope = {"visual_scope_pass": False, "visual_scope_reason": "outside_pixel_budget"}
        result = visual_scope.apply_baseline_restoration_fast_path(
            scope, _solid((10, 10), (0, 0, 0)), _solid((10, 10), (0, 0, 0))
        )
        self.assertTrue(result["visual_scope_pass"])
        self.assertEqual(
            result["visual_scope_reason"], "baseline_restoration_fast_path"
        )
        self.assertFalse(result["visual_scope_pass_region_check"])
        self.assertTrue(result["baseline_restoration"]["baseline_fast_path_pass"])
        self.assertFalse(scope["visual_scope_pass"])  # input left untouched

    def test_region_pass_survives_fast_path_failure(self):
        scope = {"visual_scope_pass": True, "visual_scope_reason": "pass"}
        fixed = _solid((10, 10), (0, 0, 0))
        baseline = _solid((10, 10), (250, 250, 250))
        result = visual_scope.apply_baseline_restoration_fast_path(
            scope, fixed, baseline
        )
        self.assertTrue(result["visual_scope_pass"])
        self.assertEqual(result["visual_scope_reason"], "pass")
        self.assertTrue(result["visual_scope_pass_region_check"])
        self.assertFalse(
            result["baseline_restoration"]["baseline_fast_path_pass"]
        )

    def test_double_failure_keeps_reason(self):
        scope = {"visual_scope_pass": False, "visual_scope_reason": "far_component"}
        result = visual_scope.apply_baseline_restoration_fast_path(
            scope,
            _solid((10, 10), (0, 0, 0)),
            _solid((10, 10), (250, 250, 250)),
        )
        self.assertFalse(result["visual_scope_pass"])
        self.assertEqual(result["visual_scope_reason"], "far_component")


if __name__ == "__main__":
    unittest.main()
