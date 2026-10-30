"""Offline parser/metric tests for the Task 2 frozen protocol.

Ported from the reference implementation's test_metrics.py; the module under
test (tasks/task2/metrics.py) is a byte-identical copy of the frozen file.
"""

import unittest

from tasks.task2.metrics import (clipped_iou_matrix, label_macro_metrics,
                                 localization_metrics, max_visible_slice_numbers,
                                 merge_gt_regions,
                                 parse_dimensions, parse_joint_diagnosis,
                                 parse_locations, parse_version)

DIMS = {
    "Typography": "排版",
    "Color": "色彩",
    "Spacing": "间距",
    "Layout & Reading Flow": "布局阅读流",
    "Visual Consistency": "视觉一致性",
}


class StrictParserTests(unittest.TestCase):
    def test_joint_diagnosis_parses_all_three_fields(self):
        parsed = parse_joint_diagnosis(
            '{"degraded_version":2,"dimensions":["Color","Spacing"],'
            '"locations":[{"slice":2,"box":[100,200,300,400]}]}', DIMS, 2)
        self.assertTrue(parsed["object_valid"])
        self.assertTrue(parsed["schema_valid"])
        self.assertTrue(parsed["dimensions_schema_valid"])
        self.assertTrue(parsed["locations_schema_valid"])
        self.assertEqual(parsed["version"], 2)
        self.assertEqual(parsed["dimensions"], {"色彩", "间距"})
        self.assertEqual(parsed["locations"], [(2, [100.0, 200.0, 300.0, 400.0])])

    def test_joint_diagnosis_rejects_bad_outer_contract(self):
        invalid = [
            '{"degraded_version":1,"dimensions":[],"locations":[],"reason":"x"}',
            '{"degraded_version":1,"dimensions":[]}',
            '{"degraded_version":1,"degraded_version":2,"dimensions":[],"locations":[]}',
            '```json\n{"degraded_version":1,"dimensions":[],"locations":[]}\n```',
        ]
        for value in invalid:
            with self.subTest(value=value):
                parsed = parse_joint_diagnosis(value, DIMS, 2)
                self.assertFalse(parsed["object_valid"])
                self.assertFalse(parsed["schema_valid"])
                self.assertIsNone(parsed["version"])
                self.assertIsNone(parsed["dimensions"])
                self.assertIsNone(parsed["locations"])

    def test_joint_diagnosis_keeps_valid_fields_when_one_value_is_bad(self):
        parsed = parse_joint_diagnosis(
            '{"degraded_version":1,"dimensions":["Color"],'
            '"locations":[{"slice":3,"box":[100,200,300,400]}]}', DIMS, 2)
        self.assertTrue(parsed["object_valid"])
        self.assertFalse(parsed["schema_valid"])
        self.assertEqual(parsed["version"], 1)
        self.assertEqual(parsed["dimensions"], {"色彩"})
        self.assertIsNone(parsed["locations"])

    def test_joint_schema_validity_is_distinct_from_coordinate_semantics(self):
        parsed = parse_joint_diagnosis(
            '{"degraded_version":1,"dimensions":["Color"],'
            '"locations":[{"slice":1,"box":[300,200,100,400]}]}', DIMS, 2)
        self.assertTrue(parsed["object_valid"])
        self.assertTrue(parsed["schema_valid"])
        self.assertTrue(parsed["locations_schema_valid"])
        self.assertIsNone(parsed["locations"])

        duplicate_dimension = parse_joint_diagnosis(
            '{"degraded_version":1,"dimensions":["Color","Color"],'
            '"locations":[]}', DIMS, 2)
        self.assertTrue(duplicate_dimension["schema_valid"])
        self.assertTrue(duplicate_dimension["dimensions_schema_valid"])
        self.assertIsNone(duplicate_dimension["dimensions"])

    def test_version_accepts_only_canonical_json(self):
        self.assertEqual(parse_version('{"degraded_version": 1}'), 1)
        self.assertEqual(parse_version(' \n{"degraded_version": 2}\t'), 2)
        invalid = [
            "Version 1",
            'answer: {"degraded_version": 1}',
            '```json\n{"degraded_version": 1}\n```',
            '{"degraded_version": "1"}',
            '{"degraded_version": 1.0}',
            '{"degraded_version": true}',
            '{"degraded_version": 3}',
            '{"degraded_version": 1, "reason": "x"}',
            '{"degraded_version": 2, "degraded_version": 1}',
            '{}',
            '[]',
        ]
        for value in invalid:
            with self.subTest(value=value):
                self.assertIsNone(parse_version(value))

    def test_dimensions_rejects_prose_oov_and_partial_acceptance(self):
        self.assertEqual(
            parse_dimensions(
                '{"dimensions":["Typography","Visual Consistency"]}', DIMS),
            {"排版", "视觉一致性"})
        self.assertEqual(parse_dimensions('{"dimensions": []}', DIMS), set())
        invalid = [
            "Typography: Yes; Color: No",
            '{"dimensions":["Typography","Imagery"]}',
            '{"dimensions":["Typography","Typography"]}',
            '{"dimensions":["typography"]}',
            '{"dimensions":"Typography"}',
            '{"dimensions":["Typography"],"reason":"x"}',
        ]
        for value in invalid:
            with self.subTest(value=value):
                self.assertIsNone(parse_dimensions(value, DIMS))

    def test_locations_are_all_or_nothing(self):
        valid = '{"locations":[{"slice":1,"box":[100,200,300,400]}]}'
        self.assertEqual(parse_locations(valid, 2), [(1, [100.0, 200.0, 300.0, 400.0])])
        self.assertEqual(parse_locations('{"locations":[]}', 2), [])
        invalid = [
            '```json\n' + valid + '\n```',
            '{"locations":[{"slice":0,"box":[100,200,300,400]}]}',
            '{"locations":[{"slice":1,"box":[100.0,200,300,400]}]}',
            '{"locations":[{"slice":1,"box":[100,200,300]}]}',
            '{"locations":[{"slice":1,"box":[100,200,300,400,500]}]}',
            '{"locations":[{"slice":1,"box":[' + '1' + '0' * 400 + ',200,300,400]}]}',
            '{"locations":[{"slice":1,"box":[-1,200,300,400]}]}',
            '{"locations":[{"slice":1,"box":[100,200,300,1001]}]}',
            '{"locations":[{"slice":1,"box":[300,200,100,400]}]}',
            '{"locations":[{"slice":1,"box":[100,200,300,400]},'
            '{"slice":1,"box":[100,200,100,400]}]}',
        ]
        for value in invalid:
            with self.subTest(value=value):
                self.assertIsNone(parse_locations(value, 2))


class AttributionMetricTests(unittest.TestCase):
    def test_constant_prior_predictor_has_zero_macro_j(self):
        pred = ["排版", "布局阅读流"]
        records = [
            {"gt_dims": ["排版"], "attr_pred": pred} for _ in range(5)
        ] + [{"gt_dims": ["色彩", "间距", "布局阅读流", "视觉一致性"],
              "attr_pred": pred}]
        result = label_macro_metrics(records, list(DIMS.values()))
        self.assertAlmostEqual(result["macro_j"], 0.0)
        for metric in result["per_dimension"].values():
            self.assertAlmostEqual(metric["j"], 0.0)

    def test_perfect_and_complement_predictions_bound_macro_j(self):
        gts = [["排版"], ["色彩"], ["间距"], ["布局阅读流"], ["视觉一致性"]]
        perfect = [{"gt_dims": gt, "attr_pred": gt} for gt in gts]
        complement = [{"gt_dims": gt,
                       "attr_pred": [d for d in DIMS.values() if d not in gt]}
                      for gt in gts]
        self.assertAlmostEqual(label_macro_metrics(perfect, list(DIMS.values()))["macro_j"], 1.0)
        self.assertAlmostEqual(label_macro_metrics(complement, list(DIMS.values()))["macro_j"], -1.0)

    def test_macro_j_is_na_without_both_classes(self):
        result = label_macro_metrics(
            [{"gt_dims": ["排版"], "attr_pred": ["排版"]}], list(DIMS.values()))
        self.assertIsNone(result["macro_j"])
        self.assertEqual(result["n_valid_j_dimensions"], 0)


class LocalizationMetricTests(unittest.TestCase):
    def test_gt_region_merge_is_conservative(self):
        parent = (0.0, 0.0, 100.0, 100.0)
        near_child = (1.0, 1.0, 99.0, 99.0)
        unrelated = (2.0, 2.0, 98.0, 98.0)
        regions = merge_gt_regions([
            (parent, ["/main/p[1]"]),
            (parent, ["/unrelated"]),  # identical rule-level boxes are one region
            (near_child, ["/main/p[1]/span[1]"]),
            (unrelated, ["/other/span[1]"]),  # unrelated paths never merge by overlap
        ])
        self.assertEqual(len(regions), 2)
        self.assertIn(parent, regions)
        self.assertIn(unrelated, regions)

    def test_duplicate_prediction_is_false_positive(self):
        metrics = localization_metrics([[1.0], [1.0]], n_pred=2, n_gt=1, threshold=0.5)
        self.assertEqual((metrics["tp"], metrics["fp"], metrics["fn"]), (1, 1, 0))
        self.assertAlmostEqual(metrics["f1"], 2 / 3)

    def test_giant_centered_box_fails_iou(self):
        scores = clipped_iou_matrix(
            [((0, 0, 1000, 1000), (0, 0, 1000, 1000))],
            [(450, 450, 550, 550)])
        metrics = localization_metrics(scores, n_pred=1, n_gt=1, threshold=0.5)
        self.assertEqual((metrics["tp"], metrics["fp"], metrics["fn"]), (0, 1, 1))

    def test_matching_is_global_not_first_fit(self):
        scores = [[0.6, 0.6], [1.0, 0.333]]
        metrics = localization_metrics(scores, n_pred=2, n_gt=2, threshold=0.5)
        self.assertEqual((metrics["tp"], metrics["fp"], metrics["fn"]), (2, 0, 0))

    def test_gt_is_clipped_to_visible_slice_before_iou(self):
        predictions = [((0, 100, 100, 200), (0, 100, 100, 200))]
        # The page-level GT extends past the slice; its visible part in this
        # slice matches the prediction exactly.
        scores = clipped_iou_matrix(predictions, [(0, 50, 100, 250)])
        self.assertAlmostEqual(scores[0][0], 1.0)
        metrics = localization_metrics(scores, n_pred=1, n_gt=1, threshold=0.5)
        self.assertEqual((metrics["tp"], metrics["fp"], metrics["fn"]), (1, 0, 0))

    def test_only_a_maximally_visible_slice_is_eligible(self):
        allowed = max_visible_slice_numbers(
            [(0, 70, 100, 100)],
            [(0, 0, 100, 100), (0, 80, 100, 180)])
        self.assertEqual(allowed, [{1}])

    def test_parse_failure_as_empty_prediction_keeps_all_false_negatives(self):
        metrics = localization_metrics([], n_pred=0, n_gt=3, threshold=0.5)
        self.assertEqual((metrics["tp"], metrics["fp"], metrics["fn"]), (0, 0, 3))
        self.assertEqual(metrics["f1"], 0.0)


if __name__ == "__main__":
    unittest.main()
