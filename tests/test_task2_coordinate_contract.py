"""Declared-coordinate contract tests for the Task 2 v8 protocol.

Ported from the reference implementation's test_coordinate_contract.py,
keeping the offline schema/conversion/parser cases. The smoke-audit,
cross-prompt-version, and manifest preflight cases are not ported: the first
helper does not exist in this repository, and the others rely on files that
are not part of the migrated protocol surface.
"""

import json
import unittest

from tasks.task2 import coordinate_contract
from tasks.task2.prompts import PROMPTS

LABELS = {
    "Typography": "排版",
    "Color": "色彩",
    "Spacing": "间距",
    "Layout & Reading Flow": "布局阅读流",
    "Visual Consistency": "视觉一致性",
}
WIDTH = 1280
HEIGHT = 1800


def response(**overrides) -> str:
    value = {
        "degraded_version": 1,
        "dimensions": ["Color"],
        "coordinate_space": "normalized_1000",
        "box_format": "xyxy",
        "locations": [{"slice": 2, "box": [100, 200, 300, 400]}],
    }
    value.update(overrides)
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


class DeclaredCoordinateSchemaTests(unittest.TestCase):
    def test_schema_requires_exact_five_key_contract(self):
        fmt = coordinate_contract.response_format(
            3, WIDTH, HEIGHT, list(LABELS))
        schema = fmt["json_schema"]["schema"]
        expected = {
            "degraded_version", "dimensions", "coordinate_space",
            "box_format", "locations",
        }
        self.assertEqual(set(schema["required"]), expected)
        self.assertEqual(set(schema["properties"]), expected)
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(
            schema["properties"]["coordinate_space"]["enum"],
            ["normalized_1000", "slice_pixels"],
        )
        self.assertEqual(
            schema["properties"]["box_format"]["enum"], ["xyxy", "xywh"])
        box_items = schema["properties"]["locations"]["items"][
            "properties"]["box"]["items"]
        # Conditional x/y bounds are enforced locally after reading the two
        # declarations; the provider receives only a safe common upper bound.
        self.assertEqual(box_items, {
            "type": "integer", "minimum": 0, "maximum": HEIGHT,
        })

    def test_schema_builder_rejects_invalid_geometry_or_labels(self):
        bad_args = [
            (0, WIDTH, HEIGHT, list(LABELS)),
            (2, 0, HEIGHT, list(LABELS)),
            (2, WIDTH, 0, list(LABELS)),
            (2, WIDTH, HEIGHT, []),
            (2, WIDTH, HEIGHT, ["Color", "Color"]),
            (2, WIDTH, HEIGHT, [1]),
        ]
        for args in bad_args:
            with self.subTest(args=args), self.assertRaises(ValueError):
                coordinate_contract.response_format(*args)


class DeclaredCoordinateConversionTests(unittest.TestCase):
    def test_normalized_xyxy_and_xywh(self):
        self.assertEqual(
            coordinate_contract.convert_box(
                [100, 200, 300, 400], "normalized_1000", "xyxy", WIDTH, HEIGHT),
            [100.0, 200.0, 300.0, 400.0],
        )
        self.assertEqual(
            coordinate_contract.convert_box(
                [100, 200, 200, 200], "normalized_1000", "xywh", WIDTH, HEIGHT),
            [100.0, 200.0, 300.0, 400.0],
        )

    def test_pixel_xyxy_and_xywh_convert_without_rounding(self):
        expected = [100.0, 100.0, 500.0, 500.0]
        self.assertEqual(
            coordinate_contract.convert_box(
                [128, 180, 640, 900], "slice_pixels", "xyxy", WIDTH, HEIGHT),
            expected,
        )
        self.assertEqual(
            coordinate_contract.convert_box(
                [128, 180, 512, 720], "slice_pixels", "xywh", WIDTH, HEIGHT),
            expected,
        )

    def test_non_divisible_width_is_not_rounded(self):
        converted = coordinate_contract.convert_box(
            [1, 1, 2, 2], "slice_pixels", "xyxy", 1365, HEIGHT)
        self.assertEqual(converted, [
            1000.0 / 1365, 1000.0 / HEIGHT,
            2000.0 / 1365, 2000.0 / HEIGHT,
        ])

    def test_right_and_bottom_edges_may_equal_declared_bounds(self):
        for space, fmt, box in (
            ("normalized_1000", "xyxy", [0, 0, 1000, 1000]),
            ("normalized_1000", "xywh", [0, 0, 1000, 1000]),
            ("slice_pixels", "xyxy", [0, 0, WIDTH, HEIGHT]),
            ("slice_pixels", "xywh", [0, 0, WIDTH, HEIGHT]),
        ):
            with self.subTest(space=space, fmt=fmt):
                self.assertEqual(
                    coordinate_contract.convert_box(
                        box, space, fmt, WIDTH, HEIGHT),
                    [0.0, 0.0, 1000.0, 1000.0],
                )

    def test_invalid_boxes_are_rejected_without_guessing_or_clipping(self):
        cases = [
            ([0, 1750, 1800, 1800], "normalized_1000", "xyxy"),
            ([0, 1750, 1800, 1800], "normalized_1000", "xywh"),
            ([0, 1750, 1800, 1800], "slice_pixels", "xyxy"),
            ([0, 1750, 1800, 1800], "slice_pixels", "xywh"),
            ([0, 0, WIDTH + 1, HEIGHT], "slice_pixels", "xyxy"),
            ([1000, 0, 1, 1], "normalized_1000", "xywh"),
            ([5, 5, 0, 1], "normalized_1000", "xywh"),
            ([5, 5, 4, 6], "normalized_1000", "xyxy"),
            ([False, 0, 1, 1], "normalized_1000", "xyxy"),
            ([0.0, 0, 1, 1], "normalized_1000", "xyxy"),
        ]
        for box, space, fmt in cases:
            with self.subTest(box=box, space=space, fmt=fmt):
                self.assertIsNone(coordinate_contract.convert_box(
                    box, space, fmt, WIDTH, HEIGHT))
        self.assertIsNone(coordinate_contract.convert_box(
            [0, 0, 1, 1], "slice_pixels", "xyxy", 0, HEIGHT))


class DeclaredCoordinateParserTests(unittest.TestCase):
    def parse(self, text: str) -> dict:
        return coordinate_contract.parse_declared_diagnosis(
            text, LABELS, 3, WIDTH, HEIGHT)

    def test_valid_pixel_xywh_is_canonicalized(self):
        parsed = self.parse(response(
            coordinate_space="slice_pixels",
            box_format="xywh",
            locations=[{"slice": 2, "box": [128, 180, 512, 720]}],
        ))
        self.assertTrue(parsed["object_valid"])
        self.assertTrue(parsed["schema_valid"])
        self.assertTrue(parsed["declared_coordinates_valid"])
        self.assertEqual(parsed["version"], 1)
        self.assertEqual(parsed["dimensions"], {"色彩"})
        self.assertEqual(parsed["locations"], [(2, [100.0, 100.0, 500.0, 500.0])])

    def test_empty_locations_still_requires_declarations(self):
        parsed = self.parse(response(locations=[]))
        self.assertTrue(parsed["schema_valid"])
        self.assertTrue(parsed["declared_coordinates_valid"])
        self.assertEqual(parsed["locations"], [])
        missing = json.loads(response(locations=[]))
        del missing["coordinate_space"]
        self.assertFalse(self.parse(json.dumps(missing))["object_valid"])

    def test_schema_shape_and_local_semantics_remain_distinct(self):
        out_of_declared_space = self.parse(response(
            locations=[{"slice": 2, "box": [0, 1001, 20, 1200]}]))
        self.assertTrue(out_of_declared_space["schema_valid"])
        self.assertTrue(out_of_declared_space["locations_schema_valid"])
        self.assertFalse(out_of_declared_space["declared_coordinates_valid"])
        self.assertIsNone(out_of_declared_space["locations"])

        duplicate_dims = self.parse(response(dimensions=["Color", "Color"]))
        self.assertTrue(duplicate_dims["schema_valid"])
        self.assertIsNone(duplicate_dims["dimensions"])

    def test_unknown_declaration_bad_types_and_bad_root_fail_closed(self):
        invalid = [
            response(coordinate_space="pixels"),
            response(box_format="ltrb"),
            response(degraded_version=True),
            response(locations=[{"slice": True, "box": [0, 0, 1, 1]}]),
            response(locations=[{"slice": 1, "box": [0, 0, 1.0, 1]}]),
            "[" + response() + "]",
            response()[:-1] + ',"extra":1}',
            response().replace('"dimensions":', '"dimensions":[],"dimensions":'),
        ]
        for text in invalid:
            with self.subTest(text=text):
                parsed = self.parse(text)
                self.assertFalse(
                    parsed["schema_valid"] and parsed["dimensions"] is not None
                    and parsed["locations"] is not None)


class DeclaredCoordinatePromptTests(unittest.TestCase):
    def test_v8_joint_prompt_declares_the_coordinate_contract(self):
        for locale in ("en", "zh"):
            prompt = PROMPTS["pair_joint"][locale]
            self.assertIn("coordinate_space", prompt)
            self.assertIn("box_format", prompt)
            self.assertIn("normalized_1000", prompt)
            self.assertIn("slice_pixels", prompt)
            self.assertIn("xyxy", prompt)
            self.assertIn("xywh", prompt)
            self.assertIn("{W}", prompt)
            self.assertIn("{H}", prompt)


if __name__ == "__main__":
    unittest.main()
