"""Offline tests for Task 3 GT-build pure functions and grading helpers."""

import json
import tempfile
import unittest
from pathlib import Path

from tasks.task3 import build_gt, grading
from tasks.task3.export_defect_specs import SPEC_SCHEMA_VERSION


def _elements():
    return [
        {"xpath": "/html/body/div[1]", "tag": "div", "cls": "a", "id": "",
         "text": "x", "bbox": [0, 0, 100, 20]},
        {"xpath": "/html/body/img[1]", "tag": "img", "cls": "", "id": "i",
         "text": "", "bbox": [0, 20, 50, 50], "natural": [400, 300]},
    ]


class TestBuildGtHelpers(unittest.TestCase):
    def setUp(self):
        self.ei = {"elements": _elements(), "peer_groups": [{"gid": "g1"}]}
        self.by_xpath = build_gt._elements_by_xpath(self.ei, "s1")

    def test_bboxes_and_physical_conversion(self):
        boxes = build_gt._bboxes(self.by_xpath, ["/html/body/div[1]"], "s1", "v")
        self.assertEqual(boxes, [[0.0, 0.0, 100.0, 20.0]])
        self.assertEqual(
            build_gt._to_physical(boxes, 2.0), [[0.0, 0.0, 200.0, 40.0]]
        )

    def test_missing_xpath_fails_closed(self):
        with self.assertRaises(build_gt.GTBuildError):
            build_gt._bboxes(self.by_xpath, ["/html/body/nope[1]"], "s1", "v")

    def test_fingerprint_identity_fields_only(self):
        fingerprints = build_gt._fingerprints(
            self.by_xpath, ["/html/body/img[1]"], "s1", "v"
        )
        self.assertEqual(
            fingerprints, [{"tag": "img", "text": "", "id": "i", "class": ""}]
        )

    def test_baseline_context(self):
        context = build_gt._baseline_context(self.ei, "s1")
        self.assertEqual(context["peer_groups"], [{"gid": "g1"}])
        self.assertEqual(
            context["image_natural_by_xpath"], {"/html/body/img[1]": [400, 300]}
        )

    def test_load_specs_schema_gate(self):
        tmp = Path(tempfile.mkdtemp())
        path = tmp / "specs.jsonl"
        path.write_text(json.dumps({
            "schema_version": SPEC_SCHEMA_VERSION,
            "sample_id": "s1",
            "layer": "diag",
            "viewport": [1920, 900],
            "dpr": 2.0,
            "target_rule_ids": ["COLOR_SEMANTIC"],
            "scoring_rule_ids": ["COLOR_SEMANTIC"],
            "safety_gate_rule_ids": [],
            "safety_report_only_rule_ids": [],
            "defects": [{
                "defect_id": "d1",
                "rule_id": "COLOR_SEMANTIC",
                "victim_xpaths_degraded": ["/html/body/div[1]"],
            }],
        }) + "\n")
        self.assertEqual(list(build_gt.load_specs(path)), ["s1"])
        bad = tmp / "bad.jsonl"
        bad.write_text(json.dumps({"schema_version": "other"}) + "\n")
        with self.assertRaises(build_gt.GTBuildError):
            build_gt.load_specs(bad)


class TestGradingPureHelpers(unittest.TestCase):
    def test_provider_output_failure(self):
        self.assertEqual(
            grading.provider_output_failure(None),
            "provider_content_missing_or_non_string",
        )
        self.assertIsNone(grading.provider_output_failure("{}"))
        self.assertIsNotNone(
            grading.provider_output_failure("x" * (grading.MAX_RESPONSE_CHARS + 1))
        )

    def test_message_content_parts_decodes_data_urls(self):
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "hello"},
                {"type": "image_url",
                 "image_url": {"url": "data:image/png;base64,aGVsbG8="}},
                {"type": "image_url",
                 "image_url": {"url": "https://example.com/x.png"}},
            ],
        }]
        parts = grading.message_content_parts(messages)
        self.assertEqual(parts[0], {"type": "text", "text": "hello"})
        self.assertEqual(parts[1]["type"], "image")
        self.assertEqual(parts[1]["data"], b"hello")
        self.assertEqual(parts[1]["media_type"], "image/png")
        self.assertEqual(
            parts[2], {"type": "image", "url": "https://example.com/x.png"}
        )

    def test_result_jsonl_roundtrip_and_summary(self):
        tmp = Path(tempfile.mkdtemp())
        path = tmp / "results.jsonl"
        record = {
            "schema_version": grading.RESULT_SCHEMA_VERSION,
            "eval_version": grading.EVAL_VERSION,
            "sample_id": "s1",
            "status": "complete",
            "api_attempted": True,
            "api_attempted_now": True,
            "fix_pass": True,
            "collateral_pass": True,
            "content_preservation_pass": False,
            "visual_scope_pass": True,
            grading.METRIC_MAIN: False,
            "all_pass": False,
        }
        grading.append_result_record(path, record)
        loaded = grading.load_result_jsonl(path, selected_ids=["s1"])
        self.assertEqual(loaded["s1"]["sample_id"], "s1")
        summary = grading.summarize_records(list(loaded.values()))
        self.assertEqual(summary["n"], 1)
        self.assertEqual(summary["repair_pass_count"], 0)
        self.assertEqual(summary["fix_pass_count"], 1)

    def test_result_jsonl_rejects_formula_drift(self):
        tmp = Path(tempfile.mkdtemp())
        path = tmp / "results.jsonl"
        record = {
            "schema_version": grading.RESULT_SCHEMA_VERSION,
            "eval_version": grading.EVAL_VERSION,
            "sample_id": "s1",
            "fix_pass": False,
            "collateral_pass": False,
            "content_preservation_pass": False,
            "visual_scope_pass": False,
            grading.METRIC_MAIN: True,
            "all_pass": True,
        }
        path.write_text(json.dumps(record) + "\n")
        with self.assertRaises(grading.Task3Error):
            grading.load_result_jsonl(path, selected_ids=["s1"])

    def test_local_model_input_geometry(self):
        reference = {
            "bundle": {"roles": {"degraded": {"grading_render": {
                "screenshot": {"size_px": [2880, 5000]},
            }}}},
            "manifest_record": {"screenshots": {"degraded": {
                "path": "results/x.png", "bytes": 10,
            }}},
            "screenshot_paths": {"degraded": "/tmp/x.png"},
        }
        request = grading.RequestInput(
            sample_id="s1",
            task2_handoff=None,
            geometry=None,
            selected_version=1,
            selected_source={"html": "p", "html_path": "/tmp/p.html"},
            task2_provenance={},
            viewport=(1440, 900),
            dpr=2,
        )
        local_input = grading.local_model_input_from_reference(request, reference)
        geometry = local_input["geometry"]
        self.assertEqual(geometry["canvas_width"], 2880)
        self.assertEqual(geometry["common_height"], 5000)
        self.assertEqual(geometry["slice_offsets"], [0, 1620, 3200])
        self.assertEqual(geometry["n_slices"], 3)
        self.assertEqual(local_input["viewport"], [1440, 900])
        self.assertEqual(local_input["dpr"], 2)


if __name__ == "__main__":
    unittest.main()
