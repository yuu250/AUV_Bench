"""Offline tests for the Task 3 formal cohort builder (no browser, no API)."""

import json
import random
import tempfile
import unittest
from pathlib import Path

from tasks.task3 import cohort
from tasks.task3.grading import GT_SCHEMA_VERSION

IDS = ["s001", "s002", "s003"]


def _gt_row(sample_id):
    return {
        "sample_id": sample_id,
        "schema_version": GT_SCHEMA_VERSION,
        "layer": "diag",
        "viewport": [1920, 900],
        "dpr": 2,
        "degraded_version": cohort.actual_degraded_version(sample_id),
        "defects": [{
            "rule_id": "R1",
            "prop": "color",
            "base_value": "a",
            "new_value": "b",
            "victim_xpaths_degraded": ["/html/body/div[1]"],
            "degraded_bboxes_px": [[0, 0, 10, 10]],
        }],
        "baseline_context": {"peer_groups": [], "image_natural_by_xpath": {}},
        "safety_gate_rule_ids": [],
        "safety_report_only_rule_ids": [],
    }


def _advisory_row(sample_id):
    version = cohort.actual_degraded_version(sample_id)
    locations = [{"slice": 1, "box": [1, 2, 3, 4]}]
    canonical = [{"slice": 1, "box": [1.0, 2.0, 3.0, 4.0]}]
    return {
        "sample_id": sample_id,
        "status": "complete",
        "layer": "diag",
        "n_defects": 1,
        "degraded_version": version,
        "eval_version": cohort.ADVISORY_EVAL_VERSION,
        "protocol": "single-call",
        "model": "m",
        "run_meta": {"model": "m", "slice_h": 1800},
        "det_valid": True,
        "det_pred": version,
        "det_correct": True,
        "joint_raw": json.dumps({
            "degraded_version": version,
            "dimensions": ["Color"],
            "coordinate_space": "normalized_1000",
            "box_format": "xyxy",
            "locations": locations,
        }),
        "joint_object_valid": True,
        "joint_schema_valid": True,
        "joint_valid": True,
        "attr_schema_valid": True,
        "attr_valid": True,
        "attr_pred_raw_labels": ["Color"],
        "coordinate_space": "normalized_1000",
        "box_format": "xyxy",
        "coordinate_declaration_valid": True,
        "loc_schema_valid": True,
        "loc_coordinate_contract_valid": True,
        "loc_valid": True,
        "loc_pred_declared": locations,
        "loc_pred_raw": canonical,
        "loc_pred_canonical": canonical,
        "loc_pred_raw_representation": "normalized_1000_xyxy_float",
        "loc_n_pred": 1,
        "loc_exact": False,
        "loc_iou50_tp": 0,
        "loc_iou50_fp": 1,
        "loc_iou50_fn": 1,
        "loc_gt_regions": 1,
        "input_geometry": {"canvas_width": 1440, "common_height": 4000, "n_slices": 3},
        "n_slices": 3,
    }


def _det_invalid_row(sample_id):
    version = cohort.actual_degraded_version(sample_id)
    return {
        "sample_id": sample_id,
        "status": "complete",
        "layer": "diag",
        "n_defects": 1,
        "degraded_version": version,
        "eval_version": cohort.ADVISORY_EVAL_VERSION,
        "protocol": "single-call",
        "model": "m",
        "run_meta": {"model": "m", "slice_h": 1800},
        "det_valid": False,
        "det_pred": None,
        "det_correct": False,
        "loc_n_pred": 0,
        "loc_exact": False,
        "loc_iou50_tp": 0,
        "loc_iou50_fp": 0,
        "loc_iou50_fn": 1,
        "loc_gt_regions": 1,
    }


class CohortFixture(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.cohort_path = self.tmp / "cohort.json"
        self.cohort_path.write_text(json.dumps({
            "schema_version": cohort.COHORT_SCHEMA_VERSION,
            "n_samples": len(IDS),
            "sample_ids": IDS,
        }))
        self.gt_path = self.tmp / "unified_gt.jsonl"
        self.gt_path.write_text(
            "".join(json.dumps(_gt_row(s)) + "\n" for s in IDS)
        )
        self.meta_path = self.tmp / "metadata.json"
        self.meta_path.write_text(
            json.dumps({"schema_version": GT_SCHEMA_VERSION, "n_rows": len(IDS)})
        )
        self.pages = self.tmp / "data"
        for sample_id in IDS:
            page_dir = self.pages / "pages" / f"{sample_id}_combo_output"
            page_dir.mkdir(parents=True)
            (page_dir / "index.html").write_text("<html>degraded</html>")
            (page_dir / "baseline.html").write_text("<html>clean</html>")

    def build(self, advisory_path=None):
        return cohort.build_formal_items(
            cohort_path=self.cohort_path,
            gt_path=self.gt_path,
            gt_metadata_path=self.meta_path,
            pages_root=self.pages,
            advisory_path=advisory_path,
        )

    def write_advisory(self, rows):
        path = self.tmp / "advisory.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        return path


class TestCohortBuild(CohortFixture):
    def test_valid_advisory_handoff(self):
        advisory = self.write_advisory([
            _advisory_row("s001"), _det_invalid_row("s002"),
        ])
        items, audit = self.build(advisory)
        self.assertEqual(len(items), 3)
        by_id = {item.sample_id: item for item in items}
        version = cohort.actual_degraded_version("s001")
        handoff = by_id["s001"].task2_handoff
        self.assertEqual(handoff["degraded_version"], version)
        self.assertEqual(handoff["locations"], [{"slice": 1, "box": [1, 2, 3, 4]}])
        self.assertEqual(by_id["s001"].task2_provenance["advisory_status"], "valid")
        self.assertEqual(
            by_id["s001"].request.geometry,
            {"canvas_width": 1440, "common_height": 4000, "n_slices": 3},
        )
        self.assertEqual(by_id["s001"].selection["selected_version"], version)
        self.assertEqual(by_id["s001"].task3_record["n_defects"], 1)

        invalid = by_id["s002"]
        self.assertIsNone(invalid.task2_handoff)
        self.assertEqual(
            invalid.task2_provenance["advisory_status"], "det_invalid"
        )
        self.assertEqual(invalid.task2_metric_gates["loc_iou50_fn"], 1)
        self.assertTrue(invalid.call_eligible)

        absent = by_id["s003"]
        self.assertIsNone(absent.task2_handoff)
        self.assertEqual(absent.task2_provenance["advisory_status"], "absent")
        self.assertEqual(audit["advisory"]["valid_handoffs"], 1)

    def test_unusable_advisory_stays_callable(self):
        row = _advisory_row("s001")
        row["attr_valid"] = False
        row["joint_valid"] = False
        items, _ = self.build(self.write_advisory([row]))
        item = items[0]
        self.assertIsNone(item.task2_handoff)
        self.assertTrue(
            item.task2_provenance["advisory_status"].startswith("unusable:")
        )
        self.assertTrue(item.call_eligible)
        self.assertEqual(item.task2_metric_gates["loc_gt_regions"], 1)

    def test_det_pred_joint_raw_mismatch_is_unusable(self):
        row = _advisory_row("s001")
        raw = json.loads(row["joint_raw"])
        raw["degraded_version"] = 3 - raw["degraded_version"]
        row["joint_raw"] = json.dumps(raw)
        items, _ = self.build(self.write_advisory([row]))
        self.assertIsNone(items[0].task2_handoff)
        self.assertIn(
            "degraded_version differs",
            items[0].task2_provenance["advisory_status"],
        )

    def test_quote_wrapped_declaration_yields_no_locations(self):
        row = _advisory_row("s001")
        raw = json.loads(row["joint_raw"])
        raw["coordinate_space"] = '"normalized_1000"'
        row["joint_raw"] = json.dumps(raw)
        row["coordinate_declaration_valid"] = False
        row["joint_schema_valid"] = False
        row["coordinate_space"] = None
        row["box_format"] = None
        row["loc_pred_canonical"] = []
        row["loc_pred_raw"] = []
        row["loc_coordinate_contract_valid"] = False
        row["loc_valid"] = False
        row["joint_valid"] = False
        items, _ = self.build(self.write_advisory([row]))
        handoff = items[0].task2_handoff
        self.assertIsNotNone(handoff, items[0].task2_provenance)
        self.assertEqual(handoff["coordinate_space"], "normalized_1000")
        self.assertEqual(handoff["locations"], [])

    def test_gate_count_inconsistency_is_unusable(self):
        row = _advisory_row("s001")
        row["loc_iou50_fn"] = 0
        items, _ = self.build(self.write_advisory([row]))
        self.assertIsNone(items[0].task2_handoff)
        self.assertIn("TP/FN", items[0].task2_provenance["advisory_status"])

    def test_no_advisory_file(self):
        items, audit = self.build()
        self.assertTrue(all(item.task2_handoff is None for item in items))
        self.assertEqual(audit["advisory"]["rows"], 0)
        self.assertTrue(all(item.call_eligible for item in items))

    def test_gt_degraded_version_drift_fails(self):
        rows = [_gt_row(s) for s in IDS]
        rows[0]["degraded_version"] = 3 - rows[0]["degraded_version"]
        self.gt_path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        with self.assertRaises(cohort.CohortError):
            self.build()

    def test_duplicate_advisory_id_fails(self):
        path = self.write_advisory([_det_invalid_row("s002"), _det_invalid_row("s002")])
        with self.assertRaises(cohort.CohortError):
            self.build(path)

    def test_select_items(self):
        items, _ = self.build()
        selected = cohort.select_items(items, sample_ids=["s003"])
        self.assertEqual([item.sample_id for item in selected], ["s003"])
        self.assertEqual(len(cohort.select_items(items, limit=2)), 2)
        with self.assertRaises(cohort.CohortError):
            cohort.select_items(items, sample_ids=["nope"])

    def test_deterministic_pairing_matches_task2(self):
        self.assertEqual(
            cohort.actual_degraded_version("s001"),
            random.Random("s001").choice([1, 2]),
        )


def _spec_row(sample_id, dpr=2):
    return {
        "schema_version": "task3-defect-spec-v1",
        "sample_id": sample_id,
        "layer": "diag",
        "viewport": [1920, 900],
        "dpr": dpr,
        "target_rule_ids": ["R1"],
        "scoring_rule_ids": ["R1"],
        "safety_gate_rule_ids": ["R1"],
        "safety_report_only_rule_ids": [],
        "defects": [{
            "defect_id": f"{sample_id}:0:R1",
            "rule_id": "R1",
            "dim": "Color",
            "prop": "color",
            "base_value": "a",
            "new_value": "b",
            "victim_xpaths_degraded": ["/html/body/div[1]"],
        }],
    }


class TestSourceItems(CohortFixture):
    def write_specs(self, rows):
        path = self.tmp / "defect_specs.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        return path

    def build_source(self, rows):
        return cohort.build_source_items(
            cohort_path=self.cohort_path,
            specs_path=self.write_specs(rows),
            pages_root=self.pages,
        )

    def test_source_items_from_specs_without_gt(self):
        items = self.build_source([_spec_row(s, dpr=2.0) for s in IDS])
        self.assertEqual([item.sample_id for item in items], IDS)
        item = items[0]
        self.assertEqual(item.viewport, (1920, 900))
        self.assertEqual(item.dpr, 2)
        self.assertEqual(item.n_lesions, 1)
        self.assertIsNone(item.task2_handoff)
        self.assertIsNone(item.request.geometry)
        self.assertEqual(item.request.selected_source["html"].endswith("index.html"), True)
        self.assertEqual(item.task3_record["dpr"], 2)
        self.assertEqual(item.task3_record["n_defects"], 1)
        self.assertEqual(
            item.selection["selected_version"],
            cohort.actual_degraded_version(item.sample_id),
        )

    def test_missing_spec_row_fails(self):
        with self.assertRaises(cohort.CohortError):
            self.build_source([_spec_row("s001")])

    def test_non_integral_dpr_fails(self):
        with self.assertRaises(cohort.CohortError):
            self.build_source([_spec_row(s, dpr=1.5) for s in IDS])

    def test_duplicate_spec_id_fails(self):
        rows = [_spec_row(s) for s in IDS] + [_spec_row("s001")]
        with self.assertRaises(cohort.CohortError):
            self.build_source(rows)


if __name__ == "__main__":
    unittest.main()
