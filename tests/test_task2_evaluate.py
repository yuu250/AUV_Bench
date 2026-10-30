"""Aggregation tests for the Task 2 v8 evaluator (hand-computed fixtures)."""

import json
import tempfile
import unittest
from pathlib import Path

from tasks.task2 import evaluate
from tasks.task2.scoring import EVAL_VERSION

RUN_META = {
    "model": "fake-model",
    "eval_version": EVAL_VERSION,
    "protocol": "single-call",
    "single_call_version": "v8",
}


def make_record(sid, version, pred, *, gt_dim, attr, loc, chain=False):
    tp, fp, fn = loc
    record = {
        "sample_id": sid, "layer": "diagnostic", "eval_version": EVAL_VERSION,
        "run_meta": dict(RUN_META), "status": "complete",
        "degraded_version": version, "det_pred": pred,
        "det_correct": pred == version, "det_parse_fail": pred is None,
        "gt_dims": [gt_dim],
        "attr_pred": list(attr), "attr_pred_raw": list(attr),
        "attr_exact": chain, "attr_parse_fail": False, "attr_schema_valid": True,
        "joint_object_valid": True, "joint_schema_valid": True, "joint_valid": True,
        "coordinate_declaration_valid": True,
        "coordinate_space": "normalized_1000", "box_format": "xyxy",
        "loc_schema_valid": True, "loc_coordinate_contract_valid": True,
        "loc_parse_fail": False, "loc_exact": chain, "loc_gt_regions": 1,
        "chain_success": chain,
    }
    for suffix in ("50", "30"):
        for prefix in ("loc_iou", "loc_ungated_loc_iou"):
            record[f"{prefix}{suffix}_tp"] = tp
            record[f"{prefix}{suffix}_fp"] = fp
            record[f"{prefix}{suffix}_fn"] = fn
    return record


# r1: full-chain hit on version 1. r2: wrong version on version 2 (valid boxes
# become FP, GT stays FN). r3: full-chain hit on version 2 with another dim.
RECORDS = [
    make_record("p1", 1, 1, gt_dim="色彩", attr=["色彩"], loc=(1, 0, 0), chain=True),
    make_record("p2", 2, 1, gt_dim="色彩", attr=[], loc=(0, 1, 1)),
    make_record("p3", 2, 2, gt_dim="排版", attr=["排版"], loc=(1, 0, 0), chain=True),
]


def make_manifest_entry(sid, gt_dim):
    deg = {
        "sample_id": sid, "source": "degraded", "layer": "diagnostic",
        "image": f"{sid}.png", "image_size": [1000, 1800],
        "has_defect": True, "n_defects": 1, "gt_dims": [gt_dim],
        "defects": [{
            "rule_id": "r1", "dim": "Color", "dim_cn": gt_dim,
            "bboxes": [[180, 90, 360, 180]],
            "localization_xpaths": ["/main/div[1]"],
        }],
    }
    clean = {
        "sample_id": sid + "_clean", "source": "paired_control",
        "layer": "diagnostic",
        "image": f"{sid}_clean.png", "image_size": [1000, 1800],
    }
    return deg, clean


class EvaluateFixture:
    def __init__(self, root: Path):
        self.root = root
        entries = []
        for sid, dim in (("p1", "色彩"), ("p2", "色彩"), ("p3", "排版")):
            entries.extend(make_manifest_entry(sid, dim))
        self.manifest = root / "manifest.jsonl"
        with self.manifest.open("w", encoding="utf-8") as handle:
            for entry in entries:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self.predictions = root / "predictions.jsonl"
        self.write_predictions(RECORDS)

    def write_predictions(self, records):
        with self.predictions.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")


class AggregationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.fx = EvaluateFixture(Path(self.tmp.name))

    def tearDown(self):
        self.tmp.cleanup()

    def test_hand_computed_headline_metrics(self):
        artifact = evaluate.build_summary(self.fx.predictions, self.fx.manifest)
        metrics = artifact["metrics"]
        self.assertAlmostEqual(
            metrics["degradation_detection_balanced_accuracy"], 0.75)
        self.assertAlmostEqual(
            metrics["detection_gated_attribution_macro_j"], 0.75)
        self.assertAlmostEqual(
            metrics["version_aware_localization_micro_f1_at_0_5"], 2 / 3)
        self.assertAlmostEqual(metrics["exact_chain_success_rate"], 2 / 3)

    def test_summary_content(self):
        artifact = evaluate.build_summary(self.fx.predictions, self.fx.manifest)
        summary = artifact["context"]["summary"]
        self.assertEqual(summary["n"], 3)
        self.assertEqual(summary["model"], "fake-model")
        self.assertEqual(summary["run_meta"], RUN_META)
        self.assertAlmostEqual(summary["detection"]["accuracy"], 2 / 3)
        self.assertAlmostEqual(summary["detection"]["balanced_accuracy"], 0.75)
        self.assertAlmostEqual(summary["detection"]["chance_corrected_score"], 50.0)
        per_dim = summary["attribution_detection_gated"]["per_dimension"]
        self.assertAlmostEqual(per_dim["色彩"]["j"], 0.5)
        self.assertAlmostEqual(per_dim["排版"]["j"], 1.0)
        self.assertIsNone(per_dim["间距"]["j"])
        self.assertEqual(
            summary["attribution_detection_gated"]["n_valid_j_dimensions"], 2)
        loc50 = summary["location_version_aware_iou50_e2e"]
        self.assertEqual((loc50["tp"], loc50["fp"], loc50["fn"]), (2, 1, 1))
        self.assertAlmostEqual(loc50["f1"], 2 / 3)
        self.assertEqual(summary["layers"], {"diagnostic": {
            "n": 3, "det_acc": round(2 / 3, 4), "chain_success": round(2 / 3, 4)}})
        self.assertEqual(summary["coordinate_space_counts"],
                         {"normalized_1000": 3, "slice_pixels": 0})
        self.assertEqual(summary["box_format_counts"], {"xywh": 0, "xyxy": 3})
        self.assertNotIn("result_jsonl_sha256", summary)
        self.assertEqual(artifact["schema_version"], "design-bench-formal-metrics-v2")
        self.assertEqual(artifact["task"], "task2")
        self.assertEqual(sorted(artifact["metrics"]),
                         sorted(evaluate._METRIC_RANGES))

    def test_incomplete_cohort_is_rejected(self):
        self.fx.write_predictions(RECORDS[:2])
        with self.assertRaises(evaluate.EvaluationError):
            evaluate.build_summary(self.fx.predictions, self.fx.manifest)

    def test_wrong_eval_version_is_rejected(self):
        bad = [dict(RECORDS[0], eval_version="task2-v4"), *RECORDS[1:]]
        self.fx.write_predictions(bad)
        with self.assertRaises(evaluate.EvaluationError):
            evaluate.build_summary(self.fx.predictions, self.fx.manifest)

    def test_mixed_models_are_rejected(self):
        bad = [dict(r) for r in RECORDS]
        bad[1]["run_meta"] = dict(RUN_META, model="other-model")
        self.fx.write_predictions(bad)
        with self.assertRaises(evaluate.EvaluationError):
            evaluate.build_summary(self.fx.predictions, self.fx.manifest)

    def test_main_writes_the_artifact(self):
        code = evaluate.main(
            [str(self.fx.predictions), "--manifest", str(self.fx.manifest)])
        self.assertEqual(code, 0)
        artifact = json.loads(
            (self.fx.root / "predictions.formal_metrics.json").read_text(
                encoding="utf-8"))
        self.assertEqual(artifact["task"], "task2")
        self.assertIn("provenance", artifact)


if __name__ == "__main__":
    unittest.main()
