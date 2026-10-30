"""Offline tests for the Judgment-Action Association (JAA) metric."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from analysis import jaa


class TestGroupMetrics(unittest.TestCase):
    def test_hand_computed_independence(self):
        j = np.array([True, True, False, False])
        r = np.array([True, False, True, False])
        g = jaa.group_metrics(j, r)
        self.assertEqual((g["N"], g["n11"], g["n10"], g["n01"], g["n00"]),
                         (4, 1, 1, 1, 1))
        self.assertEqual((g["pJ"], g["pR"], g["PARS"]), (0.5, 0.5, 0.25))
        self.assertEqual(g["phi"], 0.0)
        self.assertEqual(g["p_r_given_j1"], 0.5)
        self.assertEqual(g["p_r_given_j0"], 0.5)
        self.assertEqual(g["p_j_given_r1"], 0.5)
        self.assertEqual(g["delta_r"], 0.0)
        self.assertEqual(g["na_reasons"], [])

    def test_perfect_positive_and_negative(self):
        j = np.array([True, True, False, False])
        g = jaa.group_metrics(j, j.copy())
        self.assertEqual(g["phi"], 1.0)
        g = jaa.group_metrics(j, ~j)
        self.assertEqual(g["phi"], -1.0)
        self.assertEqual(g["delta_r"], -1.0)

    def test_zero_variance_phi_undefined(self):
        j = np.array([True, True, True])
        r = np.array([True, False, True])
        g = jaa.group_metrics(j, r)
        self.assertIsNone(g["phi"])
        self.assertTrue(any("zero variance" in reason for reason in g["na_reasons"]))
        g = jaa.group_metrics(j, np.array([False, False, False]))
        self.assertIsNone(g["phi"])
        self.assertIsNone(g["p_j_given_r1"])  # denominator 0 (no R=1)
        self.assertEqual(g["p_r_given_j1"], 0.0)
        self.assertIsNone(g["p_r_given_j0"])  # denominator 0 (no J=0)
        self.assertIsNone(g["delta_r"])

    def test_empty_group(self):
        g = jaa.group_metrics(np.array([], dtype=bool), np.array([], dtype=bool))
        self.assertEqual(g["N"], 0)
        self.assertIsNone(g["phi"])
        self.assertIsNone(g["pJ"])


class TestBuildPerSample(unittest.TestCase):
    def _write(self, directory: Path, name: str, rows: list[dict]) -> Path:
        path = directory / name
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        return path

    def test_join_and_mapping(self):
        directory = Path(tempfile.mkdtemp())
        task2 = self._write(directory, "t2.jsonl", [
            {"sample_id": "s1", "det_correct": True,
             "attr_pred_raw": ["a", "b"], "gt_dims": ["b", "a"]},
            {"sample_id": "s2", "det_correct": False,
             "attr_pred_raw": ["a"], "gt_dims": ["a"]},
            {"sample_id": "s3", "det_correct": True,
             "attr_pred_raw": ["a"], "gt_dims": ["a", "c"]},
            {"sample_id": "s4", "det_correct": True,
             "attr_pred_raw": None, "gt_dims": ["a"]},
        ])
        task3 = self._write(directory, "t3.jsonl", [
            {"sample_id": "s1", "status": "complete", "fix_pass": True,
             "collateral_pass": True, "content_preservation_pass": True,
             "visual_scope_pass": True},
            {"sample_id": "s2", "status": "complete", "fix_pass": True,
             "collateral_pass": False, "content_preservation_pass": True,
             "visual_scope_pass": True},
            {"sample_id": "s3", "status": "api_transport_failure"},
            {"sample_id": "s4", "status": "complete", "fix_pass": True,
             "collateral_pass": True, "content_preservation_pass": True,
             "visual_scope_pass": True},
        ])
        rows, splits = jaa.build_per_sample(
            [("m1", task2, task3)], {"s1": "diag", "s2": "combo",
                                     "s3": "diag", "s4": "combo"}
        )
        self.assertEqual(splits, ["diag", "combo"])
        by_id = {row["sample_id"]: row for row in rows}
        self.assertTrue(by_id["s1"]["J"])          # D and exact set match
        self.assertFalse(by_id["s2"]["J"])         # D False
        self.assertFalse(by_id["s3"]["J"])         # set mismatch
        self.assertFalse(by_id["s4"]["J"])         # pred not a list
        self.assertTrue(by_id["s1"]["R"])          # all four components
        self.assertFalse(by_id["s2"]["R"])         # C False
        self.assertFalse(by_id["s3"]["R"])         # failure row -> all False
        self.assertEqual(by_id["s3"]["task3_status"], "api_transport_failure")
        self.assertTrue(by_id["s4"]["R"])
        # J is never a gate for R: s4 has J False but R True
        self.assertFalse(by_id["s4"]["J"])
        self.assertTrue(by_id["s4"]["R"])

    def test_id_set_mismatch_fails_closed(self):
        directory = Path(tempfile.mkdtemp())
        task2 = self._write(directory, "t2.jsonl", [{"sample_id": "s1"}])
        task3 = self._write(directory, "t3.jsonl", [{"sample_id": "s2"}])
        with self.assertRaises(jaa.JAAError):
            jaa.build_per_sample([("m1", task2, task3)], None)

    def test_duplicate_ids_rejected(self):
        directory = Path(tempfile.mkdtemp())
        task2 = self._write(directory, "t2.jsonl",
                            [{"sample_id": "s1"}, {"sample_id": "s1"}])
        with self.assertRaises(jaa.JAAError):
            jaa.load_jsonl(task2, "task2")

    def test_missing_layer_rejected(self):
        directory = Path(tempfile.mkdtemp())
        task2 = self._write(directory, "t2.jsonl", [{"sample_id": "s1"}])
        task3 = self._write(directory, "t3.jsonl", [{"sample_id": "s1"}])
        with self.assertRaises(jaa.JAAError):
            jaa.build_per_sample([("m1", task2, task3)], {})

    def test_parse_run(self):
        label, t2, t3 = jaa.parse_run("m=outputs/t2.jsonl:outputs/t3.jsonl")
        self.assertEqual(label, "m")
        self.assertEqual(str(t2), "outputs/t2.jsonl")
        for bad in ("m:t2:t3", "=t2:t3", "m=t2"):
            with self.assertRaises(jaa.JAAError):
                jaa.parse_run(bad)


class TestBootstrap(unittest.TestCase):
    def test_deterministic_with_seed(self):
        rows = [{"J": i % 2 == 0, "R": i % 3 == 0} for i in range(50)]
        first = jaa.bootstrap(rows, np.random.default_rng(42), 200)
        second = jaa.bootstrap(rows, np.random.default_rng(42), 200)
        self.assertEqual(first, second)

    def test_degenerate_zero_variance(self):
        rows = [{"J": True, "R": False} for _ in range(20)]
        out = jaa.bootstrap(rows, np.random.default_rng(42), 100)
        self.assertEqual(out["phi"]["n_valid"], 0)
        self.assertTrue(out["phi"]["degenerate"])
        self.assertIsNone(out["phi"]["ci_lo"])
        # conditional probabilities stay defined here
        self.assertEqual(out["p_r_given_j1"]["n_valid"], 100)


class TestEndToEnd(unittest.TestCase):
    def _write_pack(self, directory: Path):
        layers_path = directory / "layers.jsonl"
        with layers_path.open("w", encoding="utf-8") as handle:
            for i in range(1, 41):
                handle.write(json.dumps({
                    "sample_id": f"s{i:02d}",
                    "layer": "diag" if i <= 25 else "combo",
                }) + "\n")
        runs = []
        for model, j_rate, r_rate in (("mA", 0.5, 0.5), ("mB", 0.25, 0.1)):
            t2_rows, t3_rows = [], []
            for i in range(1, 41):
                sid = f"s{i:02d}"
                j_hit = i / 40 <= j_rate
                r_hit = i / 40 <= r_rate
                t2_rows.append({
                    "sample_id": sid,
                    "det_correct": j_hit,
                    "attr_pred_raw": ["x"] if j_hit else [],
                    "gt_dims": ["x"],
                })
                t3_rows.append({
                    "sample_id": sid,
                    "status": "complete",
                    "fix_pass": r_hit,
                    "collateral_pass": True,
                    "content_preservation_pass": True,
                    "visual_scope_pass": True,
                })
            t2 = directory / f"{model}_t2.jsonl"
            t3 = directory / f"{model}_t3.jsonl"
            t2.write_text("".join(json.dumps(r) + "\n" for r in t2_rows))
            t3.write_text("".join(json.dumps(r) + "\n" for r in t3_rows))
            runs.append(f"{model}={t2}:{t3}")
        return layers_path, runs

    def test_run_writes_all_artifacts(self):
        directory = Path(tempfile.mkdtemp())
        layers_path, runs = self._write_pack(directory)
        out_dir = directory / "out"
        args = jaa.parse_args([
            *[f"--run={value}" for value in runs],
            "--layers", str(layers_path),
            "--out-dir", str(out_dir),
            "--n-boot", "200",
        ])
        jaa.run(args)

        with (out_dir / "per_sample.csv").open() as handle:
            per_sample = list(csv.DictReader(handle))
        self.assertEqual(len(per_sample), 80)
        self.assertEqual(per_sample[0]["model"], "mA")

        with (out_dir / "summary.csv").open() as handle:
            summary = {row["model"]: row for row in csv.DictReader(handle)}
        self.assertEqual(set(summary), {"mA", "mB", jaa.POOLED_LABEL})
        # mA: J and R hit the same first 20 samples -> perfect association
        self.assertEqual(summary["mA"]["phi"], "1.0")
        self.assertEqual(summary["mA"]["n11"], "20")
        self.assertEqual(summary["mB"]["N"], "40")

        with (out_dir / "by_split.csv").open() as handle:
            splits = {(row["model"], row["split"]) for row in csv.DictReader(handle)}
        self.assertIn(("mA", "diag"), splits)
        self.assertIn(("mA", "combo"), splits)
        self.assertIn((jaa.POOLED_LABEL, "overall"), splits)
        self.assertIn((jaa.POOLED_LABEL, "diag"), splits)

        detail = json.loads((out_dir / "bootstrap_detail.json").read_text())
        self.assertEqual(detail["seed"], 42)
        self.assertEqual(detail["n_boot"], 200)
        self.assertIn("mA|overall", detail["groups"])
        self.assertIn("n_valid", detail["groups"]["mA|overall"]["phi"])

        for path in out_dir.iterdir():
            self.assertNotIn("sha256", path.read_text(errors="ignore"))

    def test_sparse_flags_on_small_cells(self):
        directory = Path(tempfile.mkdtemp())
        layers_path, runs = self._write_pack(directory)
        out_dir = directory / "out2"
        args = jaa.parse_args([
            *[f"--run={value}" for value in runs],
            "--layers", str(layers_path),
            "--out-dir", str(out_dir),
            "--n-boot", "50",
        ])
        summary_rows = jaa.run(args)
        combo = next(row for row in summary_rows
                     if row["model"] == "mB" and row["split"] == "combo")
        self.assertIn("sparse J=1", combo["flags"])
        self.assertIn("sparse R=1", combo["flags"])
        self.assertIn("n11=", combo["flags"])


if __name__ == "__main__":
    unittest.main()
