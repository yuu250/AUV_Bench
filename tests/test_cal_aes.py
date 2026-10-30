"""Offline tests for the Task 4 Calibrated Aesthetic Judge calibration."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from analysis import cal_aes

DIMS = list(cal_aes.DIMS)
N_IDS = 30


def _scores(value_by_dim: dict[str, int]) -> dict:
    return {dim: {"score": value_by_dim[dim]} for dim in DIMS}


def _uniform_scores(value: int) -> dict:
    return _scores({dim: value for dim in DIMS})


def _synth_ids() -> list[str]:
    return [f"s{i:03d}" for i in range(N_IDS)]


def _synth_gpt() -> dict[str, dict[str, int]]:
    # every dimension cycles 1..5 deterministically
    return {
        sid: {dim: 1 + (i + d) % 5 for d, dim in enumerate(DIMS)}
        for i, sid in enumerate(_synth_ids())
    }


def _synth_human() -> dict[str, dict[str, float]]:
    # human = raw - 1 clamped to [1, 5]: exact monotone shift, raw bias +1
    return {
        sid: {dim: float(max(1, value - 1)) for dim, value in dims.items()}
        for sid, dims in _synth_gpt().items()
    }


def _write_manifest(directory: Path, ids: list[str]) -> Path:
    path = directory / "manifest.jsonl"
    path.write_text("".join(json.dumps({"id": sid}) + "\n" for sid in ids), encoding="utf-8")
    return path


def _write_mos(directory: Path, human: dict[str, dict[str, float]]) -> Path:
    path = directory / "mos.csv"
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["output_folder", *DIMS])
        writer.writeheader()
        for sid, dims in human.items():
            writer.writerow({"output_folder": f"{sid}_output", **dims})
    return path


def _write_scores(directory: Path, name: str, records: list[dict]) -> Path:
    path = directory / name
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    return path


def _synth_pack(directory: Path) -> tuple[Path, Path, Path]:
    ids = _synth_ids()
    manifest = _write_manifest(directory, ids)
    mos = _write_mos(directory, _synth_human())
    gpt = _synth_gpt()
    scores = _write_scores(
        directory,
        "task1.jsonl",
        [{"id": sid, "scores": _scores(gpt[sid])} for sid in ids],
    )
    return manifest, mos, scores


class TestValidScores(unittest.TestCase):
    def test_accepts_int_1_to_5(self):
        for value in (1, 2, 3, 4, 5):
            self.assertTrue(cal_aes._valid_scores(_uniform_scores(value)))

    def test_rejects_bool_float_out_of_range_missing(self):
        self.assertFalse(cal_aes._valid_scores(_uniform_scores(True)))
        self.assertFalse(cal_aes._valid_scores(_uniform_scores(3.0)))
        self.assertFalse(cal_aes._valid_scores(_uniform_scores(0)))
        self.assertFalse(cal_aes._valid_scores(_uniform_scores(6)))
        broken = _uniform_scores(3)
        del broken[DIMS[0]]
        self.assertFalse(cal_aes._valid_scores(broken))
        self.assertFalse(cal_aes._valid_scores({}))


class TestLoadTask1(unittest.TestCase):
    def test_join_and_first_valid_wins_across_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ids = ["a", "b", "c", "d"]
            manifest = _write_manifest(root, ids)
            human = {
                sid: {dim: 2.0 for dim in DIMS} for sid in ("a", "b", "c")
            }  # d has no MOS
            mos = _write_mos(root, human)
            first = _write_scores(
                root,
                "first.jsonl",
                [
                    {"id": "a", "scores": None, "error": "boom"},
                    {"id": "b", "scores": _uniform_scores(3)},
                    {"id": "c", "scores": _uniform_scores(4)},
                ],
            )
            second = _write_scores(
                root,
                "second.jsonl",
                [
                    {"id": "a", "scores": _uniform_scores(2)},
                    {"id": "b", "scores": _uniform_scores(5)},  # first file wins
                ],
            )
            joined, gpt, loaded_human, anomalies = cal_aes.load_task1(
                [first, second], mos, manifest
            )
            self.assertEqual(joined, ["a", "b", "c"])
            self.assertEqual(gpt["a"][DIMS[0]], 2)  # rescued by the supplement
            self.assertEqual(gpt["b"][DIMS[0]], 3)  # earlier file wins
            self.assertEqual(loaded_human["a"][DIMS[0]], 2.0)
            notes = "\n".join(anomalies)
            self.assertIn("rescued", notes)
            self.assertIn("excluded, no valid judge scores", notes)
            self.assertIn("excluded, no human MOS", notes)
            self.assertIn("joined samples=3 of 4", notes)

    def test_mos_out_of_range_flagged_but_kept(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = _write_manifest(root, ["a"])
            mos = _write_mos(root, {"a": {dim: 9.0 for dim in DIMS}})
            scores = _write_scores(root, "s.jsonl", [{"id": "a", "scores": _uniform_scores(3)}])
            joined, _, _, anomalies = cal_aes.load_task1([scores], mos, manifest)
            self.assertEqual(joined, ["a"])
            self.assertTrue(any("MOS out of range" in note for note in anomalies))

    def test_empty_join_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = _write_manifest(root, ["a"])
            mos = _write_mos(root, {"b": {dim: 2.0 for dim in DIMS}})
            scores = _write_scores(root, "s.jsonl", [{"id": "c", "scores": _uniform_scores(3)}])
            with self.assertRaises(cal_aes.CalibrationError):
                cal_aes.load_task1([scores], mos, manifest)

    def test_invalid_json_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = _write_manifest(root, ["a"])
            mos = _write_mos(root, {"a": {dim: 2.0 for dim in DIMS}})
            bad = root / "bad.jsonl"
            bad.write_text("{not json}\n", encoding="utf-8")
            with self.assertRaises(cal_aes.CalibrationError):
                cal_aes.load_task1([bad], mos, manifest)


class TestCrossValidateAndMappings(unittest.TestCase):
    def _fit_inputs(self):
        joined = _synth_ids()
        return joined, _synth_gpt(), _synth_human()

    def test_deterministic_and_macro_is_mean_of_dims(self):
        joined, gpt, human = self._fit_inputs()
        first = cal_aes.cross_validate(joined, gpt, human)
        second = cal_aes.cross_validate(joined, gpt, human)
        self.assertEqual(json.dumps(first, sort_keys=True), json.dumps(second, sort_keys=True))
        macro = first[-1]
        self.assertEqual(macro["dim"], "MACRO")
        for branch in ("raw", "calibrated"):
            for key in ("mae", "bias", "srcc"):
                self.assertAlmostEqual(
                    macro[branch][key],
                    float(np.mean([entry[branch][key] for entry in first[:-1]])),
                )

    def test_shift_calibration_removes_bias_and_mae(self):
        joined, gpt, human = self._fit_inputs()
        table = cal_aes.cross_validate(joined, gpt, human)
        for entry in table[:-1]:
            self.assertAlmostEqual(entry["calibrated"]["mae"], 0.0, places=7)
            self.assertAlmostEqual(entry["calibrated"]["bias"], 0.0, places=7)
            self.assertGreater(entry["raw"]["mae"], 0.5)
            self.assertGreater(entry["raw"]["bias"], 0.5)

    def test_final_mappings_are_monotone_bounded_shift(self):
        joined, gpt, human = self._fit_inputs()
        mappings = cal_aes.fit_final_mappings(joined, gpt, human)
        self.assertEqual(sorted(mappings), sorted(DIMS))
        for dim in DIMS:
            values = [mappings[dim][str(raw)] for raw in (1, 2, 3, 4, 5)]
            self.assertEqual(values, sorted(values))  # non-decreasing
            self.assertTrue(all(1.0 <= value <= 5.0 for value in values))
            self.assertAlmostEqual(mappings[dim]["5"], 4.0, places=7)  # shift -1
            self.assertAlmostEqual(mappings[dim]["1"], 1.0, places=7)  # clipped

    def test_n_splits_guard(self):
        joined, gpt, human = self._fit_inputs()
        with self.assertRaises(cal_aes.CalibrationError):
            cal_aes.cross_validate(joined, gpt, human, n_splits=len(joined) + 1)


IDENTITY_MAPPINGS = {dim: {str(raw): float(raw) for raw in (1, 2, 3, 4, 5)} for dim in DIMS}
SHIFT_MAPPINGS = {
    dim: {str(raw): float(max(1, raw - 1)) for raw in (1, 2, 3, 4, 5)} for dim in DIMS
}


def _task4_rows() -> list[dict]:
    def complete(label: str, sid: str, dims: int, overall: int) -> dict:
        scores = _uniform_scores(dims)
        scores["overall"] = {"score": overall}
        return {"participant": label, "sample_id": sid, "status": "complete", "scores": scores}

    return [
        complete("alpha", "s1", 3, 4),
        {"participant": "alpha", "sample_id": "s2", "status": "judge_error", "scores": None},
        complete("beta", "s1", 5, 5),
        complete("beta", "s2", 1, 2),
    ]


class TestApplyTask4(unittest.TestCase):
    def test_sums_and_status_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_scores(Path(tmp), "task4.jsonl", _task4_rows())
            per_participant, meta = cal_aes.apply_task4(IDENTITY_MAPPINGS, [path])
            self.assertEqual(meta["total_rows"], 4)
            alpha = per_participant["alpha"]
            self.assertEqual(alpha["n_valid"], 1)
            self.assertEqual(alpha["status_counts"], {"complete": 1, "judge_error": 1})
            self.assertAlmostEqual(alpha["raw_dimmean_sum"], 3.0)
            self.assertAlmostEqual(alpha["cal_dimmean_sum"], 3.0)
            self.assertAlmostEqual(alpha["raw_overall_sum"], 4.0)
            beta = per_participant["beta"]
            self.assertAlmostEqual(beta["raw_dimmean_sum"], 6.0)

    def test_calibration_mapping_applied_per_dimension(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_scores(Path(tmp), "task4.jsonl", _task4_rows())
            per_participant, _ = cal_aes.apply_task4(SHIFT_MAPPINGS, [path])
            # all-3 dims map to 2; all-5 -> 4, all-1 -> 1
            self.assertAlmostEqual(per_participant["alpha"]["cal_dimmean_sum"], 2.0)
            self.assertAlmostEqual(per_participant["beta"]["cal_dimmean_sum"], 5.0)

    def test_duplicate_participant_sample_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = _task4_rows()
            path = _write_scores(Path(tmp), "task4.jsonl", rows + [rows[0]])
            with self.assertRaises(cal_aes.CalibrationError):
                cal_aes.apply_task4(IDENTITY_MAPPINGS, [path])

    def test_complete_row_without_valid_scores_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_scores(
                Path(tmp),
                "task4.jsonl",
                [{"participant": "alpha", "sample_id": "s1", "status": "complete",
                  "scores": None}],
            )
            with self.assertRaises(cal_aes.CalibrationError):
                cal_aes.apply_task4(IDENTITY_MAPPINGS, [path])


class TestLeaderboard(unittest.TestCase):
    def _buckets(self) -> dict[str, dict]:
        return {
            "alpha": {"n_valid": 1, "raw_dimmean_sum": 3.0, "cal_dimmean_sum": 2.0,
                      "raw_overall_sum": 4.0, "status_counts": {}},
            "beta": {"n_valid": 2, "raw_dimmean_sum": 6.0, "cal_dimmean_sum": 5.0,
                     "raw_overall_sum": 7.0, "status_counts": {}},
        }

    def test_denominator_math_and_ranks(self):
        rows = cal_aes.build_leaderboard(self._buckets(), None, denominator=4)
        by_model = {row["model"]: row for row in rows}
        alpha = by_model["alpha"]
        # failure row scores 0: sums are divided by the fixed denominator
        self.assertAlmostEqual(alpha["raw_aes5"], 0.75)
        self.assertAlmostEqual(alpha["cal_aes5"], 0.5)
        self.assertAlmostEqual(alpha["coverage"], 0.25)
        self.assertAlmostEqual(alpha["raw_valid_only"], 3.0)
        self.assertAlmostEqual(alpha["cal_valid_only"], 2.0)
        self.assertAlmostEqual(alpha["overall_based_aes5"], 1.0)
        self.assertIsNone(alpha["pairwise_wr"])
        # sorted by cal_aes5 descending: beta (1.25) before alpha (0.5)
        self.assertEqual([row["model"] for row in rows], ["beta", "alpha"])
        self.assertEqual(by_model["beta"]["rank_cal"], 1)
        self.assertEqual(by_model["alpha"]["rank_raw"], 2)

    def test_pairwise_wr_join_and_missing_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "formal_metrics.json"
            path.write_text(json.dumps(
                {"metrics": {"overall_pairwise_win_rate": {"alpha": 0.5, "beta": 0.7}}}
            ), encoding="utf-8")
            rows = cal_aes.build_leaderboard(self._buckets(), path, denominator=4)
            self.assertEqual({row["model"]: row["pairwise_wr"] for row in rows},
                             {"beta": 0.7, "alpha": 0.5})
            path.write_text(json.dumps(
                {"metrics": {"overall_pairwise_win_rate": {"alpha": 0.5}}}
            ), encoding="utf-8")
            with self.assertRaises(cal_aes.CalibrationError):
                cal_aes.build_leaderboard(self._buckets(), path, denominator=4)


class TestCliEndToEnd(unittest.TestCase):
    def test_report_and_determinism(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest, mos, scores = _synth_pack(root)
            # fitted shift mapping: raw v -> v-1 (clip 1)
            def complete(label: str, sid: str, dims: int, overall: int) -> dict:
                s = _uniform_scores(dims)
                s["overall"] = {"score": overall}
                return {"participant": label, "sample_id": sid,
                        "status": "complete", "scores": s}

            task4 = _write_scores(
                root,
                "task4.jsonl",
                [
                    complete("alpha", "s1", 4, 4),
                    complete("alpha", "s2", 2, 2),
                    {"participant": "alpha", "sample_id": "s3",
                     "status": "render_failed", "scores": None},
                    complete("beta", "s1", 5, 5),
                    complete("beta", "s2", 5, 5),
                    complete("beta", "s3", 5, 5),
                ],
            )
            pairwise = root / "formal_metrics.json"
            pairwise.write_text(json.dumps(
                {"metrics": {"overall_pairwise_win_rate": {"alpha": 0.4, "beta": 0.8}}}
            ), encoding="utf-8")

            def run(out_dir: Path) -> dict:
                rc = cal_aes.main([
                    "--task1-scores", str(scores),
                    "--task1-mos", str(mos),
                    "--task1-manifest", str(manifest),
                    "--task4-pointwise", str(task4),
                    "--pairwise-wr", str(pairwise),
                    "--denominator", "3",
                    "--out-dir", str(out_dir),
                ])
                self.assertEqual(rc, 0)
                return json.loads((out_dir / "calibration_report.json").read_text())

            first = run(root / "out1")
            second = run(root / "out2")
            self.assertEqual(
                json.dumps(first, sort_keys=True), json.dumps(second, sort_keys=True)
            )
            self.assertEqual(first["protocol"], "task4-cal-aes-judge-isotonic-v1")
            self.assertEqual(first["task1_joined_samples"], N_IDS)
            self.assertEqual(first["seed"], cal_aes.DEFAULT_SEED)
            board = {row["model"]: row for row in first["task4_leaderboard"]}
            # alpha: mean8(cal) per page = 3 and 1, failure = 0 -> (3+1+0)/3
            self.assertAlmostEqual(board["alpha"]["cal_aes5"], (3.0 + 1.0) / 3.0)
            self.assertAlmostEqual(board["alpha"]["raw_aes5"], (4.0 + 2.0) / 3.0)
            self.assertAlmostEqual(board["alpha"]["coverage"], 2 / 3)
            self.assertAlmostEqual(board["alpha"]["pairwise_wr"], 0.4)
            # beta: every page 5 -> cal 4
            self.assertAlmostEqual(board["beta"]["cal_aes5"], 4.0)
            self.assertEqual(board["beta"]["rank_cal"], 1)
            self.assertEqual(board["alpha"]["rank_cal"], 2)
            cv_macro = first["cv_table"][-1]
            self.assertAlmostEqual(cv_macro["calibrated"]["mae"], 0.0, places=7)


if __name__ == "__main__":
    unittest.main()
