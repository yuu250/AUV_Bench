"""Offline tests for the Task 4 pointwise evaluator (Calibrated Aesthetic Score)."""

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

from tasks.task4 import eval_common
from tasks.task4 import eval_pointwise as pt
from tasks.task4.cohort import EvaluationError
from tasks.task4.prompts import (
    INPUT_PROTOCOL,
    OUTPUT_AUDIT_PROTOCOL,
    RENDER_PROTOCOL,
    SCORE_KEYS,
)
from tasks.task4.scoring import strict_scores


class FakeClient:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    async def complete(self, **kwargs):
        self.calls.append(kwargs)
        item = self.outputs.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _api_error(status: int) -> Exception:
    exc = Exception(f"HTTP {status}")
    exc.status_code = status
    return exc


def _scores_payload(**overrides):
    scores = {}
    for index, key in enumerate(SCORE_KEYS):
        scores[key] = {"score": overrides.get(key, 1 + index % 5), "reason": f"r-{key}"}
    return scores


def _write_sample(run_dir: Path, sid: str, *, valid=True, full_size=(3840, 4320)):
    sample = run_dir / sid
    sample.mkdir(parents=True)
    (sample / "answer.html").write_text("<html></html>", encoding="utf-8")
    (sample / "input_manifest.json").write_text(json.dumps({
        "protocol_version": INPUT_PROTOCOL,
        "asset_mapping": [{"alias_src": "assets/v01.jpg"}],
        "allowed_asset_paths": ["assets/v01.jpg"],
    }))
    (sample / "output_audit.json").write_text(json.dumps({
        "protocol_version": OUTPUT_AUDIT_PROTOCOL,
        "violations": [],
    }))
    (sample / "assets").mkdir(exist_ok=True)
    (sample / "assets/v01.jpg").write_bytes(b"x")
    if valid:
        Image.new("RGB", full_size, (7, 7, 7)).save(sample / "full.png")
        (sample / "render_meta.json").write_text(json.dumps({
            "protocol_version": RENDER_PROTOCOL,
            "answer_bytes": (sample / "answer.html").stat().st_size,
            "crop_size": [3840, 4320],
            "full_size": list(full_size),
            "java_script_enabled": True,
        }))


class TestStrictScores(unittest.TestCase):
    def test_valid_payload(self):
        scores = strict_scores(json.dumps(_scores_payload()), SCORE_KEYS)
        self.assertEqual(set(scores), set(SCORE_KEYS))

    def test_matrix(self):
        bad = []
        payload = _scores_payload()
        del payload["overall"]
        bad.append(payload)                                  # missing key
        bad.append({**_scores_payload(), "extra": {"score": 1, "reason": "x"}})
        bad.append({**_scores_payload(), "overall": {"score": True, "reason": "x"}})
        bad.append({**_scores_payload(), "overall": {"score": 0, "reason": "x"}})
        bad.append({**_scores_payload(), "overall": {"score": 6, "reason": "x"}})
        bad.append({**_scores_payload(), "overall": {"score": 3, "reason": "  "}})
        bad.append({**_scores_payload(), "overall": {"score": 3}})
        for payload in bad:
            with self.assertRaises(ValueError):
                strict_scores(json.dumps(payload), SCORE_KEYS)
        with self.assertRaises(ValueError):  # duplicate JSON key
            strict_scores('{"overall": 1, "overall": 2}', SCORE_KEYS)


class ScoreFixture(unittest.TestCase):
    def setUp(self):
        eval_common._input_record_cache.clear()
        self.run_dir = Path(tempfile.mkdtemp()) / "run-a"
        _write_sample(self.run_dir, "s1")
        _write_sample(self.run_dir, "s2")
        self.participants = {
            "a": eval_common.Participant("a", "run-a", self.run_dir),
        }

    def _spec(self, ordinal=0, sid="s1"):
        return pt.ScoreSpec(ordinal=ordinal, participant="a", sample_id=sid)

    def _score(self, client, spec=None):
        return asyncio.run(pt.evaluate_score(
            spec or self._spec(), self.participants, client, "judge-x"
        ))


class TestEvaluateScore(ScoreFixture):
    def test_complete(self):
        client = FakeClient([json.dumps(_scores_payload(overall=4))])
        record = self._score(client)
        self.assertEqual(record["status"], "complete")
        self.assertEqual(record["scores"]["overall"]["score"], 4)
        self.assertTrue(record["api"]["called"])
        self.assertEqual(record["api"]["response_fingerprint"]["chars"],
                         len(json.dumps(_scores_payload(overall=4))))
        self.assertEqual(record["transport"]["n_slices"], 1)
        self.assertEqual(record["input"]["height_1x"], 2160)
        self.assertGreater(record["input"]["full_bytes"], 0)
        self.assertIsNone(record["error"])
        pt.validate_result_record(record, self._spec())
        self.assertNotIn("sha256", json.dumps(record))
        parts = client.calls[0]["parts"]
        self.assertEqual(parts[0]["type"], "text")
        self.assertEqual(parts[1]["text"], "Page screenshot:")

    def test_input_invalid_no_api_call(self):
        (self.run_dir / "s1" / "full.png").unlink()
        (self.run_dir / "s1" / "render_meta.json").unlink()
        eval_common._input_record_cache.clear()
        client = FakeClient([])
        record = self._score(client)
        self.assertEqual(record["status"], "input_invalid")
        self.assertFalse(record["api"]["called"])
        self.assertIsNone(record["transport"])
        self.assertEqual(record["error"]["category"], "input_invalid")
        self.assertEqual(client.calls, [])
        pt.validate_result_record(record, self._spec())

    def test_judge_failure(self):
        client = FakeClient([_api_error(500)] * 3)
        with mock.patch("asyncio.sleep", new=mock.AsyncMock()):
            record = self._score(client)
        self.assertEqual(record["status"], "judge_failure")
        self.assertIsNone(record["scores"])
        self.assertIsNone(record["api"]["response_fingerprint"])
        pt.validate_result_record(record, self._spec())

    def test_invalid_judge_output(self):
        client = FakeClient(['{"winner": "A"}'])
        record = self._score(client)
        self.assertEqual(record["status"], "invalid_judge_output")
        self.assertIsNotNone(record["api"]["response_fingerprint"])
        pt.validate_result_record(record, self._spec())

    def test_multi_slice_overview_message(self):
        _write_sample(self.run_dir, "tall", full_size=(3840, 8640))
        client = FakeClient([json.dumps(_scores_payload())])
        record = self._score(client, self._spec(sid="tall"))
        self.assertEqual(record["transport"]["slice_offsets"], [0, 1980, 2160])
        self.assertTrue(record["transport"]["overview_included"])
        parts = client.calls[0]["parts"]
        self.assertEqual(parts[1]["text"], "Full-page overview (navigation only):")
        self.assertEqual(parts[3]["text"], "Slice 1/3:")


class TestValidateResultRecord(ScoreFixture):
    def test_matrix(self):
        client = FakeClient([json.dumps(_scores_payload())])
        record = self._score(client)
        pt.validate_result_record(record, self._spec())
        record["scores"]["overall"]["score"] = 9
        with self.assertRaises(EvaluationError):
            pt.validate_result_record(record, self._spec())
        client = FakeClient([json.dumps(_scores_payload())])
        record = self._score(client)
        record["input"]["height_1x"] = 9999
        with self.assertRaises(EvaluationError):
            pt.validate_result_record(record, self._spec())
        client = FakeClient([json.dumps(_scores_payload())])
        record = self._score(client)
        record["api"]["response_fingerprint"] = {"chars": 1, "sha256": "0" * 64}
        with self.assertRaises(EvaluationError):
            pt.validate_result_record(record, self._spec())


class TestLoadExistingResults(ScoreFixture):
    def test_roundtrip_and_corruption(self):
        schedule = pt.build_schedule(
            [self.participants["a"]], ["s1", "s2"]
        )
        client = FakeClient([json.dumps(_scores_payload())])
        record = self._score(client, schedule[0])
        path = Path(tempfile.mkdtemp()) / "results.jsonl"
        path.write_text(json.dumps(record) + "\n", encoding="utf-8")
        loaded = pt.load_existing_results(path, schedule)
        self.assertEqual(loaded, {("a", "s1"): record})
        path.write_text(json.dumps(record), encoding="utf-8")  # no trailing NL
        with self.assertRaises(EvaluationError):
            pt.load_existing_results(path, schedule)
        wrong_slot = dict(record, sample_id="s2", ordinal=0)
        path.write_text(json.dumps(wrong_slot) + "\n", encoding="utf-8")
        with self.assertRaises(EvaluationError):
            pt.load_existing_results(path, schedule)


class TestBuildSummary(ScoreFixture):
    def _fabricate(self, spec, status, overall=None):
        base = {
            "ordinal": spec.ordinal,
            "participant": spec.participant,
            "sample_id": spec.sample_id,
            "status": status,
            "scores": None,
        }
        if status == "complete":
            base["scores"] = _scores_payload(overall=overall)
        return (spec.participant, spec.sample_id), base

    def test_hand_computed_mean_with_failures_in_denominator(self):
        participant = self.participants["a"]
        ids = ["s1", "s2", "s3"]
        schedule = pt.build_schedule([participant], ids)
        records = dict([
            self._fabricate(schedule[0], "complete", overall=4),
            self._fabricate(schedule[1], "complete", overall=2),
            self._fabricate(schedule[2], "judge_failure"),
        ])
        summary = pt.build_summary(records, schedule, [participant], ids)
        row = summary["participants"]["a"]
        self.assertEqual(row["denominator"], 3)
        self.assertEqual(row["n_valid"], 2)
        self.assertEqual(row["overall_score_sum"], 6)
        self.assertEqual(row["mean_overall_aesthetic_score"], 2.0)  # 6/3, failure=0
        self.assertEqual(row["valid_only_mean_overall"], 3.0)
        self.assertEqual(row["coverage"], 2 / 3)
        self.assertEqual(row["status_counts"], {"complete": 2, "judge_failure": 1})
        self.assertIn("sum(overall scores of complete rows) / evaluated_count",
                      summary["mean_formula"])
        self.assertEqual(summary["schema_version"], "task4-pointwise-summary-v1")
        self.assertNotIn("sha256", json.dumps(summary))
        dim = row["dimension_means_valid_only"]
        self.assertEqual(set(dim), set(SCORE_KEYS) - {"overall"})
        self.assertIsNotNone(dim["color"])

    def test_no_valid_rows(self):
        participant = self.participants["a"]
        schedule = pt.build_schedule([participant], ["s1"])
        records = dict([self._fabricate(schedule[0], "input_invalid")])
        summary = pt.build_summary(records, schedule, [participant], ["s1"])
        row = summary["participants"]["a"]
        self.assertEqual(row["mean_overall_aesthetic_score"], 0.0)
        self.assertIsNone(row["valid_only_mean_overall"])
        self.assertIsNone(row["dimension_means_valid_only"]["color"])

    def test_incomplete_rejected(self):
        participant = self.participants["a"]
        schedule = pt.build_schedule([participant], ["s1", "s2"])
        records = dict([self._fabricate(schedule[0], "complete", overall=3)])
        with self.assertRaises(EvaluationError):
            pt.build_summary(records, schedule, [participant], ["s1", "s2"])

    def test_formal_artifact(self):
        participant = self.participants["a"]
        schedule = pt.build_schedule([participant], ["s1"])
        records = dict([self._fabricate(schedule[0], "complete", overall=5)])
        summary = pt.build_summary(records, schedule, [participant], ["s1"])
        artifact = pt.build_formal_metric_artifact(summary, Path("/tmp/summary.json"))
        self.assertEqual(artifact["task"], "task4_pointwise")
        self.assertEqual(
            artifact["metrics"]["mean_overall_aesthetic_score"], {"a": 5.0}
        )
        self.assertEqual(artifact["context"]["eval_version"], "task4-pointwise-visual-v2")


class TestCliAndOutDir(unittest.TestCase):
    def test_sample_id_and_removed_subset_flags(self):
        with self.assertRaises(SystemExit):
            pt.parse_args(["--participant", "a=run-a", "--fraction", "0.5"])
        with self.assertRaises(SystemExit):
            pt.parse_args([])
        args = pt.parse_args(["--participant", "a=run-a", "--sample-id", "s1"])
        self.assertEqual(args.sample_id, ["s1"])

    def test_default_out_dir(self):
        participants = [eval_common.Participant("alpha", "ra", Path("/x/ra"))]
        out = pt._default_out_dir(participants, "gpt-5.4-2026-03-05")
        self.assertEqual(
            out.name, "task4-pointwise-visual-v2-gpt-5.4-2026-03-05-alpha"
        )
        out = pt._default_out_dir(participants, "judge")
        self.assertTrue(out.name.endswith("alpha"))

    def test_schedule_order(self):
        participants = [
            eval_common.Participant("a", "ra", Path("/x/ra")),
            eval_common.Participant("b", "rb", Path("/x/rb")),
        ]
        schedule = pt.build_schedule(participants, ["s1", "s2"])
        self.assertEqual(
            [(s.participant, s.sample_id) for s in schedule],
            [("a", "s1"), ("a", "s2"), ("b", "s1"), ("b", "s2")],
        )
        self.assertEqual([s.ordinal for s in schedule], [0, 1, 2, 3])


if __name__ == "__main__":
    unittest.main()
