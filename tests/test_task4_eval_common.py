"""Offline tests for the shared Task 4 evaluation machinery."""

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

from src.api.client import InvalidResponseError
from tasks.task4 import eval_common
from tasks.task4.cohort import EvaluationError
from tasks.task4.prompts import INPUT_PROTOCOL, OUTPUT_AUDIT_PROTOCOL, RENDER_PROTOCOL


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


class TestCallJudge(unittest.TestCase):
    def _call(self, client, audit=None):
        return asyncio.run(eval_common.call_judge(
            client, "judge-x", [{"type": "text", "text": "hi"}],
            response_format={"type": "json_schema"}, response_audit=audit,
        ))

    def test_success_contract(self):
        client = FakeClient(['{"winner": "A"}'])
        audit = {}
        reply = self._call(client, audit)
        self.assertEqual(reply, '{"winner": "A"}')
        kwargs = client.calls[0]
        self.assertEqual(kwargs["temperature"], 0)
        self.assertIsNone(kwargs["max_tokens"])
        self.assertFalse(kwargs["disable_thinking"])
        self.assertTrue(kwargs["require_stop"])
        self.assertEqual(kwargs["extra_payload"], {"reasoning_effort": "none"})
        self.assertEqual(audit, {"attempt": 1, "finish_reason_stop": True})

    def test_retryable_then_success(self):
        client = FakeClient([_api_error(429), '{"winner": "tie"}'])
        audit = {}
        with mock.patch("asyncio.sleep", new=mock.AsyncMock()):
            reply = self._call(client, audit)
        self.assertEqual(reply, '{"winner": "tie"}')
        self.assertEqual(len(client.calls), 2)
        # audit is cleared per attempt; only the final successful attempt remains
        self.assertEqual(audit, {"attempt": 2, "finish_reason_stop": True})

    def test_persistent_retryable_exhausts(self):
        client = FakeClient([_api_error(500)] * 3)
        with mock.patch("asyncio.sleep", new=mock.AsyncMock()):
            with self.assertRaises(RuntimeError):
                self._call(client)
        self.assertEqual(len(client.calls), 3)

    def test_non_retryable_is_terminal_immediately(self):
        client = FakeClient([InvalidResponseError("bad"), '{"winner": "A"}'])
        with self.assertRaises(RuntimeError):
            self._call(client)
        self.assertEqual(len(client.calls), 1)
        client = FakeClient([_api_error(400), '{"winner": "A"}'])
        with self.assertRaises(RuntimeError):
            self._call(client)
        self.assertEqual(len(client.calls), 1)


class TestApiAuditHelpers(unittest.TestCase):
    def test_sanitize(self):
        audit = {"attempt": 1, "http_status": 200, "unknown": "x", "nested": {"a": 1}}
        self.assertEqual(
            eval_common._sanitize_api_audit(audit),
            {"attempt": 1, "http_status": 200},
        )

    def test_valid_matrix(self):
        self.assertTrue(eval_common._api_audit_valid(
            {"attempt": 1, "finish_reason_stop": True}, successful=True))
        self.assertTrue(eval_common._api_audit_valid(
            {"attempt": 3, "http_status": 500}, successful=False))
        self.assertFalse(eval_common._api_audit_valid(
            {"attempt": 4}, successful=False))
        self.assertFalse(eval_common._api_audit_valid(
            {"attempt": 1}, successful=True))
        self.assertFalse(eval_common._api_audit_valid(
            {"attempt": 1, "http_status": 99}, successful=False))
        self.assertFalse(eval_common._api_audit_valid(
            {"attempt": 1, "extra": 1}, successful=False))


class TestRunMetaAndSummaryIO(unittest.TestCase):
    def test_ensure_run_meta_roundtrip_and_drift(self):
        path = Path(tempfile.mkdtemp()) / "run_meta.json"
        config = {"a": 1, "b": [1, 2]}
        meta = eval_common.ensure_run_meta(path, "schema-v1", config)
        self.assertEqual(meta, {"schema_version": "schema-v1", "config": config})
        eval_common.ensure_run_meta(path, "schema-v1", config)  # same is fine
        with self.assertRaises(EvaluationError):
            eval_common.ensure_run_meta(path, "schema-v1", {"a": 2})

    def test_publish_summary(self):
        path = Path(tempfile.mkdtemp()) / "summary.json"
        summary = {"x": 1}
        eval_common.publish_summary(path, summary)
        eval_common.publish_summary(path, summary)
        with self.assertRaises(EvaluationError):
            eval_common.publish_summary(path, {"x": 2})

    def test_drop_failed_records(self):
        path = Path(tempfile.mkdtemp()) / "results.jsonl"
        rows = [
            {"ordinal": 0, "status": "complete"},
            {"ordinal": 1, "status": "judge_failure"},
            {"ordinal": 2, "status": "complete"},
            {"ordinal": 3, "status": "invalid_judge_output"},
        ]
        with path.open("w") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        existing = {row["ordinal"]: row for row in rows}
        dropped = eval_common.drop_failed_records(path, existing)
        self.assertEqual(dropped, 2)
        self.assertEqual(sorted(existing), [0, 2])
        kept = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual([row["ordinal"] for row in kept], [0, 2])
        archive = path.with_name("results.rejudged.jsonl")
        archived = [json.loads(line) for line in archive.read_text().splitlines()]
        self.assertEqual([row["ordinal"] for row in archived], [1, 3])
        self.assertEqual(eval_common.drop_failed_records(path, existing), 0)

    def test_canonical_json_deterministic(self):
        self.assertEqual(
            eval_common.canonical_json({"b": 1, "a": [2, 1]}),
            '{"a":[2,1],"b":1}',
        )


def _write_generation_sample(sample: Path, *, full_size=(3840, 4320),
                             meta_overrides=None, write_full=True):
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
    (sample / "assets").mkdir()
    (sample / "assets/v01.jpg").write_bytes(b"x")
    if write_full:
        Image.new("RGB", full_size, (7, 7, 7)).save(sample / "full.png")
    meta = {
        "protocol_version": RENDER_PROTOCOL,
        "answer_bytes": (sample / "answer.html").stat().st_size,
        "crop_size": [3840, 4320],
        "full_size": list(full_size),
        "java_script_enabled": True,
    }
    meta.update(meta_overrides or {})
    (sample / "render_meta.json").write_text(json.dumps(meta))


class TestAuditParticipantInput(unittest.TestCase):
    def setUp(self):
        self.run_dir = Path(tempfile.mkdtemp())
        self.participant = eval_common.Participant("lab", "run-x", self.run_dir)

    def test_valid_input(self):
        _write_generation_sample(self.run_dir / "s1")
        record = eval_common.audit_participant_input(self.participant, "s1")
        self.assertTrue(record["valid"])
        self.assertEqual(record["errors"], [])
        self.assertEqual(record["normalized"],
                         {"width": 1920, "height": 2160, "mode": "RGB"})
        self.assertEqual(record["source"]["format"], "PNG")
        self.assertEqual(record["source"]["size"], [3840, 4320])
        self.assertEqual(record["render"]["full_bytes"],
                         (self.run_dir / "s1/full.png").stat().st_size)
        self.assertEqual(record["schema_version"], "task4-pairwise-input-record-v3")
        self.assertNotIn("sha256", json.dumps(record))
        self.assertEqual(eval_common.input_height_1x(record), 2160)
        self.assertGreater(eval_common.input_full_bytes(record), 0)

    def test_missing_render(self):
        _write_generation_sample(self.run_dir / "s1", write_full=False)
        (self.run_dir / "s1/render_meta.json").unlink()
        record = eval_common.audit_participant_input(self.participant, "s1")
        self.assertFalse(record["valid"])
        categories = {e["category"] for e in record["errors"]}
        self.assertEqual(categories, {"render"})
        self.assertIsNone(eval_common.input_height_1x(record))
        self.assertIsNone(eval_common.input_full_bytes(record))

    def test_render_protocol_drift(self):
        _write_generation_sample(
            self.run_dir / "s1", meta_overrides={"protocol_version": "old"}
        )
        record = eval_common.audit_participant_input(self.participant, "s1")
        self.assertFalse(record["valid"])
        self.assertTrue(any("protocol" in e["message"] for e in record["errors"]))

    def test_declared_size_rules(self):
        _write_generation_sample(self.run_dir / "s1", full_size=(1920, 4320))
        record = eval_common.audit_participant_input(self.participant, "s1")
        self.assertFalse(record["valid"])
        _write_generation_sample(self.run_dir / "s2", full_size=(3840, 4321))
        record = eval_common.audit_participant_input(self.participant, "s2")
        self.assertFalse(record["valid"])
        self.assertTrue(any("even" in e["message"] for e in record["errors"]))

    def test_generation_failure_propagates(self):
        _write_generation_sample(self.run_dir / "s1")
        (self.run_dir / "s1/output_audit.json").write_text(json.dumps({
            "protocol_version": OUTPUT_AUDIT_PROTOCOL,
            "violations": ["x"],
        }))
        record = eval_common.audit_participant_input(self.participant, "s1")
        self.assertFalse(record["valid"])
        self.assertIn("generation", {e["category"] for e in record["errors"]})

    def test_memoized(self):
        _write_generation_sample(self.run_dir / "s1")
        first = eval_common.participant_input(self.participant, "s1")
        second = eval_common.participant_input(self.participant, "s1")
        self.assertIs(first, second)


class TestBuildFormalArtifact(unittest.TestCase):
    def test_pointwise_artifact(self):
        artifact = eval_common.build_formal_artifact(
            "task4_pointwise",
            {"mean_overall_aesthetic_score": {"gpt": 3.25, "claude": 4.0}},
            context={"eval_version": "v1"},
            provenance={"source_summary_path": "/x/summary.json"},
        )
        self.assertEqual(artifact["schema_version"], "design-bench-formal-metrics-v2")
        self.assertEqual(artifact["task"], "task4_pointwise")
        self.assertEqual(
            artifact["metric_definitions"]["mean_overall_aesthetic_score"]["abbreviation"],
            "AES",
        )
        self.assertNotIn("sha256", json.dumps(artifact))

    def test_pairwise_artifact(self):
        artifact = eval_common.build_formal_artifact(
            "task4_pairwise",
            {"overall_pairwise_win_rate": {"a": 0.5}},
            context={},
            provenance={},
        )
        self.assertEqual(artifact["metrics"]["overall_pairwise_win_rate"], {"a": 0.5})

    def test_validation_matrix(self):
        with self.assertRaises(EvaluationError):
            eval_common.build_formal_artifact(
                "task4_rules", {"x": {"a": 1.0}}, context={}, provenance={})
        with self.assertRaises(EvaluationError):
            eval_common.build_formal_artifact(
                "task4_pointwise", {"wrong": {"a": 1.0}}, context={}, provenance={})
        with self.assertRaises(EvaluationError):
            eval_common.build_formal_artifact(
                "task4_pairwise", {"overall_pairwise_win_rate": {"a": 1.5}},
                context={}, provenance={})
        with self.assertRaises(EvaluationError):
            eval_common.build_formal_artifact(
                "task4_pairwise", {"overall_pairwise_win_rate": {"a": float("nan")}},
                context={}, provenance={})
        with self.assertRaises(EvaluationError):
            eval_common.build_formal_artifact(
                "task4_pairwise", {"overall_pairwise_win_rate": {}},
                context={}, provenance={})


if __name__ == "__main__":
    unittest.main()
