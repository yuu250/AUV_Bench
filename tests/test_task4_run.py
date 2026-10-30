"""Offline tests for the Task 4 generation pipeline."""

import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

from tasks.task4 import run as task4_run
from tasks.task4.cohort import EvaluationError
from tasks.task4.prompts import INPUT_PROTOCOL, OUTPUT_AUDIT_PROTOCOL


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


def _montage(path: Path, n_assets: int) -> None:
    rows = (n_assets + 3) // 4
    Image.new("RGB", (760, 48 + rows * 170), (200, 210, 220)).save(path)


def _write_input_pack(root: Path, sid: str, n_assets: int = 2) -> None:
    sample = root / sid
    sample.mkdir(parents=True)
    caption = {
        "f": ["id", "src", "w", "h", "cap"],
        "a": [
            [f"A{i:04d}", f"assets/img_{i:02d}.jpg", 100 + i, 50 + i, f"cap {i}"]
            for i in range(1, n_assets + 1)
        ],
    }
    (sample / "assets.caption.min.json").write_text(json.dumps(caption))
    (sample / "selected_assets.json").write_text(json.dumps({
        "assets": [
            {"id": row[0], "path": row[1], "width": row[2], "height": row[3]}
            for row in caption["a"]
        ]
    }))
    _montage(sample / "topk.jpg", n_assets)
    for row in caption["a"]:
        target = sample / row[1]
        target.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (row[2], row[3]), (10, 20, 30)).save(target)


GOOD_HTML = "```html\n<html><body><img src=\"assets/v01.jpg\"></body></html>\n```"
TRUNCATED_HTML = "```html\n<html><body>never closes"


def _cfg(tmp: Path, *, model="m1"):
    queries = tmp / "custom.jsonl"
    queries.write_text(
        json.dumps({"sample_id": "s1", "target_lang": "en", "query_en": "Q1"}) + "\n"
        + json.dumps({"sample_id": "s2", "target_lang": "de", "query_en": "Q2"}) + "\n",
        encoding="utf-8",
    )
    inputs_root = tmp / "inputs"
    _write_input_pack(inputs_root, "s1")
    _write_input_pack(inputs_root, "s2")
    return {
        "model": model,
        "protocol": "openai",
        "base_url": "http://fake",
        "api_key_env": "TASK4_TEST_KEY",
        "input": {
            "queries": str(queries),
            "inputs_root": str(inputs_root),
            "assets_copy": False,
        },
        "request": {
            "temperature": 0.0,
            "disable_thinking": True,
            "concurrency": 4,
            "max_retries": 4,
            "timeout": 10,
            "max_tokens": None,
        },
        "output": {
            "runs_root": str(tmp / "runs"),
            "run_name": "{model}-topk-v5",
        },
    }


class TestHtmlHelpers(unittest.TestCase):
    def test_extract_html_fenced(self):
        self.assertEqual(
            task4_run.extract_html("pre\n```html\n<html></html>\n```\npost"),
            "<html></html>",
        )

    def test_extract_html_truncated_strips_opening_fence(self):
        self.assertEqual(
            task4_run.extract_html("```html\n<html><body>"),
            "<html><body>",
        )
        self.assertEqual(task4_run.extract_html("raw text"), "raw text")

    def test_is_complete_tail_window(self):
        self.assertTrue(task4_run.is_complete("<HTML></HTML>"))
        self.assertTrue(task4_run.is_complete("</html>" + "x" * 1990))
        self.assertFalse(task4_run.is_complete("</html>" + "x" * 2001))
        self.assertFalse(task4_run.is_complete("<html><body>"))

    def test_retry_ladder_constant(self):
        self.assertEqual(task4_run.RETRY_WAITS, [5, 15, 45])


class TestRunPipeline(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.cfg = _cfg(self.tmp)
        os.environ["TASK4_TEST_KEY"] = "fake"

    def tearDown(self):
        os.environ.pop("TASK4_TEST_KEY", None)

    def _run(self, cfg=None):
        asyncio.run(task4_run.run(cfg or self.cfg))

    def test_dry_run_writes_requests_without_client(self):
        self.cfg["dry_run"] = True
        self._run()
        run_dir = self.tmp / "runs" / "m1-topk-v5"
        for sid in ("s1", "s2"):
            self.assertTrue((run_dir / sid / "request_prompt.txt").is_file())
            self.assertTrue((run_dir / sid / "input_manifest.json").is_file())
            self.assertTrue((run_dir / sid / "contact_sheet_1.jpg").is_file())
            self.assertFalse((run_dir / sid / "answer.html").exists())
        events = (run_dir / "_events.jsonl").read_text().splitlines()
        self.assertEqual(len(events), 2)
        self.assertTrue(all(json.loads(e)["status"] == "dry_run" for e in events))
        meta_rows = (run_dir / "_meta.jsonl").read_text().splitlines()
        self.assertEqual(len(meta_rows), 2)
        cohort_doc = json.loads((run_dir / "_cohort.json").read_text())
        self.assertEqual(cohort_doc["cohort"], "custom:custom.jsonl")

    def test_full_generation_success(self):
        client = FakeClient([GOOD_HTML, GOOD_HTML])
        with mock.patch.object(task4_run, "make_client", return_value=client):
            self._run()
        self.assertEqual(len(client.calls), 2)
        call = client.calls[0]
        self.assertEqual(call["model"], "m1")
        self.assertEqual(call["temperature"], 0.0)
        self.assertTrue(call["disable_thinking"])
        self.assertIsNone(call["max_tokens"])
        run_dir = self.tmp / "runs" / "m1-topk-v5"
        for sid in ("s1", "s2"):
            sample = run_dir / sid
            self.assertTrue((sample / "answer.html").is_file())
            self.assertTrue((sample / "assets/v01.jpg").is_symlink())
            audit = json.loads((sample / "output_audit.json").read_text())
            self.assertEqual(audit["protocol_version"], OUTPUT_AUDIT_PROTOCOL)
            self.assertEqual(audit["violations"], [])
            self.assertEqual(audit["used_asset_paths"], ["assets/v01.jpg"])
            self.assertEqual(
                audit["answer_bytes"], (sample / "answer.html").stat().st_size
            )
            self.assertNotIn("sha256", json.dumps(audit))
            manifest = json.loads((sample / "input_manifest.json").read_text())
            self.assertEqual(manifest["protocol_version"], INPUT_PROTOCOL)
        statuses = [json.loads(line)["status"]
                    for line in (run_dir / "_meta.jsonl").read_text().splitlines()]
        self.assertEqual(statuses, ["ok", "ok"])
        self.assertFalse((run_dir / "_errors.jsonl").read_text().strip())

    def test_truncation_then_success_retries(self):
        client = FakeClient([TRUNCATED_HTML, GOOD_HTML, GOOD_HTML])
        with mock.patch.object(task4_run, "make_client", return_value=client), \
                mock.patch("asyncio.sleep", new=mock.AsyncMock()):
            self._run()
        self.assertEqual(len(client.calls), 3)  # s1 retried once, s2 first try
        rows = {
            json.loads(line)["sample_id"]: json.loads(line)
            for line in (self.tmp / "runs/m1-topk-v5/_meta.jsonl").read_text().splitlines()
        }
        self.assertEqual(rows["s1"]["status"], "ok")

    def test_persistent_truncation_errors_after_max_attempts(self):
        client = FakeClient([TRUNCATED_HTML] * 8)
        with mock.patch.object(task4_run, "make_client", return_value=client), \
                mock.patch("asyncio.sleep", new=mock.AsyncMock()):
            self._run()
        self.assertEqual(len(client.calls), 8)  # 2 samples x 4 attempts
        errors = (self.tmp / "runs/m1-topk-v5/_errors.jsonl").read_text().splitlines()
        self.assertEqual(len(errors), 2)
        self.assertIn("truncated", json.loads(errors[0])["error"])

    def test_asset_policy_violation_retries_then_errors(self):
        bad = "```html\n<html><body><img src=\"https://cdn.example.com/x.png\"></body></html>\n```"
        client = FakeClient([bad] * 8)
        with mock.patch.object(task4_run, "make_client", return_value=client), \
                mock.patch("asyncio.sleep", new=mock.AsyncMock()):
            self._run()
        errors = (self.tmp / "runs/m1-topk-v5/_errors.jsonl").read_text().splitlines()
        self.assertEqual(len(errors), 2)
        self.assertIn("asset_policy:external_resource", json.loads(errors[0])["error"])

    def test_api_error_retried_then_recorded(self):
        client = FakeClient([RuntimeError("boom")] * 8)
        with mock.patch.object(task4_run, "make_client", return_value=client), \
                mock.patch("asyncio.sleep", new=mock.AsyncMock()):
            self._run()
        self.assertEqual(len(client.calls), 8)
        rows = [json.loads(line)
                for line in (self.tmp / "runs/m1-topk-v5/_meta.jsonl").read_text().splitlines()]
        self.assertTrue(all(row["status"] == "error" for row in rows))
        self.assertIn("RuntimeError", rows[0]["error"])

    def test_resume_skips_valid_samples(self):
        client = FakeClient([GOOD_HTML, GOOD_HTML])
        with mock.patch.object(task4_run, "make_client", return_value=client):
            self._run()
        client2 = FakeClient([])
        with mock.patch.object(task4_run, "make_client", return_value=client2):
            self._run()
        self.assertEqual(client2.calls, [])
        rows = [json.loads(line)
                for line in (self.tmp / "runs/m1-topk-v5/_meta.jsonl").read_text().splitlines()]
        self.assertTrue(all(row["status"] == "ok" for row in rows))
        events = (self.tmp / "runs/m1-topk-v5/_events.jsonl").read_text().splitlines()
        self.assertEqual(len(events), 4)  # 2 ok + 2 skipped

    def test_cohort_drift_fails_closed(self):
        client = FakeClient([GOOD_HTML, GOOD_HTML])
        with mock.patch.object(task4_run, "make_client", return_value=client):
            self._run()
        cohort_path = self.tmp / "runs/m1-topk-v5/_cohort.json"
        doc = json.loads(cohort_path.read_text())
        doc["sample_count"] = 999
        cohort_path.write_text(json.dumps(doc))
        with self.assertRaises(EvaluationError):
            self._run()

    def test_unknown_sample_ids_rejected(self):
        self.cfg["sample_ids"] = ["nope"]
        with self.assertRaises(EvaluationError):
            self._run()

    def test_limit_and_sample_ids(self):
        self.cfg["sample_ids"] = ["s2"]
        self.cfg["dry_run"] = True
        self._run()
        run_dir = self.tmp / "runs" / "m1-topk-v5"
        self.assertTrue((run_dir / "s2/request_prompt.txt").is_file())
        self.assertFalse((run_dir / "s1").exists())

    def test_bad_run_name_rejected(self):
        self.cfg["output"]["run_name"] = "../escape"
        with self.assertRaises(Exception):
            self._run()


if __name__ == "__main__":
    unittest.main()
