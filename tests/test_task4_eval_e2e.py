"""End-to-end smoke tests for both Task 4 evaluators on a synthetic data pack."""

import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml
from PIL import Image

from tasks.task4 import eval_common
from tasks.task4 import eval_pairwise
from tasks.task4 import eval_pointwise
from tasks.task4.prompts import (
    INPUT_PROTOCOL,
    OUTPUT_AUDIT_PROTOCOL,
    RENDER_PROTOCOL,
    SCORE_KEYS,
)

IDS = [f"s{i:04d}" for i in range(1, 1396)]


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


def _write_data_pack(root: Path) -> None:
    root.mkdir(parents=True)
    (root / "cohort.json").write_text(json.dumps({
        "schema_version": "design-bench-task14-dedup-cohort-v1",
        "cohort": "task14-dedup-v1",
        "included_ids": IDS,
        "excluded_ids": [],
    }))
    (root / "freeze.json").write_text(json.dumps({
        "freeze_version": "task4-topk-v5-dedup-v1",
        "status": "active_default",
        "cohort": {
            "name": "task14-dedup-v1",
            "schema_version": "design-bench-task14-dedup-cohort-v1",
            "formal_items": 1395,
        },
        "protocols": {
            "input": INPUT_PROTOCOL,
            "output_audit": OUTPUT_AUDIT_PROTOCOL,
            "render": RENDER_PROTOCOL,
        },
    }))
    with (root / "queries.jsonl").open("w", encoding="utf-8") as handle:
        for sample_id in IDS:
            handle.write(json.dumps({"sample_id": sample_id}) + "\n")


def _write_run(runs_root: Path, run_name: str, sample_ids) -> None:
    run_dir = runs_root / run_name
    run_dir.mkdir(parents=True)
    (run_dir / "_cohort.json").write_text(json.dumps({
        "schema_version": "task4-run-cohort-v1",
        "cohort": "task14-dedup-v1",
        "cohort_schema_version": "design-bench-task14-dedup-cohort-v1",
        "included_ids": IDS,
        "sample_count": 1395,
    }))
    for sid in sample_ids:
        sample = run_dir / sid
        sample.mkdir()
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
        Image.new("RGB", (3840, 4320), (7, 7, 7)).save(sample / "full.png")
        (sample / "render_meta.json").write_text(json.dumps({
            "protocol_version": RENDER_PROTOCOL,
            "answer_bytes": (sample / "answer.html").stat().st_size,
            "crop_size": [3840, 4320],
            "full_size": [3840, 4320],
            "java_script_enabled": True,
        }))


def _scores_payload(overall=4):
    return json.dumps({
        key: {"score": overall if key == "overall" else 3, "reason": f"r-{key}"}
        for key in SCORE_KEYS
    })


class EndToEndFixture(unittest.TestCase):
    def setUp(self):
        eval_common._input_record_cache.clear()
        self.tmp = Path(tempfile.mkdtemp())
        self.data_root = self.tmp / "data"
        _write_data_pack(self.data_root)
        self.runs_root = self.tmp / "runs"
        _write_run(self.runs_root, "run-a", ["s0001"])
        _write_run(self.runs_root, "run-b", ["s0001"])
        os.environ["TASK4_E2E_KEY"] = "fake"
        self.config_path = self.tmp / "judge.yaml"
        self.config_path.write_text(yaml.safe_dump({
            "model": "judge-x",
            "protocol": "openai",
            "base_url": "http://fake",
            "api_key_env": "TASK4_E2E_KEY",
            "request": {"concurrency": 2, "timeout": 10},
            "eval": {
                "runs_root": str(self.runs_root),
                "data_root": str(self.data_root),
            },
        }))

    def tearDown(self):
        os.environ.pop("TASK4_E2E_KEY", None)


class TestPairwiseEndToEnd(EndToEndFixture):
    def test_single_sample_battle_publishes_summary(self):
        client = FakeClient(['{"winner": "A"}'])
        args = eval_pairwise.parse_args([
            "--config", str(self.config_path),
            "--participant", "alpha=run-a",
            "--participant", "beta=run-b",
            "--sample-id", "s0001",
            "--out-dir", str(self.tmp / "eval-out"),
        ])
        with mock.patch.object(eval_pairwise, "make_client", return_value=client):
            rc = asyncio.run(eval_pairwise.run_evaluation(args))
        self.assertIsNone(rc)
        out = self.tmp / "eval-out"
        battles = [json.loads(line)
                   for line in (out / "battles.jsonl").read_text().splitlines()]
        self.assertEqual(len(battles), 1)
        self.assertEqual(battles[0]["status"], "judged")
        summary = json.loads((out / "summary.json").read_text())
        self.assertEqual(summary["schema_version"], "task4-pairwise-summary-v3")
        self.assertEqual(summary["selected_count"], 1)
        a_side = battles[0]["assignment"]["A"]
        expected = {"alpha": 0.0, "beta": 0.0}
        expected[a_side] = 1.0
        self.assertEqual(
            {k: v["overall_win_rate"] for k, v in summary["participants"].items()},
            expected,
        )
        artifact = json.loads((out / "formal_metrics.json").read_text())
        self.assertEqual(artifact["task"], "task4_pairwise")
        self.assertEqual(
            artifact["metrics"]["overall_pairwise_win_rate"], expected
        )
        run_meta = json.loads((out / "run_meta.json").read_text())
        self.assertEqual(run_meta["schema_version"], "task4-pairwise-run-v3")
        self.assertNotIn("sha256", (out / "run_meta.json").read_text())
        # rerun: everything resumes, no new judge calls
        with mock.patch.object(eval_pairwise, "make_client", return_value=FakeClient([])):
            asyncio.run(eval_pairwise.run_evaluation(args))


class TestPointwiseEndToEnd(EndToEndFixture):
    def test_single_sample_score_publishes_summary(self):
        client = FakeClient([_scores_payload(overall=4)])
        args = eval_pointwise.parse_args([
            "--config", str(self.config_path),
            "--participant", "alpha=run-a",
            "--sample-id", "s0001",
            "--out-dir", str(self.tmp / "eval-out"),
        ])
        with mock.patch.object(eval_pointwise, "make_client", return_value=client):
            asyncio.run(eval_pointwise.run_evaluation(args))
        out = self.tmp / "eval-out"
        results = [json.loads(line)
                   for line in (out / "results.jsonl").read_text().splitlines()]
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["status"], "complete")
        summary = json.loads((out / "summary.json").read_text())
        self.assertEqual(summary["schema_version"], "task4-pointwise-summary-v1")
        self.assertEqual(
            summary["participants"]["alpha"]["mean_overall_aesthetic_score"], 4.0
        )
        self.assertEqual(summary["selection_mode"], "explicit_ids")
        artifact = json.loads((out / "formal_metrics.json").read_text())
        self.assertEqual(artifact["task"], "task4_pointwise")
        self.assertEqual(
            artifact["metrics"]["mean_overall_aesthetic_score"], {"alpha": 4.0}
        )

    def test_dry_run_audits_without_writing(self):
        args = eval_pointwise.parse_args([
            "--config", str(self.config_path),
            "--participant", "alpha=run-a",
            "--sample-id", "s0001",
            "--dry-run",
            "--out-dir", str(self.tmp / "eval-out"),
        ])
        asyncio.run(eval_pointwise.run_evaluation(args))
        self.assertFalse((self.tmp / "eval-out").exists())

    def test_full_cohort_is_default(self):
        args = eval_pointwise.parse_args([
            "--config", str(self.config_path),
            "--participant", "alpha=run-a",
            "--dry-run",
        ])
        # no --sample-id: the full formal cohort is evaluated by default
        with mock.patch.object(eval_pointwise, "dry_run_audit") as audit:
            asyncio.run(eval_pointwise.run_evaluation(args))
        evaluated_ids = audit.call_args[0][1]
        self.assertEqual(evaluated_ids, IDS)


if __name__ == "__main__":
    unittest.main()
