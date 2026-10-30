"""Pipeline tests for the Task 2 v8 runner (fake client, no network)."""

import asyncio
import json
import os
import random
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src.api.client import (
    ImageTooLargeError,
    InvalidResponseError,
    is_retryable_api_error,
)
from src.utils.config import ConfigError
from tasks.task2 import run as task2_run
from tasks.task2.coordinate_contract import RESPONSE_SCHEMA_NAME
from tasks.task2.scoring import EVAL_VERSION

WIDTH = 1000
HEIGHT = 1800  # exactly one slice: no overview branch
GT_XYWH = [180, 90, 360, 180]           # xyxy (180,90,540,270) on the page
GT_BOX_NORMALIZED = [180, 50, 540, 150]  # same region in slice-normalized xyxy


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


def api_error(status: int) -> Exception:
    exc = Exception(f"HTTP {status}")
    exc.status_code = status
    return exc


def correct_version(sample_id: str) -> int:
    return random.Random(sample_id).choice([1, 2])


def joint_reply(version: int, *, dims=("Color",),
                locations=((1, GT_BOX_NORMALIZED),)) -> str:
    return json.dumps({
        "degraded_version": version,
        "dimensions": list(dims),
        "coordinate_space": "normalized_1000",
        "box_format": "xyxy",
        "locations": [{"slice": s, "box": b} for s, b in locations],
    })


def make_pair(root: Path, sample_id: str, *, height: int = HEIGHT,
              gt_xywh=GT_XYWH, gt_dims=("色彩",)) -> tuple[dict, dict]:
    deg_image = root / f"{sample_id}.png"
    clean_image = root / f"{sample_id}_clean.png"
    Image.new("RGB", (WIDTH, height), (240, 240, 240)).save(deg_image)
    Image.new("RGB", (WIDTH, height), (245, 245, 245)).save(clean_image)
    defects = []
    if gt_xywh is not None:
        defects = [{
            "rule_id": "r1", "dim": "Color", "dim_cn": "色彩",
            "bboxes": [gt_xywh],
            "localization_xpaths": ["/main/div[1]"],
            "victim_xpaths": ["/main/div[1]"],
        }]
    deg = {
        "sample_id": sample_id, "source": "degraded", "layer": "diagnostic",
        "image": deg_image.name, "image_size": [WIDTH, height],
        "has_defect": bool(defects), "n_defects": len(defects),
        "gt_dims": list(gt_dims), "defects": defects,
    }
    clean = {
        "sample_id": sample_id + "_clean", "source": "paired_control",
        "layer": "diagnostic",
        "image": clean_image.name, "image_size": [WIDTH, height],
    }
    return deg, clean


def make_cfg(root: Path) -> dict:
    return {
        "model": "fake-model",
        "protocol": "openai",
        "input": {"images_root": str(root)},
        "request": {
            "temperature": 0.0, "max_tokens": None, "disable_thinking": True,
            "concurrency": 1, "max_retries": 3, "timeout": 1,
        },
    }


def diagnose(client, cfg, pair):
    run_meta = task2_run.build_run_meta(cfg["model"], 1, None)
    return asyncio.run(task2_run.diagnose_pair(
        client, cfg, pair, run_meta, None, asyncio.Semaphore(1)))


class RetryTableTests(unittest.TestCase):
    def test_status_code_classification(self):
        for status in (408, 429, 500, 503):
            self.assertTrue(is_retryable_api_error(api_error(status)), status)
        for status in (400, 401, 403, 404, 422):
            self.assertFalse(is_retryable_api_error(api_error(status)), status)

    def test_terminal_and_transport_classification(self):
        self.assertFalse(is_retryable_api_error(ImageTooLargeError("big")))
        self.assertFalse(is_retryable_api_error(InvalidResponseError("bad")))
        self.assertFalse(is_retryable_api_error(ValueError("bad json")))
        self.assertTrue(is_retryable_api_error(OSError("connection reset")))
        self.assertTrue(is_retryable_api_error(TimeoutError("timed out")))
        connection_error = type("APIConnectionError", (Exception,), {})("down")
        self.assertTrue(is_retryable_api_error(connection_error))


class DiagnosePairTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.cfg = make_cfg(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def test_correct_diagnosis_scores_the_full_chain(self):
        pair = make_pair(self.root, "s1")
        reply = joint_reply(correct_version("s1"))
        client = FakeClient([reply])
        rec = diagnose(client, self.cfg, pair)

        self.assertEqual(rec["status"], "complete")
        self.assertTrue(rec["det_correct"])
        self.assertEqual(rec["det_pred"], correct_version("s1"))
        self.assertTrue(rec["attr_exact"])
        self.assertEqual(rec["attr_pred"], ["色彩"])
        self.assertEqual((rec["loc_iou50_tp"], rec["loc_iou50_fp"], rec["loc_iou50_fn"]),
                         (1, 0, 0))
        self.assertTrue(rec["loc_exact"])
        self.assertTrue(rec["chain_success"])
        self.assertTrue(rec["joint_object_valid"])
        self.assertTrue(rec["joint_schema_valid"])
        self.assertTrue(rec["joint_valid"])
        self.assertTrue(rec["coordinate_declaration_valid"])
        self.assertTrue(rec["loc_coordinate_contract_valid"])
        self.assertEqual(rec["coordinate_space"], "normalized_1000")
        self.assertEqual(rec["box_format"], "xyxy")
        self.assertEqual(rec["loc_pred_canonical"],
                         [{"slice": 1, "box": [180.0, 50.0, 540.0, 150.0]}])
        self.assertEqual(rec["loc_pred_raw"], rec["loc_pred_canonical"])
        self.assertEqual(rec["det_raw"], reply)
        self.assertEqual(rec["attr_raw"], reply)
        self.assertEqual(rec["loc_raw"], reply)
        self.assertEqual(rec["eval_version"], EVAL_VERSION)
        self.assertEqual(rec["n_slices"], 1)
        self.assertEqual(rec["input_geometry"]["policy"], "pair-max-height-v1")
        self.assertEqual(len(rec["input_image_audit"]), 2)  # one slice, two versions
        for image_audit in rec["input_image_audit"]:
            self.assertNotIn("encoded_sha256", image_audit)

        call = client.calls[0]
        self.assertEqual(call["response_format"]["json_schema"]["name"],
                         RESPONSE_SCHEMA_NAME)
        self.assertIsNone(call["max_tokens"])
        self.assertTrue(call["require_stop"])
        self.assertEqual(call["temperature"], 0.0)

    def test_wrong_version_gates_attribution_and_localization(self):
        pair = make_pair(self.root, "s2")
        wrong = 3 - correct_version("s2")
        client = FakeClient([joint_reply(wrong)])
        rec = diagnose(client, self.cfg, pair)

        self.assertEqual(rec["status"], "complete")
        self.assertFalse(rec["det_correct"])
        self.assertEqual(rec["attr_pred"], [])
        self.assertFalse(rec["attr_exact"])
        # Ungated diagnostics keep the raw answer for error analysis.
        self.assertEqual(rec["attr_pred_raw"], ["色彩"])
        self.assertTrue(rec["attr_ungated_attr_exact"])
        self.assertEqual((rec["loc_iou50_tp"], rec["loc_iou50_fp"], rec["loc_iou50_fn"]),
                         (0, 1, 1))
        self.assertFalse(rec["loc_exact"])
        self.assertEqual(rec["loc_ungated_loc_iou50_tp"], 1)
        self.assertFalse(rec["chain_success"])

    def test_garbage_reply_is_a_terminal_parse_failure(self):
        pair = make_pair(self.root, "s3")
        client = FakeClient(["this is not json"])
        rec = diagnose(client, self.cfg, pair)

        self.assertEqual(rec["status"], "complete")  # parse failure is not an API error
        self.assertTrue(rec["det_parse_fail"])
        self.assertFalse(rec["joint_object_valid"])
        self.assertFalse(rec["det_correct"])
        self.assertEqual(rec["loc_iou50_fn"], 1)  # fixed denominator keeps the GT
        self.assertFalse(rec["chain_success"])
        self.assertEqual(len(client.calls), 1)  # parse failures never retry

    def test_retryable_error_retries_then_succeeds(self):
        pair = make_pair(self.root, "s4")
        client = FakeClient([api_error(500), joint_reply(correct_version("s4"))])
        original_sleep = asyncio.sleep

        async def no_sleep(_seconds):
            return None

        asyncio.sleep = no_sleep
        try:
            rec = diagnose(client, self.cfg, pair)
        finally:
            asyncio.sleep = original_sleep
        self.assertEqual(rec["status"], "complete")
        self.assertEqual(len(client.calls), 2)

    def test_terminal_error_is_not_retried_and_stays_in_denominator(self):
        pair = make_pair(self.root, "s5")
        client = FakeClient([api_error(400)])
        rec = diagnose(client, self.cfg, pair)

        self.assertEqual(rec["status"], "error")
        self.assertIn("error", rec)
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(rec["loc_iou50_fn"], 1)
        self.assertFalse(rec["chain_success"])

    def test_exhausted_retries_become_an_error_record(self):
        pair = make_pair(self.root, "s6")
        client = FakeClient([api_error(500), api_error(500), api_error(500)])
        original_sleep = asyncio.sleep

        async def no_sleep(_seconds):
            return None

        asyncio.sleep = no_sleep
        try:
            rec = diagnose(client, self.cfg, pair)
        finally:
            asyncio.sleep = original_sleep
        self.assertEqual(rec["status"], "error")
        self.assertEqual(len(client.calls), 3)


class RunFlowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        os.environ["TASK2_TEST_API_KEY"] = "test-only"

    def tearDown(self):
        self.tmp.cleanup()
        os.environ.pop("TASK2_TEST_API_KEY", None)

    def _manifest_and_cfg(self) -> tuple[Path, dict]:
        eligible = make_pair(self.root, "ok1")
        too_tall = make_pair(self.root, "big1", height=20000)  # 13 slices > 12
        manifest = self.root / "manifest.jsonl"
        with manifest.open("w", encoding="utf-8") as handle:
            for entry in (*eligible, *too_tall):
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        cfg = make_cfg(self.root)
        cfg["api_key_env"] = "TASK2_TEST_API_KEY"
        cfg["input"]["manifest"] = str(manifest)
        cfg["output"] = {"predictions": str(self.root / "predictions.jsonl")}
        return manifest, cfg

    def _run(self, cfg, client):
        original_make_client = task2_run.make_client
        task2_run.make_client = lambda *a, **k: client
        try:
            asyncio.run(task2_run.run(cfg))
        finally:
            task2_run.make_client = original_make_client

    def test_end_to_end_run_filters_cohort_and_resumes(self):
        _, cfg = self._manifest_and_cfg()
        client = FakeClient([joint_reply(correct_version("ok1"))])
        self._run(cfg, client)

        lines = (self.root / "predictions.jsonl").read_text(
            encoding="utf-8").strip().splitlines()
        self.assertEqual(len(lines), 1)  # the 13-slice pair never reaches the API
        rec = json.loads(lines[0])
        self.assertEqual(rec["sample_id"], "ok1")
        self.assertEqual(rec["status"], "complete")
        self.assertTrue(rec["chain_success"])
        self.assertEqual(len(client.calls), 1)

        # Rerun: every written record is terminal, so the client is untouched.
        second = FakeClient([])
        self._run(cfg, second)
        self.assertEqual(len(second.calls), 0)
        lines = (self.root / "predictions.jsonl").read_text(
            encoding="utf-8").strip().splitlines()
        self.assertEqual(len(lines), 1)

    def test_output_with_foreign_sample_id_is_rejected(self):
        _, cfg = self._manifest_and_cfg()
        foreign = {
            "sample_id": "zzz",
            "eval_version": EVAL_VERSION,
            "run_meta": task2_run.build_run_meta("fake-model", 1, None),
        }
        (self.root / "predictions.jsonl").write_text(
            json.dumps(foreign) + "\n", encoding="utf-8")
        with self.assertRaises(ConfigError):
            self._run(cfg, FakeClient([]))

    def test_output_from_another_model_is_rejected(self):
        _, cfg = self._manifest_and_cfg()
        foreign = {
            "sample_id": "ok1",
            "eval_version": EVAL_VERSION,
            "run_meta": task2_run.build_run_meta("other-model", 1, None),
        }
        (self.root / "predictions.jsonl").write_text(
            json.dumps(foreign) + "\n", encoding="utf-8")
        with self.assertRaises(ConfigError):
            self._run(cfg, FakeClient([]))


if __name__ == "__main__":
    unittest.main()
