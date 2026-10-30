"""Offline tests for the Task 4 renderer skip logic and constants."""

import json
import tempfile
import unittest
from pathlib import Path

from tasks.task4 import render
from tasks.task4.prompts import RENDER_PROTOCOL


class TestConstants(unittest.TestCase):
    def test_frozen_render_contract(self):
        self.assertEqual(RENDER_PROTOCOL, "task4-render-1920x2160-dpr2-v2")
        self.assertEqual(render.CROP_CSS_H, 2160)
        self.assertEqual(render.CROP_SIZE, (3840, 4320))
        self.assertEqual(render.BROWSER_RELAUNCH_LIMIT, 10)


class TestExistingRenderValid(unittest.TestCase):
    def setUp(self):
        self.sample_dir = Path(tempfile.mkdtemp())
        (self.sample_dir / "answer.html").write_text(
            "<html></html>", encoding="utf-8"
        )
        self.answer_bytes = (self.sample_dir / "answer.html").stat().st_size

    def _write_outputs(self, *, meta_overrides=None, crop=True, full=True):
        if crop:
            (self.sample_dir / "crop4320.png").write_bytes(b"png")
        if full:
            (self.sample_dir / "full.png").write_bytes(b"png")
        meta = {
            "protocol_version": RENDER_PROTOCOL,
            "answer_bytes": self.answer_bytes,
            "crop_size": [3840, 4320],
            "full_size": [3840, 8640],
            "java_script_enabled": True,
        }
        meta.update(meta_overrides or {})
        (self.sample_dir / "render_meta.json").write_text(json.dumps(meta))

    def test_valid_prior_render_skips(self):
        self._write_outputs()
        self.assertTrue(
            render.existing_render_valid(self.sample_dir, self.answer_bytes)
        )

    def test_missing_outputs_do_not_skip(self):
        self.assertFalse(
            render.existing_render_valid(self.sample_dir, self.answer_bytes)
        )
        self._write_outputs(crop=False)
        self.assertFalse(
            render.existing_render_valid(self.sample_dir, self.answer_bytes)
        )

    def test_protocol_drift_renders_again(self):
        self._write_outputs(meta_overrides={"protocol_version": "old"})
        self.assertFalse(
            render.existing_render_valid(self.sample_dir, self.answer_bytes)
        )

    def test_answer_change_renders_again(self):
        self._write_outputs()
        (self.sample_dir / "answer.html").write_text(
            "<html><body>changed</body></html>", encoding="utf-8"
        )
        new_bytes = (self.sample_dir / "answer.html").stat().st_size
        self.assertFalse(render.existing_render_valid(self.sample_dir, new_bytes))

    def test_crop_size_drift_renders_again(self):
        self._write_outputs(meta_overrides={"crop_size": [3840, 4000]})
        self.assertFalse(
            render.existing_render_valid(self.sample_dir, self.answer_bytes)
        )

    def test_broken_meta_renders_again(self):
        self._write_outputs()
        (self.sample_dir / "render_meta.json").write_text("not json")
        self.assertFalse(
            render.existing_render_valid(self.sample_dir, self.answer_bytes)
        )

    def test_meta_has_no_digest_fields(self):
        self._write_outputs()
        self.assertNotIn(
            "sha256", (self.sample_dir / "render_meta.json").read_text()
        )


if __name__ == "__main__":
    unittest.main()
