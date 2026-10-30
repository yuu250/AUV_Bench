"""Offline tests for Task 4 Top-K request materialization."""

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from tasks.task4 import inputs
from tasks.task4.prompts import INPUT_PROTOCOL, OUTPUT_AUDIT_PROTOCOL


def _montage(path: Path, n_assets: int) -> None:
    cols, cell_h, header_h = 4, 170, 48
    rows = (n_assets + cols - 1) // cols
    image = Image.new("RGB", (cols * 190, header_h + rows * cell_h), (200, 210, 220))
    image.save(path, format="JPEG")


def _caption(n_assets: int, ext: str = ".jpg") -> dict:
    return {
        "f": ["id", "src", "w", "h", "cap"],
        "a": [
            [f"A{i:04d}", f"assets/img_{i:02d}{ext}", 100 + i, 50 + i, f"cap {i}"]
            for i in range(1, n_assets + 1)
        ],
    }


def _write_inputs(root: Path, sid: str, n_assets: int, *, ext: str = ".jpg") -> None:
    sample = root / sid
    sample.mkdir(parents=True)
    caption = _caption(n_assets, ext)
    (sample / "assets.caption.min.json").write_text(
        json.dumps(caption), encoding="utf-8"
    )
    (sample / "selected_assets.json").write_text(
        json.dumps({
            "assets": [
                {"id": row[0], "path": row[1], "width": row[2], "height": row[3]}
                for row in caption["a"]
            ]
        }),
        encoding="utf-8",
    )
    _montage(sample / "topk.jpg", n_assets)
    for row in caption["a"]:
        target = sample / row[1]
        target.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (row[2], row[3]), (10, 20, 30)).save(target)


class TestBuildCopyBlock(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())

    def test_missing_file_no_block(self):
        self.assertEqual(inputs.build_copy_block("s1", self.root), "")

    def test_wrap_dedup_verbatim_lines(self):
        sample = self.root / "s1"
        sample.mkdir()
        # extractor-shaped lines: already normalized, exact duplicate, and a
        # trailing space left by the upstream 300-char cut
        (sample / "copy.txt").write_text(
            "Hello world \nHello world \nHello   world\nSecond line\n\n",
            encoding="utf-8",
        )
        block = inputs.build_copy_block("s1", self.root)
        self.assertIn("use VERBATIM", block)
        self.assertEqual(block.count("Hello world "), 1)  # exact-match dedup
        self.assertIn("Hello world \n", block)  # trailing space preserved
        self.assertIn("Hello   world\n", block)  # whitespace variants not merged
        self.assertIn("Second line", block)

    def test_total_cap_truncation_marker(self):
        sample = self.root / "s1"
        sample.mkdir()
        (sample / "copy.txt").write_text(
            "\n".join(f"line {i} " + "x" * 200 for i in range(100)),
            encoding="utf-8",
        )
        block = inputs.build_copy_block("s1", self.root, total_cap=15000)
        self.assertIn("... (truncated, ", block)
        self.assertIn("more lines)", block)

    def test_empty_file_no_block(self):
        sample = self.root / "s1"
        sample.mkdir()
        (sample / "copy.txt").write_text("\n  \n", encoding="utf-8")
        self.assertEqual(inputs.build_copy_block("s1", self.root), "")


class TestBuildRequestAssets(unittest.TestCase):
    def test_aliases_and_mapping(self):
        request, mapping = inputs.build_request_assets(_caption(2, ".png"))
        self.assertEqual(
            request["a"],
            [["V01", "assets/v01.png", 101, 51, "cap 1"],
             ["V02", "assets/v02.png", 102, 52, "cap 2"]],
        )
        self.assertEqual(mapping[0]["source_src"], "assets/img_01.png")
        self.assertEqual(mapping[1]["id"], "V02")

    def test_unsafe_extension_rejected(self):
        caption = _caption(1)
        caption["a"][0][1] = "assets/no_extension"
        with self.assertRaises(ValueError):
            inputs.build_request_assets(caption)
        caption["a"][0][1] = "assets/x.JP$G"
        with self.assertRaises(ValueError):
            inputs.build_request_assets(caption)


class TestBuildContactSheets(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.sid = "s1"
        _write_inputs(self.root, self.sid, 5)
        self.sample_dir = Path(tempfile.mkdtemp())

    def test_single_sheet_geometry(self):
        caption = json.loads(
            (self.root / self.sid / "assets.caption.min.json").read_text()
        )
        request, _ = inputs.build_request_assets(caption)
        sheets = inputs.build_contact_sheets(
            self.sid, request["a"], self.sample_dir, self.root
        )
        self.assertEqual(len(sheets), 1)
        self.assertEqual(sheets[0].name, "contact_sheet_1.jpg")
        with Image.open(sheets[0]) as sheet:
            self.assertEqual(sheet.size, (760, 48 + 2 * 170))
            self.assertEqual(sheet.format, "JPEG")

    def test_two_sheets_for_17_assets(self):
        sid = "s17"
        _write_inputs(self.root, sid, 17)
        caption = json.loads(
            (self.root / sid / "assets.caption.min.json").read_text()
        )
        request, _ = inputs.build_request_assets(caption)
        sheets = inputs.build_contact_sheets(
            sid, request["a"], self.sample_dir, self.root
        )
        self.assertEqual([p.name for p in sheets],
                         ["contact_sheet_1.jpg", "contact_sheet_2.jpg"])
        with Image.open(sheets[1]) as sheet:
            self.assertEqual(sheet.size, (760, 48 + 1 * 170))

    def test_bad_montage_geometry_rejected(self):
        (self.root / self.sid / "topk.jpg").unlink()
        _montage(self.root / self.sid / "topk.jpg", 9)  # 3 rows; caption has 5 (2 rows)
        caption = json.loads(
            (self.root / self.sid / "assets.caption.min.json").read_text()
        )
        request, _ = inputs.build_request_assets(caption)
        with self.assertRaises(ValueError):
            inputs.build_contact_sheets(
                self.sid, request["a"], self.sample_dir, self.root
            )

    def test_missing_montage_rejected(self):
        with self.assertRaises(FileNotFoundError):
            inputs.build_contact_sheets(
                "missing", [[1]], self.sample_dir, self.root
            )


class TestBuildTopkRequest(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.sid = "s1"
        _write_inputs(self.root, self.sid, 3)
        (self.root / self.sid / "copy.txt").write_text(
            "Some headline\n", encoding="utf-8"
        )
        self.sample_dir = Path(tempfile.mkdtemp()) / self.sid

    def test_happy_path_artifacts_and_parts(self):
        parts, manifest = inputs.build_topk_request(
            self.sid, "German", "Build it", self.sample_dir, self.root,
            model="m1",
        )
        self.assertEqual(parts[0], {"type": "text",
                                    "text": (self.sample_dir / "request_prompt.txt").read_text()})
        self.assertEqual(len(parts), 2)  # text + one contact sheet
        self.assertEqual(parts[1]["type"], "image")
        self.assertEqual(parts[1]["media_type"], "image/jpeg")
        prompt = parts[0]["text"]
        self.assertIn("written in German", prompt)
        self.assertIn("## Requirement\nBuild it", prompt)
        self.assertIn("Some headline", prompt)
        self.assertIn("assets/v01.jpg", prompt)
        self.assertEqual(manifest["protocol_version"], INPUT_PROTOCOL)
        self.assertEqual(manifest["model"], "m1")
        self.assertEqual(manifest["mode"], "topk")
        self.assertEqual(manifest["asset_count"], 3)
        self.assertEqual(manifest["allowed_asset_paths"],
                         ["assets/v01.jpg", "assets/v02.jpg", "assets/v03.jpg"])
        self.assertEqual(manifest["message_order"], ["text", "contact_sheet_1.jpg"])
        self.assertEqual(manifest["prompt_bytes"], len(prompt.encode("utf-8")))
        serialized = json.dumps(manifest)
        self.assertNotIn("sha256", serialized)
        on_disk = json.loads((self.sample_dir / "input_manifest.json").read_text())
        self.assertEqual(on_disk, manifest)
        compact = json.loads(
            (self.sample_dir / "request_assets.caption.min.json").read_text()
        )
        self.assertEqual(compact["a"][0][:2], ["V01", "assets/v01.jpg"])
        self.assertTrue((self.sample_dir / "contact_sheet_1.jpg").is_file())

    def test_empty_caption_uses_empty_prompt(self):
        sid = "empty"
        sample = self.root / sid
        sample.mkdir()
        (sample / "assets.caption.min.json").write_text(
            json.dumps({"f": ["id", "src", "w", "h", "cap"], "a": []}),
            encoding="utf-8",
        )
        (sample / "selected_assets.json").write_text(
            json.dumps({"assets": []}), encoding="utf-8"
        )
        parts, manifest = inputs.build_topk_request(
            sid, "English", "Q", self.sample_dir, self.root, model="m1"
        )
        self.assertEqual(len(parts), 1)  # no contact sheet
        self.assertIn("array below is empty", parts[0]["text"])
        self.assertEqual(manifest["asset_count"], 0)
        self.assertEqual(manifest["contact_sheets"], [])

    def test_missing_inputs_rejected(self):
        with self.assertRaises(FileNotFoundError):
            inputs.build_topk_request(
                "missing", "English", "Q", self.sample_dir, self.root, model="m1"
            )

    def test_caption_schema_rejected(self):
        (self.root / self.sid / "assets.caption.min.json").write_text(
            json.dumps({"f": ["wrong"], "a": []}), encoding="utf-8"
        )
        with self.assertRaises(ValueError):
            inputs.build_topk_request(
                self.sid, "English", "Q", self.sample_dir, self.root, model="m1"
            )

    def test_caption_selected_mismatch_rejected(self):
        selected = json.loads(
            (self.root / self.sid / "selected_assets.json").read_text()
        )
        selected["assets"][0]["width"] += 1
        (self.root / self.sid / "selected_assets.json").write_text(
            json.dumps(selected), encoding="utf-8"
        )
        with self.assertRaises(ValueError):
            inputs.build_topk_request(
                self.sid, "English", "Q", self.sample_dir, self.root, model="m1"
            )


class TestMaterializeTopkAssets(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.sid = "s1"
        _write_inputs(self.root, self.sid, 2)
        caption = json.loads(
            (self.root / self.sid / "assets.caption.min.json").read_text()
        )
        _, self.mapping = inputs.build_request_assets(caption)
        self.sample_dir = Path(tempfile.mkdtemp())

    def test_symlink_default(self):
        inputs.materialize_topk_assets(
            self.sid, self.sample_dir, self.mapping, self.root
        )
        alias = self.sample_dir / "assets/v01.jpg"
        self.assertTrue(alias.is_symlink())
        self.assertEqual(alias.resolve(), (self.root / self.sid / "assets/img_01.jpg").resolve())

    def test_copy_mode(self):
        inputs.materialize_topk_assets(
            self.sid, self.sample_dir, self.mapping, self.root, assets_copy=True
        )
        alias = self.sample_dir / "assets/v02.jpg"
        self.assertTrue(alias.is_file())
        self.assertFalse(alias.is_symlink())
        self.assertEqual(
            alias.read_bytes(),
            (self.root / self.sid / "assets/img_02.jpg").read_bytes(),
        )

    def test_missing_source_rejected(self):
        (self.root / self.sid / "assets/img_01.jpg").unlink()
        with self.assertRaises(FileNotFoundError):
            inputs.materialize_topk_assets(
                self.sid, self.sample_dir, self.mapping, self.root
            )


class TestCanSkipTopk(unittest.TestCase):
    def setUp(self):
        self.sample_dir = Path(tempfile.mkdtemp())
        self.answer = self.sample_dir / "answer.html"
        self.answer.write_text("<html></html>", encoding="utf-8")
        (self.sample_dir / "assets").mkdir()
        (self.sample_dir / "assets/v01.jpg").write_bytes(b"x")
        (self.sample_dir / "input_manifest.json").write_text(json.dumps({
            "protocol_version": INPUT_PROTOCOL,
            "asset_mapping": [{"alias_src": "assets/v01.jpg"}],
        }), encoding="utf-8")
        (self.sample_dir / "output_audit.json").write_text(json.dumps({
            "protocol_version": OUTPUT_AUDIT_PROTOCOL,
            "violations": [],
        }), encoding="utf-8")

    def test_skip_when_everything_valid(self):
        self.assertTrue(inputs.can_skip_topk(self.sample_dir, self.answer))

    def test_no_skip_matrix(self):
        cases = []
        cases.append(("input_manifest.json", {"protocol_version": "old",
                                              "asset_mapping": []}))
        cases.append(("output_audit.json", {"protocol_version": "old",
                                            "violations": []}))
        cases.append(("output_audit.json", {"protocol_version": OUTPUT_AUDIT_PROTOCOL,
                                            "violations": ["x"]}))
        for name, payload in cases:
            backup = (self.sample_dir / name).read_bytes()
            (self.sample_dir / name).write_text(json.dumps(payload), encoding="utf-8")
            self.assertFalse(inputs.can_skip_topk(self.sample_dir, self.answer), name)
            (self.sample_dir / name).write_bytes(backup)
        self.answer.write_text("", encoding="utf-8")
        self.assertFalse(inputs.can_skip_topk(self.sample_dir, self.answer))
        self.answer.write_text("<html></html>", encoding="utf-8")
        (self.sample_dir / "assets/v01.jpg").unlink()
        self.assertFalse(inputs.can_skip_topk(self.sample_dir, self.answer))


class TestRecordMerging(unittest.TestCase):
    def test_merge_result_records(self):
        existing = {"a": {"sample_id": "a", "status": "ok", "tokens": 5},
                    "b": {"sample_id": "b", "status": "error"}}
        results = [{"sample_id": "a", "status": "skipped"},
                   {"sample_id": "b", "status": "ok", "tokens": 7},
                   {"sample_id": "c", "status": "error"}]
        merged = inputs.merge_result_records(existing, results)
        self.assertEqual(merged["a"]["tokens"], 5)  # skipped keeps original ok row
        self.assertEqual(merged["b"]["status"], "ok")
        self.assertEqual(merged["c"]["status"], "error")

    def test_load_latest_records_last_wins(self):
        path = Path(tempfile.mkdtemp()) / "r.jsonl"
        path.write_text(
            '{"sample_id": "a", "status": "error"}\n'
            'not json\n'
            '{"sample_id": "a", "status": "ok"}\n',
            encoding="utf-8",
        )
        latest = inputs.load_latest_records(path, path.with_name("missing.jsonl"))
        self.assertEqual(latest["a"]["status"], "ok")


if __name__ == "__main__":
    unittest.main()
