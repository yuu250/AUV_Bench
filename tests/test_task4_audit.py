"""Offline tests for the Task 4 output-policy audit."""

import json
import tempfile
import unittest
from pathlib import Path

from tasks.task4 import audit
from tasks.task4.prompts import INPUT_PROTOCOL, OUTPUT_AUDIT_PROTOCOL

ALLOWED = ["assets/v01.jpg", "assets/v02.png"]


def check(html, allowed=ALLOWED):
    return audit.validate_topk_output(html, allowed)


class TestValidateTopkOutput(unittest.TestCase):
    def test_allowed_asset_used(self):
        used, violations = check('<img src="assets/v01.jpg"><img src="assets/v02.png">')
        self.assertEqual(used, ["assets/v01.jpg", "assets/v02.png"])
        self.assertEqual(violations, [])

    def test_query_suffix_still_counts(self):
        used, violations = check('<img src="assets/v01.jpg?v=2">')
        self.assertEqual(used, ["assets/v01.jpg"])
        self.assertEqual(violations, [])

    def test_external_resources(self):
        _, violations = check('<img src="https://cdn.example.com/x.jpg">')
        self.assertTrue(any(v.startswith("external_resource:") for v in violations))
        _, violations = check('<img src="//cdn.example.com/x.jpg">')
        self.assertTrue(any(v.startswith("external_resource:") for v in violations))

    def test_data_url_only_flagged_in_media_attributes(self):
        _, violations = check('<img src="data:image/png;base64,AAA">')
        self.assertTrue(any(v.startswith("inline_data_media:") for v in violations))
        # Frozen semantics: data: inside CSS url() is silently allowed.
        _, violations = check(
            '<div style="background:url(data:image/png;base64,AAA)"></div>'
            '<img src="assets/v01.jpg">'
        )
        self.assertEqual(violations, [])
        _, violations = check('<source srcset="data:image/png;base64,AAA 1x">')
        self.assertTrue(any(v.startswith("inline_data_media:") for v in violations))

    def test_anchor_ignored(self):
        used, violations = check('<a href="#top">x</a><img src="assets/v01.jpg">')
        self.assertEqual(violations, [])

    def test_unsafe_and_outside_paths(self):
        _, violations = check('<img src="assets/../secret.jpg">')
        self.assertTrue(any(v.startswith("unsafe_asset_path:") for v in violations))
        _, violations = check('<img src="assets/v99.jpg">')
        self.assertTrue(any(v.startswith("outside_topk:") for v in violations))
        _, violations = check('<img src="plain.jpg">')
        self.assertTrue(any(v.startswith("non_asset_resource:") for v in violations))

    def test_srcset_candidates(self):
        # Frozen semantics: the srcset parser marks both candidates used, but
        # the quoted-literal scan re-flags the full attribute value (with
        # descriptors) as outside_topk.  Verified against the reference scorer.
        used, violations = check(
            '<img srcset="assets/v01.jpg 1x, assets/v02.png 2x">'
        )
        self.assertEqual(used, ["assets/v01.jpg", "assets/v02.png"])
        self.assertEqual(
            violations, ["outside_topk:assets/v01.jpg 1x, assets/v02.png 2x"]
        )

    def test_link_and_import_checked(self):
        _, violations = check('<link href="https://cdn.example.com/a.css">')
        self.assertTrue(any(v.startswith("external_resource:") for v in violations))
        _, violations = check(
            '<style>@import "https://cdn.example.com/a.css";</style>'
        )
        self.assertTrue(any(v.startswith("external_resource:") for v in violations))

    def test_css_url_external(self):
        _, violations = check(
            '<style>.a{background: url("https://cdn.example.com/x.png")}</style>'
        )
        self.assertTrue(any(v.startswith("external_resource:") for v in violations))

    def test_svg_image_href(self):
        used, violations = check('<svg><image xlink:href="assets/v02.png"/></svg>')
        self.assertEqual(used, ["assets/v02.png"])
        self.assertEqual(violations, [])

    def test_js_asset_literal_counts_as_used(self):
        used, violations = check('<script>var a = ["assets/v01.jpg"];</script>')
        self.assertEqual(used, ["assets/v01.jpg"])
        self.assertEqual(violations, [])

    def test_js_external_image_literal(self):
        _, violations = check(
            '<script>var u = "https://cdn.example.com/x.png";</script>'
        )
        self.assertTrue(any(v.startswith("external_resource:") for v in violations))

    def test_source_tag_markers(self):
        _, violations = check("<p>[h1] Hello</p><img src='assets/v01.jpg'>")
        self.assertTrue(
            any(v.startswith("rendered_source_tag_markers:") for v in violations)
        )

    def test_no_selected_asset_used(self):
        _, violations = check("<p>no images at all</p>")
        self.assertIn("no_selected_asset_used", violations)
        # Empty allowed list: the rule does not fire.
        _, violations = check("<p>no images at all</p>", allowed=[])
        self.assertEqual(violations, [])

    def test_used_sorted_and_violations_deduped(self):
        _, violations = check(
            '<img src="https://a.com/1.png"><img src="https://a.com/1.png">'
        )
        self.assertEqual(violations, sorted(set(violations)))


class TestInspectGeneration(unittest.TestCase):
    def _write_sample(self, sample_dir: Path, *, manifest=None, audit_doc=None,
                      answer=True):
        sample_dir.mkdir(parents=True, exist_ok=True)
        if answer:
            (sample_dir / "answer.html").write_text("<html></html>", encoding="utf-8")
        (sample_dir / "input_manifest.json").write_text(json.dumps(manifest if manifest is not None else {
            "protocol_version": INPUT_PROTOCOL,
            "asset_mapping": [{"alias_src": "assets/v01.jpg"}],
            "allowed_asset_paths": ["assets/v01.jpg"],
        }), encoding="utf-8")
        (sample_dir / "output_audit.json").write_text(json.dumps(audit_doc if audit_doc is not None else {
            "protocol_version": OUTPUT_AUDIT_PROTOCOL,
            "violations": [],
        }), encoding="utf-8")
        (sample_dir / "assets").mkdir(exist_ok=True)
        (sample_dir / "assets/v01.jpg").write_bytes(b"x")

    def test_happy_path(self):
        sample_dir = Path(tempfile.mkdtemp())
        self._write_sample(sample_dir)
        fingerprint, error = audit.inspect_generation(sample_dir)
        self.assertIsNone(error)
        self.assertTrue(fingerprint["answer_exists"])
        self.assertEqual(fingerprint["answer_bytes"], len("<html></html>"))
        self.assertNotIn("sha256", json.dumps(fingerprint))

    def test_missing_files(self):
        sample_dir = Path(tempfile.mkdtemp())
        fingerprint, error = audit.inspect_generation(sample_dir)
        self.assertEqual(error, "missing answer.html")
        self.assertFalse(fingerprint["answer_exists"])
        sample_dir = Path(tempfile.mkdtemp())
        (sample_dir / "answer.html").write_text("x", encoding="utf-8")
        _, error = audit.inspect_generation(sample_dir)
        self.assertEqual(error, "missing input_manifest.json")

    def test_protocol_mismatch(self):
        sample_dir = Path(tempfile.mkdtemp())
        self._write_sample(sample_dir, manifest={"protocol_version": "old"})
        _, error = audit.inspect_generation(sample_dir)
        self.assertEqual(error, "input manifest protocol mismatch")
        sample_dir = Path(tempfile.mkdtemp())
        self._write_sample(sample_dir, audit_doc={"protocol_version": "old",
                                                  "violations": []})
        _, error = audit.inspect_generation(sample_dir)
        self.assertEqual(error, "output audit protocol mismatch")

    def test_violations_rejected(self):
        sample_dir = Path(tempfile.mkdtemp())
        self._write_sample(sample_dir, audit_doc={
            "protocol_version": OUTPUT_AUDIT_PROTOCOL,
            "violations": ["external_resource:x"],
        })
        _, error = audit.inspect_generation(sample_dir)
        self.assertEqual(error, "output audit contains violations")

    def test_bad_asset_mapping(self):
        sample_dir = Path(tempfile.mkdtemp())
        self._write_sample(sample_dir, manifest={
            "protocol_version": INPUT_PROTOCOL,
            "asset_mapping": [{"alias_src": "assets/../x"}],
            "allowed_asset_paths": ["assets/v01.jpg"],
        })
        _, error = audit.inspect_generation(sample_dir)
        self.assertEqual(error, "input manifest contains an invalid asset mapping")
        sample_dir = Path(tempfile.mkdtemp())
        self._write_sample(sample_dir, manifest={
            "protocol_version": INPUT_PROTOCOL,
            "asset_mapping": [{"alias_src": "assets/v99.jpg"}],
            "allowed_asset_paths": ["assets/v01.jpg"],
        })
        _, error = audit.inspect_generation(sample_dir)
        self.assertEqual(error, "input manifest contains an invalid asset mapping")

    def test_missing_alias_file(self):
        sample_dir = Path(tempfile.mkdtemp())
        self._write_sample(sample_dir)
        (sample_dir / "assets/v01.jpg").unlink()
        _, error = audit.inspect_generation(sample_dir)
        self.assertEqual(error, "missing allowed asset: assets/v01.jpg")


if __name__ == "__main__":
    unittest.main()
