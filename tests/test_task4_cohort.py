"""Offline tests for the Task 4 formal-cohort guards and explicit selection."""

import json
import tempfile
import unittest
from pathlib import Path

from tasks.task4 import cohort


def _ids(n=1395):
    return [f"s{i:04d}" for i in range(1, n + 1)]


def _write_pack(root: Path, ids, *, freeze_overrides=None, cohort_overrides=None,
                queries_ids=None):
    root.mkdir(parents=True, exist_ok=True)
    cohort_doc = {
        "schema_version": "design-bench-task14-dedup-cohort-v1",
        "cohort": "task14-dedup-v1",
        "included_ids": ids,
        "excluded_ids": [],
    }
    cohort_doc.update(cohort_overrides or {})
    (root / "cohort.json").write_text(json.dumps(cohort_doc), encoding="utf-8")
    freeze_doc = {
        "freeze_version": "task4-topk-v5-dedup-v1",
        "status": "active_default",
        "cohort": {
            "name": "task14-dedup-v1",
            "schema_version": "design-bench-task14-dedup-cohort-v1",
            "formal_items": 1395,
        },
        "protocols": {
            "input": "task4-topk-contact-sheets-v5",
            "output_audit": "task4-topk-output-audit-v3",
            "render": "task4-render-1920x2160-dpr2-v2",
        },
    }
    _deep_update(freeze_doc, freeze_overrides or {})
    (root / "freeze.json").write_text(json.dumps(freeze_doc), encoding="utf-8")
    with (root / "queries.jsonl").open("w", encoding="utf-8") as handle:
        for sample_id in (queries_ids if queries_ids is not None else ids):
            handle.write(json.dumps({"sample_id": sample_id}) + "\n")


def _deep_update(target, patch):
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value


class TestLoadFormalCohort(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        _write_pack(self.root, _ids())

    def test_happy_path(self):
        ids, freeze, cohort_doc = cohort.load_formal_cohort(self.root)
        self.assertEqual(ids, _ids())
        self.assertEqual(freeze["freeze_version"], "task4-topk-v5-dedup-v1")
        self.assertEqual(cohort_doc["cohort"], "task14-dedup-v1")

    def _expect_drift(self, *, freeze_overrides=None, cohort_overrides=None,
                      queries_ids=None):
        root = Path(tempfile.mkdtemp())
        _write_pack(root, _ids(),
                    freeze_overrides=freeze_overrides,
                    cohort_overrides=cohort_overrides,
                    queries_ids=queries_ids)
        with self.assertRaises(cohort.EvaluationError):
            cohort.load_formal_cohort(root)

    def test_status_drift(self):
        self._expect_drift(freeze_overrides={"status": "archived"})

    def test_freeze_version_drift(self):
        self._expect_drift(freeze_overrides={"freeze_version": "other"})

    def test_cohort_schema_drift(self):
        self._expect_drift(cohort_overrides={"schema_version": "other"})

    def test_cohort_name_drift(self):
        self._expect_drift(cohort_overrides={"cohort": "other"})

    def test_cohort_count_drift(self):
        self._expect_drift(cohort_overrides={"included_ids": _ids(1394)})

    def test_cohort_duplicate_ids(self):
        ids = _ids()
        ids[-1] = ids[-2]
        self._expect_drift(cohort_overrides={"included_ids": ids})

    def test_query_ids_drift(self):
        ids = _ids()
        ids[0] = "other"
        self._expect_drift(queries_ids=ids)

    def test_input_protocol_drift(self):
        self._expect_drift(freeze_overrides={"protocols": {"input": "x"}})

    def test_output_audit_protocol_drift(self):
        self._expect_drift(freeze_overrides={"protocols": {"output_audit": "x"}})

    def test_render_protocol_drift(self):
        self._expect_drift(freeze_overrides={"protocols": {"render": "x"}})


class TestLoadQueryRecords(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        _write_pack(self.root, _ids())
        self.default_queries = self.root / "queries.jsonl"
        self.cohort_path = self.root / "cohort.json"

    def test_default_validates_and_carries_full_id_list(self):
        records, meta = cohort.load_query_records(
            self.default_queries,
            default_queries_path=self.default_queries,
            cohort_path=self.cohort_path,
        )
        self.assertEqual(len(records), 1395)
        self.assertEqual(meta["schema_version"], "task4-run-cohort-v1")
        self.assertEqual(meta["cohort"], "task14-dedup-v1")
        self.assertEqual(meta["included_ids"], _ids())
        self.assertEqual(meta["sample_count"], 1395)
        self.assertNotIn("included_ids_sha256", json.dumps(meta))

    def test_default_drift_fails_closed(self):
        with self.default_queries.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"sample_id": "extra"}) + "\n")
        with self.assertRaises(cohort.EvaluationError):
            cohort.load_query_records(
                self.default_queries,
                default_queries_path=self.default_queries,
                cohort_path=self.cohort_path,
            )

    def test_custom_queries_skip_cohort_check(self):
        custom = self.root / "custom.jsonl"
        custom.write_text(
            json.dumps({"sample_id": "x"}) + "\n"
            + json.dumps({"sample_id": "y"}) + "\n",
            encoding="utf-8",
        )
        records, meta = cohort.load_query_records(
            custom, default_queries_path=self.default_queries
        )
        self.assertEqual(len(records), 2)
        self.assertEqual(meta["cohort"], "custom:custom.jsonl")
        self.assertEqual(meta["sample_count"], 2)

    def test_duplicate_ids_rejected(self):
        custom = self.root / "dup.jsonl"
        custom.write_text(
            json.dumps({"sample_id": "x"}) + "\n"
            + json.dumps({"sample_id": "x"}) + "\n",
            encoding="utf-8",
        )
        with self.assertRaises(cohort.EvaluationError):
            cohort.load_query_records(
                custom, default_queries_path=self.default_queries
            )


class TestValidateRunCohort(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        _write_pack(self.root, _ids())
        _, self.freeze, self.cohort_doc = cohort.load_formal_cohort(self.root)
        self.run_dir = Path(tempfile.mkdtemp())

    def _write_sidecar(self, **overrides):
        sidecar = {
            "schema_version": "task4-run-cohort-v1",
            "cohort": "task14-dedup-v1",
            "cohort_schema_version": "design-bench-task14-dedup-cohort-v1",
            "included_ids": _ids(),
            "sample_count": 1395,
        }
        sidecar.update(overrides)
        (self.run_dir / "_cohort.json").write_text(
            json.dumps(sidecar), encoding="utf-8"
        )

    def test_matching_sidecar_passes(self):
        self._write_sidecar()
        cohort.validate_run_cohort(self.run_dir, self.freeze, self.cohort_doc)

    def test_drifted_ids_rejected(self):
        ids = _ids()
        ids[0] = "tampered"
        self._write_sidecar(included_ids=ids)
        with self.assertRaises(cohort.EvaluationError):
            cohort.validate_run_cohort(self.run_dir, self.freeze, self.cohort_doc)

    def test_drifted_count_rejected(self):
        self._write_sidecar(sample_count=1394)
        with self.assertRaises(cohort.EvaluationError):
            cohort.validate_run_cohort(self.run_dir, self.freeze, self.cohort_doc)

    def test_missing_key_rejected(self):
        sidecar = {"schema_version": "task4-run-cohort-v1"}
        (self.run_dir / "_cohort.json").write_text(
            json.dumps(sidecar), encoding="utf-8"
        )
        with self.assertRaises(cohort.EvaluationError):
            cohort.validate_run_cohort(self.run_dir, self.freeze, self.cohort_doc)


class TestSelectIds(unittest.TestCase):
    def test_mutex(self):
        with self.assertRaises(cohort.EvaluationError):
            cohort.select_ids(["a", "b"], True, ["a"])
        with self.assertRaises(cohort.EvaluationError):
            cohort.select_ids(["a", "b"], False, [])

    def test_all(self):
        self.assertEqual(cohort.select_ids(["b", "a"], True, []), ["b", "a"])

    def test_unknown_and_duplicate(self):
        with self.assertRaises(cohort.EvaluationError):
            cohort.select_ids(["a"], False, ["z"])
        with self.assertRaises(cohort.EvaluationError):
            cohort.select_ids(["a", "b"], False, ["a", "a"])

    def test_formal_order_preserved(self):
        self.assertEqual(
            cohort.select_ids(["a", "b", "c"], False, ["c", "a"]), ["a", "c"]
        )


if __name__ == "__main__":
    unittest.main()
