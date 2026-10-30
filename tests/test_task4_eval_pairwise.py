"""Offline tests for the Task 4 pairwise evaluator (protocol v3, byte-sum A/B)."""

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

from tasks.task4 import eval_common
from tasks.task4 import eval_pairwise as pw
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


class TestAbAssignment(unittest.TestCase):
    def test_deterministic(self):
        parity1 = pw.ab_assignment_parity("ident", "s1", ("a", "b"))
        parity2 = pw.ab_assignment_parity("ident", "s1", ("a", "b"))
        self.assertEqual(parity1, parity2)
        self.assertIn(parity1, (0, 1))

    def test_sensitivity_to_inputs(self):
        base = pw.ab_assignment_parity("ident", "s1", ("a", "b"))
        flipped = {
            pw.ab_assignment_parity("ident2", "s1", ("a", "b")),
            pw.ab_assignment_parity("ident", "s2", ("a", "b")),
            pw.ab_assignment_parity("ident", "s1", ("a", "c")),
        }
        self.assertTrue(len(flipped) > 1 or base in flipped)

    def test_distribution_on_formal_sized_cohort(self):
        ids = [f"s{i:04d}" for i in range(1, 1396)]
        ones = sum(
            pw.ab_assignment_parity("ident", sid, ("alpha", "beta")) for sid in ids
        )
        ratio = ones / len(ids)
        self.assertTrue(0.45 <= ratio <= 0.55, ratio)

    def test_no_digest_inputs(self):
        # byte-sum parity needs only the canonical string; nothing is hashed
        canonical = f"{pw.AB_NAMESPACE}|ident|s1|a|b"
        expected = sum(canonical.encode("utf-8")) % 2
        self.assertEqual(pw.ab_assignment_parity("ident", "s1", ("a", "b")), expected)


class TestSchedule(unittest.TestCase):
    def test_enumeration_and_ids(self):
        schedule = pw.build_schedule(["s1", "s2"], ["a", "b", "c"], "ident")
        self.assertEqual(len(schedule), 2 * 3)
        self.assertEqual([spec.ordinal for spec in schedule], list(range(6)))
        self.assertEqual(schedule[0].sample_id, "s1")
        self.assertEqual(schedule[0].pair, ("a", "b"))
        self.assertEqual(schedule[0].battle_id, "battle|s1|a|b")
        self.assertEqual(len({spec.battle_id for spec in schedule}), 6)
        for spec in schedule:
            self.assertEqual(
                {spec.a_label, spec.b_label}, set(spec.pair)
            )

    def test_identity_changes_assignment_but_not_ids(self):
        first = pw.build_schedule(["s1"], ["a", "b"], "ident-1")[0]
        second = pw.build_schedule(["s1"], ["a", "b"], "ident-2")[0]
        self.assertEqual(first.battle_id, second.battle_id)
        self.assertEqual(
            {first.a_label, first.b_label}, {second.a_label, second.b_label}
        )


class BattleFixture(unittest.TestCase):
    def setUp(self):
        eval_common._input_record_cache.clear()
        self.runs_root = Path(tempfile.mkdtemp())
        self.run_a = self.runs_root / "run-a"
        self.run_b = self.runs_root / "run-b"
        _write_sample(self.run_a, "s1")
        _write_sample(self.run_b, "s1")
        self.participants = {
            "a": eval_common.Participant("a", "run-a", self.run_a),
            "b": eval_common.Participant("b", "run-b", self.run_b),
        }

    def _spec(self, sid="s1", pair=("a", "b"), a="a", b="b"):
        return pw.BattleSpec(
            ordinal=0,
            battle_id=pw.battle_id_for(sid, pair),
            sample_id=sid,
            pair=pair,
            a_label=a,
            b_label=b,
            assignment_parity=0,
        )

    def _battle(self, client, spec=None):
        return asyncio.run(pw.evaluate_battle(
            spec or self._spec(), self.participants, client, "judge-x"
        ))


class TestEvaluateBattle(BattleFixture):
    def test_judged_a_wins(self):
        client = FakeClient(['{"winner": "A"}'])
        record = self._battle(client)
        self.assertEqual(record["status"], "judged")
        self.assertEqual(record["result"], {"winner": "A", "winner_label": "a"})
        self.assertEqual(record["points"], {"A": 1.0, "B": 0.0})
        self.assertTrue(record["api"]["called"])
        self.assertEqual(record["api"]["response_fingerprint"],
                         {"chars": len('{"winner": "A"}')})
        self.assertEqual(record["transport"]["n_slices"], 1)
        self.assertFalse(record["transport"]["truncated"])
        self.assertFalse(record["transport"]["overview_included"])
        self.assertIsNone(record["error"])
        pw.validate_battle_record(record, self._spec())
        self.assertNotIn("sha256", json.dumps(record))

    def test_judged_tie(self):
        client = FakeClient(['{"winner": "tie"}'])
        record = self._battle(client)
        self.assertEqual(record["points"], {"A": 0.5, "B": 0.5})
        self.assertIsNone(record["result"]["winner_label"])

    def test_invalid_judge_output(self):
        client = FakeClient(['{"winner": "C"}'])
        record = self._battle(client)
        self.assertEqual(record["status"], "invalid_judge_output")
        self.assertEqual(record["points"], {"A": 0.0, "B": 0.0})
        self.assertEqual(record["api"]["response_fingerprint"],
                         {"chars": len('{"winner": "C"}')})
        pw.validate_battle_record(record, self._spec())

    def test_judge_failure(self):
        client = FakeClient([_api_error(500)] * 3)
        with mock.patch("asyncio.sleep", new=mock.AsyncMock()):
            record = self._battle(client)
        self.assertEqual(record["status"], "judge_failure")
        self.assertEqual(record["points"], {"A": 0.0, "B": 0.0})
        self.assertIsNone(record["api"]["response_fingerprint"])
        pw.validate_battle_record(record, self._spec())

    def test_auto_win(self):
        (self.run_b / "s1" / "full.png").unlink()
        (self.run_b / "s1" / "render_meta.json").unlink()
        eval_common._input_record_cache.clear()
        client = FakeClient([])
        record = self._battle(client)
        self.assertEqual(record["status"], "auto_win")
        self.assertEqual(record["result"], {"winner": "A", "winner_label": "a"})
        self.assertEqual(record["points"], {"A": 1.0, "B": 0.0})
        self.assertFalse(record["api"]["called"])
        self.assertIsNone(record["transport"])
        self.assertEqual(client.calls, [])
        pw.validate_battle_record(record, self._spec())

    def test_both_invalid(self):
        for run in (self.run_a, self.run_b):
            (run / "s1" / "full.png").unlink()
            (run / "s1" / "render_meta.json").unlink()
        eval_common._input_record_cache.clear()
        record = self._battle(FakeClient([]))
        self.assertEqual(record["status"], "both_invalid")
        self.assertEqual(record["points"], {"A": 0.0, "B": 0.0})
        pw.validate_battle_record(record, self._spec())

    def test_multi_slice_messages(self):
        _write_sample(self.run_a, "tall", full_size=(3840, 8640))
        _write_sample(self.run_b, "tall", full_size=(3840, 4320))
        spec = self._spec(sid="tall")
        client = FakeClient(['{"winner": "B"}'])
        record = self._battle(client, spec)
        self.assertEqual(record["transport"]["common_height"], 4320)
        self.assertEqual(record["transport"]["slice_offsets"], [0, 1980, 2160])
        self.assertTrue(record["transport"]["overview_included"])
        parts = client.calls[0]["parts"]
        self.assertEqual(parts[0]["type"], "text")
        kinds = [part["type"] for part in parts]
        # prompt + overview pair (2 text + 2 image) + 3 slice pairs (6 text + 6 image)
        self.assertEqual(len(parts), 1 + 4 + 12)
        self.assertEqual(kinds.count("image"), 8)
        self.assertIn("Screenshot A - full-page overview (navigation only):",
                      parts[1]["text"])


class TestValidateBattleRecord(BattleFixture):
    def _valid_record(self):
        client = FakeClient(['{"winner": "A"}'])
        return self._battle(client)

    def test_valid_record_passes(self):
        pw.validate_battle_record(self._valid_record(), self._spec())

    def test_envelope_drift(self):
        record = self._valid_record()
        record["pair"] = ["b", "a"]
        with self.assertRaises(EvaluationError):
            pw.validate_battle_record(record, self._spec())

    def test_extra_key_rejected(self):
        record = self._valid_record()
        record["run_config_sha256"] = "x" * 64
        with self.assertRaises(EvaluationError):
            pw.validate_battle_record(record, self._spec())

    def test_winner_label_misaligned(self):
        record = self._valid_record()
        record["result"]["winner_label"] = "b"
        with self.assertRaises(EvaluationError):
            pw.validate_battle_record(record, self._spec())

    def test_points_mismatch(self):
        record = self._valid_record()
        record["points"] = {"A": 0.5, "B": 0.5}
        with self.assertRaises(EvaluationError):
            pw.validate_battle_record(record, self._spec())

    def test_transport_recomputed(self):
        record = self._valid_record()
        record["transport"]["n_slices"] = 5
        with self.assertRaises(EvaluationError):
            pw.validate_battle_record(record, self._spec())

    def test_fingerprint_shape(self):
        record = self._valid_record()
        record["api"]["response_fingerprint"] = {"chars": 5, "sha256": "0" * 64}
        with self.assertRaises(EvaluationError):
            pw.validate_battle_record(record, self._spec())


class TestLoadExistingBattles(BattleFixture):
    def _write_records(self, path, records):
        with path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")

    def test_roundtrip(self):
        schedule = pw.build_schedule(["s1"], ["a", "b"], "ident")
        record = self._battle(FakeClient(['{"winner": "A"}']), schedule[0])
        path = Path(tempfile.mkdtemp()) / "battles.jsonl"
        self._write_records(path, [record])
        loaded = pw.load_existing_battles(path, schedule)
        self.assertEqual(loaded, {record["battle_id"]: record})

    def test_corruption_matrix(self):
        schedule = pw.build_schedule(["s1"], ["a", "b"], "ident")
        record = self._battle(FakeClient(['{"winner": "A"}']), schedule[0])
        path = Path(tempfile.mkdtemp()) / "battles.jsonl"
        cases = {
            "truncated": json.dumps(record),
            "blank_line": json.dumps(record) + "\n\n",
            "too_many": (json.dumps(record) + "\n") * 2 + json.dumps(record) + "\n",
        }
        for name, content in cases.items():
            path.write_text(content, encoding="utf-8")
            with self.assertRaises(EvaluationError, msg=name):
                pw.load_existing_battles(path, schedule)
        unknown = dict(record, battle_id="battle|s1|a|c")
        path.write_text(json.dumps(unknown) + "\n", encoding="utf-8")
        with self.assertRaises(EvaluationError):
            pw.load_existing_battles(path, schedule)
        dup = json.dumps(record) + "\n" + json.dumps(record) + "\n"
        path.write_text(dup, encoding="utf-8")
        with self.assertRaises(EvaluationError):
            pw.load_existing_battles(path, schedule)
        wrong_ordinal = dict(record, ordinal=1)
        path.write_text(json.dumps(wrong_ordinal) + "\n", encoding="utf-8")
        with self.assertRaises(EvaluationError):
            pw.load_existing_battles(path, schedule)


class TestBuildSummary(unittest.TestCase):
    def _fabricate(self, spec, winner, status="judged"):
        from tasks.task4.scoring import expected_points

        winner_label = spec.a_label if winner == "A" else spec.b_label if winner == "B" else None
        return spec.battle_id, {
            "pair": list(spec.pair),
            "assignment": {"A": spec.a_label, "B": spec.b_label},
            "status": status,
            "result": {"winner": winner, "winner_label": winner_label},
            "points": expected_points(winner),
        }

    def test_two_participants_hand_computed(self):
        schedule = pw.build_schedule(["s1", "s2"], ["a", "b"], "ident")
        spec1, spec2 = schedule
        a_side1 = "A" if spec1.a_label == "a" else "B"
        records = dict([
            self._fabricate(spec1, a_side1),       # "a" wins battle 1 outright
            self._fabricate(spec2, "tie"),         # battle 2 is a tie
        ])
        summary = pw.build_summary(records, schedule, ["a", "b"], 2, False)
        self.assertEqual(summary["schema_version"], "task4-pairwise-summary-v3")
        self.assertEqual(summary["denominator"], 2)
        self.assertEqual(summary["expected_battles"], 2)
        pair = summary["pairs"][0]
        self.assertEqual(pair["participants"], ["a", "b"])
        self.assertEqual(pair["points"], {"a": 1.5, "b": 0.5})
        self.assertEqual(pair["win_rate"], {"a": 0.75, "b": 0.25})
        self.assertEqual(summary["participants"]["a"]["overall_win_rate"], 0.75)
        self.assertEqual(summary["participants"]["b"]["overall_win_rate"], 0.25)
        self.assertNotIn("sha256", json.dumps(summary))

    def test_formal_all_uses_formal_denominator(self):
        ids = [f"s{i}" for i in range(3)]
        schedule = pw.build_schedule(ids, ["a", "b"], "ident")
        records = dict(
            self._fabricate(spec, "tie") for spec in schedule
        )
        summary = pw.build_summary(records, schedule, ["a", "b"], 3, True)
        self.assertEqual(summary["denominator"], 1395)
        self.assertEqual(summary["selected_count"], 3)
        self.assertAlmostEqual(
            summary["participants"]["a"]["overall_win_rate"], 1.5 / 1395
        )

    def test_three_participants_overall_denominator(self):
        schedule = pw.build_schedule(["s1"], ["a", "b", "c"], "ident")
        records = dict(
            self._fabricate(spec, "A") for spec in schedule
        )
        summary = pw.build_summary(records, schedule, ["a", "b", "c"], 1, False)
        self.assertEqual(
            summary["overall_win_rate_formula"],
            "participant total points / (1 * (participant_count - 1))",
        )
        for label in ("a", "b", "c"):
            self.assertEqual(summary["participants"][label]["denominator"], 2)

    def test_incomplete_rejected(self):
        schedule = pw.build_schedule(["s1", "s2"], ["a", "b"], "ident")
        records = dict([self._fabricate(schedule[0], "tie")])
        with self.assertRaises(EvaluationError):
            pw.build_summary(records, schedule, ["a", "b"], 2, False)

    def test_formal_artifact_from_summary(self):
        schedule = pw.build_schedule(["s1"], ["a", "b"], "ident")
        records = dict([self._fabricate(schedule[0], "A")])
        summary = pw.build_summary(records, schedule, ["a", "b"], 1, False)
        artifact = pw.build_formal_metric_artifact(summary, Path("/tmp/summary.json"))
        self.assertEqual(artifact["task"], "task4_pairwise")
        rates = artifact["metrics"]["overall_pairwise_win_rate"]
        self.assertEqual(set(rates), {"a", "b"})
        self.assertEqual(
            artifact["provenance"]["source_summary_schema_version"],
            "task4-pairwise-summary-v3",
        )


class TestDefaultOutDir(unittest.TestCase):
    def test_naming(self):
        participants = [
            eval_common.Participant("beta", "rb", Path("/x/rb")),
            eval_common.Participant("alpha", "ra", Path("/x/ra")),
        ]
        out = pw._default_out_dir(participants, "gpt-5.4-2026-03-05")
        self.assertEqual(
            out.name, "task4-pairwise-visual-v3-gpt-5.4-2026-03-05-beta+alpha"
        )
        self.assertNotIn("sha256", str(out))


if __name__ == "__main__":
    unittest.main()
