#!/usr/bin/env python3
"""Evaluate Task 4 runs with deterministic pairwise visual judgments.

Run from the repository root:

    python -m tasks.task4.eval_pairwise --config configs/task4_pairwise.yaml \
        --participant alpha=qwen3-vl-plus-topk-v5 --participant beta=gpt-topk-v5 --all

Battles pair the same query across participants; the A/B roles are assigned
deterministically per battle by byte-sum parity (protocol v3 — no digests are
computed anywhere, and assignments are NOT battle-level comparable with the
reference implementation's SHA-256 assignments).  Scores: win 1 / tie 0.5 /
loss 0; a participant with an invalid input auto-loses without a judge call.
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import PIL

from src.api.client import make_client
from src.utils.config import load_config, require_mapping, resolve_api_key
from tasks.task4.cohort import (
    DEFAULT_DATA_ROOT,
    FORMAL_DENOMINATOR,
    EvaluationError,
    load_formal_cohort,
    select_ids,
    validate_run_cohort,
)
from tasks.task4.eval_common import (
    INPUT_RECORD_SCHEMA,
    RETRY_CONTRACT,
    Participant,
    _api_audit_valid,
    _error,
    _safe_component,
    _sanitize_api_audit,
    append_record,
    build_formal_artifact,
    call_judge,
    canonical_json,
    drop_failed_records,
    ensure_run_meta,
    input_full_bytes,
    input_height_1x,
    participant_input,
    publish_summary,
    write_json_atomic,
)
from tasks.task4.prompts import (
    PAIRWISE_PROMPT,
    PAIRWISE_PROMPT_VERSION,
    PAIRWISE_RESPONSE_FORMAT,
    PAIRWISE_RESPONSE_SCHEMA_VERSION,
)
from tasks.task4.scoring import expected_points, strict_winner
from tasks.task4.views import (
    PREPROCESS_CONTRACT,
    SLICE_H,
    encode_view_pair,
    fixed_canvas_slice,
    load_1x_image,
    overview_image,
    slice_offsets,
)

EVAL_VERSION = "task4-pairwise-visual-v3"
RUN_META_SCHEMA = "task4-pairwise-run-v3"
BATTLE_SCHEMA = "task4-pairwise-battle-v3"
SUMMARY_SCHEMA = "task4-pairwise-summary-v3"
AB_NAMESPACE = "task4-pairwise-ab-assignment-v3"


@dataclass(frozen=True)
class BattleSpec:
    ordinal: int
    battle_id: str
    sample_id: str
    pair: tuple[str, str]
    a_label: str
    b_label: str
    assignment_parity: int


def parse_participants(values: list[str], runs_root: Path) -> list[Participant]:
    if len(values) < 2:
        raise EvaluationError("at least two --participant LABEL=RUN_NAME values are required")
    parsed: list[tuple[str, str]] = []
    for value in values:
        if value.count("=") != 1:
            raise EvaluationError("each --participant must have the form LABEL=RUN_NAME")
        label, run_name = value.split("=", 1)
        _safe_component(label, "participant label")
        _safe_component(run_name, "participant run name")
        parsed.append((label, run_name))
    labels = [label for label, _ in parsed]
    runs = [run_name for _, run_name in parsed]
    if len(labels) != len(set(labels)):
        raise EvaluationError("duplicate participant labels are not allowed")
    if len(runs) != len(set(runs)):
        raise EvaluationError("duplicate participant runs are not allowed")

    root = runs_root.resolve()
    participants: list[Participant] = []
    for label, run_name in sorted(parsed):
        run_dir = (runs_root / run_name).resolve()
        try:
            relative = run_dir.relative_to(root)
        except ValueError as exc:
            raise EvaluationError(f"participant run escapes RUNS_ROOT: {run_name}") from exc
        if len(relative.parts) != 1 or not run_dir.is_dir():
            raise EvaluationError(f"Task 4 run directory does not exist: {run_name}")
        participants.append(Participant(label, run_name, run_dir))
    if len({item.run_dir for item in participants}) != len(participants):
        raise EvaluationError("duplicate resolved participant runs are not allowed")
    return participants


def ab_assignment_parity(
    evaluation_identity: str, sample_id: str, pair: tuple[str, str]
) -> int:
    """Deterministic A/B coin flip: byte-sum parity of the canonical string.

    Worker-independent and stable across processes; assigns each participant
    to the A side in ~50% of battles so judge position bias cancels out.
    """

    canonical = f"{AB_NAMESPACE}|{evaluation_identity}|{sample_id}|{pair[0]}|{pair[1]}"
    return sum(canonical.encode("utf-8")) % 2


def battle_id_for(sample_id: str, pair: tuple[str, str]) -> str:
    return f"battle|{sample_id}|{pair[0]}|{pair[1]}"


def build_schedule(
    selected_ids: list[str], labels: list[str], evaluation_identity: str
) -> list[BattleSpec]:
    schedule: list[BattleSpec] = []
    for sample_id in selected_ids:
        for pair in itertools.combinations(labels, 2):
            parity = ab_assignment_parity(evaluation_identity, sample_id, pair)
            a_label, b_label = pair if parity == 0 else (pair[1], pair[0])
            schedule.append(BattleSpec(
                ordinal=len(schedule),
                battle_id=battle_id_for(sample_id, pair),
                sample_id=sample_id,
                pair=pair,
                a_label=a_label,
                b_label=b_label,
                assignment_parity=parity,
            ))
    return schedule


def _image_part(payload: tuple[bytes, str]) -> dict:
    data, media_type = payload
    return {"type": "image", "data": data, "media_type": media_type}


def _messages(a_image, b_image, offsets: list[int]) -> list[dict]:
    """Interleaved overview + slices; offsets come from the symmetric geometry."""

    n_slices = len(offsets)
    common_height = max(a_image.height, b_image.height)
    parts: list[dict] = [{"type": "text", "text": PAIRWISE_PROMPT}]
    if n_slices > 1:
        overview_a, overview_b = (
            overview_image(a_image, common_height),
            overview_image(b_image, common_height),
        )
        payload_a, payload_b = encode_view_pair(overview_a, overview_b)
        parts.append({"type": "text", "text": "Screenshot A - full-page overview (navigation only):"})
        parts.append(_image_part(payload_a))
        parts.append({"type": "text", "text": "Screenshot B - full-page overview (navigation only):"})
        parts.append(_image_part(payload_b))
    for index, top in enumerate(offsets):
        slice_a, slice_b = (
            fixed_canvas_slice(a_image, top),
            fixed_canvas_slice(b_image, top),
        )
        payload_a, payload_b = encode_view_pair(slice_a, slice_b)
        if n_slices > 1:
            label_a = f"Screenshot A - slice {index + 1}/{n_slices}:"
            label_b = f"Screenshot B - slice {index + 1}/{n_slices}:"
        else:
            label_a, label_b = "Screenshot A:", "Screenshot B:"
        parts.append({"type": "text", "text": label_a})
        parts.append(_image_part(payload_a))
        parts.append({"type": "text", "text": label_b})
        parts.append(_image_part(payload_b))
    return parts


def _base_record(spec: BattleSpec, inputs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": BATTLE_SCHEMA,
        "eval_version": EVAL_VERSION,
        "battle_id": spec.battle_id,
        "ordinal": spec.ordinal,
        "sample_id": spec.sample_id,
        "pair": list(spec.pair),
        "assignment": {"A": spec.a_label, "B": spec.b_label},
        "assignment_parity": spec.assignment_parity,
        "inputs": inputs,
    }


def _outcome(
    base: dict[str, Any],
    transport: dict[str, Any] | None,
    status: str,
    winner: str | None,
    points_a: float,
    points_b: float,
    called: bool,
    response_audit: dict[str, Any] | None,
    response_fingerprint: dict[str, Any] | None,
    error: dict[str, str] | None,
) -> dict[str, Any]:
    winner_label = base["assignment"].get(winner) if winner in {"A", "B"} else None
    return {
        **base,
        "transport": transport,
        "status": status,
        "result": {"winner": winner, "winner_label": winner_label},
        "points": {"A": points_a, "B": points_b},
        "api": {
            "called": called,
            "response_audit": response_audit,
            "response_fingerprint": response_fingerprint,
        },
        "error": error,
    }


async def evaluate_battle(
    spec: BattleSpec,
    participants_by_label: dict[str, Participant],
    client,
    judge_model: str,
) -> dict[str, Any]:
    a_record = participant_input(participants_by_label[spec.a_label], spec.sample_id)
    b_record = participant_input(participants_by_label[spec.b_label], spec.sample_id)

    inputs = {
        "A": {
            "label": spec.a_label,
            "valid": a_record["valid"],
            "height_1x": input_height_1x(a_record),
            "full_bytes": input_full_bytes(a_record),
        },
        "B": {
            "label": spec.b_label,
            "valid": b_record["valid"],
            "height_1x": input_height_1x(b_record),
            "full_bytes": input_full_bytes(b_record),
        },
    }
    base = _base_record(spec, inputs)
    a_valid = inputs["A"]["valid"]
    b_valid = inputs["B"]["valid"]
    if a_valid and not b_valid:
        return _outcome(base, None, "auto_win", "A", 1.0, 0.0, False, None, None, None)
    if b_valid and not a_valid:
        return _outcome(base, None, "auto_win", "B", 0.0, 1.0, False, None, None, None)
    if not a_valid and not b_valid:
        return _outcome(base, None, "both_invalid", None, 0.0, 0.0, False, None, None, None)

    a_image = load_1x_image(participants_by_label[spec.a_label].run_dir, spec.sample_id)
    b_image = load_1x_image(participants_by_label[spec.b_label].run_dir, spec.sample_id)
    if a_image.height != inputs["A"]["height_1x"] or b_image.height != inputs["B"]["height_1x"]:
        raise EvaluationError(f"input height drift on reload: {spec.battle_id}")
    common_height = max(a_image.height, b_image.height)
    offsets = slice_offsets(common_height)
    transport = {
        "common_height": common_height,
        "n_slices": len(offsets),
        "slice_offsets": list(offsets),
        "truncated": offsets[-1] + SLICE_H < common_height,
        "overview_included": len(offsets) > 1,
    }

    response_audit: dict[str, Any] = {"attempt": 1}
    try:
        reply = await call_judge(
            client,
            judge_model,
            _messages(a_image, b_image, offsets),
            response_format=PAIRWISE_RESPONSE_FORMAT,
            response_audit=response_audit,
        )
    except Exception as exc:
        return _outcome(
            base,
            transport,
            "judge_failure",
            None,
            0.0,
            0.0,
            True,
            _sanitize_api_audit(response_audit),
            None,
            _error("judge_failure", f"judge call failed ({type(exc).__name__})"),
        )

    sanitized_audit = _sanitize_api_audit(response_audit)
    response_fingerprint = {"chars": len(reply)}
    try:
        winner = strict_winner(reply)
    except Exception:
        return _outcome(
            base,
            transport,
            "invalid_judge_output",
            None,
            0.0,
            0.0,
            True,
            sanitized_audit,
            response_fingerprint,
            _error("invalid_judge_output", "judge response failed the strict winner schema"),
        )
    points = expected_points(winner)
    return _outcome(
        base,
        transport,
        "judged",
        winner,
        points["A"],
        points["B"],
        True,
        sanitized_audit,
        response_fingerprint,
        None,
    )


def _transport_matches(transport: object, height_a: int, height_b: int) -> bool:
    if not isinstance(transport, dict):
        return False
    if set(transport) != {
        "common_height", "n_slices", "slice_offsets", "truncated", "overview_included"
    }:
        return False
    common_height = max(height_a, height_b)
    offsets = slice_offsets(common_height)
    expected = {
        "common_height": common_height,
        "n_slices": len(offsets),
        "slice_offsets": offsets,
        "truncated": offsets[-1] + SLICE_H < common_height,
        "overview_included": len(offsets) > 1,
    }
    return transport == expected


def validate_battle_record(record: dict[str, Any], spec: BattleSpec) -> None:
    expected_keys = {
        "schema_version", "eval_version", "battle_id", "ordinal", "sample_id",
        "pair", "assignment", "assignment_parity", "inputs", "transport",
        "status", "result", "points", "api", "error",
    }
    if set(record) != expected_keys:
        raise EvaluationError(f"battle record shape mismatch: {spec.battle_id}")
    expected_envelope = {
        "schema_version": BATTLE_SCHEMA,
        "eval_version": EVAL_VERSION,
        "battle_id": spec.battle_id,
        "ordinal": spec.ordinal,
        "sample_id": spec.sample_id,
        "pair": list(spec.pair),
        "assignment": {"A": spec.a_label, "B": spec.b_label},
        "assignment_parity": spec.assignment_parity,
    }
    if any(record.get(key) != value for key, value in expected_envelope.items()):
        raise EvaluationError(f"battle schedule/config drift: {spec.battle_id}")
    inputs = record.get("inputs")
    if (
        not isinstance(inputs, dict)
        or set(inputs) != {"A", "B"}
        or any(
            not isinstance(inputs[side], dict)
            or set(inputs[side]) != {"label", "valid", "height_1x", "full_bytes"}
            or not isinstance(inputs[side].get("label"), str)
            or type(inputs[side].get("valid")) is not bool
            for side in ("A", "B")
        )
    ):
        raise EvaluationError(f"invalid battle inputs: {spec.battle_id}")
    if inputs["A"]["label"] != spec.a_label or inputs["B"]["label"] != spec.b_label:
        raise EvaluationError(f"battle input labels drift: {spec.battle_id}")
    for side in ("A", "B"):
        height = inputs[side].get("height_1x")
        full_bytes = inputs[side].get("full_bytes")
        if inputs[side]["valid"]:
            if type(height) is not int or height <= 0:
                raise EvaluationError(f"valid input lacks 1x height: {spec.battle_id}")
            if type(full_bytes) is not int or full_bytes <= 0:
                raise EvaluationError(f"valid input lacks full bytes: {spec.battle_id}")
        elif height is not None or full_bytes is not None:
            raise EvaluationError(f"invalid input carries render details: {spec.battle_id}")

    status = record.get("status")
    if status not in {
        "judged", "auto_win", "both_invalid", "judge_failure", "invalid_judge_output"
    }:
        raise EvaluationError(f"invalid terminal battle status: {spec.battle_id}")
    result = record.get("result")
    points = record.get("points")
    api = record.get("api")
    error = record.get("error")
    transport = record.get("transport")
    if not isinstance(result, dict) or set(result) != {"winner", "winner_label"}:
        raise EvaluationError(f"invalid battle result: {spec.battle_id}")
    if not isinstance(points, dict) or set(points) != {"A", "B"}:
        raise EvaluationError(f"invalid battle points: {spec.battle_id}")
    if any(type(points[side]) is not float for side in ("A", "B")):
        raise EvaluationError(f"battle points must be floats: {spec.battle_id}")
    if not isinstance(api, dict) or set(api) != {
        "called", "response_audit", "response_fingerprint"
    }:
        raise EvaluationError(f"invalid battle API audit: {spec.battle_id}")
    if type(api["called"]) is not bool:
        raise EvaluationError(f"invalid battle API called flag: {spec.battle_id}")
    response_fingerprint = api["response_fingerprint"]
    fingerprint_valid = (
        isinstance(response_fingerprint, dict)
        and set(response_fingerprint) == {"chars"}
        and type(response_fingerprint["chars"]) is int
        and response_fingerprint["chars"] >= 0
    )
    if error is not None and (
        not isinstance(error, dict)
        or set(error) != {"category", "message"}
        or not all(isinstance(error[key], str) for key in error)
    ):
        raise EvaluationError(f"invalid sanitized battle error: {spec.battle_id}")

    winner = result["winner"]
    expected_winner_label = (
        spec.a_label if winner == "A" else spec.b_label if winner == "B" else None
    )
    if result["winner_label"] != expected_winner_label or points != expected_points(winner):
        raise EvaluationError(f"inconsistent battle winner/points: {spec.battle_id}")
    a_valid = inputs["A"]["valid"]
    b_valid = inputs["B"]["valid"]
    audit = api["response_audit"]
    transport_valid = (
        a_valid and b_valid
        and _transport_matches(transport, inputs["A"]["height_1x"], inputs["B"]["height_1x"])
    )

    if status == "judged":
        valid = (
            transport_valid and winner in {"A", "B", "tie"}
            and api["called"] and fingerprint_valid and error is None
            and _api_audit_valid(audit, successful=True)
        )
    elif status == "auto_win":
        valid_side = "A" if a_valid and not b_valid else "B" if b_valid and not a_valid else None
        valid = (
            valid_side is not None and winner == valid_side
            and transport is None
            and not api["called"] and audit is None
            and response_fingerprint is None and error is None
        )
    elif status == "both_invalid":
        valid = (
            not a_valid and not b_valid and winner is None
            and transport is None
            and not api["called"] and audit is None
            and response_fingerprint is None and error is None
        )
    elif status == "judge_failure":
        valid = (
            transport_valid and winner is None and api["called"]
            and response_fingerprint is None
            and _api_audit_valid(audit, successful=False)
            and isinstance(error, dict) and error.get("category") == "judge_failure"
        )
    else:
        valid = (
            transport_valid and winner is None and api["called"]
            and fingerprint_valid
            and _api_audit_valid(audit, successful=True)
            and isinstance(error, dict)
            and error.get("category") == "invalid_judge_output"
        )
    if not valid:
        raise EvaluationError(f"inconsistent status/result/API semantics: {spec.battle_id}")


def load_existing_battles(
    path: Path, schedule: list[BattleSpec]
) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise EvaluationError(f"{path}: truncated final JSONL line")
    records: dict[str, dict[str, Any]] = {}
    known_ids = {spec.battle_id for spec in schedule}
    for line_number, line in enumerate(raw.splitlines(), 1):
        if not line.strip():
            raise EvaluationError(f"{path}:{line_number}: blank lines are not allowed")
        if line_number > len(schedule):
            raise EvaluationError(f"{path}:{line_number}: more rows than scheduled battles")
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise EvaluationError(f"{path}:{line_number}: malformed JSON") from exc
        if not isinstance(record, dict):
            raise EvaluationError(f"{path}:{line_number}: battle row must be an object")
        battle_id = record.get("battle_id")
        if battle_id not in known_ids:
            raise EvaluationError(f"{path}:{line_number}: unknown battle ID")
        if battle_id in records:
            raise EvaluationError(f"{path}:{line_number}: duplicate battle ID")
        # Slot binding, not row-order binding: rejudged rows leave holes and
        # their replacements append at the end; each row's own ordinal locks
        # the schedule slot and validate_battle_record checks the semantics.
        ordinal = record.get("ordinal")
        if type(ordinal) is not int or not 0 <= ordinal < len(schedule):
            raise EvaluationError(f"{path}:{line_number}: battle ordinal out of range")
        spec = schedule[ordinal]
        if battle_id != spec.battle_id:
            raise EvaluationError(f"{path}:{line_number}: battle ordinal/ID slot mismatch")
        validate_battle_record(record, spec)
        records[battle_id] = record
    return records


def build_summary(
    records: dict[str, dict[str, Any]],
    schedule: list[BattleSpec],
    labels: list[str],
    selected_count: int,
    formal_all: bool,
) -> dict[str, Any]:
    missing = [spec.battle_id for spec in schedule if spec.battle_id not in records]
    if missing:
        raise EvaluationError(
            f"cannot publish an incomplete summary: {len(missing)} battles missing"
        )
    ordered = [records[spec.battle_id] for spec in schedule]
    pair_count = len(labels) * (len(labels) - 1) // 2
    formal_battle_count = FORMAL_DENOMINATOR * pair_count
    denominator = FORMAL_DENOMINATOR if formal_all else selected_count
    participant_points = {label: 0.0 for label in labels}
    pair_rows: list[dict[str, Any]] = []
    for left, right in itertools.combinations(labels, 2):
        pair_records = [record for record in ordered if record["pair"] == [left, right]]
        points = {left: 0.0, right: 0.0}
        for record in pair_records:
            points[record["assignment"]["A"]] += record["points"]["A"]
            points[record["assignment"]["B"]] += record["points"]["B"]
        participant_points[left] += points[left]
        participant_points[right] += points[right]
        pair_rows.append({
            "participants": [left, right],
            "selected_battles": len(pair_records),
            "formal_denominator": FORMAL_DENOMINATOR,
            "denominator": denominator,
            "points": points,
            "win_rate": {
                left: points[left] / denominator,
                right: points[right] / denominator,
            },
        })
    overall_denominator = denominator * (len(labels) - 1)
    return {
        "schema_version": SUMMARY_SCHEMA,
        "eval_version": EVAL_VERSION,
        "formal_all": formal_all,
        "formal_denominator": FORMAL_DENOMINATOR,
        "denominator": denominator,
        "selected_count": selected_count,
        "participant_count": len(labels),
        "expected_battles": len(schedule),
        "completed_battles": len(ordered),
        "formal_expected_battles": formal_battle_count,
        "coverage": {
            "selected": len(ordered) / len(schedule),
            "formal": len(ordered) / formal_battle_count,
        },
        "status_counts": dict(sorted(Counter(row["status"] for row in ordered).items())),
        "pair_win_rate_formula": f"participant pair points / {denominator}",
        "overall_win_rate_formula": f"participant total points / ({denominator} * (participant_count - 1))",
        "pairs": pair_rows,
        "participants": {
            label: {
                "points": participant_points[label],
                "denominator": overall_denominator,
                "overall_win_rate": participant_points[label] / overall_denominator,
            }
            for label in labels
        },
    }


def build_formal_metric_artifact(
    summary: dict[str, Any], summary_path: Path
) -> dict[str, Any]:
    if summary.get("schema_version") != SUMMARY_SCHEMA:
        raise EvaluationError("unexpected Task4 pairwise summary schema")
    rates = {
        label: values["overall_win_rate"]
        for label, values in sorted(summary["participants"].items())
    }
    return build_formal_artifact(
        "task4_pairwise",
        {"overall_pairwise_win_rate": rates},
        context={
            "eval_version": summary["eval_version"],
            "formal_all": summary["formal_all"],
            "formal_denominator_per_opponent": summary["formal_denominator"],
            "denominator_per_opponent": summary["denominator"],
            "participant_count": summary["participant_count"],
            "selected_count": summary["selected_count"],
            "status_counts": summary["status_counts"],
        },
        provenance={
            "source_summary_path": str(summary_path.resolve()),
            "source_summary_schema_version": summary["schema_version"],
        },
    )


def _slug(value: str) -> str:
    import re as _re

    return _re.sub(r"[^A-Za-z0-9._-]", "_", value)


def _default_out_dir(
    participants: list[Participant], judge_model: str
) -> Path:
    labels = "+".join(item.label for item in participants)
    return (
        Path("outputs/task4/eval/pairwise")
        / f"{EVAL_VERSION}-{_slug(judge_model)}-{labels}"
    )


def _validate_output_dir(out_dir: Path, runs_root: Path) -> None:
    resolved = out_dir.resolve()
    try:
        resolved.relative_to(runs_root.resolve())
    except ValueError:
        return
    raise EvaluationError("--out-dir must not be inside the frozen Task 4 runs directory")


def build_run_config(
    participants: list[Participant],
    selected_ids: list[str],
    formal_ids: list[str],
    formal_all: bool,
    freeze: dict[str, Any],
    cohort: dict[str, Any],
    judge_model: str,
    judge_base_url: str | None,
) -> dict[str, Any]:
    pair_count = len(participants) * (len(participants) - 1) // 2
    return {
        "eval_version": EVAL_VERSION,
        "schemas": {
            "run_meta": RUN_META_SCHEMA,
            "battle": BATTLE_SCHEMA,
            "summary": SUMMARY_SCHEMA,
            "input_record": INPUT_RECORD_SCHEMA,
            "response": PAIRWISE_RESPONSE_SCHEMA_VERSION,
        },
        "formal_all": formal_all,
        "formal_denominator": FORMAL_DENOMINATOR,
        "formal_count": len(formal_ids),
        "selected_count": len(selected_ids),
        "participants": [
            {"label": item.label, "run_name": item.run_name}
            for item in participants
        ],
        "schedule": {
            "enumeration": "formal-query-order-then-itertools.combinations(sorted-labels,2)",
            "pair_count_per_sample": pair_count,
            "expected_battles": len(selected_ids) * pair_count,
            "ab_namespace": AB_NAMESPACE,
        },
        "freeze": {
            "version": freeze.get("freeze_version"),
            "cohort_name": cohort.get("cohort"),
            "cohort_schema_version": cohort.get("schema_version"),
        },
        "judge": {
            "model": judge_model,
            "base_url": judge_base_url,
        },
        "prompt": {
            "version": PAIRWISE_PROMPT_VERSION,
            "message_format": "text-(overview-A,B)-interleaved-slices-v2",
        },
        "response_format": {"version": PAIRWISE_RESPONSE_SCHEMA_VERSION},
        "preprocess": PREPROCESS_CONTRACT,
        "retry": RETRY_CONTRACT,
        "runtime_versions": {
            "python": sys.version.split()[0],
            "pillow": PIL.__version__,
        },
    }


async def run_evaluation(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    eval_cfg = cfg.get("eval") or {}
    runs_root = Path(
        args.runs_root or eval_cfg.get("runs_root") or "outputs/task4/runs"
    ).expanduser().resolve()
    participants = parse_participants(args.participant, runs_root)
    data_root = Path(eval_cfg.get("data_root") or DEFAULT_DATA_ROOT)
    formal_ids, freeze, cohort = load_formal_cohort(data_root)
    if len(formal_ids) != FORMAL_DENOMINATOR:
        raise EvaluationError("formal Task 4 denominator drift")
    selected_ids = select_ids(formal_ids, args.all, args.sample_id)
    for participant in participants:
        validate_run_cohort(participant.run_dir, freeze, cohort)
    formal_all = selected_ids == formal_ids
    judge_model = cfg["model"]
    out_dir = (
        Path(args.out_dir)
        if args.out_dir is not None
        else Path(eval_cfg.get("out_dir") or _default_out_dir(participants, judge_model))
    )
    _validate_output_dir(out_dir, runs_root)
    run_meta_path = out_dir / "run_meta.json"
    battles_path = out_dir / "battles.jsonl"
    summary_path = out_dir / "summary.json"
    formal_metrics_path = out_dir / "formal_metrics.json"

    config = build_run_config(
        participants,
        selected_ids,
        formal_ids,
        formal_all,
        freeze,
        cohort,
        judge_model,
        cfg.get("base_url"),
    )
    ensure_run_meta(run_meta_path, RUN_META_SCHEMA, config)
    evaluation_identity = canonical_json({
        "eval_version": EVAL_VERSION,
        "participants": config["participants"],
        "judge_model": judge_model,
        "formal_all": formal_all,
        "selected_ids": selected_ids,
    })
    labels = [participant.label for participant in participants]
    schedule = build_schedule(selected_ids, labels, evaluation_identity)
    existing = load_existing_battles(battles_path, schedule)
    if args.rejudge_failures and existing:
        dropped = drop_failed_records(battles_path, existing)
        if dropped:
            for stale_artifact in (summary_path, formal_metrics_path):
                if stale_artifact.exists():
                    stale_artifact.unlink()
            print(
                f"[pairwise] rejudging {dropped} failed battles "
                f"(stale summary artifacts removed)",
                flush=True,
            )
    pending = [spec for spec in schedule if spec.battle_id not in existing]
    if (summary_path.exists() or formal_metrics_path.exists()) and pending:
        raise EvaluationError(
            "summary artifact exists before all scheduled battles are complete"
        )
    request_cfg = require_mapping(cfg, "request")
    workers = args.workers or int(request_cfg.get("concurrency", 4))
    print(
        f"Task4 pairwise | participants={len(participants)} | samples={len(selected_ids)} "
        f"| battles={len(schedule)} | resume={len(existing)} | pending={len(pending)} "
        f"| workers={workers}",
        flush=True,
    )
    client = make_client(
        cfg.get("protocol", "openai"),
        base_url=cfg.get("base_url"),
        api_key=resolve_api_key(cfg),
        timeout=request_cfg.get("timeout", 600),
    )
    participants_by_label = {item.label: item for item in participants}
    semaphore = asyncio.Semaphore(workers)
    specs_by_id = {spec.battle_id: spec for spec in schedule}
    completed_now = 0

    async def judge(spec: BattleSpec) -> None:
        nonlocal completed_now
        async with semaphore:
            record = await evaluate_battle(
                spec, participants_by_label, client, judge_model
            )
        validate_battle_record(record, specs_by_id[record["battle_id"]])
        append_record(battles_path, record)
        existing[record["battle_id"]] = record
        completed_now += 1
        if completed_now % 25 == 0 or completed_now == len(pending):
            print(f"[pairwise] completed {len(existing)}/{len(schedule)}", flush=True)

    await asyncio.gather(*(judge(spec) for spec in pending))

    if len(existing) != len(schedule):
        raise EvaluationError(
            "evaluation ended before every selected battle reached a terminal state"
        )
    summary = build_summary(
        existing, schedule, labels, len(selected_ids), formal_all
    )
    publish_summary(summary_path, summary)
    formal_artifact = build_formal_metric_artifact(summary, summary_path)
    write_json_atomic(formal_metrics_path, formal_artifact)
    print(
        f"Complete | Pairwise WR written | statuses={summary['status_counts']} "
        f"| output={out_dir}",
        flush=True,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/task4_pairwise.yaml")
    parser.add_argument(
        "--participant",
        action="append",
        default=[],
        metavar="LABEL=RUN_NAME",
        help="participant label and directory under the runs root; repeat at least twice",
    )
    parser.add_argument(
        "--runs-root",
        help="override eval.runs_root (default: outputs/task4/runs)",
    )
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--all", action="store_true",
                           help="evaluate the formal 1,395-item cohort")
    selection.add_argument(
        "--sample-id",
        action="append",
        default=[],
        help="evaluate one formal sample ID; repeat for multiple IDs",
    )
    parser.add_argument("--workers", type=int, default=None,
                        help="override request.concurrency")
    parser.add_argument(
        "--rejudge-failures",
        action="store_true",
        help="before resuming, drop judge_failure/invalid_judge_output rows (archived to "
             "battles.rejudged.jsonl) so this round re-judges them",
    )
    parser.add_argument("--out-dir")
    args = parser.parse_args(argv)
    if args.workers is not None and args.workers < 1:
        parser.error("--workers must be at least 1")
    if len(args.participant) < 2:
        parser.error("--participant must be repeated at least twice")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        asyncio.run(run_evaluation(args))
    except Exception as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
