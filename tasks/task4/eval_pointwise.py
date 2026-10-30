#!/usr/bin/env python3
"""Evaluate Task 4 runs with pointwise LLM aesthetic scoring.

Every page of the formal cohort is judged on the Task 1 8-dimension 1-5
rubric plus an overall 1-5 score; pages with missing renders or failed
generations stay terminal and remain in the denominator (overall counts
as 0). Use --sample-id to judge explicit samples only (debugging).

Run from the repository root:

    python -m tasks.task4.eval_pointwise --config configs/task4_pointwise.yaml \
        --participant alpha=qwen3-vl-plus-topk-v5
    python -m tasks.task4.eval_pointwise --config configs/task4_pointwise.yaml \
        --participant alpha=qwen3-vl-plus-topk-v5 --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import PIL

from src.api.client import make_client
from src.utils.config import load_config, require_mapping, resolve_api_key
from tasks.task4.cohort import (
    DEFAULT_DATA_ROOT,
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
    drop_failed_records,
    ensure_run_meta,
    input_full_bytes,
    input_height_1x,
    participant_input,
    publish_summary,
    write_json_atomic,
)
from tasks.task4.prompts import (
    DIMENSION_KEYS,
    POINTWISE_PROMPT,
    POINTWISE_PROMPT_VERSION,
    POINTWISE_RESPONSE_FORMAT,
    POINTWISE_RESPONSE_SCHEMA_VERSION,
    SCORE_KEYS,
)
from tasks.task4.scoring import scores_valid, strict_scores
from tasks.task4.views import (
    PREPROCESS_CONTRACT,
    SLICE_H,
    encode_view,
    fixed_canvas_slice,
    load_1x_image,
    overview_image,
    slice_offsets,
)

EVAL_VERSION = "task4-pointwise-visual-v2"
RUN_META_SCHEMA = "task4-pointwise-run-v1"
RESULT_SCHEMA = "task4-pointwise-result-v1"
SUMMARY_SCHEMA = "task4-pointwise-summary-v1"


@dataclass(frozen=True)
class ScoreSpec:
    ordinal: int
    participant: str
    sample_id: str


def parse_participants(values: list[str], runs_root: Path) -> list[Participant]:
    if not values:
        raise EvaluationError("at least one --participant LABEL=RUN_NAME value is required")
    parsed: list[tuple[str, str]] = []
    for value in values:
        if value.count("=") != 1:
            raise EvaluationError("each --participant must have the form LABEL=RUN_NAME")
        label, run_name = value.split("=", 1)
        _safe_component(label, "participant label")
        _safe_component(run_name, "participant run name")
        parsed.append((label, run_name))
    labels = [label for label, _ in parsed]
    if len(labels) != len(set(labels)):
        raise EvaluationError("duplicate participant labels are not allowed")
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
    return participants


def build_schedule(
    participants: list[Participant], evaluated_ids: list[str]
) -> list[ScoreSpec]:
    specs: list[ScoreSpec] = []
    for participant in participants:
        for sample_id in evaluated_ids:
            specs.append(ScoreSpec(len(specs), participant.label, sample_id))
    return specs


def _messages(image, offsets: list[int]) -> list[dict]:
    n_slices = len(offsets)
    parts: list[dict] = [{"type": "text", "text": POINTWISE_PROMPT}]
    if n_slices > 1:
        overview = overview_image(image, image.height)
        data, media_type = encode_view(overview)
        parts.append({"type": "text", "text": "Full-page overview (navigation only):"})
        parts.append({"type": "image", "data": data, "media_type": media_type})
    for index, top in enumerate(offsets):
        slice_image = fixed_canvas_slice(image, top)
        label = (
            f"Slice {index + 1}/{n_slices}:" if n_slices > 1 else "Page screenshot:"
        )
        data, media_type = encode_view(slice_image)
        parts.append({"type": "text", "text": label})
        parts.append({"type": "image", "data": data, "media_type": media_type})
    return parts


def _base_record(
    spec: ScoreSpec, input_summary: dict[str, Any]
) -> dict[str, Any]:
    return {
        "schema_version": RESULT_SCHEMA,
        "eval_version": EVAL_VERSION,
        "ordinal": spec.ordinal,
        "participant": spec.participant,
        "sample_id": spec.sample_id,
        "input": input_summary,
    }


def _outcome(
    base: dict[str, Any],
    transport: dict[str, Any] | None,
    status: str,
    scores: dict[str, dict[str, Any]] | None,
    called: bool,
    response_audit: dict[str, Any] | None,
    response_fingerprint: dict[str, Any] | None,
    error: dict[str, str] | None,
) -> dict[str, Any]:
    return {
        **base,
        "transport": transport,
        "status": status,
        "scores": scores,
        "api": {
            "called": called,
            "response_audit": response_audit,
            "response_fingerprint": response_fingerprint,
        },
        "error": error,
    }


def _input_summary(input_record: dict[str, Any]) -> dict[str, Any]:
    return {
        "valid": bool(input_record.get("valid")),
        "height_1x": input_height_1x(input_record),
        "full_bytes": input_full_bytes(input_record),
    }


async def evaluate_score(
    spec: ScoreSpec,
    participants_by_label: dict[str, Participant],
    client,
    judge_model: str,
) -> dict[str, Any]:
    participant = participants_by_label[spec.participant]
    input_record = participant_input(participant, spec.sample_id)
    input_summary = _input_summary(input_record)
    base = _base_record(spec, input_summary)
    if not input_summary["valid"] or input_summary["height_1x"] is None:
        reasons = "; ".join(
            item.get("message", "") for item in input_record.get("errors", [])
        )
        return _outcome(
            base,
            None,
            "input_invalid",
            None,
            False,
            None,
            None,
            _error("input_invalid", reasons or "input audit failed"),
        )
    image = load_1x_image(participant.run_dir, spec.sample_id)
    if image.height != input_summary["height_1x"]:
        raise EvaluationError(
            f"input height drift on reload: {spec.participant}/{spec.sample_id}"
        )
    offsets = slice_offsets(image.height)
    transport = {
        "height_1x": image.height,
        "n_slices": len(offsets),
        "slice_offsets": list(offsets),
        "truncated": offsets[-1] + SLICE_H < image.height,
        "overview_included": len(offsets) > 1,
    }
    response_audit: dict[str, Any] = {"attempt": 1}
    try:
        reply = await call_judge(
            client,
            judge_model,
            _messages(image, offsets),
            response_format=POINTWISE_RESPONSE_FORMAT,
            response_audit=response_audit,
        )
    except Exception as exc:
        return _outcome(
            base,
            transport,
            "judge_failure",
            None,
            True,
            _sanitize_api_audit(response_audit),
            None,
            _error("judge_failure", f"judge call failed ({type(exc).__name__})"),
        )
    sanitized_audit = _sanitize_api_audit(response_audit)
    response_fingerprint = {"chars": len(reply)}
    try:
        scores = strict_scores(reply, SCORE_KEYS)
    except Exception:
        return _outcome(
            base,
            transport,
            "invalid_judge_output",
            None,
            True,
            sanitized_audit,
            response_fingerprint,
            _error(
                "invalid_judge_output", "judge response failed the strict rubric schema"
            ),
        )
    return _outcome(
        base, transport, "complete", scores, True, sanitized_audit, response_fingerprint, None
    )


def _transport_valid(transport: object, height_1x: int) -> bool:
    if not isinstance(transport, dict):
        return False
    if set(transport) != {
        "height_1x", "n_slices", "slice_offsets", "truncated", "overview_included"
    }:
        return False
    offsets = slice_offsets(height_1x)
    return transport == {
        "height_1x": height_1x,
        "n_slices": len(offsets),
        "slice_offsets": offsets,
        "truncated": offsets[-1] + SLICE_H < height_1x,
        "overview_included": len(offsets) > 1,
    }


def validate_result_record(record: dict[str, Any], spec: ScoreSpec) -> None:
    expected_keys = {
        "schema_version", "eval_version", "ordinal", "participant", "sample_id",
        "input", "transport", "status", "scores", "api", "error",
    }
    if set(record) != expected_keys:
        raise EvaluationError(
            f"result record shape mismatch: {spec.participant}/{spec.sample_id}"
        )
    for key, expected in (
        ("schema_version", RESULT_SCHEMA),
        ("eval_version", EVAL_VERSION),
        ("ordinal", spec.ordinal),
        ("participant", spec.participant),
        ("sample_id", spec.sample_id),
    ):
        if record.get(key) != expected:
            raise EvaluationError(
                f"result envelope mismatch on {key}: {spec.participant}/{spec.sample_id}"
            )
    input_summary = record["input"]
    if not isinstance(input_summary, dict) or set(input_summary) != {
        "valid", "height_1x", "full_bytes"
    }:
        raise EvaluationError(f"invalid input summary: {spec.participant}/{spec.sample_id}")
    api = record["api"]
    if not isinstance(api, dict) or set(api) != {
        "called", "response_audit", "response_fingerprint"
    }:
        raise EvaluationError(f"invalid api block: {spec.participant}/{spec.sample_id}")
    status = record["status"]
    fingerprint = api["response_fingerprint"]
    fingerprint_valid = (
        isinstance(fingerprint, dict)
        and set(fingerprint) == {"chars"}
        and type(fingerprint["chars"]) is int
        and fingerprint["chars"] >= 0
    )
    audit = api["response_audit"]
    if status == "complete":
        valid = (
            input_summary["valid"] and isinstance(input_summary["height_1x"], int)
            and _transport_valid(record["transport"], input_summary["height_1x"])
            and scores_valid(record["scores"], SCORE_KEYS)
            and api["called"] and fingerprint_valid and record["error"] is None
            and _api_audit_valid(audit, successful=True)
        )
    elif status == "input_invalid":
        valid = (
            not input_summary["valid"] and record["transport"] is None
            and record["scores"] is None
            and not api["called"] and audit is None and fingerprint is None
            and isinstance(record["error"], dict)
            and record["error"].get("category") == "input_invalid"
        )
    elif status == "judge_failure":
        valid = (
            input_summary["valid"] and isinstance(input_summary["height_1x"], int)
            and _transport_valid(record["transport"], input_summary["height_1x"])
            and record["scores"] is None
            and api["called"] and fingerprint is None
            and _api_audit_valid(audit, successful=False)
            and isinstance(record["error"], dict)
            and record["error"].get("category") == "judge_failure"
        )
    elif status == "invalid_judge_output":
        valid = (
            input_summary["valid"] and isinstance(input_summary["height_1x"], int)
            and _transport_valid(record["transport"], input_summary["height_1x"])
            and record["scores"] is None
            and api["called"] and fingerprint_valid
            and _api_audit_valid(audit, successful=True)
            and isinstance(record["error"], dict)
            and record["error"].get("category") == "invalid_judge_output"
        )
    else:
        valid = False
    if not valid:
        raise EvaluationError(
            f"inconsistent status/result/API semantics: {spec.participant}/{spec.sample_id}"
        )


def load_existing_results(
    path: Path, schedule: list[ScoreSpec]
) -> dict[tuple[str, str], dict[str, Any]]:
    if not path.exists():
        return {}
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise EvaluationError(f"{path}: truncated final JSONL line")
    records: dict[tuple[str, str], dict[str, Any]] = {}
    known_slots = {(spec.participant, spec.sample_id) for spec in schedule}
    for line_number, line in enumerate(raw.splitlines(), 1):
        if not line.strip():
            raise EvaluationError(f"{path}:{line_number}: blank lines are not allowed")
        if line_number > len(schedule):
            raise EvaluationError(f"{path}:{line_number}: more rows than scheduled scores")
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise EvaluationError(f"{path}:{line_number}: malformed JSON") from exc
        if not isinstance(record, dict):
            raise EvaluationError(f"{path}:{line_number}: result row must be an object")
        key = (record.get("participant"), record.get("sample_id"))
        if key not in known_slots:
            raise EvaluationError(f"{path}:{line_number}: unknown participant/sample slot")
        if key in records:
            raise EvaluationError(f"{path}:{line_number}: duplicate participant/sample slot")
        ordinal = record.get("ordinal")
        if type(ordinal) is not int or not 0 <= ordinal < len(schedule):
            raise EvaluationError(f"{path}:{line_number}: result ordinal out of range")
        spec = schedule[ordinal]
        if (spec.participant, spec.sample_id) != key:
            raise EvaluationError(f"{path}:{line_number}: result ordinal/slot mismatch")
        validate_result_record(record, spec)
        records[key] = record
    return records


def build_run_config(
    participants: list[Participant],
    evaluated_ids: list[str],
    formal_ids: list[str],
    judge_model: str,
    judge_base_url: str | None,
    selection_mode: str | None = None,
) -> dict[str, Any]:
    cohort_block: dict[str, Any] = {
        "size": len(evaluated_ids),
        "ids": evaluated_ids,
        "formal_cohort_size": len(formal_ids),
    }
    if selection_mode is not None:
        cohort_block = {"mode": selection_mode, **cohort_block}
    return {
        "eval_version": EVAL_VERSION,
        "schemas": {
            "run_meta": RUN_META_SCHEMA,
            "result": RESULT_SCHEMA,
            "summary": SUMMARY_SCHEMA,
            "input_record": INPUT_RECORD_SCHEMA,
        },
        "participants": [
            {"label": item.label, "run_name": item.run_name} for item in participants
        ],
        "judge": {"model": judge_model, "base_url": judge_base_url},
        "cohort": cohort_block,
        "preprocess": {**PREPROCESS_CONTRACT, "geometry": "single-page-v1"},
        "prompt_version": POINTWISE_PROMPT_VERSION,
        "response_schema_version": POINTWISE_RESPONSE_SCHEMA_VERSION,
        "retry": RETRY_CONTRACT,
        "runtime_versions": {
            "python": sys.version.split()[0],
            "pillow": PIL.__version__,
        },
    }


def build_summary(
    records: dict[tuple[str, str], dict[str, Any]],
    schedule: list[ScoreSpec],
    participants: list[Participant],
    evaluated_ids: list[str],
    selection_mode: str | None = None,
) -> dict[str, Any]:
    missing = [
        spec for spec in schedule if (spec.participant, spec.sample_id) not in records
    ]
    if missing:
        raise EvaluationError(
            f"cannot publish an incomplete summary: {len(missing)} scores missing"
        )
    denominator = len(evaluated_ids)
    per_participant: dict[str, Any] = {}
    for participant in participants:
        rows = [
            records[(participant.label, spec.sample_id)]
            for spec in schedule
            if spec.participant == participant.label
        ]
        status_counts = dict(sorted(Counter(row["status"] for row in rows).items()))
        complete = [row for row in rows if row["status"] == "complete"]
        n_valid = len(complete)
        overall_sum = sum(row["scores"]["overall"]["score"] for row in complete)
        dimension_means = {}
        for key in DIMENSION_KEYS:
            dimension_means[key] = (
                round(sum(row["scores"][key]["score"] for row in complete) / n_valid, 6)
                if n_valid
                else None
            )
        per_participant[participant.label] = {
            "denominator": denominator,
            "n_valid": n_valid,
            "coverage": n_valid / denominator,
            "status_counts": status_counts,
            "overall_score_sum": overall_sum,
            "mean_overall_aesthetic_score": overall_sum / denominator,
            "valid_only_mean_overall": round(overall_sum / n_valid, 6) if n_valid else None,
            "dimension_means_valid_only": dimension_means,
        }
    summary = {
        "schema_version": SUMMARY_SCHEMA,
        "eval_version": EVAL_VERSION,
        "evaluated_ids": evaluated_ids,
        "evaluated_count": denominator,
        "participant_count": len(participants),
        "expected_scores": len(schedule),
        "completed_scores": len(records),
        "status_counts": dict(
            sorted(Counter(row["status"] for row in records.values()).items())
        ),
        "mean_formula": "sum(overall scores of complete rows) / evaluated_count; failed rows contribute 0",
        "participants": per_participant,
    }
    if selection_mode is not None:
        summary["selection_mode"] = selection_mode
    return summary


def build_formal_metric_artifact(
    summary: dict[str, Any], summary_path: Path
) -> dict[str, Any]:
    if summary.get("schema_version") != SUMMARY_SCHEMA:
        raise EvaluationError("unexpected Task4 pointwise summary schema")
    means = {
        label: values["mean_overall_aesthetic_score"]
        for label, values in sorted(summary["participants"].items())
    }
    return build_formal_artifact(
        "task4_pointwise",
        {"mean_overall_aesthetic_score": means},
        context={
            "eval_version": summary["eval_version"],
            "evaluated_count": summary["evaluated_count"],
            "participant_count": summary["participant_count"],
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
    participants: list[Participant],
    judge_model: str,
) -> Path:
    labels = "+".join(item.label for item in participants)
    name = f"{EVAL_VERSION}-{_slug(judge_model)}-{labels}"
    return Path("outputs/task4/eval/pointwise") / name


def _validate_output_dir(out_dir: Path, runs_root: Path) -> None:
    resolved = out_dir.resolve()
    try:
        resolved.relative_to(runs_root.resolve())
    except ValueError:
        return
    raise EvaluationError("--out-dir must not be inside the frozen Task 4 runs directory")


def dry_run_audit(
    participants: list[Participant],
    evaluated_ids: list[str],
    workers: int,
) -> None:
    def audit_one(job: tuple[Participant, str]) -> tuple[str, bool, list[str]]:
        participant, sample_id = job
        record = participant_input(participant, sample_id)
        messages = [item.get("message", "") for item in record.get("errors", [])]
        return participant.label, bool(record.get("valid")), messages

    jobs = [(p, sid) for p in participants for sid in evaluated_ids]
    counts: dict[str, Counter] = {p.label: Counter() for p in participants}
    examples: dict[str, list[str]] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for label, valid, messages in executor.map(audit_one, jobs):
            counts[label]["valid" if valid else "invalid"] += 1
            if not valid and label not in examples:
                examples[label] = messages
    print(f"DRY-RUN | samples={len(evaluated_ids)} | participants={len(participants)}")
    for label in sorted(counts):
        counter = counts[label]
        line = f"  {label:<14} valid={counter['valid']:<4} invalid={counter['invalid']}"
        if label in examples:
            line += f" | first errors: {examples[label][:2]}"
        print(line)
    print("DRY-RUN complete (no API calls, nothing written)", flush=True)


async def run_evaluation(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    eval_cfg = cfg.get("eval") or {}
    runs_root = Path(
        args.runs_root or eval_cfg.get("runs_root") or "outputs/task4/runs"
    ).expanduser().resolve()
    participants = parse_participants(args.participant, runs_root)
    data_root = Path(eval_cfg.get("data_root") or DEFAULT_DATA_ROOT)
    formal_ids, freeze, cohort = load_formal_cohort(data_root)
    for participant in participants:
        validate_run_cohort(participant.run_dir, freeze, cohort)
    evaluated_ids = (
        select_ids(formal_ids, False, args.sample_id)
        if args.sample_id
        else list(formal_ids)
    )
    selection_mode = "explicit_ids" if args.sample_id else None
    request_cfg = require_mapping(cfg, "request")
    workers = args.workers or int(request_cfg.get("concurrency", 4))
    if args.dry_run:
        dry_run_audit(participants, evaluated_ids, workers)
        return
    judge_model = cfg["model"]
    out_dir = (
        Path(args.out_dir)
        if args.out_dir is not None
        else Path(eval_cfg.get("out_dir") or _default_out_dir(
            participants, judge_model
        ))
    )
    _validate_output_dir(out_dir, runs_root)
    run_meta_path = out_dir / "run_meta.json"
    results_path = out_dir / "results.jsonl"
    summary_path = out_dir / "summary.json"
    formal_metrics_path = out_dir / "formal_metrics.json"

    config = build_run_config(
        participants,
        evaluated_ids,
        formal_ids,
        judge_model,
        cfg.get("base_url"),
        selection_mode,
    )
    ensure_run_meta(run_meta_path, RUN_META_SCHEMA, config)
    schedule = build_schedule(participants, evaluated_ids)
    existing = load_existing_results(results_path, schedule)
    if args.rejudge_failures and existing:
        dropped = drop_failed_records(results_path, existing)
        if dropped:
            for stale_artifact in (summary_path, formal_metrics_path):
                if stale_artifact.exists():
                    stale_artifact.unlink()
            print(
                f"[pointwise] rejudging {dropped} failed scores "
                f"(stale summary artifacts removed)",
                flush=True,
            )
    pending = [
        spec for spec in schedule
        if (spec.participant, spec.sample_id) not in existing
    ]
    if (summary_path.exists() or formal_metrics_path.exists()) and pending:
        raise EvaluationError(
            "summary artifact exists before all scheduled scores are complete"
        )
    if args.limit is not None:
        pending = pending[: args.limit]
    print(
        f"Task4 pointwise | participants={len(participants)} | samples={len(evaluated_ids)} "
        f"| scores={len(schedule)} | resume={len(existing)} | pending={len(pending)} "
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
    specs_by_ordinal = {spec.ordinal: spec for spec in schedule}
    completed_now = 0

    async def judge(spec: ScoreSpec) -> None:
        nonlocal completed_now
        async with semaphore:
            record = await evaluate_score(
                spec, participants_by_label, client, judge_model
            )
        validate_result_record(record, specs_by_ordinal[record["ordinal"]])
        append_record(results_path, record)
        existing[(record["participant"], record["sample_id"])] = record
        completed_now += 1
        if completed_now % 25 == 0 or completed_now == len(pending):
            print(f"[pointwise] completed {len(existing)}/{len(schedule)}", flush=True)

    await asyncio.gather(*(judge(spec) for spec in pending))

    if len(existing) != len(schedule):
        print(
            f"[pointwise] partial run: {len(existing)}/{len(schedule)} terminal; "
            "re-run without --limit to finish and publish summary",
            flush=True,
        )
        return
    summary = build_summary(
        existing,
        schedule,
        participants,
        evaluated_ids,
        selection_mode,
    )
    publish_summary(summary_path, summary)
    formal_artifact = build_formal_metric_artifact(summary, summary_path)
    write_json_atomic(formal_metrics_path, formal_artifact)
    print(
        f"Complete | pointwise AES-5 written | statuses={summary['status_counts']} "
        f"| output={out_dir}",
        flush=True,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/task4_pointwise.yaml")
    parser.add_argument(
        "--participant",
        action="append",
        default=[],
        metavar="LABEL=RUN_NAME",
        help="participant label and directory under the runs root; repeatable",
    )
    parser.add_argument("--runs-root",
                        help="override eval.runs_root (default: outputs/task4/runs)")
    parser.add_argument(
        "--sample-id",
        action="append",
        default=[],
        metavar="SAMPLE_ID",
        help="evaluate explicit formal sample IDs instead of the full cohort; "
             "repeatable (debugging)",
    )
    parser.add_argument("--workers", type=int, default=None,
                        help="override request.concurrency")
    parser.add_argument(
        "--rejudge-failures",
        action="store_true",
        help="before resuming, drop judge_failure/invalid_judge_output rows (archived to "
             "results.rejudged.jsonl) so this round re-judges them",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="audit cohort inputs only; no API calls, nothing written",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="judge at most N pending scores this run (smoke; summary is skipped)",
    )
    parser.add_argument("--out-dir")
    args = parser.parse_args(argv)
    if args.workers is not None and args.workers < 1:
        parser.error("--workers must be at least 1")
    if not args.participant:
        parser.error("--participant must be given at least once")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
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
