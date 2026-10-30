#!/usr/bin/env python3
"""Shared machinery for the two Task 4 evaluators (pointwise and pairwise).

Covers participant resolution, input audits, the judge call contract, run
metadata, JSONL persistence, and the formal-metrics artifact.  No digests are
computed anywhere: configuration identity is established by direct comparison
of the full run_meta.json document plus deterministic schedule recomputation.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.api.client import is_retryable_api_error
from tasks.task4.audit import inspect_generation
from tasks.task4.cohort import EvaluationError, read_json
from tasks.task4.prompts import RENDER_PROTOCOL
from tasks.task4.views import (
    MIN_SOURCE_HEIGHT,
    NORMALIZED_WIDTH,
    PREPROCESS_VERSION,
    SOURCE_WIDTH,
)

INPUT_RECORD_SCHEMA = "task4-pairwise-input-record-v3"
API_RETRIES = 3
RETRY_CONTRACT_VERSION = "task4-pairwise-call-api-retries-3-v1"
SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

# The judge never sends enable_thinking (providers that reject unknown
# parameters would 400); reasoning is pinned off via reasoning_effort instead.
JUDGE_EXTRA_PAYLOAD = {"reasoning_effort": "none"}

RETRY_CONTRACT = {
    "version": RETRY_CONTRACT_VERSION,
    "logical_calls_per_battle": 1,
    "retries": API_RETRIES,
    "temperature": 0,
    "enable_thinking_preferred": False,
    "reasoning_effort": "none",
    "structured_output_omits_max_tokens": True,
}

ALLOWED_API_AUDIT_FIELDS = {"attempt", "http_status", "finish_reason_stop"}

FORMAL_METRICS_SCHEMA = "design-bench-formal-metrics-v2"
TASK4_METRIC_DEFINITIONS = {
    "mean_overall_aesthetic_score": {
        "formal_name": "Calibrated Aesthetic Score",
        "abbreviation": "AES",
        "direction": "higher_is_better",
        "role": "main",
    },
    "overall_pairwise_win_rate": {
        "formal_name": "Pairwise Win Rate",
        "abbreviation": "WR",
        "direction": "higher_is_better",
        "role": "main",
    },
}
TASK4_METRIC_RANGES = {
    "mean_overall_aesthetic_score": (0.0, 5.0),
    "overall_pairwise_win_rate": (0.0, 1.0),
}
TASK4_METRIC_BY_TASK = {
    "task4_pointwise": "mean_overall_aesthetic_score",
    "task4_pairwise": "overall_pairwise_win_rate",
}


@dataclass(frozen=True)
class Participant:
    label: str
    run_name: str
    run_dir: Path


def _safe_component(value: str, kind: str) -> None:
    if value in {".", ".."} or not SAFE_COMPONENT.fullmatch(value):
        raise EvaluationError(
            f"{kind} must be one safe path component using letters, digits, '.', '_', or '-'"
        )


def _bounded_text(value: object, limit: int = 1000) -> str:
    text = str(value).replace("\x00", " ").replace("\r", " ").replace("\n", " ")
    return text[:limit]


def _error(category: str, message: str) -> dict[str, str]:
    return {"category": category, "message": _bounded_text(message)}


def canonical_json(value: object) -> str:
    """Deterministic serialization for in-memory identity strings (never a digest)."""

    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def append_record(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def publish_summary(path: Path, summary: dict[str, Any]) -> None:
    if path.exists():
        existing = read_json(path)
        if existing != summary:
            raise EvaluationError(
                f"{path}: existing summary does not match current results"
            )
        return
    write_json_atomic(path, summary)


def ensure_run_meta(path: Path, schema: str, config: dict[str, Any]) -> dict[str, Any]:
    """Bind an output directory to one exact invocation configuration."""

    meta = {"schema_version": schema, "config": config}
    if path.exists():
        if read_json(path) != meta:
            raise EvaluationError(
                f"{path}: existing run metadata does not match this invocation"
            )
    else:
        write_json_atomic(path, meta)
    return meta


def drop_failed_records(path: Path, existing: dict) -> int:
    """Drop judge_failure/invalid_judge_output rows so they are re-judged.

    Dropped rows are appended to <stem>.rejudged.jsonl (audit trail); the
    survivors are atomically rewritten in ordinal order and removed from
    ``existing`` in place.  Returns the number of dropped rows.
    """

    doomed_keys = {
        key
        for key, record in existing.items()
        if record["status"] in {"judge_failure", "invalid_judge_output"}
    }
    if not doomed_keys:
        return 0
    doomed = sorted(
        (existing[key] for key in doomed_keys), key=lambda record: record["ordinal"]
    )
    archive_path = path.with_name(path.stem + ".rejudged.jsonl")
    with archive_path.open("a", encoding="utf-8") as handle:
        for record in doomed:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    kept = sorted(
        (record for key, record in existing.items() if key not in doomed_keys),
        key=lambda record: record["ordinal"],
    )
    tmp_path = path.with_name(path.stem + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        for record in kept:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    os.replace(tmp_path, path)
    for key in doomed_keys:
        del existing[key]
    return len(doomed)


# ---------------------------------------------------------------------------
# Judge call contract
# ---------------------------------------------------------------------------


async def call_judge(
    client,
    model: str,
    parts: list[dict],
    *,
    response_format: dict[str, Any],
    response_audit: dict[str, Any] | None,
    retries: int = API_RETRIES,
) -> str:
    """One logical judge call: temperature 0, json_schema, reasoning off.

    Only HTTP 408/429/5xx and transport errors retry (5s, 10s ladder); an
    invalid response is immediately terminal, mirroring the reference scorer.
    """

    if response_audit is not None:
        response_audit.clear()
    failure = "unknown API failure"
    for attempt in range(retries):
        if response_audit is not None:
            response_audit.clear()
            response_audit["attempt"] = attempt + 1
        try:
            text = await client.complete(
                model=model,
                parts=parts,
                temperature=0,
                max_tokens=None,  # json_schema calls omit max_tokens
                disable_thinking=False,  # never sent for the judge
                response_format=response_format,
                require_stop=True,
                extra_payload=JUDGE_EXTRA_PAYLOAD,
            )
            if response_audit is not None:
                response_audit["finish_reason_stop"] = True
            return text
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            if (
                response_audit is not None
                and isinstance(status, int)
                and not isinstance(status, bool)
            ):
                response_audit["http_status"] = status
            failure = type(exc).__name__
            if not is_retryable_api_error(exc):
                break
        if attempt + 1 < retries:
            await asyncio.sleep(5 * (attempt + 1))
    raise RuntimeError(f"judge call failed: {failure}")


def _sanitize_api_audit(value: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in sorted(ALLOWED_API_AUDIT_FIELDS):
        if key not in value:
            continue
        item = value[key]
        if item is None or type(item) in {bool, int, float, str}:
            result[key] = item
    return result


def _api_audit_valid(value: object, *, successful: bool) -> bool:
    if not isinstance(value, dict) or set(value) - ALLOWED_API_AUDIT_FIELDS:
        return False
    attempt = value.get("attempt")
    if type(attempt) is not int or not 1 <= attempt <= API_RETRIES:
        return False
    http_status = value.get("http_status")
    if http_status is not None and (
        type(http_status) is not int or not 100 <= http_status <= 599
    ):
        return False
    finish = value.get("finish_reason_stop")
    if finish is not None and type(finish) is not bool:
        return False
    if successful and finish is not True:
        return False
    return True


# ---------------------------------------------------------------------------
# Participant input audit
# ---------------------------------------------------------------------------


def audit_participant_input(
    participant: Participant, sample_id: str
) -> dict[str, Any]:
    """Validate one participant/sample input: generation artifacts + render."""

    sample_dir = participant.run_dir / sample_id
    errors: list[dict[str, str]] = []
    try:
        generation_fingerprint, generation_error = inspect_generation(sample_dir)
    except Exception as exc:
        generation_fingerprint = {}
        generation_error = f"{type(exc).__name__}: {exc}"
    if generation_error is not None:
        errors.append(_error("generation", generation_error))

    full_path = sample_dir / "full.png"
    crop_path = sample_dir / "crop4320.png"
    meta_path = sample_dir / "render_meta.json"
    render: dict[str, Any] = {
        "full_exists": full_path.is_file(),
        "crop_exists": crop_path.is_file(),
        "render_meta_exists": meta_path.is_file(),
    }
    source_audit: dict[str, Any] | None = None
    if full_path.is_file():
        source_audit = {
            "bytes": full_path.stat().st_size,
            "format": None,
            "size": None,
        }
        render["full_bytes"] = source_audit["bytes"]
    else:
        errors.append(_error("render", "missing full.png"))
    meta: dict[str, Any] | None = None
    if meta_path.is_file():
        try:
            meta = read_json(meta_path)
            render["metadata"] = {
                "protocol_version": meta.get("protocol_version"),
                "answer_bytes": meta.get("answer_bytes"),
                "crop_size": meta.get("crop_size"),
                "full_size": meta.get("full_size"),
                "java_script_enabled": meta.get("java_script_enabled"),
            }
        except Exception as exc:
            errors.append(_error("render", f"invalid render_meta.json: {type(exc).__name__}: {exc}"))
    else:
        errors.append(_error("render", "missing render_meta.json"))

    declared_size: list[int] | None = None
    if meta is not None:
        if meta.get("protocol_version") != RENDER_PROTOCOL:
            errors.append(_error("render", "render protocol mismatch"))
        full_size = meta.get("full_size")
        if (
            not isinstance(full_size, list)
            or len(full_size) != 2
            or any(type(value) is not int or value <= 0 for value in full_size)
        ):
            errors.append(_error("render", "declared full size is invalid"))
        else:
            declared_size = full_size
            if full_size[0] != SOURCE_WIDTH:
                errors.append(_error("render", "declared full width mismatch"))
            if full_size[1] < MIN_SOURCE_HEIGHT:
                errors.append(_error("render", "declared full height below one viewport"))
            if full_size[1] % 2 != 0:
                errors.append(_error("render", "declared full height is not even"))
        if type(meta.get("java_script_enabled")) is not bool:
            errors.append(_error("render", "JavaScript render flag is invalid"))

    if full_path.is_file():
        try:
            from PIL import Image

            Image.MAX_IMAGE_PIXELS = None
            with Image.open(full_path) as source:
                image_format = source.format
                actual_size = source.size
                source.load()
            render["full_format"] = image_format
            render["actual_full_size"] = list(actual_size)
            if source_audit is not None:
                source_audit["format"] = image_format
                source_audit["size"] = list(actual_size)
            if image_format != "PNG":
                errors.append(_error("render", "full.png is not a PNG image"))
            if declared_size is not None and list(actual_size) != list(declared_size):
                errors.append(_error("render", "actual full size mismatch"))
            if actual_size[0] != SOURCE_WIDTH:
                errors.append(_error("render", "actual full width mismatch"))
            if actual_size[1] % 2 != 0:
                errors.append(_error("render", "actual full height is not even"))
        except Exception as exc:
            errors.append(_error("render", f"PNG decode failed: {type(exc).__name__}: {exc}"))

    normalized: dict[str, Any] | None = None
    if not errors and declared_size is not None:
        normalized = {
            "width": NORMALIZED_WIDTH,
            "height": declared_size[1] // 2,
            "mode": "RGB",
        }

    return {
        "schema_version": INPUT_RECORD_SCHEMA,
        "sample_id": sample_id,
        "label": participant.label,
        "run_name": participant.run_name,
        "generation": {
            "fingerprint": generation_fingerprint,
            "error": generation_error,
        },
        "render": render,
        "source": source_audit,
        "preprocess_version": PREPROCESS_VERSION,
        "normalized": normalized,
        "valid": not errors,
        "errors": errors,
    }


_input_record_cache: dict[tuple[str, str], dict[str, Any]] = {}
_input_record_cache_lock = threading.Lock()


def participant_input(participant: Participant, sample_id: str) -> dict[str, Any]:
    """Validate one participant input lazily, memoized per run.

    Image bytes are not cached: each battle re-reads and re-slices so P
    participants' full-page images never stay resident in memory.
    """

    key = (participant.label, sample_id)
    with _input_record_cache_lock:
        hit = _input_record_cache.get(key)
    if hit is not None:
        return hit
    loaded = audit_participant_input(participant, sample_id)
    with _input_record_cache_lock:
        return _input_record_cache.setdefault(key, loaded)


def input_height_1x(record: dict[str, Any]) -> int | None:
    normalized = record.get("normalized")
    if not record.get("valid") or not isinstance(normalized, dict):
        return None
    height = normalized.get("height")
    return height if type(height) is int and height > 0 else None


def input_full_bytes(record: dict[str, Any]) -> int | None:
    if not record.get("valid"):
        return None
    render = record.get("render")
    if not isinstance(render, dict):
        return None
    value = render.get("full_bytes")
    return value if type(value) is int and value > 0 else None


# ---------------------------------------------------------------------------
# Formal metrics artifact (task1/2 envelope conventions)
# ---------------------------------------------------------------------------


def build_formal_artifact(
    task: str,
    metrics: dict[str, dict[str, float]],
    *,
    context: dict[str, Any],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    import math

    expected = TASK4_METRIC_BY_TASK.get(task)
    if expected is None:
        raise EvaluationError(f"unexpected task for the formal artifact: {task}")
    if set(metrics) != {expected}:
        raise EvaluationError(f"metric keys must be ['{expected}'], got {sorted(metrics)}")
    values = metrics[expected]
    if not isinstance(values, dict) or not values:
        raise EvaluationError(f"{expected} must be a non-empty label->value mapping")
    lower, upper = TASK4_METRIC_RANGES[expected]
    for label, value in values.items():
        if not isinstance(label, str) or not label:
            raise EvaluationError(f"{expected} labels must be non-empty strings")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise EvaluationError(f"{expected}[{label}] must be numeric")
        number = float(value)
        if not math.isfinite(number):
            raise EvaluationError(f"{expected}[{label}] must be finite")
        if not lower <= number <= upper:
            raise EvaluationError(
                f"{expected}[{label}] must be within [{lower}, {upper}]"
            )
    return {
        "schema_version": FORMAL_METRICS_SCHEMA,
        "task": task,
        "metrics": {expected: dict(values)},
        "metric_definitions": {expected: dict(TASK4_METRIC_DEFINITIONS[expected])},
        "context": dict(context),
        "provenance": dict(provenance),
    }


__all__ = [
    "ALLOWED_API_AUDIT_FIELDS",
    "API_RETRIES",
    "FORMAL_METRICS_SCHEMA",
    "INPUT_RECORD_SCHEMA",
    "JUDGE_EXTRA_PAYLOAD",
    "RETRY_CONTRACT",
    "RETRY_CONTRACT_VERSION",
    "TASK4_METRIC_BY_TASK",
    "TASK4_METRIC_DEFINITIONS",
    "TASK4_METRIC_RANGES",
    "EvaluationError",
    "Participant",
    "_api_audit_valid",
    "_bounded_text",
    "_error",
    "_safe_component",
    "_sanitize_api_audit",
    "append_record",
    "audit_participant_input",
    "build_formal_artifact",
    "call_judge",
    "canonical_json",
    "drop_failed_records",
    "ensure_run_meta",
    "input_full_bytes",
    "input_height_1x",
    "participant_input",
    "publish_summary",
    "read_json",
    "write_json_atomic",
]
