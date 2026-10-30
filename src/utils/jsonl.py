"""JSONL helpers: manifest loading, prediction appending, resume bookkeeping."""

from __future__ import annotations

import json
from pathlib import Path
from typing import IO, Any


class ManifestError(ValueError):
    pass


def load_manifest(path: str | Path) -> list[dict[str, Any]]:
    """Load the sample manifest.

    Each line must be a JSON object with a unique non-empty ``id`` and either
    ``image`` (path relative to the images root) or ``image_url`` (an
    already-hosted remote image).
    """
    items: list[dict[str, Any]] = []
    seen: set[str] = set()
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ManifestError(
                    f"manifest line {line_number} is invalid JSON"
                ) from error
            if not isinstance(record, dict):
                raise ManifestError(f"manifest line {line_number} must be an object")
            sample_id = record.get("id")
            if not isinstance(sample_id, str) or not sample_id:
                raise ManifestError(f"manifest line {line_number} has invalid id")
            if sample_id in seen:
                raise ManifestError(f"manifest contains duplicate id: {sample_id}")
            if not record.get("image") and not record.get("image_url"):
                raise ManifestError(
                    f"manifest line {line_number} ({sample_id}) needs image or image_url"
                )
            seen.add(sample_id)
            items.append(record)
    if not items:
        raise ManifestError("manifest is empty")
    return items


def load_done_ids(output_path: str | Path) -> set[str]:
    """Ids already finished in a previous run.

    A record counts as done when it carries scores or was excluded by the
    slicing protocol; failed records are retried on the next run.
    """
    done: set[str] = set()
    path = Path(output_path)
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "id" in record and (
                record.get("scores")
                or record.get("error") == "excluded_too_many_slices"
            ):
                done.add(record["id"])
    return done


def append_record(handle: IO[str], record: dict[str, Any]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    handle.flush()
