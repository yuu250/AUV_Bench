"""Fail-closed provenance for the hermetic Task 3 font environment.

Every Task 3 render (references, model input, patched-page grading) must run
under the pinned font set shipped in the data package.  This module proves the
environment is intact before any render starts:

* ``FONTCONFIG_FILE`` must resolve to ``<render_fonts_dir>/fonts.conf``;
* ``MANIFEST.json`` must be a strict JSON object with the expected schema;
* ``fonts/`` must contain exactly the manifest inventory (no missing, no
  extra, no symlinks), and every file's byte length must match.

Integrity is established by direct comparison (file inventory and byte
lengths), never by digest.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "task3-render-environment-provenance-v1"
POLICY_VERSION = "task3-required-local-fonts-v1"
FONT_SUFFIXES = frozenset({".ttf", ".ttc", ".otf", ".otc", ".dfont"})


class RenderEnvironmentError(RuntimeError):
    """The required local font environment cannot be proven intact."""


def _strict_json_object(path: Path) -> dict[str, Any]:
    def pairs_hook(pairs):
        output = {}
        for key, value in pairs:
            if key in output:
                raise RenderEnvironmentError(
                    f"font manifest contains duplicate key {key!r}"
                )
            output[key] = value
        return output

    def reject_constant(token: str):
        raise RenderEnvironmentError(f"font manifest contains {token}")

    try:
        value = json.loads(
            path.read_text(encoding="utf-8", errors="strict"),
            object_pairs_hook=pairs_hook,
            parse_constant=reject_constant,
        )
    except RenderEnvironmentError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RenderEnvironmentError(f"cannot parse font manifest {path}: {error}") from error
    if not isinstance(value, dict):
        raise RenderEnvironmentError("font manifest must be one JSON object")
    return value


def _relative(path: Path, anchor: Path) -> str:
    try:
        return path.relative_to(anchor).as_posix()
    except ValueError as error:
        raise RenderEnvironmentError(
            f"render environment path escapes {anchor}: {path}"
        ) from error


def _file_record(path: Path, anchor: Path) -> dict[str, object]:
    if not path.is_file():
        raise RenderEnvironmentError(f"required render environment file is missing: {path}")
    return {
        "path": _relative(path, anchor),
        "bytes": path.stat().st_size,
    }


def validate_manifest(value: object) -> list[dict[str, object]]:
    expected_keys = {"purpose", "source", "n_fonts", "families", "files"}
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise RenderEnvironmentError("font manifest has invalid top-level fields")
    if not isinstance(value["purpose"], str) or not value["purpose"]:
        raise RenderEnvironmentError("font manifest purpose is invalid")
    if not isinstance(value["source"], str) or not value["source"]:
        raise RenderEnvironmentError("font manifest source is invalid")
    n_fonts = value["n_fonts"]
    files = value["files"]
    if (
        isinstance(n_fonts, bool)
        or not isinstance(n_fonts, int)
        or n_fonts <= 0
        or not isinstance(files, list)
        or len(files) != n_fonts
    ):
        raise RenderEnvironmentError("font manifest n_fonts/files count is invalid")

    records: list[dict[str, object]] = []
    names: set[str] = set()
    for index, record in enumerate(files):
        if not isinstance(record, Mapping) or set(record) != {"file", "bytes"}:
            raise RenderEnvironmentError(f"font manifest file record {index} is invalid")
        name = record["file"]
        size = record["bytes"]
        if (
            not isinstance(name, str)
            or not name
            or Path(name).name != name
            or Path(name).suffix.lower() not in FONT_SUFFIXES
            or name in names
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size <= 0
        ):
            raise RenderEnvironmentError(f"font manifest file record {index} is invalid")
        names.add(name)
        records.append({"file": name, "bytes": size})
    if [record["file"] for record in records] != sorted(names):
        raise RenderEnvironmentError("font manifest file inventory is not sorted")

    families = value["families"]
    if not isinstance(families, Mapping) or not families:
        raise RenderEnvironmentError("font manifest families are invalid")
    for family, family_files in families.items():
        if (
            not isinstance(family, str)
            or not family
            or not isinstance(family_files, list)
            or not family_files
            or any(not isinstance(name, str) or name not in names for name in family_files)
            or len(family_files) != len(set(family_files))
        ):
            raise RenderEnvironmentError(f"font manifest family {family!r} is invalid")
    return records


def render_environment_provenance(
    *,
    render_fonts_dir: str | Path,
    environ: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Validate the required local fonts and return canonical provenance."""

    try:
        render_fonts_dir = Path(render_fonts_dir).resolve(strict=True)
    except OSError as error:
        raise RenderEnvironmentError(
            f"render environment root cannot be resolved: {error}"
        ) from error
    fonts_conf = render_fonts_dir / "fonts.conf"
    manifest_path = render_fonts_dir / "MANIFEST.json"
    fonts_dir = render_fonts_dir / "fonts"
    environment = os.environ if environ is None else environ
    configured = environment.get("FONTCONFIG_FILE")
    if not isinstance(configured, str) or not configured:
        raise RenderEnvironmentError("FONTCONFIG_FILE is required")
    try:
        configured_path = Path(configured).expanduser().resolve(strict=True)
        expected_path = fonts_conf.resolve(strict=True)
    except OSError as error:
        raise RenderEnvironmentError(f"FONTCONFIG_FILE cannot be resolved: {error}") from error
    if configured_path != expected_path:
        raise RenderEnvironmentError(
            f"FONTCONFIG_FILE must resolve to {expected_path}, got {configured_path}"
        )
    if not manifest_path.is_file():
        raise RenderEnvironmentError(f"font manifest is missing: {manifest_path}")
    if not fonts_dir.is_dir():
        raise RenderEnvironmentError(f"font directory is missing: {fonts_dir}")

    manifest = _strict_json_object(manifest_path)
    manifest_records = validate_manifest(manifest)
    font_candidates = [
        path for path in fonts_dir.iterdir() if path.suffix.lower() in FONT_SUFFIXES
    ]
    symlinked_fonts = sorted(path.name for path in font_candidates if path.is_symlink())
    if symlinked_fonts:
        raise RenderEnvironmentError(
            f"font directory contains symlinked font files: {symlinked_fonts}"
        )
    actual_paths = sorted(
        (path for path in font_candidates if path.is_file()),
        key=lambda path: path.name,
    )
    actual_names = [path.name for path in actual_paths]
    manifest_names = [str(record["file"]) for record in manifest_records]
    if actual_names != manifest_names:
        missing = sorted(set(manifest_names) - set(actual_names))
        extra = sorted(set(actual_names) - set(manifest_names))
        raise RenderEnvironmentError(
            f"font directory inventory mismatch: missing={missing}, extra={extra}"
        )

    for path, expected in zip(actual_paths, manifest_records, strict=True):
        actual = {"file": path.name, "bytes": path.stat().st_size}
        if actual != expected:
            raise RenderEnvironmentError(f"font file integrity mismatch: {path.name}")

    record: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "policy_version": POLICY_VERSION,
        "fontconfig_file": _file_record(expected_path, render_fonts_dir),
        "manifest_file": _file_record(manifest_path, render_fonts_dir),
        "font_directory": {
            "path": _relative(fonts_dir, render_fonts_dir),
            "n_fonts": len(manifest_records),
            "total_font_bytes": sum(int(row["bytes"]) for row in manifest_records),
        },
    }
    return json.loads(json.dumps(record, ensure_ascii=False, allow_nan=False))


__all__ = [
    "POLICY_VERSION",
    "SCHEMA_VERSION",
    "RenderEnvironmentError",
    "render_environment_provenance",
    "validate_manifest",
]
