#!/usr/bin/env python3
"""Model-output-independent baseline/degraded render cache for Task 3.

The cache lives at a fixed location (``results/task3_reference_cache/`` by
default).  Staleness is detected by direct structural comparison: the manifest
embeds the full build configuration (grading render policy, browser version,
per-sample input records) and the loader rebuilds that configuration from the
current checkout and requires exact equality.
"""

from __future__ import annotations

import gzip
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent

MANIFEST_SCHEMA_VERSION = "task3-repair-pass-reference-cache-manifest-v3"
BUNDLE_SCHEMA_VERSION = "task3-repair-pass-reference-cache-bundle-v3"
POLICY_VERSION = "task3-repair-pass-render-reference-open-v1"
EXPECTED_FORMAL_SAMPLES = 660
ROLES = ("baseline", "degraded")

DEFAULT_CACHE_ROOT = REPO_ROOT / "results" / "task3_reference_cache"
DEFAULT_MANIFEST = DEFAULT_CACHE_ROOT / "manifest.json"


class ReferenceCacheError(RuntimeError):
    """The shared grading-reference cache is missing, stale, or corrupt."""


def canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ReferenceCacheError(f"non-canonical JSON value: {error}") from error


def canonical_equal(left: object, right: object) -> bool:
    return canonical_bytes(left) == canonical_bytes(right)


def repo_relative(path: Path, *, repo_root: Path = REPO_ROOT) -> str:
    try:
        return Path(path).resolve(strict=True).relative_to(
            Path(repo_root).resolve(strict=True)
        ).as_posix()
    except (OSError, ValueError) as error:
        raise ReferenceCacheError(f"cache artifact escapes repository: {path}") from error


def resolved_repo_path(
    raw: object,
    *,
    repo_root: Path = REPO_ROOT,
    require_file: bool = True,
) -> Path:
    if not isinstance(raw, str) or not raw or Path(raw).is_absolute():
        raise ReferenceCacheError(f"invalid repository-relative path: {raw!r}")
    root = Path(repo_root).resolve(strict=True)
    try:
        path = (root / raw).resolve(strict=True)
        path.relative_to(root)
    except (OSError, ValueError) as error:
        raise ReferenceCacheError(f"cache path is missing or escapes repository: {raw}") from error
    if require_file and not path.is_file():
        raise ReferenceCacheError(f"cache path is not a file: {raw}")
    return path


def resolve_relocatable_path(
    raw: object,
    *,
    repo_root: Path = REPO_ROOT,
) -> Path:
    # Bundle audits record repository-relative POSIX paths.  Absolute paths
    # from a foreign checkout are rebased at the pages/ or results/ anchor.
    path = Path(str(raw))
    if not path.is_absolute():
        return (Path(repo_root) / path).resolve(strict=True)
    try:
        return path.resolve(strict=True)
    except (OSError, RuntimeError):
        parts = path.parts
        for anchor in ("pages", "results"):
            if anchor not in parts:
                continue
            index = len(parts) - 1 - parts[::-1].index(anchor)
            try:
                return (Path(repo_root) / Path(*parts[index:])).resolve(strict=True)
            except (OSError, RuntimeError):
                continue
        raise


def file_record(path: Path, *, repo_root: Path = REPO_ROOT) -> dict[str, object]:
    path = Path(path).resolve(strict=True)
    return {
        "path": repo_relative(path, repo_root=repo_root),
        "bytes": path.stat().st_size,
    }


def verify_file_record(
    record: object,
    *,
    repo_root: Path = REPO_ROOT,
) -> Path:
    if not isinstance(record, Mapping) or set(record) != {"path", "bytes"}:
        raise ReferenceCacheError("invalid cache file record")
    if type(record["bytes"]) is not int or record["bytes"] < 0:
        raise ReferenceCacheError("invalid cache file record")
    return resolved_repo_path(record["path"], repo_root=repo_root)


def strict_json(path: Path, label: str) -> dict[str, Any]:
    def pairs_hook(pairs):
        output = {}
        for key, value in pairs:
            if key in output:
                raise ReferenceCacheError(f"{label} contains duplicate key {key!r}")
            output[key] = value
        return output

    try:
        value = json.loads(
            Path(path).read_text(encoding="utf-8", errors="strict"),
            object_pairs_hook=pairs_hook,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ReferenceCacheError(f"{label} contains {token}")
            ),
        )
    except ReferenceCacheError:
        raise
    except Exception as error:  # noqa: BLE001
        raise ReferenceCacheError(f"cannot read {label}: {error}") from error
    if not isinstance(value, dict):
        raise ReferenceCacheError(f"{label} must be an object")
    return value


def read_gzip_json(path: Path, label: str) -> dict[str, Any]:
    try:
        raw = gzip.decompress(Path(path).read_bytes()).decode("utf-8", errors="strict")
        value = json.loads(raw)
    except Exception as error:  # noqa: BLE001
        raise ReferenceCacheError(f"cannot read {label}: {error}") from error
    if not isinstance(value, dict):
        raise ReferenceCacheError(f"{label} must be an object")
    return value


def write_gzip_json_atomic(path: Path, value: Mapping[str, object]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = gzip.compress(canonical_bytes(dict(value)), compresslevel=6, mtime=0)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def write_json_exclusive(path: Path, value: Mapping[str, object]) -> Path:
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise ReferenceCacheError(f"refusing to overwrite cache manifest: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        dict(value), ensure_ascii=False, sort_keys=True, indent=2
    ).encode("utf-8") + b"\n"
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise ReferenceCacheError(
                f"refusing to overwrite cache manifest: {path}"
            ) from error
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _item_input(item: object, *, repo_root: Path = REPO_ROOT) -> dict[str, object]:
    from tasks.task3 import grading as runner

    task3 = getattr(item, "task3_record", None)
    sample_id = getattr(item, "sample_id", None)
    if not isinstance(task3, Mapping) or not isinstance(sample_id, str):
        raise ReferenceCacheError("invalid formal item for reference cache")
    artifacts = task3.get("artifact_paths")
    if not isinstance(artifacts, Mapping):
        raise ReferenceCacheError(f"{sample_id}: missing Task3 artifacts")
    role_inputs = {}
    for role, html_key in (
        ("baseline", "clean_html"),
        ("degraded", "degraded_html"),
    ):
        html = Path(str(artifacts[html_key])).resolve(strict=True)
        role_inputs[role] = {
            "html": file_record(html, repo_root=repo_root),
            "relative_assets_closure": runner.selected_html_asset_closure(html),
        }
    return {
        "sample_id": sample_id,
        "viewport_width_css": int(task3["viewport"][0]),
        "dpr": float(task3["dpr"]),
        "local_render_policy": runner.grading_render_policy(),
        "roles": role_inputs,
    }


def config_mode(n_samples: int) -> str:
    return "formal" if n_samples == EXPECTED_FORMAL_SAMPLES else "subset"


def build_config(
    items: Sequence[object],
    *,
    browser_version: str,
    mode: str,
    repo_root: Path = REPO_ROOT,
) -> dict[str, object]:
    from tasks.task3 import grading as runner

    inputs = [_item_input(item, repo_root=repo_root) for item in items]
    return {
        "policy_version": POLICY_VERSION,
        "browser": {"engine": "chromium", "version": browser_version},
        "grading_render_policy": runner.grading_render_policy(),
        "mode": mode,
        "n_samples": len(items),
        "sample_ids": [str(getattr(item, "sample_id")) for item in items],
        "sample_inputs": inputs,
    }


def validate_bundle(
    bundle_path: Path,
    *,
    manifest_record: Mapping[str, object],
    item: object,
    config: Mapping[str, object],
    repo_root: Path = REPO_ROOT,
) -> dict[str, object]:
    from tasks.task3 import grading as runner
    from tasks.task3.render_transport import require_same_browser_character_set

    recorded_bundle_path = verify_file_record(
        manifest_record.get("bundle"), repo_root=repo_root
    )
    if recorded_bundle_path != Path(bundle_path).resolve(strict=True):
        raise ReferenceCacheError("reference bundle path mismatch")
    bundle = read_gzip_json(bundle_path, "reference bundle")
    sample_id = str(getattr(item, "sample_id"))
    expected_input = _item_input(item, repo_root=repo_root)
    if (
        bundle.get("schema_version") != BUNDLE_SCHEMA_VERSION
        or bundle.get("policy_version") != POLICY_VERSION
        or bundle.get("sample_id") != sample_id
        or not canonical_equal(bundle.get("input"), expected_input)
    ):
        raise ReferenceCacheError(f"{sample_id}: reference bundle identity drift")
    roles = bundle.get("roles")
    screenshots = manifest_record.get("screenshots")
    if not isinstance(roles, Mapping) or set(roles) != set(ROLES):
        raise ReferenceCacheError(f"{sample_id}: incomplete reference roles")
    if not isinstance(screenshots, Mapping) or set(screenshots) != set(ROLES):
        raise ReferenceCacheError(f"{sample_id}: incomplete reference screenshots")
    expected_policy = config.get("grading_render_policy")
    if not isinstance(expected_policy, Mapping):
        raise ReferenceCacheError(f"{sample_id}: local render policy is missing")
    screenshot_paths = {}
    screenshot_records = {}
    audits = {}
    for role in ROLES:
        payload = roles[role]
        if not isinstance(payload, Mapping) or set(payload) != {
            "element_index", "grading_render"
        }:
            raise ReferenceCacheError(f"{sample_id}/{role}: invalid reference payload")
        element_index = payload["element_index"]
        audit = payload["grading_render"]
        if not isinstance(element_index, Mapping) or not isinstance(audit, Mapping):
            raise ReferenceCacheError(f"{sample_id}/{role}: invalid reference objects")
        screenshot_record = screenshots[role]
        screenshot_path = verify_file_record(screenshot_record, repo_root=repo_root)
        screenshot = audit.get("screenshot")
        html_record = expected_input["roles"][role]["html"]
        html_path = verify_file_record(html_record, repo_root=repo_root)
        try:
            expected_width = runner.expected_full_page_physical_width(
                audit.get("css_width"), audit.get("scroll_width_css"), audit.get("dpr")
            )
            audit_html_path = resolve_relocatable_path(
                audit.get("html_path", ""), repo_root=repo_root
            )
            audit_screenshot_path = resolve_relocatable_path(
                screenshot.get("path", "") if isinstance(screenshot, Mapping) else "",
                repo_root=repo_root,
            )
        except Exception as error:  # noqa: BLE001
            raise ReferenceCacheError(
                f"{sample_id}/{role}: invalid local render geometry or path"
            ) from error
        if (
            audit.get("policy_version") != runner.GRADING_RENDER_POLICY_VERSION
            or audit.get("render_environment")
            != expected_policy.get("render_environment")
            or audit.get("role") != role
            or audit.get("browser_version") != config["browser"]["version"]
            or audit.get("css_width") != expected_input["viewport_width_css"]
            or audit.get("dpr") != expected_input["dpr"]
            or audit.get("expected_physical_width") != expected_width
            or audit_html_path != html_path
            or not isinstance(screenshot, Mapping)
            or audit_screenshot_path != screenshot_path
            or screenshot.get("png_bytes") != screenshot_record.get("bytes")
            or screenshot.get("format") != "PNG"
            or not isinstance(screenshot.get("size_px"), list)
            or len(screenshot["size_px"]) != 2
            or screenshot["size_px"][0] != expected_width
            or type(screenshot["size_px"][1]) is not int
            or screenshot["size_px"][1] <= 0
        ):
            raise ReferenceCacheError(f"{sample_id}/{role}: render provenance drift")
        if not isinstance(element_index.get("elements"), list):
            raise ReferenceCacheError(f"{sample_id}/{role}: element index missing")
        screenshot_paths[role] = screenshot_path
        screenshot_records[role] = dict(screenshot_record)
        audits[role] = {
            **audit,
            "html_path": audit_html_path.as_posix(),
            "screenshot": {**screenshot, "path": audit_screenshot_path.as_posix()},
        }
        # Downstream grading reads the parsed bundle's grading_render directly,
        # so normalize its paths to this checkout the same way.  The bundle is
        # a freshly parsed object here; in-place rewriting shares no state.
        bundle_grading_render = bundle["roles"][role]["grading_render"]
        bundle_grading_render["html_path"] = audit_html_path.as_posix()
        bundle_grading_render["screenshot"]["path"] = audit_screenshot_path.as_posix()
    if audits["baseline"].get("render_contract") != audits["degraded"].get(
        "render_contract"
    ):
        raise ReferenceCacheError(f"{sample_id}: cached role contracts differ")
    request_section = bundle.get("request")
    if not isinstance(request_section, Mapping) or set(request_section) != {
        "target_catalog",
        "catalog_navigation_audit",
        "local_model_input",
    }:
        raise ReferenceCacheError(f"{sample_id}: frozen request section is invalid")
    local_input = request_section["local_model_input"]
    degraded_size = audits["degraded"]["screenshot"]["size_px"]
    try:
        from tasks.task3.target_ids import validate_pagewide_target_catalog

        validate_pagewide_target_catalog(request_section["target_catalog"])
        navigation = request_section["catalog_navigation_audit"]
        font_mode = navigation.get("font_mode") if isinstance(navigation, Mapping) else None
        if (
            not isinstance(navigation, Mapping)
            or not isinstance(navigation.get("original"), Mapping)
            or not isinstance(navigation.get("marker"), Mapping)
            or not isinstance(font_mode, Mapping)
            or font_mode.get("mode") != "required-local"
        ):
            raise ValueError("catalog navigation audit is incomplete")
        geometry = local_input["geometry"]
        if (
            local_input["viewport"][0] != expected_input["viewport_width_css"]
            or float(local_input["dpr"]) != expected_input["dpr"]
            or geometry["canvas_width"] != degraded_size[0]
            or geometry["common_height"] != degraded_size[1]
            or geometry["n_slices"] != len(geometry["slice_offsets"])
        ):
            raise ValueError("local model input geometry drifted")
        if local_input["screenshot"] != dict(screenshot_records["degraded"]):
            raise ValueError("local model input screenshot record drifted")
    except (KeyError, TypeError, ValueError, IndexError) as error:
        raise ReferenceCacheError(
            f"{sample_id}: frozen request materials are invalid: {error}"
        ) from error
    try:
        local_render_gate = runner.require_local_grading_renders({
            "policy": dict(expected_policy),
            **audits,
        })
        require_same_browser_character_set(
            {
                role: audits[role]["navigation_audit"]
                for role in ROLES
            },
            required_labels=ROLES,
        )
    except Exception as error:  # noqa: BLE001
        raise ReferenceCacheError(
            f"{sample_id}: authoritative local render gate failed: {error}"
        ) from error
    return {
        "bundle": bundle,
        "screenshot_paths": screenshot_paths,
        "screenshot_records": screenshot_records,
        "grading_render_audits": audits,
        "local_render_gate": local_render_gate,
        "manifest_record": dict(manifest_record),
    }


def load_bundle(
    bundle_path: Path,
    *,
    manifest_record: Mapping[str, object],
    repo_root: Path = REPO_ROOT,
) -> dict[str, object]:
    """Light loader for the direct run path: existence + parse only."""
    bundle = read_gzip_json(bundle_path, "reference bundle")
    roles = bundle.get("roles")
    if not isinstance(roles, Mapping) or set(roles) != set(ROLES):
        raise ReferenceCacheError("incomplete reference roles")
    for role in ROLES:
        payload = roles[role]
        if not isinstance(payload, Mapping) or set(payload) != {
            "element_index", "grading_render"
        }:
            raise ReferenceCacheError(f"{role}: invalid reference payload")
        if not isinstance(payload["element_index"], Mapping):
            raise ReferenceCacheError(f"{role}: element index missing")
    request_section = bundle.get("request")
    if not isinstance(request_section, Mapping) or set(request_section) != {
        "target_catalog",
        "catalog_navigation_audit",
        "local_model_input",
    }:
        raise ReferenceCacheError("frozen request section is missing or invalid")
    screenshots = manifest_record.get("screenshots")
    if not isinstance(screenshots, Mapping) or set(screenshots) != set(ROLES):
        raise ReferenceCacheError("incomplete reference screenshots")
    screenshot_paths = {
        role: resolved_repo_path(screenshots[role].get("path"), repo_root=repo_root)
        for role in ROLES
    }
    return {
        "bundle": bundle,
        "screenshot_paths": screenshot_paths,
        "manifest_record": dict(manifest_record),
    }


def load_manifest_subset(
    path: Path,
    *,
    items: Sequence[object],
    browser_version: str,
    repo_root: Path = REPO_ROOT,
) -> tuple[dict[str, object], dict[str, Mapping[str, object]]]:
    """Subset-run loader: the cache may cover more samples than this run.

    The build-invariant fields (policy version, browser, grading render
    policy) must equal this checkout's recomputation, and every selected
    sample's frozen inputs must match ``_item_input`` exactly.  Whole-config
    equality is only required for full-cohort runs (``load_manifest``).
    """
    manifest = strict_json(path, "reference cache manifest")
    records = manifest.get("records")
    if (
        manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION
        or manifest.get("phase") != "offline_reference_cache"
        or manifest.get("decision") != "pass"
        or manifest.get("policy_version") != POLICY_VERSION
        or not isinstance(records, list)
    ):
        raise ReferenceCacheError("reference cache manifest is stale or incomplete")
    config = manifest.get("config")
    if not isinstance(config, Mapping):
        raise ReferenceCacheError("reference cache config is missing")
    if config.get("policy_version") != POLICY_VERSION or config.get("browser") != {
        "engine": "chromium",
        "version": browser_version,
    }:
        raise ReferenceCacheError("reference cache build environment differs")
    from tasks.task3 import grading as runner

    if not canonical_equal(
        config.get("grading_render_policy"), runner.grading_render_policy()
    ):
        raise ReferenceCacheError(
            "reference cache render policy differs from this checkout"
        )
    sample_inputs = config.get("sample_inputs")
    if not isinstance(sample_inputs, list):
        raise ReferenceCacheError("reference cache config lacks sample_inputs")
    inputs_by_id = {
        entry["sample_id"]: entry
        for entry in sample_inputs
        if isinstance(entry, Mapping) and isinstance(entry.get("sample_id"), str)
    }
    by_id: dict[str, Mapping[str, object]] = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise ReferenceCacheError("reference cache manifest has a non-object record")
        sample_id = record.get("sample_id")
        if not isinstance(sample_id, str) or sample_id in by_id:
            raise ReferenceCacheError("reference cache sample IDs are invalid")
        by_id[sample_id] = record
    for item in items:
        sample_id = str(getattr(item, "sample_id"))
        if sample_id not in by_id:
            raise ReferenceCacheError(
                f"{sample_id}: missing from the reference cache"
            )
        if not canonical_equal(
            inputs_by_id.get(sample_id), _item_input(item, repo_root=repo_root)
        ):
            raise ReferenceCacheError(
                f"{sample_id}: reference cache inputs differ from this checkout"
            )
    return manifest, by_id


def build_manifest(
    *,
    config: Mapping[str, object],
    records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    ordered = sorted((dict(record) for record in records), key=lambda row: row["sample_id"])
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "phase": "offline_reference_cache",
        "decision": "pass",
        "policy_version": POLICY_VERSION,
        "config": dict(config),
        "n_samples": len(ordered),
        "records": ordered,
    }


def load_manifest(
    path: Path,
    *,
    items: Sequence[object],
    browser_version: str,
    require_formal: bool,
    expected_config: Mapping[str, object] | None = None,
    repo_root: Path = REPO_ROOT,
) -> tuple[dict[str, object], dict[str, Mapping[str, object]]]:
    manifest = strict_json(path, "reference cache manifest")
    records = manifest.get("records")
    if (
        manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION
        or manifest.get("phase") != "offline_reference_cache"
        or manifest.get("decision") != "pass"
        or manifest.get("policy_version") != POLICY_VERSION
        or not isinstance(records, list)
        or manifest.get("n_samples") != len(items)
        or (require_formal and len(items) != EXPECTED_FORMAL_SAMPLES)
    ):
        raise ReferenceCacheError("reference cache manifest is stale or incomplete")
    if expected_config is not None and not canonical_equal(
        manifest.get("config"), expected_config
    ):
        raise ReferenceCacheError(
            "reference cache config differs from the current environment"
        )
    by_id: dict[str, Mapping[str, object]] = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise ReferenceCacheError("reference cache manifest has a non-object record")
        sample_id = record.get("sample_id")
        if not isinstance(sample_id, str) or sample_id in by_id:
            raise ReferenceCacheError("reference cache sample IDs are invalid")
        by_id[sample_id] = record
    expected_ids = [str(getattr(item, "sample_id")) for item in items]
    if list(by_id) != sorted(expected_ids):
        raise ReferenceCacheError(
            "reference cache sample IDs differ or records are not canonical"
        )
    return manifest, by_id


__all__ = [
    "BUNDLE_SCHEMA_VERSION",
    "DEFAULT_CACHE_ROOT",
    "DEFAULT_MANIFEST",
    "EXPECTED_FORMAL_SAMPLES",
    "MANIFEST_SCHEMA_VERSION",
    "POLICY_VERSION",
    "ROLES",
    "ReferenceCacheError",
    "build_config",
    "build_manifest",
    "canonical_bytes",
    "canonical_equal",
    "config_mode",
    "file_record",
    "load_bundle",
    "load_manifest",
    "load_manifest_subset",
    "read_gzip_json",
    "repo_relative",
    "resolve_relocatable_path",
    "resolved_repo_path",
    "strict_json",
    "validate_bundle",
    "verify_file_record",
    "write_gzip_json_atomic",
    "write_json_exclusive",
]
