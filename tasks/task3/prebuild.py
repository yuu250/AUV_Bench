#!/usr/bin/env python3
"""Build the shipped Task 3 reference cache in one deterministic pass.

For every formal cohort sample this renders the frozen baseline and degraded
HTML under the authoritative local grading contract, extracts the GT-blind
page-wide target catalog from the degraded page, derives the local model
input, and freezes everything into one gzip bundle per sample plus a
top-level manifest.  A bundle therefore contains every model-independent
artifact the direct chain needs; ``run.py`` never re-renders request
materials.

The manifest is written with exclusive-create semantics so a completed cache
can never silently mix artifacts from two environments; an interrupted build
(no manifest yet) resumes per-sample after re-validating any existing bundle.
After writing, every bundle is re-read and validated against the freshly
computed build configuration before the command reports success.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from tasks.task3 import cohort as cohort_mod
from tasks.task3 import grading
from tasks.task3 import reference_cache
from tasks.task3.fontenv import configure_render_environment
from tasks.task3.grading import Task3Error
from tasks.task3.render_transport import CHROMIUM_RENDER_ARGS
from tasks.task3.visual_scope import slice_offsets

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent

DEFAULT_COHORT = REPO_ROOT / "data" / "task3" / "cohort.json"
DEFAULT_SPECS = REPO_ROOT / "data" / "task3" / "defect_specs.jsonl"
DEFAULT_PAGES_ROOT = REPO_ROOT / "data" / "task3"

DEFAULT_WORKERS = 4
DEFAULT_RETRIES = 3


class PrebuildError(Task3Error):
    pass


def _local_model_input_for_build(
    item: cohort_mod.FormalItem,
    degraded_audit: dict,
    screenshot_record: dict,
) -> dict:
    size = degraded_audit["screenshot"]["size_px"]
    offsets = slice_offsets(
        size[1],
        slice_height_px=grading.SLICE_H,
        slice_overlap_px=grading.OVERLAP,
    )
    return {
        "policy_version": grading.LOCAL_MODEL_INPUT_POLICY_VERSION,
        "screenshot": dict(screenshot_record),
        "geometry": {
            "policy": grading.LOCAL_MODEL_INPUT_POLICY_VERSION,
            "canvas_width": size[0],
            "common_height": size[1],
            "full_height": size[1],
            "slice_offsets": offsets,
            "n_slices": len(offsets),
        },
        "viewport": [item.viewport[0], item.viewport[1]],
        "dpr": float(item.dpr),
    }


async def _render_role(
    browser,
    *,
    html_path: Path,
    css_width: int,
    dpr: float,
    screenshot_path: Path,
    role: str,
    retries: int,
) -> tuple[dict, dict]:
    errors = []
    for attempt in range(1, retries + 1):
        try:
            return await grading.grading_render_scan_capture(
                browser,
                html_path,
                css_width=css_width,
                dpr=dpr,
                screenshot_path=screenshot_path,
                role=role,
            )
        except Exception as error:  # noqa: BLE001
            errors.append(f"{type(error).__name__}: {error}"[:500])
            if attempt == retries:
                raise PrebuildError(
                    f"{role} render failed after {retries} attempts: {errors}"
                ) from error
    raise AssertionError("unreachable")


async def _build_one_fresh(
    browser,
    item: cohort_mod.FormalItem,
    *,
    cache_root: Path,
    retries: int,
) -> dict:
    sample_id = item.sample_id
    bundle_path = cache_root / "bundles" / f"{sample_id}.json.gz"
    screenshot_paths = {
        role: cache_root / "screenshots" / f"{sample_id}_{role}.png"
        for role in reference_cache.ROLES
    }
    roles: dict[str, dict] = {}
    screenshot_records: dict[str, dict] = {}
    audits: dict[str, dict] = {}
    for role, html_key in (
        ("baseline", "clean_html"),
        ("degraded", "degraded_html"),
    ):
        html_path = Path(item.task3_record["artifact_paths"][html_key])
        screenshot_path = screenshot_paths[role]
        element_index_payload, audit = await _render_role(
            browser,
            html_path=html_path,
            css_width=item.viewport[0],
            dpr=float(item.dpr),
            screenshot_path=screenshot_path,
            role=role,
            retries=retries,
        )
        roles[role] = {
            "element_index": element_index_payload,
            "grading_render": audit,
        }
        screenshot_records[role] = reference_cache.file_record(screenshot_path)
        audits[role] = audit

    request = item.request
    if request is None:
        raise PrebuildError(f"{sample_id}: cohort item lacks a request")
    source_html = Path(
        item.task3_record["artifact_paths"]["degraded_html"]
    ).read_text(encoding="utf-8")
    target_catalog, catalog_navigation_audit = await grading.extract_target_catalog(
        browser, request, source_html
    )
    local_model_input = _local_model_input_for_build(
        item, audits["degraded"], screenshot_records["degraded"]
    )

    bundle = {
        "schema_version": reference_cache.BUNDLE_SCHEMA_VERSION,
        "policy_version": reference_cache.POLICY_VERSION,
        "sample_id": sample_id,
        "input": reference_cache._item_input(item),
        "roles": roles,
        "request": {
            "target_catalog": target_catalog,
            "catalog_navigation_audit": catalog_navigation_audit,
            "local_model_input": local_model_input,
        },
    }
    reference_cache.write_gzip_json_atomic(bundle_path, bundle)
    return {
        "sample_id": sample_id,
        "bundle": reference_cache.file_record(bundle_path),
        "screenshots": screenshot_records,
    }


async def build_one(
    browser,
    item: cohort_mod.FormalItem,
    *,
    cache_root: Path,
    config: dict,
    semaphore: asyncio.Semaphore,
    retries: int,
) -> dict:
    async with semaphore:
        sample_id = item.sample_id
        bundle_path = cache_root / "bundles" / f"{sample_id}.json.gz"
        screenshot_paths = {
            role: cache_root / "screenshots" / f"{sample_id}_{role}.png"
            for role in reference_cache.ROLES
        }
        if bundle_path.is_file() and all(
            path.is_file() for path in screenshot_paths.values()
        ):
            record = {
                "sample_id": sample_id,
                "bundle": reference_cache.file_record(bundle_path),
                "screenshots": {
                    role: reference_cache.file_record(path)
                    for role, path in screenshot_paths.items()
                },
            }
            try:
                reference_cache.validate_bundle(
                    bundle_path,
                    manifest_record=record,
                    item=item,
                    config=config,
                )
            except reference_cache.ReferenceCacheError:
                pass
            else:
                return record
        errors = []
        for attempt in range(1, retries + 1):
            try:
                return await _build_one_fresh(
                    browser, item, cache_root=cache_root, retries=retries
                )
            except Exception as error:  # noqa: BLE001
                errors.append(f"{type(error).__name__}: {error}"[:500])
                if attempt == retries:
                    raise PrebuildError(
                        f"{sample_id}: build failed after {retries} attempts: "
                        f"{errors}"
                    ) from error
        raise AssertionError("unreachable")


async def build_cache(
    items: list[cohort_mod.FormalItem],
    *,
    cache_root: Path,
    workers: int,
    retries: int,
) -> Path:
    if workers < 1:
        raise PrebuildError("workers must be positive")
    if retries < 1:
        raise PrebuildError("retries must be positive")
    manifest_path = cache_root / "manifest.json"
    if manifest_path.exists():
        raise PrebuildError(
            f"cache manifest already exists: {manifest_path}; remove the cache "
            "root to rebuild from scratch"
        )
    (cache_root / "bundles").mkdir(parents=True, exist_ok=True)
    (cache_root / "screenshots").mkdir(parents=True, exist_ok=True)

    from playwright.async_api import async_playwright

    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(
            headless=True, args=list(CHROMIUM_RENDER_ARGS)
        )
        try:
            config = reference_cache.build_config(
                items,
                browser_version=browser.version,
                mode=reference_cache.config_mode(len(items)),
            )
            semaphore = asyncio.Semaphore(workers)
            records = await asyncio.gather(*(
                build_one(
                    browser,
                    item,
                    cache_root=cache_root,
                    config=config,
                    semaphore=semaphore,
                    retries=retries,
                )
                for item in items
            ))
        finally:
            await browser.close()

    manifest = reference_cache.build_manifest(config=config, records=records)
    reference_cache.write_json_exclusive(manifest_path, manifest)
    return manifest_path


def revalidate_cache(
    items: list[cohort_mod.FormalItem],
    *,
    cache_root: Path,
    browser_version: str,
) -> None:
    """Fail-closed read-back: every bundle must survive the runtime loader."""

    config = reference_cache.build_config(
        items,
        browser_version=browser_version,
        mode=reference_cache.config_mode(len(items)),
    )
    manifest_path = cache_root / "manifest.json"
    _, records_by_id = reference_cache.load_manifest(
        manifest_path,
        items=items,
        browser_version=browser_version,
        require_formal=len(items) == reference_cache.EXPECTED_FORMAL_SAMPLES,
        expected_config=config,
    )
    for item in items:
        record = records_by_id[item.sample_id]
        bundle_path = reference_cache.resolved_repo_path(record["bundle"]["path"])
        reference_cache.validate_bundle(
            bundle_path,
            manifest_record=record,
            item=item,
            config=config,
        )


async def _browser_version() -> str:
    from playwright.async_api import async_playwright

    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch()
        try:
            return browser.version
        finally:
            await browser.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, default=DEFAULT_COHORT)
    parser.add_argument("--specs", type=Path, default=DEFAULT_SPECS)
    parser.add_argument("--pages-root", type=Path, default=DEFAULT_PAGES_ROOT)
    parser.add_argument(
        "--cache-root", type=Path, default=reference_cache.DEFAULT_CACHE_ROOT
    )
    parser.add_argument("--sample-id", action="append", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--retries", type=int, default=DEFAULT_RETRIES)
    args = parser.parse_args(argv)
    configure_render_environment()

    items = cohort_mod.build_source_items(
        cohort_path=args.cohort,
        specs_path=args.specs,
        pages_root=args.pages_root,
    )
    items = cohort_mod.select_items(
        items, sample_ids=args.sample_id, limit=args.limit
    )
    cache_root = args.cache_root.resolve()
    try:
        cache_root.relative_to(REPO_ROOT)
    except ValueError:
        raise PrebuildError("cache root must live inside the repository")

    print(f"prebuild: {len(items)} sample(s) -> {cache_root}", flush=True)
    manifest_path = asyncio.run(
        build_cache(
            items,
            cache_root=cache_root,
            workers=args.workers,
            retries=args.retries,
        )
    )
    print("prebuild: renders frozen; re-validating read-back", flush=True)
    browser_version = asyncio.run(_browser_version())
    revalidate_cache(items, cache_root=cache_root, browser_version=browser_version)
    print(f"prebuild: OK — manifest at {manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (Task3Error, reference_cache.ReferenceCacheError) as exc:
        print(f"prebuild failed: {exc}", file=sys.stderr)
        sys.exit(1)
