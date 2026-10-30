#!/usr/bin/env python3
"""Task 3 inference: structured repair of the degraded page (direct chain).

Run from the repository root (after ``python -m tasks.task3.prebuild`` and
``python -m tasks.task3.build_gt`` have produced the frozen data)::

    python -m tasks.task3.run --config configs/task3.yaml
    python -m tasks.task3.run --config configs/task3.yaml --limit 5

Every formal sample is bound to the benchmark's actual degraded page.  A
valid five-field Task 2 handoff from the advisory predictions file is shown
to the model as advisory evidence; a missing or unusable advisory row falls
back to ``{}`` and never changes the page, the editable targets, or the
scoring scope.

Each sample makes exactly one API call (transport-level retries only).
Records are appended to the result JSONL one by one and every written
record is terminal — rerunning the same command resumes by skipping
existing sample IDs.  Scoring is fixed-denominator: API failures, provider
output failures, protocol failures, and grading failures all stay in the
cohort and count as ``repair_pass = false``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from src.api.client import make_client
from src.utils.config import ConfigError, load_config, require_mapping, resolve_api_key
from tasks.task3 import cohort as cohort_mod
from tasks.task3 import grading
from tasks.task3 import reference_cache
from tasks.task3.render_transport import CHROMIUM_RENDER_ARGS
from tasks.task3.fontenv import configure_render_environment
from tasks.task3.grading import Task3Error
from tasks.task3.html_evidence import DEFAULT_MAX_MODEL_HTML_CHARS

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent

def _write_run_meta(path: Path, *, model: str, protocol: str) -> None:
    payload = {
        "model": model,
        "protocol": protocol,
        "eval_version": grading.EVAL_VERSION,
        "result_schema_version": grading.RESULT_SCHEMA_VERSION,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        stored = json.loads(path.read_text(encoding="utf-8"))
        if stored != payload:
            raise ConfigError(
                f"{path} was produced by a different model/protocol/version; "
                "use a fresh results file to avoid mixing runs"
            )
        return
    grading.atomic_write_json(path, payload)


async def run(cfg: dict) -> None:
    configure_render_environment()
    input_cfg = require_mapping(cfg, "input")
    request_cfg = require_mapping(cfg, "request")
    output_cfg = require_mapping(cfg, "output")
    model = cfg["model"]
    protocol = cfg.get("protocol", "openai")

    advisory_raw = input_cfg.get("advisory_predictions")
    advisory_path = Path(advisory_raw) if advisory_raw else None
    items, _ = cohort_mod.build_formal_items(
        cohort_path=Path(input_cfg["cohort"]),
        gt_path=Path(input_cfg["gt"]),
        gt_metadata_path=Path(input_cfg["gt_metadata"]),
        pages_root=Path(input_cfg["pages_root"]),
        advisory_path=advisory_path,
    )
    limit = cfg.get("limit") or 0
    if limit > 0:
        items = cohort_mod.select_items(items, limit=limit)
    sample_ids = [item.sample_id for item in items]
    n_valid_handoffs = sum(
        1 for item in items if item.task2_handoff is not None
    )
    if advisory_path:
        print(
            f"[INFO] {len(items)} samples | advisory: {n_valid_handoffs} "
            f"trusted handoffs from {advisory_path}",
            flush=True,
        )
    else:
        print(
            f"[INFO] {len(items)} samples | no advisory predictions configured",
            flush=True,
        )

    output_path = Path(output_cfg["results"].format(model=model))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    runs_root = Path(output_cfg.get("runs_dir", "outputs/task3/runs"))
    run_meta_path = output_path.with_suffix(".run_meta.json")
    _write_run_meta(run_meta_path, model=model, protocol=protocol)

    existing = grading.load_result_jsonl(output_path, selected_ids=sample_ids)
    todo = [item for item in items if item.sample_id not in existing]
    if existing:
        print(
            f"[INFO] resuming: {len(existing)} terminal records skipped",
            flush=True,
        )
    print(f"[INFO] {len(todo)} samples to repair | model {model}", flush=True)

    cache_root = Path(input_cfg.get(
        "cache_root", reference_cache.DEFAULT_CACHE_ROOT
    ))
    max_model_html_chars = int(
        request_cfg.get("max_model_html_chars", DEFAULT_MAX_MODEL_HTML_CHARS)
    )
    retries = int(request_cfg.get("max_retries", 3))
    concurrency = int(request_cfg.get("concurrency", 4))

    from playwright.async_api import async_playwright

    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(
            headless=True, args=list(CHROMIUM_RENDER_ARGS)
        )
        try:
            if len(items) == reference_cache.EXPECTED_FORMAL_SAMPLES:
                expected_config = reference_cache.build_config(
                    items,
                    browser_version=browser.version,
                    mode=reference_cache.config_mode(len(items)),
                )
                _, records_by_id = reference_cache.load_manifest(
                    cache_root / "manifest.json",
                    items=items,
                    browser_version=browser.version,
                    require_formal=True,
                    expected_config=expected_config,
                )
            else:
                _, records_by_id = reference_cache.load_manifest_subset(
                    cache_root / "manifest.json",
                    items=items,
                    browser_version=browser.version,
                )
        except reference_cache.ReferenceCacheError as exc:
            raise ConfigError(
                f"reference cache is missing or stale: {exc}; "
                "run python -m tasks.task3.prebuild"
            ) from exc

        if not todo:
            records = [existing[sample_id] for sample_id in sample_ids]
            _write_summary(output_path, records)
            return

        client = make_client(
            protocol,
            base_url=cfg.get("base_url"),
            api_key=resolve_api_key(cfg),
            timeout=float(request_cfg.get("timeout", 600)),
        )
        semaphore = asyncio.Semaphore(concurrency)
        total = len(todo)
        counts = {"done": 0, "failures": 0, "repair_pass": 0}

        async def process(item: cohort_mod.FormalItem) -> None:
            async with semaphore:
                manifest_record = records_by_id[item.sample_id]
                reference = reference_cache.load_bundle(
                    reference_cache.resolved_repo_path(
                        manifest_record["bundle"]["path"]
                    ),
                    manifest_record=manifest_record,
                )
                request_materials = grading.request_materials_from_reference(
                    reference
                )
                prepared = grading.prepare_request(
                    item.request,
                    provider=protocol,
                    max_model_html_chars=max_model_html_chars,
                    request_materials=request_materials,
                )
                run_dir = runs_root / model / item.sample_id
                record = await grading.run_one_model_sample(
                    browser,
                    item,
                    prepared,
                    reference,
                    client=client,
                    model=model,
                    run_dir=run_dir,
                    retries=retries,
                )
                grading.append_result_record(output_path, record)
                counts["done"] += 1
                if record.get("status") != "complete":
                    counts["failures"] += 1
                if record.get(grading.METRIC_MAIN) is True:
                    counts["repair_pass"] += 1
                tag = "PASS" if record.get(grading.METRIC_MAIN) else "fail"
                print(
                    f"  [{counts['done']}/{total}] {item.sample_id} "
                    f"{record.get('status')} {tag}",
                    flush=True,
                )
                if counts["done"] % 20 == 0 or counts["done"] == total:
                    print(
                        f"[PROGRESS] {counts['done']}/{total} done, "
                        f"{counts['failures']} non-complete, "
                        f"{counts['repair_pass']} repair passes",
                        flush=True,
                    )

        await asyncio.gather(*(process(item) for item in todo))

    records = grading.load_result_jsonl(output_path, selected_ids=sample_ids)
    _write_summary(output_path, [records[sample_id] for sample_id in sample_ids])
    print(
        f"[DONE] {counts['done']} repaired this invocation | "
        f"results: {output_path}",
        flush=True,
    )


def _write_summary(output_path: Path, records: list[dict]) -> None:
    summary = grading.expected_summary(out_path=output_path, records=records)
    summary_path = output_path.with_suffix(".summary.json")
    grading.atomic_write_json(summary_path, summary)
    print(
        f"[DONE] repair_pass_rate "
        f"{summary['repair_pass_count']}/{summary['n']} "
        f"= {summary['repair_pass_rate']:.4f} | summary: {summary_path}",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/task3.yaml")
    parser.add_argument("--model", default=None, help="override the model name")
    parser.add_argument("--limit", type=int, default=0,
                        help="repair at most this many samples (0 = all)")
    parser.add_argument("--concurrency", type=int, default=None,
                        help="override request.concurrency")
    parser.add_argument("--output", default=None,
                        help="override output.results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.model:
        cfg["model"] = args.model
    if args.limit:
        cfg["limit"] = args.limit
    if args.concurrency:
        require_mapping(cfg, "request")["concurrency"] = args.concurrency
    if args.output:
        require_mapping(cfg, "output")["results"] = args.output
    try:
        asyncio.run(run(cfg))
    except (ConfigError, Task3Error, cohort_mod.CohortError) as exc:
        print(f"[FATAL] {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
