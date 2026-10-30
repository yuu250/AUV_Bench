#!/usr/bin/env python3
"""Task 4 generation: 1,395 deduped queries -> single-file HTML per sample.

Run from the repository root:

    python -m tasks.task4.run --config configs/task4.yaml
    python -m tasks.task4.run --config configs/task4.yaml --limit 5 --dry-run

Each sample sends the frozen request (English query + verbatim copy block +
compact Top-K caption JSON + up to two 4x4 contact sheets) with thinking
disabled and temperature 0.  Truncated answers (missing </html>), asset-policy
violations, and API errors retry on the frozen [5, 15, 45] second ladder, at
most request.max_retries attempts.  Rerunning resumes: samples whose
answer.html is complete and passes the output audit are skipped.

Run-level artifacts under outputs/task4/runs/<run_name>/:
_cohort.json (fail-closed cohort identity), _meta.jsonl (one row per sample,
atomically rewritten at the end), _events.jsonl (append-only event stream),
_errors.jsonl.  No digests are recorded anywhere; identity uses byte counts.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from pathlib import Path

from src.api.client import make_client
from src.utils.config import ConfigError, load_config, require_mapping, resolve_api_key
from tasks.task4.cohort import (
    DEFAULT_QUERIES_PATH,
    EvaluationError,
    load_query_records,
)
from tasks.task4.inputs import (
    build_topk_request,
    can_skip_topk,
    load_latest_records,
    materialize_topk_assets,
    merge_result_records,
)
from tasks.task4.audit import validate_topk_output
from tasks.task4.prompts import LANG_NAME, OUTPUT_AUDIT_PROTOCOL

RETRY_WAITS = [5, 15, 45]
ANTHROPIC_MAX_TOKENS = 32768  # full-page HTML peaks around 15k tokens
RUN_NAME_PATTERN = re.compile(r"[A-Za-z0-9._-]+")


def extract_html(text: str) -> str:
    match = re.search(r"```html\s*(.*?)```", text, re.S | re.I)
    if match:
        return match.group(1).strip()
    return re.sub(r"^\s*```(?:html)?\s*\n", "", text, flags=re.I).strip()  # truncated fence


def is_complete(html: str) -> bool:
    return "</html>" in html[-2000:].lower()


async def gen_sample(
    client,
    cfg: dict,
    rec: dict,
    run_dir: Path,
    inputs_root: Path,
    *,
    assets_copy: bool,
    dry_run: bool,
    semaphore: asyncio.Semaphore,
) -> dict:
    """Generate one sample with the frozen retry ladder; mirrors the reference."""

    sid = rec["sample_id"]
    model = cfg["model"]
    request_cfg = require_mapping(cfg, "request")
    sample_dir = run_dir / sid
    answer_path = sample_dir / "answer.html"
    if (
        answer_path.exists()
        and is_complete(answer_path.read_text(encoding="utf-8"))
        and can_skip_topk(sample_dir, answer_path)
    ):
        return {"sample_id": sid, "status": "skipped", "tokens": None}
    input_manifest = None
    try:
        lang_code = rec.get("target_lang")
        if lang_code not in LANG_NAME:
            raise ValueError(f"unsupported target_lang: {lang_code!r}")
        lang = LANG_NAME[lang_code]
        parts, input_manifest = build_topk_request(
            sid, lang, rec["query_en"], sample_dir, inputs_root, model=model
        )
    except Exception as exc:
        err = f"input_{type(exc).__name__}: {exc}"[:300]
        print(f"[fail] {sid} | {err}", flush=True)
        return {"sample_id": sid, "status": "error", "stage": "input", "error": err}
    if dry_run:
        return {
            "sample_id": sid,
            "status": "dry_run",
            "tokens": None,
            "asset_count": input_manifest["asset_count"],
            "contact_sheets": len(input_manifest["contact_sheets"]),
        }

    max_attempts = int(request_cfg.get("max_retries", 4))
    max_tokens = request_cfg.get("max_tokens")
    if max_tokens is None and cfg.get("protocol") == "anthropic":
        max_tokens = ANTHROPIC_MAX_TOKENS
    html, err, used_assets = "", None, []
    async with semaphore:
        for attempt in range(max_attempts):
            try:
                text = await client.complete(
                    model=model,
                    parts=parts,
                    temperature=float(request_cfg.get("temperature", 0.0)),
                    max_tokens=max_tokens,
                    disable_thinking=bool(request_cfg.get("disable_thinking", True)),
                )
                html = extract_html(text)
                if is_complete(html):
                    used_assets, violations = validate_topk_output(
                        html, input_manifest["allowed_asset_paths"]
                    )
                    if violations:
                        err = "asset_policy:" + ",".join(violations[:8])
                    else:
                        err = None
                        break
                else:
                    err = f"truncated({len(html) // 1024}KB)"
            except Exception as exc:  # noqa: BLE001 - frozen ladder retries everything
                err = f"{type(exc).__name__}: {exc}"[:200]
            if attempt < max_attempts - 1:
                wait = RETRY_WAITS[min(attempt, 2)]
                print(f"[retry] {sid} attempt {attempt + 1} failed ({err}), "
                      f"retrying in {wait}s", flush=True)
                await asyncio.sleep(wait)
    if err:
        print(f"[fail] {sid} | {err}", flush=True)
        return {"sample_id": sid, "status": "error", "error": err}

    sample_dir.mkdir(parents=True, exist_ok=True)
    answer_path.write_text(html, encoding="utf-8")
    materialize_topk_assets(
        sid, sample_dir, input_manifest["asset_mapping"], inputs_root,
        assets_copy=assets_copy,
    )
    (sample_dir / "output_audit.json").write_text(json.dumps({
        "protocol_version": OUTPUT_AUDIT_PROTOCOL,
        "answer_bytes": answer_path.stat().st_size,
        "allowed_asset_count": len(input_manifest["allowed_asset_paths"]),
        "used_asset_count": len(used_assets),
        "used_asset_paths": used_assets,
        "violations": [],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] {sid} | html {len(html) // 1024}KB", flush=True)
    return {
        "sample_id": sid,
        "status": "ok",
        "tokens": None,  # the unified client does not expose usage; see README
        "html_kb": len(html) // 1024,
        "asset_count": input_manifest["asset_count"],
        "contact_sheets": len(input_manifest["contact_sheets"]),
        "used_asset_count": len(used_assets),
    }


async def run(cfg: dict) -> None:
    input_cfg = require_mapping(cfg, "input")
    request_cfg = require_mapping(cfg, "request")
    output_cfg = require_mapping(cfg, "output")

    run_name = (output_cfg.get("run_name") or "{model}-topk-v5").format(
        model=cfg["model"]
    )
    if not RUN_NAME_PATTERN.fullmatch(run_name) or run_name in {".", ".."}:
        raise ConfigError(
            "run_name must contain only letters, numbers, dot, underscore, or hyphen"
        )
    run_dir = Path(output_cfg.get("runs_root", "outputs/task4/runs")) / run_name
    inputs_root = Path(input_cfg["inputs_root"])

    all_recs, cohort_meta = load_query_records(
        Path(input_cfg["queries"]),
        default_queries_path=DEFAULT_QUERIES_PATH,
        cohort_path=Path(input_cfg["cohort"]) if input_cfg.get("cohort") else None,
    )
    recs = list(all_recs)
    sample_ids = cfg.get("sample_ids")
    if sample_ids:
        wanted = set(sample_ids)
        recs = [rec for rec in recs if rec["sample_id"] in wanted]
        missing = wanted - {rec["sample_id"] for rec in recs}
        if missing:
            raise EvaluationError(f"unknown --sample-ids: {sorted(missing)}")
    limit = cfg.get("limit") or 0
    if limit > 0:
        recs = recs[:limit]

    run_dir.mkdir(parents=True, exist_ok=True)
    cohort_path = run_dir / "_cohort.json"
    if cohort_path.exists():
        existing_cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
        if existing_cohort != cohort_meta:
            raise EvaluationError(f"run cohort mismatch: {cohort_path}")
    else:
        cohort_path.write_text(
            json.dumps(cohort_meta, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )

    meta_path = run_dir / "_meta.jsonl"
    events_path = run_dir / "_events.jsonl"
    allowed_ids = {rec["sample_id"] for rec in all_recs}
    existing = {
        sid: row
        for sid, row in load_latest_records(meta_path, events_path).items()
        if sid in allowed_ids
    }
    dry_run = bool(cfg.get("dry_run"))
    print(
        f"Task4 generation | model={cfg['model']} | cohort={cohort_meta['cohort']} "
        f"| {len(recs)} samples | concurrency={request_cfg['concurrency']} "
        f"| dry_run={dry_run}",
        flush=True,
    )
    start = time.time()
    client = None if dry_run else make_client(
        cfg.get("protocol", "openai"),
        base_url=cfg.get("base_url"),
        api_key=resolve_api_key(cfg),
        timeout=request_cfg["timeout"],
    )
    semaphore = asyncio.Semaphore(int(request_cfg["concurrency"]))
    results = []

    async def process(rec: dict) -> None:
        try:
            row = await gen_sample(
                client,
                cfg,
                rec,
                run_dir,
                inputs_root,
                assets_copy=bool(input_cfg.get("assets_copy", False)),
                dry_run=dry_run,
                semaphore=semaphore,
            )
        except Exception as exc:  # noqa: BLE001 - per-sample isolation
            row = {
                "sample_id": rec["sample_id"],
                "status": "error",
                "stage": "worker",
                "error": f"{type(exc).__name__}: {exc}"[:300],
            }
            print(f"[fail] {rec['sample_id']} | {row['error']}", flush=True)
        results.append(row)
        with events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    await asyncio.gather(*(process(rec) for rec in recs))

    merged = merge_result_records(existing, results)
    ordered_ids = [rec["sample_id"] for rec in all_recs]
    ordered = [merged[sid] for sid in ordered_ids if sid in merged]
    meta_tmp = Path(str(meta_path) + ".tmp")
    meta_tmp.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in ordered),
        encoding="utf-8",
    )
    meta_tmp.replace(meta_path)
    errors = [row for row in ordered if row["status"] == "error"]
    with (run_dir / "_errors.jsonl").open("w", encoding="utf-8") as handle:
        for row in errors:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    ok = sum(1 for row in results if row["status"] == "ok")
    skipped = sum(1 for row in results if row["status"] == "skipped")
    dry = sum(1 for row in results if row["status"] == "dry_run")
    print(
        f"Done | {(time.time() - start) / 60:.0f}min | generated {ok} "
        f"| reused {skipped} | dry-run {dry} | failed {len(errors)}",
        flush=True,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/task4.yaml")
    parser.add_argument("--model", default=None, help="override the model name")
    parser.add_argument("--limit", type=int, default=0,
                        help="generate at most this many samples (0 = no limit)")
    parser.add_argument("--sample-ids", default=None,
                        help="comma-separated sample IDs to generate")
    parser.add_argument("--run-name", default=None,
                        help="override output.run_name")
    parser.add_argument("--concurrency", type=int, default=None,
                        help="override request.concurrency")
    parser.add_argument("--dry-run", action="store_true",
                        help="build requests and write input artifacts only; no API calls")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = load_config(args.config)
    if args.model:
        cfg["model"] = args.model
    if args.limit:
        cfg["limit"] = args.limit
    if args.sample_ids:
        cfg["sample_ids"] = args.sample_ids.split(",")
    if args.run_name:
        require_mapping(cfg, "output")["run_name"] = args.run_name
    if args.concurrency:
        require_mapping(cfg, "request")["concurrency"] = args.concurrency
    if args.dry_run:
        cfg["dry_run"] = True
    try:
        asyncio.run(run(cfg))
    except (ConfigError, EvaluationError) as exc:
        print(f"[FATAL] {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
