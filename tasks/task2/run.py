#!/usr/bin/env python3
"""Task 2 inference: paired degradation diagnosis with a VLM (v8 protocol).

Run from the repository root:

    python -m tasks.task2.run --config configs/task2.yaml
    python -m tasks.task2.run --config configs/task2.yaml --limit 5

Each eligible pair is diagnosed in ONE multimodal call: the model sees both
page versions (shared-scale overview for multi-slice pages, then fixed-canvas
slices) and answers a strict five-key JSON object (degraded version, defect
dimensions, declared coordinate space/box format, location boxes). Scoring is
fixed-denominator: API errors, schema failures, and wrong version decisions
all stay in the cohort.

Records are appended to the predictions file one by one and every written
record is terminal — rerunning the same command resumes by skipping existing
sample IDs, and a record is never re-asked to selectively erase a failure.
"""

from __future__ import annotations

import argparse
import asyncio
import random
import sys
import time
from pathlib import Path

from PIL import Image

from src.api.client import is_retryable_api_error, make_client
from src.utils.config import ConfigError, load_config, require_mapping, resolve_api_key
from src.utils.jsonl import append_record
from tasks.task2 import coordinate_contract
from tasks.task2.geometry import (
    GEOMETRY_POLICY,
    MAX_SLICES,
    OVERLAP,
    OVERVIEW_POLICY,
    SLICE_H,
    THUMB_H,
    build_pair_geometry,
    build_version_image_content,
)
from tasks.task2.prompts import DIMENSIONS, DIM_EN2ZH, render_pair_joint
from tasks.task2.scoring import (
    EVAL_VERSION,
    GT_REGION_MERGE_IOU,
    LOC_PRIMARY_IOU,
    LOC_SOFT_IOU,
    MESSAGE_FORMAT_VERSION,
    PROTOCOL,
    SINGLE_CALL_VERSION,
    attr_item_scores,
    load_manifest_pairs,
    load_records,
    pair_is_eligible,
    score_locations,
    score_locations_no_match,
)

# Giant full-page screenshots exceed PIL's default decompression-bomb limit.
Image.MAX_IMAGE_PIXELS = None

ANTHROPIC_DEFAULT_MAX_TOKENS = 8192


def build_run_meta(model: str, n_cohort: int, max_tokens: int | None) -> dict:
    """Lean protocol-constants section embedded in every record."""
    return {
        "eval_version": EVAL_VERSION,
        "protocol": PROTOCOL,
        "single_call_version": SINGLE_CALL_VERSION,
        "api_calls_per_sample": 1,
        "model": model,
        "response_format": "json_schema",
        "response_schema_version": coordinate_contract.RESPONSE_SCHEMA_VERSION,
        "coordinate_contract_version": coordinate_contract.CONTRACT_VERSION,
        "coordinate_canonical_representation": "normalized_1000_xyxy_float",
        "location_record_fields": {
            "source": "loc_pred_declared",
            "canonical": "loc_pred_canonical",
            "compatibility_alias": "loc_pred_raw=loc_pred_canonical",
        },
        "geometry_policy": GEOMETRY_POLICY,
        "message_format_version": MESSAGE_FORMAT_VERSION,
        "fixed_slice_canvas": True,
        "overview_policy": OVERVIEW_POLICY,
        "input_image_audit": "encoded-bytes-per-image-and-sequence",
        "attr_e2e_policy": "detection-gated-empty-on-wrong-or-invalid-version",
        "loc_e2e_policy": "wrong-or-invalid-version-valid-boxes-fp-all-gt-fn",
        "loc_slice_policy": "max-visible-area",
        "loc_primary_iou": LOC_PRIMARY_IOU,
        "loc_soft_iou": LOC_SOFT_IOU,
        "gt_region_merge_iou": GT_REGION_MERGE_IOU,
        "request_spec": {
            "temperature": 0,
            "enable_thinking": False,
            "response_format": "json_schema",
            "max_tokens": max_tokens,
        },
        "slice_h": SLICE_H,
        "slice_overlap": OVERLAP,
        "thumb_h": THUMB_H,
        "max_slices": MAX_SLICES,
        "sampler": "all-eligible",
        "seed": None,
        "n_cohort": n_cohort,
    }


def load_pair_images(deg: dict, clean: dict,
                     images_root: Path) -> tuple[Image.Image, Image.Image]:
    with Image.open(images_root / deg["image"]) as source:
        deg_img = source.convert("RGB")
    with Image.open(images_root / clean["image"]) as source:
        clean_img = source.convert("RGB")
    if list(deg_img.size) != list(deg["image_size"]):
        raise ValueError(f"{deg['sample_id']}: degraded image dimensions differ from manifest")
    if list(clean_img.size) != list(clean["image_size"]):
        raise ValueError(f"{clean['sample_id']}: clean image dimensions differ from manifest")
    return deg_img, clean_img


async def call_joint_api(client, *, model: str, parts: list[dict],
                         response_format: dict, request_cfg: dict,
                         max_tokens: int | None) -> str:
    """One logical API answer with the frozen retry table: only HTTP
    408/429/5xx and transport errors retry (5s, then 10s); invalid responses
    and other HTTP errors are terminal. Parse failures never retry."""
    max_retries = request_cfg["max_retries"]
    failure: Exception | None = None
    for attempt in range(max_retries):
        try:
            return await client.complete(
                model=model,
                parts=parts,
                temperature=request_cfg["temperature"],
                max_tokens=max_tokens,
                disable_thinking=request_cfg["disable_thinking"],
                response_format=response_format,
                require_stop=True,
            )
        except Exception as exc:  # noqa: BLE001 - classified below
            failure = exc
            if not is_retryable_api_error(exc):
                break
            print(f"  [WARN] attempt {attempt + 1}: retryable API error: {exc}",
                  file=sys.stderr)
            if attempt + 1 < max_retries:
                await asyncio.sleep(5 * (attempt + 1))
    raise RuntimeError(f"API call failed: {failure}")


async def diagnose_pair(client, cfg: dict, pair: tuple[dict, dict],
                        run_meta: dict, max_tokens: int | None,
                        semaphore: asyncio.Semaphore) -> dict:
    """Ask detection, attribution, and localization in one multimodal request."""
    deg, clean = pair
    request_cfg = require_mapping(cfg, "request")
    images_root = Path(require_mapping(cfg, "input")["images_root"])
    model = cfg["model"]
    sid = deg["sample_id"]
    geometry = build_pair_geometry(deg, clean)
    gt = set(deg["gt_dims"])
    # The degraded version's position (Version 1 or 2) is drawn per sample ID,
    # so it is reproducible across runs and machines.
    degraded_version = random.Random(sid).choice([1, 2])

    rec = {"sample_id": sid, "layer": deg["layer"],
           "n_defects": deg["n_defects"], "gt_dims": deg["gt_dims"],
           "degraded_version": degraded_version,
           "eval_version": EVAL_VERSION, "protocol": PROTOCOL, "model": model,
           "run_meta": run_meta, "status": "error",
           "joint_asked": True, "joint_object_valid": False,
           "joint_schema_valid": False, "joint_valid": False,
           "coordinate_space": None, "box_format": None,
           "coordinate_declaration_valid": False,
           "det_pred": None, "det_valid": False, "det_parse_fail": False,
           "det_correct": False,
           "attr_asked": True, "attr_schema_valid": False,
           "attr_valid": False, "attr_parse_fail": False,
           "attr_pred": [], "attr_pred_raw": [], "attr_pred_raw_labels": [],
           "attr_exact": False,
           "attr_item_precision": 0.0, "attr_item_recall": 0.0, "attr_item_f1": 0.0,
           "loc_asked": True, "loc_schema_valid": False,
           "loc_coordinate_contract_valid": False,
           "loc_valid": False, "loc_parse_fail": False,
           "loc_pred_declared": [], "loc_pred_canonical": [],
           # Compatibility alias: consumers must use the run_meta
           # representation declaration rather than infer semantics from name.
           "loc_pred_raw": [],
           "loc_pred_raw_representation": "normalized_1000_xyxy_float",
           "loc_n_pred": 0, "loc_exact": False}
    # Fixed-denominator default: every API/schema failure and every wrong
    # version begins with zero accepted locations and all GT regions as FN.
    score_locations(rec, [], deg, geometry.offsets)
    try:
        deg_img, clean_img = load_pair_images(deg, clean, images_root)
        version_images = {
            degraded_version: deg_img,
            3 - degraded_version: clean_img,
        }
        parts, audit = build_version_image_content(version_images, geometry)
        parts.append({"type": "text", "text": render_pair_joint(
            geometry.n_slices, geometry.canvas_width, SLICE_H)})
        rec["n_slices"] = geometry.n_slices
        rec["input_geometry"] = {
            "policy": geometry.policy,
            "canvas_width": geometry.canvas_width,
            "common_height": geometry.common_height,
            "slice_offsets": list(geometry.offsets),
            "n_slices": geometry.n_slices,
        }
        rec["input_image_audit"] = audit
        provider_response_format = coordinate_contract.response_format(
            geometry.n_slices, geometry.canvas_width, SLICE_H, DIMENSIONS.values())
        async with semaphore:
            t0 = time.time()
            reply = await call_joint_api(
                client, model=model, parts=parts,
                response_format=provider_response_format,
                request_cfg=request_cfg, max_tokens=max_tokens)
        rec["joint_latency_s"] = round(time.time() - t0, 1)
        rec["joint_raw"] = reply
        # Compatibility aliases: review/export tooling can find all three
        # answers without pretending that three requests were made.
        rec["det_raw"] = reply
        rec["attr_raw"] = reply
        rec["loc_raw"] = reply

        parsed = coordinate_contract.parse_declared_diagnosis(
            reply, DIM_EN2ZH,
            geometry.n_slices, geometry.canvas_width, SLICE_H)
        rec["joint_object_valid"] = parsed["object_valid"]
        rec["joint_schema_valid"] = parsed["schema_valid"]
        rec["coordinate_space"] = parsed["coordinate_space"]
        rec["box_format"] = parsed["box_format"]
        rec["coordinate_declaration_valid"] = parsed["coordinate_declaration_valid"]

        pred_v = parsed["version"]
        rec["det_pred"] = pred_v
        rec["det_valid"] = pred_v is not None
        rec["det_parse_fail"] = pred_v is None
        rec["det_correct"] = pred_v == degraded_version

        parsed_dims = parsed["dimensions"]
        rec["attr_schema_valid"] = parsed["dimensions_schema_valid"]
        rec["attr_valid"] = parsed_dims is not None
        rec["attr_parse_fail"] = parsed_dims is None
        raw_dims = parsed_dims if parsed_dims is not None else set()
        rec["attr_pred_raw"] = sorted(raw_dims)
        rec["attr_pred_raw_labels"] = sorted(DIMENSIONS[dim] for dim in raw_dims)
        rec.update({f"attr_ungated_{key}": value
                    for key, value in attr_item_scores(raw_dims, gt).items()})
        gated_dims = raw_dims if rec["det_correct"] else set()
        rec["attr_pred"] = sorted(gated_dims)
        rec.update(attr_item_scores(gated_dims, gt))
        if not rec["det_correct"]:
            rec["attr_exact"] = False

        locs = parsed["locations"]
        rec["loc_schema_valid"] = parsed["locations_schema_valid"]
        rec["loc_coordinate_contract_valid"] = parsed["declared_coordinates_valid"]
        rec["loc_valid"] = locs is not None
        rec["loc_parse_fail"] = locs is None
        canonical_locs = locs if locs is not None else []
        rec["loc_pred_declared"] = list(parsed["source_locations"] or [])
        rec["loc_pred_canonical"] = [
            {"slice": slice_no, "box": [float(value) for value in box]}
            for slice_no, box in canonical_locs
        ]
        rec["loc_pred_raw"] = list(rec["loc_pred_canonical"])
        ungated_loc = {}
        score_locations(ungated_loc, canonical_locs, deg, geometry.offsets)
        rec.update({f"loc_ungated_{key}": value for key, value in ungated_loc.items()})
        if rec["det_correct"]:
            score_locations(rec, canonical_locs, deg, geometry.offsets)
        elif locs is not None:
            score_locations_no_match(rec, canonical_locs, deg)

        rec["joint_valid"] = bool(rec["det_valid"] and rec["attr_valid"]
                                  and rec["loc_valid"])
        rec["status"] = "complete"
    except Exception as exc:  # noqa: BLE001 - per-sample isolation, fixed denominator
        rec["error"] = str(exc)[:500]
    rec["chain_success"] = bool(rec["det_correct"] and rec["attr_exact"]
                                and rec["loc_exact"])
    return rec


def validate_existing_output(records: dict[str, dict], run_meta: dict) -> dict:
    """Resume gate: a non-empty output keeps its stored run_meta; only the
    model/protocol/version identity is checked for consistency."""
    if not records:
        return run_meta
    rows = list(records.values())
    if any(r.get("eval_version") != EVAL_VERSION or not isinstance(r.get("run_meta"), dict)
           for r in rows):
        raise ConfigError(
            f"output contains old/unversioned results; use a fresh predictions file "
            f"for {EVAL_VERSION}")
    keys = ("model", "eval_version", "protocol", "single_call_version")
    stored = rows[0]["run_meta"]
    for record in rows:
        if any(record["run_meta"].get(key) != stored.get(key) for key in keys):
            raise ConfigError(
                "existing records disagree on model/protocol/version; "
                "use a fresh predictions file")
    if any(stored.get(key) != run_meta.get(key) for key in keys):
        raise ConfigError(
            "existing records were produced by a different model/protocol/version; "
            "use a fresh predictions file to avoid mixing runs")
    return stored


async def run(cfg: dict) -> None:
    input_cfg = require_mapping(cfg, "input")
    request_cfg = require_mapping(cfg, "request")
    model = cfg["model"]
    protocol = cfg.get("protocol", "openai")

    max_tokens = request_cfg.get("max_tokens")
    if protocol == "anthropic" and max_tokens is None:
        max_tokens = ANTHROPIC_DEFAULT_MAX_TOKENS

    pairs = load_manifest_pairs(Path(input_cfg["manifest"]))
    cohort = [pair for pair in pairs if pair_is_eligible(pair)]
    print(f"[INFO] {len(pairs)} pairs in manifest, {len(cohort)} eligible "
          f"({len(pairs) - len(cohort)} excluded)")
    limit = cfg.get("limit") or 0
    if limit > 0:
        cohort = cohort[:limit]

    output_path = Path(
        require_mapping(cfg, "output")["predictions"].format(model=model))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    run_meta = build_run_meta(model, len(cohort), max_tokens)
    existing = load_records(output_path)
    run_meta = validate_existing_output(existing, run_meta)
    cohort_ids = {pair[0]["sample_id"] for pair in cohort}
    extra_ids = set(existing) - cohort_ids
    if extra_ids:
        raise ConfigError(
            f"output file contains {len(extra_ids)} samples outside this cohort; "
            f"use a fresh predictions file")
    todo = [pair for pair in cohort if pair[0]["sample_id"] not in existing]
    if existing:
        n_errors = sum(r.get("status") == "error" for r in existing.values())
        print(f"[INFO] resuming: {len(existing)} terminal records skipped "
              f"({n_errors} with errors)")
    print(f"[INFO] {len(todo)} pairs to diagnose | model {model}")
    if not todo:
        return

    client = make_client(
        protocol,
        base_url=cfg.get("base_url"),
        api_key=resolve_api_key(cfg),
        timeout=request_cfg["timeout"],
    )
    semaphore = asyncio.Semaphore(request_cfg["concurrency"])
    total = len(todo)
    counts = {"done": 0, "errors": 0}

    async def process(pair: tuple[dict, dict]) -> None:
        rec = await diagnose_pair(client, cfg, pair, run_meta, max_tokens, semaphore)
        with output_path.open("a", encoding="utf-8") as handle:
            append_record(handle, rec)
        counts["done"] += 1
        if rec.get("status") == "error":
            counts["errors"] += 1
        tag = "ERR" if rec.get("status") == "error" else (
            "OK" if rec.get("det_correct") else "WRONG")
        print(f"  [{counts['done']}/{total}] {rec['sample_id']} "
              f"slices={rec.get('n_slices', '?')} {tag} "
              f"gt={rec['gt_dims']} attr={rec.get('attr_pred', '-')} "
              f"{rec.get('error', '')}")
        if counts["done"] % 20 == 0 or counts["done"] == total:
            print(f"[PROGRESS] {counts['done']}/{total} done, "
                  f"{counts['errors']} errors")

    await asyncio.gather(*(process(pair) for pair in todo))
    print(f"[DONE] {counts['done']} diagnosed, {counts['errors']} errors")
    print(f"[DONE] predictions written to {output_path}")
    print(f"[DONE] evaluate with: python -m tasks.task2.evaluate "
          f"{output_path} --manifest {input_cfg['manifest']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/task2.yaml")
    parser.add_argument("--model", default=None, help="override the model name")
    parser.add_argument("--limit", type=int, default=0,
                        help="diagnose at most this many eligible pairs (0 = all)")
    parser.add_argument("--concurrency", type=int, default=None,
                        help="override request.concurrency")
    parser.add_argument("--output", default=None,
                        help="override output.predictions")
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
        require_mapping(cfg, "output")["predictions"] = args.output
    try:
        asyncio.run(run(cfg))
    except ConfigError as exc:
        print(f"[FATAL] {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
