#!/usr/bin/env python3
"""Task 1 inference: score UI page screenshots with a VLM on 8 aesthetic dimensions.

Run from the repository root:

    python -m tasks.task1.run --config configs/task1.yaml
    python -m tasks.task1.run --config configs/task1.yaml --limit 5

Each sample is sent as a downscaled overview (only when the page needs more
than one slice; navigation aid) followed by fixed-height slices from top to
bottom, then the scoring prompt. Pages needing more than max_slices slices
are recorded as excluded_too_many_slices without sending any request.

Samples are appended to the predictions file one by one; rerunning the same
command resumes where it left off (samples with scores or an exclusion are
skipped, failed samples are retried).
"""

from __future__ import annotations

import argparse
import asyncio
import io
import sys
import time
import urllib.request
from pathlib import Path

from PIL import Image

from src.api.client import ImageTooLargeError, make_client
from src.image.encoding import encode_image_payload
from src.image.slicing import slice_offsets
from src.utils.config import ConfigError, load_config, require_mapping, resolve_api_key
from src.utils.jsonl import append_record, load_done_ids, load_manifest
from tasks.task1.prompts import build_prompt
from tasks.task1.scoring import parse_scores, validate_scores

INTRO_TEXT = (
    "You will be shown a web page screenshot. "
    "If the page is tall, it first includes a size-normalized overview for navigation only. "
    "The page is then shown in horizontal slices from top to bottom; "
    "use those slices for detailed visual decisions."
)


def timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def build_sliced_parts(source: Path | bytes, prompt: str, image_cfg: dict,
                       *, sample_id: str, force_jpeg: bool) -> tuple[list[dict], int]:
    """Build the neutral message parts for the sliced protocol.

    Message order: intro text, overview (only for multi-slice pages), slices
    top to bottom, scoring prompt last.
    """
    if isinstance(source, Path):
        img = Image.open(source)
    else:
        img = Image.open(io.BytesIO(source))
    with img:
        w, h = img.size
        slice_h = image_cfg["slice_height"]
        offs = slice_offsets(h, slice_h, image_cfg["slice_overlap"])
        save_dir = (
            Path(image_cfg["processed_dir"])
            if image_cfg.get("save_processed_images") else None
        )

        def encode(im: Image.Image, name: str) -> dict:
            data, mime = encode_image_payload(
                im,
                force_jpeg=force_jpeg,
                max_bytes=image_cfg["max_image_bytes"],
                jpeg_quality=image_cfg["jpeg_quality"],
                save_dir=save_dir,
                save_name=name,
            )
            return {"type": "image", "data": data, "media_type": mime}

        parts: list[dict] = [{"type": "text", "text": INTRO_TEXT}]
        if len(offs) > 1:
            tw = max(1, round(w * image_cfg["overview_height"] / h))
            parts.append({"type": "text", "text": "Page overview (scaled down):"})
            parts.append(encode(
                img.resize((tw, image_cfg["overview_height"])),
                f"{sample_id}_overview",
            ))
        for i, top in enumerate(offs):
            tile = img.crop((0, top, w, min(top + slice_h, h)))
            parts.append({"type": "text",
                          "text": f"Screenshot slice {i + 1}/{len(offs)}:"})
            parts.append(encode(tile, f"{sample_id}_slice{i + 1:02d}"))
        parts.append({"type": "text", "text": prompt})
    return parts, len(offs)


def download_image(url: str, timeout: float) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "design-bench/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


async def judge_sample(client, cfg: dict, sample: dict,
                       semaphore: asyncio.Semaphore) -> dict:
    """Score one sample; mirrors the reference implementation's retry semantics."""
    model = cfg["model"]
    input_cfg = require_mapping(cfg, "input")
    image_cfg = require_mapping(cfg, "image")
    request_cfg = require_mapping(cfg, "request")
    sample_id = sample["id"]
    prompt = build_prompt(input_cfg.get("lang", "en"))
    image_label = sample.get("image") or sample.get("image_url")

    source: Path | bytes | None = None
    url_direct: str | None = None
    image_url = sample.get("image_url")
    if image_url and input_cfg.get("url_mode", "direct") == "direct":
        url_direct = image_url
    elif image_url:
        try:
            source = download_image(image_url, timeout=request_cfg["timeout"])
        except Exception as exc:
            print(f"  [ERROR] {sample_id} image download failed: {exc}",
                  file=sys.stderr)
            return {
                "id": sample_id, "image": image_label, "model": model,
                "scores": None, "error": "image_download_failed",
                "timestamp": timestamp(),
            }
    else:
        path = Path(input_cfg["images_root"]) / sample["image"]
        if not path.exists():
            print(f"  [SKIP] {sample_id} image not found: {path}", file=sys.stderr)
            return {
                "id": sample_id, "image": image_label, "model": model,
                "scores": None, "error": "image_not_found",
                "timestamp": timestamp(),
            }
        source = path

    if url_direct is None:
        opener = Image.open(source) if isinstance(source, Path) \
            else Image.open(io.BytesIO(source))
        with opener as probe:
            n = len(slice_offsets(
                probe.height, image_cfg["slice_height"], image_cfg["slice_overlap"]))
        if n > image_cfg["max_slices"]:
            return {
                "id": sample_id, "image": image_label, "model": model,
                "scores": None, "error": "excluded_too_many_slices",
                "protocol": "sliced", "n_slices": n,
                "timestamp": timestamp(),
            }

    force_jpeg = False
    async with semaphore:
        for attempt in range(request_cfg["max_retries"]):
            try:
                if url_direct is not None:
                    parts = [
                        {"type": "text", "text": prompt},
                        {"type": "image", "url": url_direct},
                    ]
                    n_slices = None
                else:
                    parts, n_slices = build_sliced_parts(
                        source, prompt, image_cfg,
                        sample_id=sample_id, force_jpeg=force_jpeg)
                raw = await client.complete(
                    model=model,
                    parts=parts,
                    temperature=request_cfg["temperature"],
                    max_tokens=request_cfg["max_tokens"],
                    disable_thinking=request_cfg["disable_thinking"],
                )
                scores = parse_scores(raw)
                if scores and validate_scores(scores):
                    rec = {
                        "id": sample_id,
                        "image": image_label,
                        "model": model,
                        "scores": scores,
                        "protocol": "url" if url_direct else "sliced",
                        "timestamp": timestamp(),
                    }
                    if n_slices is not None:
                        rec["n_slices"] = n_slices
                    if force_jpeg:
                        rec["image_encoding"] = "sliced_jpeg"
                    return rec
                print(f"  [WARN] {sample_id} attempt {attempt + 1}: "
                      f"could not parse scores, retrying...", file=sys.stderr)
            except ImageTooLargeError:
                if not force_jpeg:
                    force_jpeg = True
                    print(f"  [WARN] {sample_id} image payload too large; "
                          f"retrying with JPEG compression", file=sys.stderr)
                    continue
                print(f"  [ERROR] {sample_id} attempt {attempt + 1}: "
                      f"image still too large after JPEG compression",
                      file=sys.stderr)
                if attempt < request_cfg["max_retries"] - 1:
                    await asyncio.sleep(2 ** attempt)
            except Exception as exc:  # noqa: BLE001 - per-sample isolation
                print(f"  [ERROR] {sample_id} attempt {attempt + 1}: {exc}",
                      file=sys.stderr)
                if attempt < request_cfg["max_retries"] - 1:
                    await asyncio.sleep(2 ** attempt)

    return {
        "id": sample_id,
        "image": image_label,
        "model": model,
        "scores": None,
        "error": "max_retries_exceeded",
        "protocol": "url" if url_direct else "sliced",
        "timestamp": timestamp(),
    }


async def run(cfg: dict) -> None:
    input_cfg = require_mapping(cfg, "input")
    request_cfg = require_mapping(cfg, "request")

    manifest = load_manifest(input_cfg["manifest"])
    output_path = Path(
        require_mapping(cfg, "output")["predictions"].format(
            model=cfg["model"], lang=input_cfg.get("lang", "en"))
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    done_ids = load_done_ids(output_path)
    todo = [s for s in manifest if s["id"] not in done_ids]
    if done_ids:
        print(f"[INFO] {len(done_ids)} samples already done, skipping")
    limit = cfg.get("limit") or 0
    if limit > 0:
        todo = todo[:limit]
    print(f"[INFO] {len(manifest)} samples in manifest, {len(todo)} to score")
    if not todo:
        return

    client = make_client(
        cfg.get("protocol", "openai"),
        base_url=cfg.get("base_url"),
        api_key=resolve_api_key(cfg),
        timeout=request_cfg["timeout"],
    )
    semaphore = asyncio.Semaphore(request_cfg["concurrency"])
    total = len(todo)
    counts = {"done": 0, "failed": 0, "excluded": 0}

    async def process(sample: dict) -> None:
        result = await judge_sample(client, cfg, sample, semaphore)
        with output_path.open("a", encoding="utf-8") as handle:
            append_record(handle, result)
        counts["done"] += 1
        if result.get("error") == "excluded_too_many_slices":
            counts["excluded"] += 1
        elif result.get("scores") is None:
            counts["failed"] += 1
        if counts["done"] % 10 == 0 or counts["done"] == total:
            print(f"[PROGRESS] {counts['done']}/{total} done, "
                  f"{counts['failed']} failed, {counts['excluded']} excluded")

    await asyncio.gather(*(process(sample) for sample in todo))
    print(f"[DONE] {counts['done']} processed: "
          f"{counts['done'] - counts['failed'] - counts['excluded']} scored, "
          f"{counts['failed']} failed, {counts['excluded']} excluded")
    print(f"[DONE] predictions written to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/task1.yaml")
    parser.add_argument("--model", default=None, help="override the model name")
    parser.add_argument("--limit", type=int, default=0,
                        help="score at most this many samples (0 = no limit)")
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
