#!/usr/bin/env python3
"""Top-K request materialization for Task 4 generation.

Builds the frozen multimodal request for one sample: verbatim copy block from
the released ``copy.txt``, the compact caption JSON with V01.. aliases, and up
to two 4x4 contact sheets re-labeled from the audited ``topk.jpg`` montage.
File identity is recorded as byte counts and direct comparisons only.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path

from PIL import Image, ImageDraw

from tasks.task4.prompts import (
    INPUT_PROTOCOL,
    OUTPUT_AUDIT_PROTOCOL,
    PROMPT,
    PROMPT_COPY,
    PROMPT_TOPK,
    PROMPT_TOPK_EMPTY,
)


def build_copy_block(sid: str, inputs_root: Path, total_cap: int = 15000) -> str:
    """Wrap the released per-sample copy lines in the frozen VERBATIM block.

    ``copy.txt`` carries the upstream extractor's per-element texts verbatim
    (already whitespace-normalized and cut to 300 chars; the cut can leave a
    trailing space). Lines are read as-is — re-normalizing here would strip
    that trailing space and break byte-identity with the frozen prompt. Only
    dedup and the total cap are applied at read time.
    """

    copy_path = Path(inputs_root) / sid / "copy.txt"
    if not copy_path.is_file():
        return ""
    raw_lines = copy_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    items = [line for line in raw_lines if line.strip()]
    seen, lines, total = set(), [], 0
    for text in items:
        if text in seen:
            continue
        seen.add(text)
        total += len(text)
        if total > total_cap:
            lines.append(f"... (truncated, {len(items) - len(lines)} more lines)")
            break
        lines.append(text)
    if not lines:
        return ""
    return PROMPT_COPY.format(listing="\n".join(lines))


def build_request_assets(compact: dict) -> tuple[dict, list[dict]]:
    """Relabel the compact caption array as V01.. with short assets/ aliases."""

    rows, mapping = [], []
    for index, source_row in enumerate(compact["a"], 1):
        source_id, source_src, width, height, caption = source_row
        suffix = Path(source_src).suffix.lower()
        if not suffix or not re.fullmatch(r"\.[a-z0-9]+", suffix):
            raise ValueError(f"unsafe or missing asset extension: {source_src}")
        request_id = f"V{index:02d}"
        alias_src = f"assets/v{index:02d}{suffix}"
        rows.append([request_id, alias_src, width, height, caption])
        mapping.append({
            "id": request_id,
            "alias_src": alias_src,
            "source_id": source_id,
            "source_src": source_src,
        })
    return {"f": compact["f"], "a": rows}, mapping


def build_contact_sheets(
    sid: str,
    request_rows: list[list],
    sample_dir: Path,
    inputs_root: Path,
) -> list[Path]:
    """Relabel the audited Top-K montage as V01..V32 in at most two 4x4 sheets."""

    asset_count = len(request_rows)
    if asset_count <= 0:
        return []
    source = Path(inputs_root) / sid / "topk.jpg"
    if not source.exists():
        raise FileNotFoundError(f"missing Top-K contact sheet: {source}")
    im = Image.open(source).convert("RGB")
    cols, cell_h, header_h = 4, 170, 48
    expected_w = cols * 190
    expected_h = header_h + ((asset_count + cols - 1) // cols) * cell_h
    if im.width != expected_w or im.height != expected_h:
        raise ValueError(
            f"unexpected Top-K sheet geometry for {sid}: {im.size}, expected {(expected_w, expected_h)}"
        )
    paths = []
    for sheet_index, start in enumerate(range(0, asset_count, 16), 1):
        rows = request_rows[start:start + 16]
        sheet_rows = (len(rows) + cols - 1) // cols
        canvas = Image.new("RGB", (expected_w, header_h + sheet_rows * cell_h), "#f3f4f6")
        draw = ImageDraw.Draw(canvas)
        draw.text((12, 14), f"{sid} selected visuals {start + 1}-{start + len(rows)}", fill="#111827")
        for local_index, row in enumerate(rows):
            source_index = start + local_index
            source_x = (source_index % cols) * 190
            source_y = header_h + (source_index // cols) * cell_h
            dest_x = (local_index % cols) * 190
            dest_y = header_h + (local_index // cols) * cell_h
            # Preserve the already-decoded visual tile (including SVGs), but replace
            # the old sparse A#### label with the frozen short alias.
            tile = im.crop((source_x + 4, source_y + 2, source_x + 186, source_y + 144))
            canvas.paste(tile, (dest_x + 4, dest_y + 2))
            label = f"{row[0]} {Path(row[1]).name} {row[2]}x{row[3]}"
            draw.text((dest_x + 5, dest_y + cell_h - 22), label[:31], fill="#111827")
        path = sample_dir / f"contact_sheet_{sheet_index}.jpg"
        canvas.save(path, quality=90, optimize=True)
        paths.append(path)
    return paths


def materialize_topk_assets(
    sid: str,
    sample_dir: Path,
    asset_mapping: list[dict],
    inputs_root: Path,
    *,
    assets_copy: bool = False,
) -> None:
    """Expose only selected visuals under deterministic, short per-run aliases."""

    assets_dir = sample_dir / "assets"
    if assets_dir.is_symlink():
        assets_dir.unlink()
    assets_dir.mkdir(parents=True, exist_ok=True)
    for item in asset_mapping:
        source = Path(inputs_root) / sid / item["source_src"]
        if not source.is_file():
            raise FileNotFoundError(f"missing selected source asset: {source}")
        alias = sample_dir / item["alias_src"]
        alias.parent.mkdir(parents=True, exist_ok=True)
        if alias.is_symlink() or alias.is_file():
            alias.unlink()
        elif alias.exists():
            raise ValueError(f"asset alias target is not a file: {alias}")
        if assets_copy:
            shutil.copy2(source, alias)
        else:
            alias.symlink_to(Path(os.path.relpath(source, alias.parent)))


def build_topk_request(
    sid: str,
    lang: str,
    query: str,
    sample_dir: Path,
    inputs_root: Path,
    *,
    model: str,
) -> tuple[list[dict], dict]:
    """Build the frozen request parts and write the per-sample input artifacts."""

    inputs_root = Path(inputs_root)
    caption_path = inputs_root / sid / "assets.caption.min.json"
    selected_path = inputs_root / sid / "selected_assets.json"
    if not caption_path.exists() or not selected_path.exists():
        raise FileNotFoundError(f"missing formal Top-K inputs for {sid}")
    compact_bytes = caption_path.read_bytes()
    compact = json.loads(compact_bytes)
    if compact.get("f") != ["id", "src", "w", "h", "cap"]:
        raise ValueError(f"unexpected caption schema for {sid}")
    selected = json.loads(selected_path.read_text(encoding="utf-8"))["assets"]
    expected = [(x["id"], x["path"], x["width"], x["height"]) for x in selected]
    actual = [tuple(x[:4]) for x in compact["a"]]
    if actual != expected:
        raise ValueError(f"caption/selected asset mismatch for {sid}")

    sample_dir.mkdir(parents=True, exist_ok=True)
    request_compact, asset_mapping = build_request_assets(compact)
    request_compact_bytes = json.dumps(
        request_compact, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    (sample_dir / "request_assets.caption.min.json").write_bytes(request_compact_bytes)
    sheets = build_contact_sheets(sid, request_compact["a"], sample_dir, inputs_root)
    asset_prompt = PROMPT_TOPK if actual else PROMPT_TOPK_EMPTY
    extras = build_copy_block(sid, inputs_root) + asset_prompt.format(
        compact_json=request_compact_bytes.decode("utf-8")
    )
    prompt = PROMPT.format(lang=lang, extras=extras, query=query)
    prompt_bytes = prompt.encode("utf-8")
    (sample_dir / "request_prompt.txt").write_bytes(prompt_bytes)
    sheet_meta = []
    for index, path in enumerate(sheets):
        data = path.read_bytes()
        start = index * 16
        sheet_meta.append({
            "file": path.name,
            "bytes": len(data),
            "asset_ids": [row[0] for row in request_compact["a"][start:start + 16]],
        })
    manifest = {
        "protocol_version": INPUT_PROTOCOL,
        "sample_id": sid,
        "model": model,
        "mode": "topk",
        "target_lang": lang,
        "query_en": query,
        "prompt_file": "request_prompt.txt",
        "prompt_bytes": len(prompt_bytes),
        "source_caption_json": str(caption_path),
        "request_caption_json": "request_assets.caption.min.json",
        "request_caption_json_bytes": len(request_compact_bytes),
        "asset_count": len(actual),
        "allowed_asset_paths": [row[1] for row in request_compact["a"]],
        "asset_mapping": asset_mapping,
        "contact_sheets": sheet_meta,
        "message_order": ["text"] + [x["file"] for x in sheet_meta],
    }
    (sample_dir / "input_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    parts = [{"type": "text", "text": prompt}] + [
        {"type": "image", "data": path.read_bytes(), "media_type": "image/jpeg"}
        for path in sheets
    ]
    return parts, manifest


def can_skip_topk(sample_dir: Path, answer_path: Path) -> bool:
    try:
        manifest = json.loads(
            (sample_dir / "input_manifest.json").read_text(encoding="utf-8")
        )
        audit = json.loads(
            (sample_dir / "output_audit.json").read_text(encoding="utf-8")
        )
        if manifest.get("protocol_version") != INPUT_PROTOCOL:
            return False
        if audit.get("protocol_version") != OUTPUT_AUDIT_PROTOCOL or audit.get("violations"):
            return False
        if not answer_path.is_file() or answer_path.stat().st_size == 0:
            return False
        for item in manifest.get("asset_mapping", []):
            if not (sample_dir / item["alias_src"]).is_file():
                return False
        return True
    except Exception:
        return False


def load_latest_records(*paths: Path) -> dict:
    latest = {}
    for path in paths:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
                if row.get("sample_id"):
                    latest[row["sample_id"]] = row
            except Exception:
                continue
    return latest


def merge_result_records(existing: dict, results: list[dict]) -> dict:
    merged = dict(existing)
    for row in results:
        sid = row["sample_id"]
        # A validated resume skip should not erase the original token and output
        # metadata from the successful generation event.
        if row.get("status") == "skipped" and existing.get(sid, {}).get("status") == "ok":
            continue
        merged[sid] = row
    return merged


__all__ = [
    "build_contact_sheets",
    "build_copy_block",
    "build_request_assets",
    "build_topk_request",
    "can_skip_topk",
    "load_latest_records",
    "materialize_topk_assets",
    "merge_result_records",
]
