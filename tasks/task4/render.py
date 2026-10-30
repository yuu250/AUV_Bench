#!/usr/bin/env python3
"""Task 4 rendering: fixed 3840x4320 (1920x2160 CSS px @2x) page captures.

Produces per sample under the run directory:
    full.png        full-page screenshot (the scored artifact)
    crop4320.png    fixed scoring viewport, always exactly 3840x4320
    render_meta.json

Run from the repository root:

    python -m tasks.task4.render qwen3-vl-plus-topk-v5 [--workers 8]
    python -m tasks.task4.render qwen3-vl-plus-topk-v5 --runs-root outputs/task4/runs

Pages are rendered with JavaScript enabled; pages whose JS hangs the
compositor retry once with JavaScript disabled.  A crashed browser process is
relaunched up to 10 times.  Reruns skip samples whose render_meta.json matches
the protocol and the current answer.html byte count.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from PIL import Image

from tasks.task4.prompts import RENDER_PROTOCOL

# playwright 1.62.0 (pinned in requirements.txt) exposes TargetClosedError only
# in the private impl module; newer versions re-export it from async_api.
try:
    from playwright.async_api import TargetClosedError
except ImportError:  # pragma: no cover - pinned-version path
    from playwright._impl._errors import TargetClosedError

Image.MAX_IMAGE_PIXELS = None  # very long pages exceed the PIL default pixel cap

CROP_CSS_H = 2160  # 3840x4320 @2x corresponds to 1920x2160 CSS px
CROP_SIZE = (3840, 4320)
BROWSER_RELAUNCH_LIMIT = 10


def existing_render_valid(sample_dir: Path, answer_bytes: int) -> bool:
    """Idempotent skip: a complete prior render for this exact answer.html."""

    sample_dir = Path(sample_dir)
    crop_path = sample_dir / "crop4320.png"
    full_path = sample_dir / "full.png"
    meta_path = sample_dir / "render_meta.json"
    if not (crop_path.exists() and full_path.exists() and meta_path.exists()):
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return (
        meta.get("protocol_version") == RENDER_PROTOCOL
        and meta.get("answer_bytes") == answer_bytes
        and tuple(meta.get("crop_size", [])) == CROP_SIZE
    )


async def render_one(browser, sample_dir: Path, counters: dict) -> None:
    sid = sample_dir.name
    html = sample_dir / "answer.html"
    crop_path = sample_dir / "crop4320.png"
    full_path = sample_dir / "full.png"
    meta_path = sample_dir / "render_meta.json"
    if not html.exists():
        return
    answer_bytes = html.stat().st_size
    if existing_render_valid(sample_dir, answer_bytes):
        counters["skip"] += 1
        return
    crop_tmp = Path(str(crop_path) + ".tmp.png")
    full_tmp = Path(str(full_path) + ".tmp.png")
    html_uri = html.resolve().as_uri()  # absolute file:/// URL; relative paths are invalid
    for js_on in (True, False):  # JS on first; JS-disabled retry on screenshot hangs
        ctx = await browser.new_context(viewport={"width": 1920, "height": 1080},
                                        device_scale_factor=2, java_script_enabled=js_on)
        page = await ctx.new_page()
        page.on("dialog", lambda dialog: asyncio.ensure_future(dialog.dismiss()))
        try:
            try:
                await page.goto(html_uri, wait_until="networkidle", timeout=45000)
            except Exception:
                await page.goto(html_uri, wait_until="domcontentloaded", timeout=20000)
            await page.wait_for_timeout(1000)
            # The scored artifact is a real 1920x2160 CSS-pixel viewport at DPR 2.
            # This stays exactly 3840x4320 even when the document itself is shorter.
            await page.set_viewport_size({"width": 1920, "height": CROP_CSS_H})
            await page.screenshot(path=str(crop_tmp), full_page=False, timeout=120000)
            await page.screenshot(path=str(full_tmp), full_page=True, timeout=120000)
            await ctx.close()
            break
        except TargetClosedError:
            try:
                await ctx.close()
            except Exception:
                pass
            crop_tmp.unlink(missing_ok=True)
            full_tmp.unlink(missing_ok=True)
            raise  # browser crash is not a page problem; main relaunches the browser
        except Exception as exc:
            try:
                await ctx.close()
            except Exception:
                pass
            crop_tmp.unlink(missing_ok=True)
            full_tmp.unlink(missing_ok=True)
            if js_on:
                print(f"[retry-nojs] {sid}: {type(exc).__name__}, retrying without JS", flush=True)
                continue
            counters["fail"] += 1
            print(f"[fail] {sid}: {type(exc).__name__} {exc}"[:150], flush=True)
            return
    with Image.open(crop_tmp) as crop_image:
        crop_size = crop_image.size
    with Image.open(full_tmp) as full_image:
        full_size = full_image.size
    if crop_size != CROP_SIZE:
        crop_tmp.unlink(missing_ok=True)
        full_tmp.unlink(missing_ok=True)
        counters["fail"] += 1
        print(f"[fail] {sid}: wrong crop size {crop_size}", flush=True)
        return
    crop_tmp.replace(crop_path)
    full_tmp.replace(full_path)
    meta_path.write_text(json.dumps({
        "protocol_version": RENDER_PROTOCOL,
        "answer_bytes": answer_bytes,
        "crop_size": list(crop_size),
        "full_size": list(full_size),
        "java_script_enabled": js_on,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    counters["done"] += 1
    if counters["done"] % 100 == 0:
        print(f"[render] progress {counters['done']} "
              f"(skipped {counters['skip']} failed {counters['fail']})", flush=True)


async def render_run(run_name: str, runs_root: Path, workers: int) -> dict:
    from playwright.async_api import async_playwright

    run_dir = Path(runs_root) / run_name
    dirs = sorted(
        d for d in run_dir.iterdir()
        if d.is_dir() and (d / "answer.html").exists()
    )
    print(f"render {run_name} | {len(dirs)} pages | workers={workers}", flush=True)
    counters = {"done": 0, "fail": 0, "skip": 0}
    semaphore = asyncio.Semaphore(workers)
    async with async_playwright() as playwright:
        for attempt in range(1, BROWSER_RELAUNCH_LIMIT + 1):
            browser = await playwright.chromium.launch()

            async def guarded(sample_dir):
                async with semaphore:
                    await render_one(browser, sample_dir, counters)

            tasks = [asyncio.ensure_future(guarded(d)) for d in dirs]
            try:
                await asyncio.gather(*tasks)
            except TargetClosedError as exc:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                try:
                    await browser.close()
                except Exception:
                    pass
                left = sum(1 for d in dirs if not (d / "crop4320.png").exists())
                print(f"[browser-relaunch] crash #{attempt} ({type(exc).__name__}), "
                      f"relaunching, {left} pages left", flush=True)
                continue
            await browser.close()
            break
        else:
            raise SystemExit("browser crashed 10 times in a row; rendering incomplete")
    print(f"Done | rendered {counters['done']} | reused {counters['skip']} "
          f"| failed {counters['fail']}", flush=True)
    return counters


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_name", help="run directory under the runs root")
    parser.add_argument("--runs-root", default="outputs/task4/runs")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    asyncio.run(render_run(args.run_name, Path(args.runs_root), args.workers))


if __name__ == "__main__":
    main()
