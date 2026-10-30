"""End-to-end tests of the Task 1 inference pipeline with a fake client."""

import asyncio
import json
from pathlib import Path

import pytest
from PIL import Image

from tasks.task1.run import run
from tasks.task1.scoring import parse_scores  # noqa: F401  (ensure import order)


def good_scores_json() -> str:
    from tasks.task1.prompts import DIM_KEYS
    scores = {dim: {"score": 4, "reason": "ok"} for dim in DIM_KEYS}
    scores["overall"] = {"score": 4, "reason": "ok"}
    return json.dumps(scores)


class FakeClient:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = 0

    async def complete(self, **kwargs):
        self.calls += 1
        out = self.outputs.pop(0) if self.outputs else good_scores_json()
        if isinstance(out, Exception):
            raise out
        return out


def make_cfg(tmp_path: Path, manifest_rows: list[dict],
             *, image_size=(800, 1000), max_slices=12) -> dict:
    images = tmp_path / "images"
    images.mkdir()
    for row in manifest_rows:
        if "image" in row and row.pop("_create_image", True):
            path = images / row["image"]
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", image_size, (10, 20, 30)).save(path)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(r) + "\n" for r in manifest_rows), encoding="utf-8"
    )
    return {
        "model": "fake-model",
        "protocol": "openai",
        "input": {"manifest": str(manifest), "images_root": str(images),
                  "lang": "en"},
        "image": {
            "slice_height": 1800,
            "slice_overlap": 180,
            "overview_height": 1500,
            "max_slices": max_slices,
            "max_image_bytes": 9 * 1024 * 1024,
            "jpeg_quality": 90,
            "save_processed_images": False,
            "processed_dir": str(tmp_path / "processed"),
        },
        "request": {"temperature": 0.0, "max_tokens": 100,
                    "disable_thinking": False, "concurrency": 2,
                    "max_retries": 3, "timeout": 10},
        "output": {"predictions": str(tmp_path / "preds.jsonl")},
    }


def read_records(path: Path) -> list[dict]:
    return [json.loads(line) for line in
            path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_run_writes_scores_and_resumes(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    fake = FakeClient([])
    monkeypatch.setattr("tasks.task1.run.make_client", lambda *a, **k: fake)
    cfg = make_cfg(tmp_path, [{"id": "s1", "image": "s1_output/x.png"}])

    asyncio.run(run(cfg))
    records = read_records(Path(cfg["output"]["predictions"]))
    assert len(records) == 1
    rec = records[0]
    assert rec["id"] == "s1"
    assert rec["model"] == "fake-model"
    assert rec["protocol"] == "sliced"
    assert rec["n_slices"] == 1
    assert rec["scores"]["overall"]["score"] == 4
    assert fake.calls == 1

    asyncio.run(run(cfg))  # resume: already-done sample is skipped
    assert fake.calls == 1
    assert len(read_records(Path(cfg["output"]["predictions"]))) == 1


def test_run_excludes_too_many_slices_without_calling_api(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    fake = FakeClient([])
    monkeypatch.setattr("tasks.task1.run.make_client", lambda *a, **k: fake)
    cfg = make_cfg(tmp_path, [{"id": "tall", "image": "tall_output/x.png"}],
                   image_size=(800, 4000), max_slices=2)

    asyncio.run(run(cfg))
    records = read_records(Path(cfg["output"]["predictions"]))
    assert records[0]["error"] == "excluded_too_many_slices"
    assert records[0]["scores"] is None
    assert records[0]["n_slices"] == 3
    assert fake.calls == 0


def test_run_records_missing_image(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    fake = FakeClient([])
    monkeypatch.setattr("tasks.task1.run.make_client", lambda *a, **k: fake)
    row = {"id": "gone", "image": "gone_output/x.png", "_create_image": False}
    cfg = make_cfg(tmp_path, [row])

    asyncio.run(run(cfg))
    records = read_records(Path(cfg["output"]["predictions"]))
    assert records[0]["error"] == "image_not_found"
    assert "protocol" not in records[0]
    assert fake.calls == 0


def test_run_retries_unparseable_output(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    fake = FakeClient(["not json at all", good_scores_json()])
    monkeypatch.setattr("tasks.task1.run.make_client", lambda *a, **k: fake)
    cfg = make_cfg(tmp_path, [{"id": "s1", "image": "s1_output/x.png"}])

    asyncio.run(run(cfg))
    records = read_records(Path(cfg["output"]["predictions"]))
    assert records[0]["scores"]["overall"]["score"] == 4
    assert fake.calls == 2


def test_run_gives_up_after_max_retries(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    fake = FakeClient([RuntimeError("boom")] * 10)
    monkeypatch.setattr("tasks.task1.run.make_client", lambda *a, **k: fake)

    async def no_sleep(_seconds):
        return None

    monkeypatch.setattr("asyncio.sleep", no_sleep)
    cfg = make_cfg(tmp_path, [{"id": "s1", "image": "s1_output/x.png"}])

    asyncio.run(run(cfg))
    records = read_records(Path(cfg["output"]["predictions"]))
    assert records[0]["error"] == "max_retries_exceeded"
    assert records[0]["scores"] is None
    assert fake.calls == 3
