"""Task 2 prompt catalog and defect-dimension labels.

``prompts.single_call.v8.json`` is part of the frozen experimental protocol
and is loaded verbatim; do not reword. The v8 main protocol uses only the
``dimensions`` table and the ``pair_joint`` prompt; the remaining keys are
kept in the file for completeness.
"""

from __future__ import annotations

import json
from pathlib import Path

PROMPTS_PATH = Path(__file__).resolve().parent / "prompts.single_call.v8.json"
PROMPTS = json.loads(PROMPTS_PATH.read_text(encoding="utf-8"))

DIMENSIONS = PROMPTS["dimensions"]  # Chinese GT label -> English response label
DIM_EN2ZH = {v: k for k, v in DIMENSIONS.items()}
ALL_DIMS = tuple(DIMENSIONS.keys())


def render_pair_joint(n_slices: int, canvas_width: int, slice_height: int) -> str:
    """Render the v8 joint prompt with the pair's geometry filled in."""
    return (PROMPTS["pair_joint"]["en"]
            .replace("{N}", str(n_slices))
            .replace("{W}", str(canvas_width))
            .replace("{H}", str(slice_height)))
