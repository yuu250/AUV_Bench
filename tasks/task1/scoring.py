"""Parsing and validation of model score outputs (Task 1).

The parsing strategy and validation rules are part of the experimental
protocol and are kept identical to the reference implementation.
"""

from __future__ import annotations

import json

from tasks.task1.prompts import DIM_KEYS


def parse_scores(text: str) -> dict | None:
    """Parse the JSON scores from a model output; tolerates ```json fences."""
    if "```" in text:
        for block in text.split("```"):
            block = block.strip()
            if block.startswith("json"):
                block = block[4:].strip()
            try:
                return json.loads(block)
            except json.JSONDecodeError:
                continue
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass
    return None


def validate_scores(scores: dict) -> bool:
    """All 8 dimensions + overall must be present with a score in [1, 5]."""
    required = DIM_KEYS + ["overall"]
    for key in required:
        if key not in scores:
            return False
        entry = scores[key]
        if not isinstance(entry, dict) or "score" not in entry:
            return False
        s = entry["score"]
        if not isinstance(s, (int, float)) or s < 1 or s > 5:
            return False
    return True
