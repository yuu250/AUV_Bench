#!/usr/bin/env python3
"""Strict judge-output parsing and point arithmetic for the Task 4 evaluators."""

from __future__ import annotations

import json
from typing import Any


def _load_strict_object(text: str) -> dict[str, Any]:
    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    value = json.loads(text, object_pairs_hook=object_pairs)
    if not isinstance(value, dict):
        raise ValueError("response must be a JSON object")
    return value


def strict_winner(text: str) -> str:
    value = _load_strict_object(text)
    if set(value) != {"winner"}:
        raise ValueError("response must contain exactly the winner key")
    winner = value["winner"]
    if winner not in {"A", "B", "tie"}:
        raise ValueError("winner must be A, B, or tie")
    return winner


def expected_points(winner: str | None) -> dict[str, float]:
    if winner == "A":
        return {"A": 1.0, "B": 0.0}
    if winner == "B":
        return {"A": 0.0, "B": 1.0}
    if winner == "tie":
        return {"A": 0.5, "B": 0.5}
    return {"A": 0.0, "B": 0.0}


def strict_scores(text: str, score_keys: list[str]) -> dict[str, dict[str, Any]]:
    value = _load_strict_object(text)
    if set(value) != set(score_keys):
        raise ValueError("response must contain exactly the rubric keys")
    for key, entry in value.items():
        if not isinstance(entry, dict) or set(entry) != {"score", "reason"}:
            raise ValueError(f"{key}: must contain exactly score and reason")
        score = entry["score"]
        if isinstance(score, bool) or not isinstance(score, int) or not 1 <= score <= 5:
            raise ValueError(f"{key}: score must be an integer within 1..5")
        if not isinstance(entry["reason"], str) or not entry["reason"].strip():
            raise ValueError(f"{key}: reason must be a non-empty string")
    return value


def scores_valid(scores: object, score_keys: list[str]) -> bool:
    if not isinstance(scores, dict) or set(scores) != set(score_keys):
        return False
    for entry in scores.values():
        if not isinstance(entry, dict) or set(entry) != {"score", "reason"}:
            return False
        score = entry["score"]
        if isinstance(score, bool) or not isinstance(score, int) or not 1 <= score <= 5:
            return False
        if not isinstance(entry["reason"], str) or not entry["reason"]:
            return False
    return True


__all__ = [
    "expected_points",
    "scores_valid",
    "strict_scores",
    "strict_winner",
]
