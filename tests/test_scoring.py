"""Tests for model-output parsing and score validation."""

import json

from tasks.task1.prompts import DIM_KEYS
from tasks.task1.scoring import parse_scores, validate_scores


def make_scores(value=3):
    scores = {dim: {"score": value, "reason": "ok"} for dim in DIM_KEYS}
    scores["overall"] = {"score": value, "reason": "ok"}
    return scores


def test_parse_plain_json():
    assert parse_scores(json.dumps(make_scores())) == make_scores()


def test_parse_fenced_json_block():
    text = f"```json\n{json.dumps(make_scores())}\n```"
    assert parse_scores(text) == make_scores()


def test_parse_fenced_block_without_language_tag():
    text = f"```\n{json.dumps(make_scores())}\n```"
    assert parse_scores(text) == make_scores()


def test_parse_prose_wrapped_json():
    text = f"Here are the scores:\n{json.dumps(make_scores())}\nHope this helps."
    assert parse_scores(text) == make_scores()


def test_parse_garbage_returns_none():
    assert parse_scores("no json here at all") is None
    assert parse_scores("") is None


def test_validate_accepts_boundary_scores():
    assert validate_scores(make_scores(1))
    assert validate_scores(make_scores(5))
    assert validate_scores(make_scores(2.5))


def test_validate_rejects_missing_dimension():
    scores = make_scores()
    del scores["color"]
    assert not validate_scores(scores)


def test_validate_rejects_missing_overall():
    scores = make_scores()
    del scores["overall"]
    assert not validate_scores(scores)


def test_validate_rejects_out_of_range():
    assert not validate_scores(make_scores(0))
    assert not validate_scores(make_scores(6))


def test_validate_rejects_non_numeric_score():
    assert not validate_scores(make_scores("3"))


def test_validate_rejects_non_object_entry():
    scores = make_scores()
    scores["color"] = 3
    assert not validate_scores(scores)


def test_validate_rejects_missing_score_key():
    scores = make_scores()
    scores["color"] = {"reason": "no score"}
    assert not validate_scores(scores)
