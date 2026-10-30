"""Offline tests for the frozen Task 4 prompt text and judge schemas."""

import json
import unittest

from tasks.task1.prompts import DIMENSIONS, RUBRIC_TEXT_EN
from tasks.task4 import prompts


class TestGenerationPrompts(unittest.TestCase):
    def test_prompt_interpolation(self):
        text = prompts.PROMPT.format(lang="German", extras="EXTRA\n", query="Do it")
        self.assertIn("All visible body text MUST be written in German", text)
        self.assertIn("EXTRA", text)
        self.assertTrue(text.endswith("## Requirement\nDo it"))
        self.assertIn("Output ONLY one ```html fenced code block", text)

    def test_prompt_copy_wraps_listing(self):
        text = prompts.PROMPT_COPY.format(listing="Hello\nWorld")
        self.assertIn("use VERBATIM", text)
        self.assertTrue(text.endswith("Hello\nWorld"))

    def test_topk_vs_empty_wording(self):
        self.assertIn("The attached contact sheets show", prompts.PROMPT_TOPK)
        self.assertIn("array below is empty", prompts.PROMPT_TOPK_EMPTY)
        for template in (prompts.PROMPT_TOPK, prompts.PROMPT_TOPK_EMPTY):
            self.assertIn("do not use external image or placeholder services",
                          template.lower() if template is prompts.PROMPT_TOPK_EMPTY else template)
            self.assertIn("{compact_json}", template)

    def test_protocol_constants(self):
        self.assertEqual(prompts.INPUT_PROTOCOL, "task4-topk-contact-sheets-v5")
        self.assertEqual(prompts.OUTPUT_AUDIT_PROTOCOL, "task4-topk-output-audit-v3")
        self.assertEqual(prompts.RENDER_PROTOCOL, "task4-render-1920x2160-dpr2-v2")

    def test_lang_name_coverage(self):
        self.assertEqual(len(prompts.LANG_NAME), 32)
        self.assertEqual(prompts.LANG_NAME["pt-BR"], "Brazilian Portuguese")
        self.assertEqual(prompts.LANG_NAME["zh"], "Chinese")


class TestPointwiseJudgePrompt(unittest.TestCase):
    def test_rubric_embedded_verbatim(self):
        self.assertIn(RUBRIC_TEXT_EN, prompts.POINTWISE_PROMPT)

    def test_dimension_keys_follow_task1_order(self):
        self.assertEqual(
            prompts.DIMENSION_KEYS, [key for key, *_ in DIMENSIONS]
        )
        self.assertEqual(len(prompts.DIMENSION_KEYS), 8)
        self.assertEqual(prompts.SCORE_KEYS, prompts.DIMENSION_KEYS + ["overall"])

    def test_prompt_lists_every_dimension(self):
        for key in prompts.DIMENSION_KEYS:
            self.assertIn(f'"{key}": {{"score": <1-5>', prompts.POINTWISE_PROMPT)
        self.assertIn('"overall": {"score": <1-5>', prompts.POINTWISE_PROMPT)

    def test_response_format_structure(self):
        fmt = prompts.POINTWISE_RESPONSE_FORMAT
        self.assertEqual(fmt["type"], "json_schema")
        schema = fmt["json_schema"]["schema"]
        self.assertTrue(fmt["json_schema"]["strict"])
        self.assertEqual(fmt["json_schema"]["name"], "task4_pointwise_visual_rubric")
        self.assertEqual(schema["required"], prompts.SCORE_KEYS)
        self.assertFalse(schema["additionalProperties"])
        entry = schema["properties"]["overall"]
        self.assertEqual(entry["properties"]["score"]["minimum"], 1)
        self.assertEqual(entry["properties"]["score"]["maximum"], 5)
        self.assertEqual(entry["required"], ["score", "reason"])
        json.dumps(fmt)  # must be JSON-serializable

    def test_versions_pinned(self):
        self.assertEqual(prompts.POINTWISE_PROMPT_VERSION,
                         "task4-pointwise-visual-rubric-v1")
        self.assertEqual(prompts.POINTWISE_RESPONSE_SCHEMA_VERSION,
                         "task4-pointwise-rubric-json-schema-v1")


class TestPairwiseJudgePrompt(unittest.TestCase):
    def test_prompt_markers(self):
        text = prompts.PAIRWISE_PROMPT
        self.assertIn("Screenshot A and Screenshot B", text)
        self.assertIn("solely by visual aesthetics", text)
        self.assertIn("Use tie only when neither has a clear aesthetic advantage.", text)
        self.assertNotIn("{", text)  # no interpolation placeholders

    def test_response_format_structure(self):
        fmt = prompts.PAIRWISE_RESPONSE_FORMAT
        schema = fmt["json_schema"]["schema"]
        self.assertTrue(fmt["json_schema"]["strict"])
        self.assertEqual(schema["properties"]["winner"]["enum"], ["A", "B", "tie"])
        self.assertEqual(schema["required"], ["winner"])
        self.assertFalse(schema["additionalProperties"])
        json.dumps(fmt)

    def test_versions_pinned(self):
        self.assertEqual(prompts.PAIRWISE_PROMPT_VERSION,
                         "task4-pairwise-pure-visual-aesthetics-v3")
        self.assertEqual(prompts.PAIRWISE_RESPONSE_SCHEMA_VERSION,
                         "task4-pairwise-winner-json-schema-v1")


if __name__ == "__main__":
    unittest.main()
