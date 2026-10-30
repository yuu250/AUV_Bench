#!/usr/bin/env python3
"""Frozen prompt text for Task 4 (UI generation) and its two VLM evaluators.

Generation prompts are byte-frozen: the generator sees the English query, the
verbatim copy block, the compact Top-K caption JSON, and up to two 4x4 contact
sheets.  The pointwise judge prompt embeds the Task 1 English rubric by import
(never copied); the pairwise judge prompt is a standalone pure-visual text.
"""

from __future__ import annotations

from typing import Any

from tasks.task1.prompts import DIMENSIONS, RUBRIC_TEXT_EN

INPUT_PROTOCOL = "task4-topk-contact-sheets-v5"
OUTPUT_AUDIT_PROTOCOL = "task4-topk-output-audit-v3"
RENDER_PROTOCOL = "task4-render-1920x2160-dpr2-v2"

LANG_NAME = {
    "ca": "Catalan", "cs": "Czech", "da": "Danish", "de": "German",
    "el": "Greek", "en": "English", "es": "Spanish", "fr": "French",
    "he": "Hebrew", "hu": "Hungarian", "id": "Indonesian", "it": "Italian",
    "ja": "Japanese", "ka": "Georgian", "ko": "Korean", "lv": "Latvian",
    "nl": "Dutch", "no": "Norwegian", "pl": "Polish", "pt": "Portuguese",
    "pt-BR": "Brazilian Portuguese", "ro": "Romanian", "ru": "Russian",
    "sk": "Slovak", "sl": "Slovenian", "sv": "Swedish", "th": "Thai",
    "tr": "Turkish", "uk": "Ukrainian", "uz": "Uzbek", "vi": "Vietnamese",
    "zh": "Chinese",
}

PROMPT = """Build a complete single-file HTML page for the product requirement below. Start coding directly.

- Single runnable HTML file, all CSS/JS inline
- All visible body text MUST be written in {lang}
- Follow the sections and style direction in the requirement
- Output ONLY one ```html fenced code block
{extras}
## Requirement
{query}"""

PROMPT_COPY = """
## Original page copy (use VERBATIM)
The original site's visible copy is listed below, one plain-text item per line. Use only the items needed for the
requested page; you do not need to display every item. Whenever you use an item, copy it verbatim. Do not invent,
translate, paraphrase or summarize visible prose. You decide the layout and styling freely.

{listing}"""

PROMPT_TOPK = """
## Selected local visual assets
The attached contact sheets show the filtered, content-bearing source visuals. Each cell is labeled with
the asset ID used by the compact JSON below. The JSON field order is `id, src, width, height, caption`.
Captions are short identification hints; use the attached visuals as the source of truth.
Every visual has a short ID such as `V01` and a matching short `src`; file extensions vary by asset.
Reference chosen images by copying the JSON `src` exactly; never guess or rewrite a path or extension.
Use these local assets for page imagery and do not use external image or placeholder services.

{compact_json}"""

PROMPT_TOPK_EMPTY = """
## Selected local visual assets
No content-bearing source visual survived the frozen asset filtering for this item, so no contact sheet is
attached and the compact asset array below is empty. Do not use external image or placeholder services.
Build the page's visual hierarchy with HTML and CSS only.

{compact_json}"""

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".avif", ".svg"}
COPY_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "a", "button",
             "figcaption", "blockquote", "th", "td", "label", "summary"}

# ---------------------------------------------------------------------------
# Pointwise judge (Calibrated Aesthetic Score)
# ---------------------------------------------------------------------------

POINTWISE_PROMPT_VERSION = "task4-pointwise-visual-rubric-v1"
POINTWISE_RESPONSE_SCHEMA_VERSION = "task4-pointwise-rubric-json-schema-v1"

DIMENSION_KEYS = [key for key, *_ in DIMENSIONS]
SCORE_KEYS = DIMENSION_KEYS + ["overall"]

_dims_json = ",\n".join(
    f'    "{key}": {{"score": <1-5>, "reason": "<one-sentence justification>"}}'
    for key in DIMENSION_KEYS
)
POINTWISE_PROMPT = (
    "You are a professional UI design reviewer. You will rate ONE web page "
    "screenshot. Pages taller than one slice are first shown as a shared-scale "
    "overview for navigation only, followed by fixed-canvas horizontal slices "
    "from top to bottom; use the slices for detailed visual decisions. Blank "
    "white space at the bottom of the last slice is padding, not page content. "
    "Rate the page solely by visual aesthetics according to the rubric below. "
    "Do not evaluate factual correctness, task compliance, functionality, "
    "implementation details, or inferred creator identity. Do not treat page "
    "length itself as an advantage or a disadvantage.\n\n"
    + RUBRIC_TEXT_EN
    + f"""
## Output Requirements

Output ONLY the following JSON, with no other text:

```json
{{
{_dims_json},
    "overall": {{"score": <1-5>, "reason": "<one-sentence overall comment>"}}
}}
```

Notes:
- Score each dimension independently; do not let one weak dimension drag down the others.
- "overall" is a holistic aesthetic judgment, not a simple average of the dimension scores.
- Keep each reason to a single sentence, in English.
"""
)

_score_entry: dict[str, Any] = {
    "type": "object",
    "properties": {
        "score": {"type": "integer", "minimum": 1, "maximum": 5},
        "reason": {"type": "string"},
    },
    "required": ["score", "reason"],
    "additionalProperties": False,
}
POINTWISE_RESPONSE_FORMAT: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "task4_pointwise_visual_rubric",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {key: _score_entry for key in SCORE_KEYS},
            "required": SCORE_KEYS,
            "additionalProperties": False,
        },
    },
}

# ---------------------------------------------------------------------------
# Pairwise judge
# ---------------------------------------------------------------------------

PAIRWISE_PROMPT_VERSION = "task4-pairwise-pure-visual-aesthetics-v3"
PAIRWISE_RESPONSE_SCHEMA_VERSION = "task4-pairwise-winner-json-schema-v1"

PAIRWISE_PROMPT = (
    "You will compare two web page screenshots, Screenshot A and Screenshot B. "
    "Pages taller than one slice are first shown as a shared-scale overview for "
    "navigation only, followed by fixed-canvas horizontal slices from top to "
    "bottom; use the slices for detailed visual decisions. When one page is "
    "shorter than the other, its later slices contain blank white space. Compare "
    "Screenshot A and Screenshot B solely by visual aesthetics, weighing: "
    "composition (balanced layout with deliberate use of space), visual "
    "hierarchy (clear primary and secondary structure that guides the eye), "
    "typography (readable, well-paired, appropriately sized and weighted text), "
    "color harmony (a cohesive, restrained palette with sufficient contrast), "
    "spacing and alignment (even rhythm, consistent gutters, no ragged or "
    "drifting edges), consistency (unified component styles across the page), "
    "and polish (finished details, with no placeholder text, broken images, or "
    "unstyled elements). Do not evaluate factual correctness, task compliance, "
    "functionality, implementation details, or inferred creator identity. Do "
    "not treat page length itself as an advantage or a disadvantage. Select the "
    "more visually appealing screenshot. Use tie only when neither has a clear "
    "aesthetic advantage."
)

PAIRWISE_RESPONSE_FORMAT: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "task4_pairwise_visual_winner",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "winner": {"type": "string", "enum": ["A", "B", "tie"]}
            },
            "required": ["winner"],
            "additionalProperties": False,
        },
    },
}


__all__ = [
    "COPY_TAGS",
    "DIMENSION_KEYS",
    "IMG_EXTS",
    "INPUT_PROTOCOL",
    "LANG_NAME",
    "OUTPUT_AUDIT_PROTOCOL",
    "PAIRWISE_PROMPT",
    "PAIRWISE_PROMPT_VERSION",
    "PAIRWISE_RESPONSE_FORMAT",
    "PAIRWISE_RESPONSE_SCHEMA_VERSION",
    "POINTWISE_PROMPT",
    "POINTWISE_PROMPT_VERSION",
    "POINTWISE_RESPONSE_FORMAT",
    "POINTWISE_RESPONSE_SCHEMA_VERSION",
    "PROMPT",
    "PROMPT_COPY",
    "PROMPT_TOPK",
    "PROMPT_TOPK_EMPTY",
    "RENDER_PROTOCOL",
    "SCORE_KEYS",
]
