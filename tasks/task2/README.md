# Aesthetic Diagnosis

Given two versions of the same web page — one with injected visual
degradations, one clean control, presented in random order — the model must,
in **one** multimodal call:

1. **detect** which version is degraded (`degraded_version`: 1 or 2);
2. **attribute** the defects to design dimensions (`dimensions`: Typography,
   Color, Spacing, Layout & Reading Flow, Visual Consistency);
3. **localize** the degraded elements (`locations`: slice index + box), after
   **declaring** its coordinate space (`normalized_1000` or `slice_pixels`)
   and box format (`xyxy` or `xywh`).

The answer is a strict five-key JSON object enforced server-side via
`json_schema` (`coordinate_contract.py`); declared boxes are canonicalized to
normalized-1000 xyxy without guessing or repair.

## Protocol highlights (frozen)

- Symmetric pair geometry (`pair-max-height-v1`): both versions share the
  pair's maximum page height, identical 1800 px fixed-canvas white-bottom
  slices (180 px overlap), and — only for multi-slice pages — a shared-scale
  top-aligned LANCZOS overview (1500 px). Both versions of one view always
  share one codec: if either PNG exceeds 9 MiB, both are sent as JPEG q90.
- Fixed denominator: API errors, schema failures, and wrong version decisions
  never leave the cohort. Attribution is detection-gated (wrong version →
  empty prediction set); localization on a wrong version counts every valid
  box as FP and every GT region as FN.
- GT regions merge rule-level boxes that share an XPath lineage and overlap
  heavily (IoU ≥ 0.9); predictions match GT per slice with GT clipped to the
  slice's visible rect, each GT accepting only its maximally visible slice(s).
- Retries: only HTTP 408/429/5xx and transport errors (5 s, then 10 s).
  Invalid responses and parse failures are terminal; every written record is
  terminal, so reruns resume by sample id and never re-ask a finished pair.

## Files

| file | role |
|---|---|
| `prompts.single_call.v8.json` | frozen prompt catalog (verbatim; do not reword) |
| `prompts.py` | catalog loader: dimensions table, label maps, prompt renderer |
| `metrics.py` | strict parsers + IoU/matching/label-metric math (verbatim) |
| `coordinate_contract.py` | declared-coordinate response schema + canonicalization |
| `geometry.py` | pair geometry, slicing, overview, pair codec, message parts |
| `scoring.py` | protocol constants, GT regions, per-record scoring, cohort IO |
| `run.py` | async inference pipeline (`python -m tasks.task2.run`) |
| `evaluate.py` | fixed-denominator aggregation (`python -m tasks.task2.evaluate`) |

## Usage

```bash
python -m tasks.task2.run --config configs/task2.yaml
python -m tasks.task2.evaluate outputs/task2/predictions/diagnosis_<model>.jsonl \
    --manifest data/task2/manifest.jsonl
```

Headline metrics in the emitted artifact (`design-bench-formal-metrics-v2`):

| metric | range |
|---|---|
| `degradation_detection_balanced_accuracy` | [0, 1] |
| `detection_gated_attribution_macro_j` | [-1, 1] |
| `version_aware_localization_micro_f1_at_0_5` | [0, 1] |
| `exact_chain_success_rate` | [0, 1] |
