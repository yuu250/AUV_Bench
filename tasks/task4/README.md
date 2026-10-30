# Text-to-UI Generation

Given one of 1,395 deduped product requirements, the model under test builds a
complete single-file HTML page from the frozen Top-K inputs (English query +
verbatim copy block + compact asset caption JSON + up to two 4×4 contact
sheets). The page is rendered at the frozen 1920×2160 CSS px @ DPR 2 viewport,
then scored by two VLM-judge metrics:

- **Cal. Aes. Score** (pointwise, `eval_pointwise`) — a judge rates each page
  on the Task 1 eight-dimension 1–5 rubric plus an `overall` 1–5 score, over
  the full 1,395-sample cohort with pinned prompt/schema versions. Headline
  number: `mean_overall_aesthetic_score` =
  `sum(overall of complete rows) / 1395`; failed rows contribute 0.
- **Pairwise** (`eval_pairwise`) — for every query, each pair of participants
  battles: the judge sees both renders interleaved (shared-scale overview +
  fixed-canvas slices) and picks `A` / `B` / `tie` on pure visual aesthetics.
  Win = 1, tie = 0.5. A participant whose input is invalid auto-loses without
  a judge call. Headline number: `overall_pairwise_win_rate` =
  `total points / (1395 × (participant_count − 1))`.

The paper's Task 4 headline (**Cal. Aes. Judge**) adds a human-anchored
post-hoc calibration on top of the pointwise scores: per-dimension isotonic
mappings fitted exclusively on Task 1 (judge raw scores vs. professional-
designer MOS), frozen, then applied per dimension to the pointwise
`results.jsonl` (8-dim mean per page; failed rows = 0; fixed denominator
1395). Run it via `python -m analysis.cal_aes` — see
[`analysis/README.md`](../../analysis/README.md). Note that
`mean_overall_aesthetic_score` in the pointwise summary is the *raw*
overall-based mean, not the calibrated headline.

## Setup

1. Install dependencies and the browser:

   ```bash
   pip install -r requirements.txt
   playwright install chromium        # needed for the rendering step
   ```

2. Unpack the Task 4 data release into `data/task4/` (see
   [`data/README.md`](../../data/README.md) for the layout). The release
   ships `queries.jsonl`, `cohort.json`, `freeze.json`, and the prepared
   per-sample Top-K `inputs/`.

3. Configure keys. The generator reads `configs/task4.yaml`
   (`api_key_env`, e.g. `DASHSCOPE_API_KEY`); the judges read
   `configs/task4_pointwise.yaml` / `configs/task4_pairwise.yaml`
   (default `OPENAI_API_KEY`). The official judge is GPT-5.4; any
   OpenAI-compatible VLM with `json_schema` support works, but scores are
   comparable only within one judge.

## Run the pipeline

```bash
# 1. generate (resumable; --dry-run builds requests without any API call)
python -m tasks.task4.run --config configs/task4.yaml
python -m tasks.task4.run --config configs/task4.yaml --limit 2   # smoke

# 2. render (per run directory; resumable)
python -m tasks.task4.render <model>-topk-v5 --workers 8

# 3a. pointwise Cal. Aes. Score (full 1,395-sample cohort)
python -m tasks.task4.eval_pointwise --config configs/task4_pointwise.yaml \
    --participant mymodel=<model>-topk-v5 --dry-run                 # input audit
python -m tasks.task4.eval_pointwise --config configs/task4_pointwise.yaml \
    --participant mymodel=<model>-topk-v5 --limit 5                 # smoke, no summary
python -m tasks.task4.eval_pointwise --config configs/task4_pointwise.yaml \
    --participant mymodel=<model>-topk-v5

# 3b. pairwise win rate (needs >= 2 participants)
python -m tasks.task4.eval_pairwise --config configs/task4_pairwise.yaml \
    --participant modelA=<run_a> --participant modelB=<run_b> --all
python -m tasks.task4.eval_pairwise --config configs/task4_pairwise.yaml \
    --participant modelA=<run_a> --participant modelB=<run_b> --sample-id <SAMPLE_ID>
```

Evaluators resume from their JSONL files; rows that ended in
`judge_failure` / `invalid_judge_output` stay terminal until you pass
`--rejudge-failures` (they are archived to `*.rejudged.jsonl` and re-judged).

Outputs:

```text
outputs/task4/
├── runs/<run_name>/<sample_id>/   # answer.html, assets/v01.., contact_sheet_*.jpg,
│   │                              # request_prompt.txt, input_manifest.json,
│   │                              # output_audit.json, full.png, crop4320.png,
│   │                              # render_meta.json
│   ├── _cohort.json  _meta.jsonl  _events.jsonl  _errors.jsonl
└── eval/{pointwise,pairwise}/<eval_version>-<judge>-<labels>/
                                 # run_meta.json, results.jsonl | battles.jsonl,
                                 # summary.json, formal_metrics.json
```

## Frozen protocol constants

| constant | value |
|---|---|
| input protocol | `task4-topk-contact-sheets-v5` |
| output audit protocol | `task4-topk-output-audit-v3` |
| render protocol | `task4-render-1920x2160-dpr2-v2` (crop always exactly 3840×4320) |
| pointwise eval | `task4-pointwise-visual-v2`; full 1,395-sample cohort |
| pairwise eval | `task4-pairwise-visual-v3`; A/B namespace `task4-pairwise-ab-assignment-v3` |
| judge calls | temperature 0, `json_schema`, `reasoning_effort: "none"`, no `max_tokens`, no `enable_thinking`; only 408/429/5xx + transport errors retry (5 s/10 s, 3 attempts) |
| generation retries | frozen ladder 5 s/15 s/45 s, at most 4 attempts per sample |
| image transport | 3840→1920 1x Lanczos, 2160 px slices, 180 px overlap, 1500 px overview, 12-slice top cap, paired views share one codec |

## Pairwise v3 and the A/B assignment

The A/B roles were historically assigned by SHA-256 parity. This repository
computes **no digests anywhere** (project rule), so v3 assigns roles by
byte-sum parity of the canonical string
`task4-pairwise-ab-assignment-v3|<identity>|<sample_id>|<label1>|<label2>`.
The assignment is deterministic, worker-independent, and puts each
participant on the A side in ~50% of battles (judge position bias cancels
out). Consequence: battle-level A/B roles are **not** comparable with
SHA-256-era results, and pairwise outcomes must not be mixed with or compared
against runs produced by the reference implementation. Win-rate statistics
are equivalent in expectation.

## No-hash identity model

Nothing in Task 4 computes or stores a digest. Identity is established by:

- byte counts (`answer_bytes`, `prompt_bytes`, `full_bytes`, …),
- full ID lists compared directly (`_cohort.json` carries all 1,395 IDs),
- whole-document equality (`run_meta.json` config is compared verbatim on
  resume; summaries must republish byte-identical content),
- deterministic schedule recomputation (every JSONL row is revalidated
  against its schedule slot on load).

## Deviations from the reference implementation

- API access goes through `src/api` (non-streaming SDK calls); the reference
  used raw SSE streams. Timeouts are governed by `request.timeout`.
  Generation with the Anthropic protocol may require streaming for large
  `max_tokens` — prefer an OpenAI-compatible endpoint for generation.
- The unified client does not expose token usage; `_meta.jsonl` keeps the
  `tokens` field as `null`.
- Only the official `topk` generation mode is ported (the legacy
  copy/assets/base modes and the asset pool are not part of the release).
- `copy.txt` ships pre-extracted in the data release; the extraction pipeline
  itself stays upstream. Prompt bytes are identical for released data.
- `formal_metrics.json` follows this repository's per-task inline artifact
  convention (same `design-bench-formal-metrics-v2` envelope as Tasks 1/2);
  provenance records paths only.
- `summary.json`/`run_meta.json`/JSONL records carry no digest fields;
  cross-configuration mixing is prevented by the whole-document `run_meta`
  comparison plus schedule recomputation.
