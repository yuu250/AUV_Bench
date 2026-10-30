# Cross-task analysis

- **JAA** (`analysis/jaa.py`) — instance-level Task 2 × Task 3
  Judgment–Action Association (φ).
- **Cal. Aes. Judge** (`analysis/cal_aes.py`) — the Task 4 headline metric:
  human-anchored isotonic calibration of the pointwise judge scores.

# JAA (Judgment–Action Association, φ)

`analysis/jaa.py` computes the instance-level Task 2 × Task 3 association
between *judging* a degradation correctly and *repairing* it successfully.

## Definitions (aligned with the recorded Task 2/Task 3 protocols)

| symbol | definition | source field |
|---|---|---|
| D | degraded version correctly identified | Task 2 `det_correct is True` |
| A | affected-dimension set exactly matches GT | `set(attr_pred_raw) == set(gt_dims)` — raw predicted set, exact set match; **localization not involved, ECS not used** |
| J | judgment success | D AND A |
| F/C/P/S | repair components | Task 3 `fix_pass` / `collateral_pass` / `content_preservation_pass` / `visual_scope_pass` (each `is True`) |
| R | full Repair Pass | F∧C∧P∧S. Failure/non-complete rows count as False and stay in the denominator. **J is never a gate for R.** |

Each model's Task 2 predictions JSONL joins its Task 3 result JSONL on
`sample_id`; the join is fail-closed (identical id sets, no duplicates, no
row-number alignment).

## Metrics

Per group (overall and per layer split): 2×2 counts n11/n10/n01/n00 (J, R),
N, `pJ`, `pR`, `PARS` (= n11/N), and

```text
φ = (PARS − pJ·pR) / sqrt(pJ(1−pJ)pR(1−pR))        # raw [-1,1], no ×100
P(R=1|J=1) = n11/(n11+n10)   P(R=1|J=0) = n01/(n01+n00)
P(J=1|R=1) = n11/(n11+n01)   ΔR = P(R=1|J=1) − P(R=1|J=0)
```

Any zero denominator → **N/A** (empty CSV cell + `flags` note); no smoothing,
no `abs()`.

Bootstrap: 5,000 resamples (seed 42, `numpy.random.default_rng`) jointly
resampling instance-level (J, R) pairs. Undefined resamples are dropped
(never filled with 0); per-metric `*_boot_n_valid` is reported; CIs are the
2.5%/97.5% percentiles of valid resamples and require ≥ 40 valid draws.
Groups are flagged (not smoothed) when n_j1 < 30, n_r1 < 30, n11 < 10, or the
bootstrap is degenerate (< 95% valid resamples for any metric).

## Usage

```bash
python -m analysis.jaa \
    --run mymodel=outputs/task2/predictions/diagnosis_mymodel.jsonl:outputs/task3/results/repair_mymodel.jsonl \
    --run other=outputs/task2/predictions/diagnosis_other.jsonl:outputs/task3/results/repair_other.jsonl \
    --layers data/task3/unified_gt.jsonl \
    --out-dir outputs/jaa
```

- `--run LABEL=TASK2_JSONL:TASK3_JSONL` — one model; repeat per model.
- `--layers` — JSONL with `sample_id` + `layer` per row (e.g. the Task 3 GT).
  Omit it to compute overall rows only.
- `--n-boot` / `--seed` — default 5000 / 42.

## Outputs

| file | content |
|---|---|
| `per_sample.csv` | instance-level D/A/J/F/C/P/S/R + layer + task3_status |
| `summary.csv` | per-model overall rows (+ a `__pooled_all_models__` reference row, not a formal metric) |
| `by_split.csv` | model × (overall / each layer) rows + pooled reference rows |
| `bootstrap_detail.json` | per-group per-metric valid resample counts and CIs |

## Interpretation caveats

- φ is an instance-level 2×2 association measure — not a causal effect and
  not a cross-task coherence score.
- Simpson-type composition effects are expected: the overall association is
  driven mostly by both J and R concentrating in the easiest layer. Within a
  difficulty layer the association can shrink or flip sign. Always cite the
  split and the flags alongside any number.
- Rows labelled `__pooled_all_models__` are reference aggregates across
  models, not a formal metric.

# Cal. Aes. Judge (Task 4 headline calibration)

`analysis/cal_aes.py` produces the paper's Task 4 headline metric. It fits
per-dimension monotonic calibration functions that map the judge's raw 1–5
scores onto the professional-designer MOS scale, freezes them, and applies
them to the Task 4 pointwise results:

```text
AesJudge_i = (1/8) · Σ_d f_d(raw_{i,d})     for a valid generation
AesJudge_i = 0                               for invalid/failed rows
AesJudge_m = (1/N) · Σ_i AesJudge_i          over the fixed denominator N (1395)
```

## Calibration protocol (frozen)

- Fitted **exclusively on Task 1 data**: judge raw scores joined with the
  cleaned professional-designer MOS over the Task 1 manifest. No Task 4
  signal participates in any fitting decision.
- One `IsotonicRegression` per dimension (`increasing=True`, `y_min=1.0`,
  `y_max=5.0`, `out_of_bounds="clip"`), evaluated on the raw grid {1..5}.
- Validated with 5-fold cross-validation (`KFold(shuffle=True,
  random_state=20260915)`, pooled out-of-fold predictions; MAE / signed
  bias / SRCC per dimension plus the equally weighted macro row), then
  refit on all joined Task 1 samples and frozen.
- Requires `scikit-learn` (declared in `requirements.txt`).

## Usage

```bash
python -m analysis.cal_aes \
    --task1-scores outputs/task1/predictions/scores_gpt-5.4.jsonl \
    --task1-mos data/task1/human_gt_cleaned_mos.csv \
    --task1-manifest data/task1/manifest.jsonl \
    --task4-pointwise outputs/task4/eval/<pointwise-out>/results.jsonl \
    --pairwise-wr outputs/task4/eval/<pairwise-out>/formal_metrics.json \
    --out-dir outputs/cal_aes
```

- `--task1-scores` — repeatable; the first valid record per sample id wins
  across files in the order given, so a supplement file rescues ids the
  earlier file failed on.
- `--task4-pointwise` — repeatable; each file may hold several participants.
  Only `status == "complete"` rows are scored; every other row stays in the
  fixed denominator as zero.
- `--pairwise-wr` — optional reference column
  (`metrics.overall_pairwise_win_rate`); never used for fitting.
- `--denominator` (default 1395), `--seed` (default 20260915), `--n-splits`
  (default 5).

## Outputs

`calibration_report.json` (plus the same tables on stdout): data-alignment
anomalies, the CV table, the frozen per-dimension lookup table, and the
Task 4 leaderboard (`cal_aes5` headline, `raw_aes5`, coverage, valid-only
means, `overall_based_aes5`, reference `pairwise_wr`, both ranks).
Deterministic: identical inputs reproduce the report byte for byte.
