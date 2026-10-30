# Aesthetic Repair

Given a degraded webpage (screenshot + HTML source) and the model's own
Task 2 diagnosis as advisory context, the model under test emits a structured
source-level patch that should restore the page's aesthetic quality. Repairs
are applied locally, re-rendered, and graded by deterministic scorers.

Headline number: **Repair Pass** = `fix_pass ∧ collateral_pass ∧
content_preservation_pass ∧ visual_scope_pass`, over the fixed 660-sample
formal cohort; failed rows count as not passed and stay in the denominator.
The visual-scope scorer includes the baseline-restoration fast path
(per-pixel RGB comparison against the frozen baseline render).

## Setup

1. Install dependencies and the browser (Task 3 renders pages locally):

   ```bash
   pip install -r requirements.txt
   playwright install chromium
   ```

2. Unpack the Task 3 data release into `data/task3/` (see
   [`data/README.md`](../../data/README.md)): `cohort.json`,
   `defect_specs.jsonl`, and `pages/`.

3. Build the environment-dependent artifacts, once per environment:

   ```bash
   python -m tasks.task3.prebuild     # reference cache (renders every page)
   python -m tasks.task3.build_gt     # data/task3/unified_gt.jsonl + metadata.json
   ```

4. Configure keys in `configs/task3.yaml` (`api_key_env`). The optional
   `input.advisory_predictions` points at the same model's Task 2 predictions;
   missing/invalid advisory rows degrade to a `{}` handoff without changing
   the page or the scoring scope.

## Run

```bash
python -m tasks.task3.run --config configs/task3.yaml
python -m tasks.task3.evaluate outputs/task3/results/repair_<model>.jsonl
```

`run.py` is resumable and writes `repair_<model>.jsonl`;
`evaluate` aggregates the per-component pass rates and the headline Repair
Pass.
