# AUV-BENCH

**Aesthetic Understanding and Generation Evaluation for User Interfaces**

AUV-BENCH evaluates how multimodal models assess, diagnose, repair, and generate
web interface designs. This repository provides inference pipelines, webpage
rendering, evaluation scripts, and cross-task analysis tools.

| Task | Description | Configuration |
| --- | --- | --- |
| **1. Aesthetic Scoring** | Rate a webpage screenshot across eight aesthetic dimensions and compare the ratings with human judgments. | [task1.yaml](configs/task1.yaml) |
| **2. Aesthetic Diagnosis** | Identify the degraded version of a webpage, classify its design defects, and locate the affected regions. | [task2.yaml](configs/task2.yaml) |
| **3. Aesthetic Repair** | Produce a structured patch for a degraded webpage using its screenshot, HTML, and Task 2 diagnosis. | [task3.yaml](configs/task3.yaml) |
| **4. Text-to-UI Generation** | Generate a webpage from requirements, text, and image assets, then evaluate its visual quality. | [task4.yaml](configs/task4.yaml) |

Task 3 uses the same model's Task 2 predictions as context. Task 4 is an
independent generation task.

## Installation

Use Python 3.10 or later. Run the following commands from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m playwright install chromium
```

Chromium is used to render webpages in Tasks 3 and 4.

## Data

Place the four task folders from the `auv_bench_data_v1/` data package under
`data/`:

```bash
cp -R /path/to/auv_bench_data_v1/task{1,2,3,4} data/
```

```text
data/
├── task1/    # screenshots and human aesthetic ratings
├── task2/    # paired screenshots and defect annotations
├── task3/    # webpages, defect specifications, and rendering fonts
└── task4/    # generation requirements, text, and image assets
```

For Task 3, include the `render_fonts/fonts/` directory supplied with the data
package. See [Data preparation](data/README.md) for file layouts and annotation
formats.

## Model configuration

Set `model`, `protocol`, `base_url`, and `api_key_env` in the corresponding
[configuration file](configs/). The example task configurations use DashScope:

```bash
export DASHSCOPE_API_KEY="your-api-key"
```

Task 4's evaluation models are configured separately in
[task4_pointwise.yaml](configs/task4_pointwise.yaml) and
[task4_pairwise.yaml](configs/task4_pairwise.yaml). These use an OpenAI endpoint:

```bash
export OPENAI_API_KEY="your-api-key"
```

The inference client supports OpenAI-compatible endpoints (`protocol: openai`)
and the Anthropic Messages API (`protocol: anthropic`). Set `api_key_env` to the
environment variable for your provider; [.env.example](.env.example) lists the
supported examples. Task 4 evaluators use the OpenAI-compatible protocol.

## Evaluation

The commands below use the model names and output paths in the example
configurations. If you change a model or output path, update the corresponding
evaluation command as well.

### Task 1: Aesthetic Scoring

Run inference and compare the predicted ratings with human mean opinion scores:

```bash
python -m tasks.task1.run --config configs/task1.yaml
python -m tasks.task1.evaluate \
    outputs/task1/predictions/scores_qwen3.7-plus_en.jsonl \
    --gt data/task1/human_gt_cleaned_mos.csv \
    --manifest data/task1/manifest.jsonl
```

Evaluation reports Spearman rank correlation (SRCC) and mean absolute error
(MAE) for each dimension and their macro averages.

### Task 2: Aesthetic Diagnosis

```bash
python -m tasks.task2.run --config configs/task2.yaml
python -m tasks.task2.evaluate \
    outputs/task2/predictions/diagnosis_qwen3-vl-plus.jsonl \
    --manifest data/task2/manifest.jsonl
```

Evaluation measures degraded-version detection, defect attribution,
localization, and success across all three steps. See the
[Task 2 documentation](tasks/task2/README.md) for metric definitions.

### Task 3: Aesthetic Repair

First, render the reference pages and build the local ground-truth annotations.
Run these steps once for each rendering environment:

```bash
python -m tasks.task3.prebuild
python -m tasks.task3.build_gt
```

Set `input.advisory_predictions` in [task3.yaml](configs/task3.yaml) to the same
model's Task 2 output, then run repair and evaluation:

```bash
python -m tasks.task3.run --config configs/task3.yaml
python -m tasks.task3.evaluate \
    outputs/task3/results/repair_qwen3-vl-plus.jsonl
```

The main metric is **Repair Pass**: a repair must fix the defects, preserve the
content, avoid introducing other defects, and satisfy the allowed editing
scope. See the [Task 3 documentation](tasks/task3/README.md) for details.

### Task 4: Text-to-UI Generation

Generate the webpages and render them:

```bash
python -m tasks.task4.run --config configs/task4.yaml
python -m tasks.task4.render qwen3-vl-plus-topk-v5 --workers 8
```

The generated HTML, assets, and screenshots are saved under
`outputs/task4/runs/qwen3-vl-plus-topk-v5/`.

Evaluate each generated page on the full 1,395-sample cohort:

```bash
python -m tasks.task4.eval_pointwise --config configs/task4_pointwise.yaml \
    --participant qwen3-vl-plus=qwen3-vl-plus-topk-v5 \
    --out-dir outputs/task4/eval/pointwise/qwen3-vl-plus
```

To compare two models, replace `run_a` and `run_b` below with their run directory
names under `outputs/task4/runs/`:

```bash
python -m tasks.task4.eval_pairwise --config configs/task4_pairwise.yaml \
    --participant model_a=run_a --participant model_b=run_b --all
```

Pointwise scores are saved in `results.jsonl`; pairwise judgments are saved in
`battles.jsonl`. Both evaluators produce `summary.json` and
`formal_metrics.json`. See the [Task 4 documentation](tasks/task4/README.md) for
the rendering and evaluation settings.

## Cross-task analysis

### Calibrated aesthetic scores

**Cal. Aes. Judge** calibrates the Task 4 judge's scores against human ratings
from Task 1. It requires Task 1 predictions from the **same judge model** used
for Task 4 pointwise evaluation.

Copy `configs/task1.yaml` to `configs/task1_judge.yaml`, then set its `model`,
`protocol`, `base_url`, and `api_key_env` to match
`configs/task4_pointwise.yaml`. Run the judge on Task 1 and apply the calibration:

```bash
python -m tasks.task1.run --config configs/task1_judge.yaml \
    --output outputs/task1/predictions/judge_scores.jsonl
python -m analysis.cal_aes \
    --task1-scores outputs/task1/predictions/judge_scores.jsonl \
    --task1-mos data/task1/human_gt_cleaned_mos.csv \
    --task1-manifest data/task1/manifest.jsonl \
    --task4-pointwise outputs/task4/eval/pointwise/qwen3-vl-plus/results.jsonl \
    --out-dir outputs/cal_aes
```

The calibrated scores are written to `outputs/cal_aes/calibration_report.json`.
The raw pointwise mean in `summary.json` is reported separately.

### Judgment–Action Association

**JAA** measures the association between correct diagnosis in Task 2 and
successful repair in Task 3:

```bash
python -m analysis.jaa \
    --run qwen3-vl-plus=outputs/task2/predictions/diagnosis_qwen3-vl-plus.jsonl:outputs/task3/results/repair_qwen3-vl-plus.jsonl \
    --layers data/task3/unified_gt.jsonl \
    --out-dir outputs/jaa
```

See [Analysis](analysis/README.md) for definitions and output formats.

## Development

```bash
pip install -r requirements-dev.txt
python -m pytest tests/
```
