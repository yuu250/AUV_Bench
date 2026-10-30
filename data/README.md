# Data directory

Benchmark data is **not** shipped with this repository. It is distributed as
a separate data release directory (`auv_bench_data_v1/`) containing one
sub-directory per task:

```text
auv_bench_data_v1/
├── task1/    ├── task2/    ├── task3/    └── task4/
└── MANIFEST.txt                # file/sample counts for verification
```

After downloading the release, place it so the configs find it — any of:

1. **Merge into the repo (default paths):** copy the four task directories
   into this `data/` directory, i.e. `cp -r auv_bench_data_v1/task* data/`.
   Every shipped `configs/*.yaml` already points at `data/task<N>/...`.
2. **Symlink:** `ln -s /path/to/auv_bench_data_v1 data` (replaces this
   directory entirely).
3. **Custom location:** edit the `input.*` paths in each `configs/*.yaml`.

Sanity checks after placement: `data/task1/manifest.jsonl` has 1,395 lines,
`data/task2/manifest.jsonl` has 1,390 lines, `data/task3/pages/` has 660
directories, and `data/task4/` passes the pipeline's own fail-closed cohort
validation on first run. The per-task layouts below are what the release
provides (paths are configurable in `configs/task1.yaml` / `configs/task2.yaml`
etc.):

```text
data/task1/
├── manifest.jsonl              # one JSON object per line
├── images/                     # local page screenshots
│   ├── 0019_output/
│   │   └── 0019_rendered.png
│   └── ...
└── human_gt_cleaned_mos.csv    # human ground-truth scores (for evaluation)
```

## manifest.jsonl

One JSON object per line. Required fields:

| field | meaning |
|---|---|
| `id` | unique sample id |
| `image` | image file path relative to `images_root` |
| `image_url` | alternative to `image`: an already-hosted remote image URL |

Each sample needs either `image` or `image_url`.

## Ground-truth CSV

Header: `output_folder,color,typography,graphics,layout,component_consistency,visual_style_consistency,copy_quality,image_text_fit`

- `output_folder` is `{id}_output` for the sample with that id.
- Each dimension column holds a human score in `[1, 5]`.

Example:

```csv
output_folder,color,typography,graphics,layout,component_consistency,visual_style_consistency,copy_quality,image_text_fit
0019_output,3.0,2.5,3.5,2.5,3.5,4.5,4.0,4.5
```

## Task 2

```text
data/task2/
├── manifest.jsonl              # one JSON object per line (see below)
└── images/                     # paired screenshots referenced by "image"
```

Task 2 works on **pairs**: every degraded page has a clean control. The
manifest holds one JSON object per line for both members of each pair.

Degraded entry — required fields:

| field | meaning |
|---|---|
| `sample_id` | unique id; the clean control must be `{sample_id}_clean` |
| `source` | `"degraded"` |
| `layer` | cohort stratum label (e.g. diagnostic / combination / stress) |
| `image` | image file path relative to `images_root` |
| `image_size` | `[width, height]` in pixels; widths must match within a pair |
| `n_defects` | number of injected degradation rules |
| `gt_dims` | non-empty list of affected dimensions, from: 排版, 色彩, 间距, 布局阅读流, 视觉一致性 |
| `defects` | one object per injected rule (see below) |

Each `defects` entry needs:

| field | meaning |
|---|---|
| `bboxes` | list of `[x, y, width, height]` page-space boxes of the degraded elements |
| `victim_xpaths` | one XPath per bbox identifying the degraded element (used to merge boxes that describe one visual region); `localization_xpaths` is accepted as an alias |

Clean control entry — required fields: `sample_id` (`{degraded_id}_clean`),
`source` (`"paired_control"`), `layer`, `image`, `image_size`.

Pairs are excluded from the formal cohort when the widths differ, the shared
page height needs more than 12 slices, or any GT box is missing, malformed,
or outside the degraded page canvas.


## Task 3

```text
data/task3/
├── cohort.json                 # frozen ordered 660 formal sample ids
├── defect_specs.jsonl          # 660 environment-independent defect specs
├── render_fonts/
│   └── fonts/                  # 25 pinned open-licensed fonts (SIL OFL / Apache-2.0)
└── pages/                      # frozen degraded/baseline webpages
    └── <sample_id>_combo_output/
        ├── index.html              # degraded page (the repair target)
        └── ...                     # assets + baseline HTML
```

`render_fonts/fonts/` is a **required input**: the hermetic font environment
must contain exactly this pinned set before any render. `MANIFEST.json`,
`fonts.conf`, and `cache/` inside `render_fonts/` are regenerated
deterministically by `configure_render_environment` — do not ship or edit
them. Override the location with `TASK3_RENDER_FONTS_DIR` if needed.

Task 3 deliberately ships only environment-independent inputs. Two artifacts
are **generated locally, once per environment** (they depend on the local
render stack and are not part of the release):

```bash
python -m tasks.task3.prebuild     # renders pages/ into the reference cache
python -m tasks.task3.build_gt     # re-derives data/task3/unified_gt.jsonl
                                   # + data/task3/metadata.json from the specs
```

The JAA analysis (`analysis/jaa.py --layers`) reads the generated
`data/task3/unified_gt.jsonl` for its layer labels.


## Task 4

```text
data/task4/
├── queries.jsonl               # 1,395 lines, one query object per line
├── cohort.json                 # the shared dedup cohort (full included_ids list)
├── freeze.json                 # the active freeze (protocol triple, cohort binding)
└── inputs/                     # prepared Top-K pack, one directory per sample
    └── <sample_id>/
        ├── topk.jpg                  # audited 4-column montage of selected visuals
        ├── assets.caption.min.json   # {"f": ["id","src","w","h","cap"], "a": [...]}
        ├── selected_assets.json      # cross-checked against the caption array
        ├── copy.txt                  # pre-extracted verbatim copy lines (optional)
        └── assets/                   # the selected source visuals referenced by src
```

`queries.jsonl` fields used by the pipeline: `sample_id`, `target_lang`
(one of 32 language codes), `query_en` (the generation always prompts in
English; `target_lang` only names the language of the page's visible copy).

The cohort is fail-closed: the default queries must contain exactly the 1,395
IDs listed in `cohort.json`, and `freeze.json` must match the frozen protocol
triple (`task4-topk-contact-sheets-v5` / `task4-topk-output-audit-v3` /
`task4-render-1920x2160-dpr2-v2`). Identity throughout is established by
direct comparison and byte counts — no digests.
