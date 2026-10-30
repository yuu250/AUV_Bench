#!/usr/bin/env python3
"""Judgment-Action Association (JAA, phi) over Task 2 and Task 3 results.

Instance-level cross-task association, aligned with the recorded protocols:

  D = Task 2 `det_correct` is True        (degraded version correctly identified)
  A = set(Task 2 `attr_pred_raw`) == set(`gt_dims`)   (exact affected-dimension
      set match; localization NOT involved; ECS is NOT used)
  J = D AND A
  F/C/P/S = Task 3 `fix_pass` / `collateral_pass` / `content_preservation_pass` /
            `visual_scope_pass` (each `is True`)
  R = F AND C AND P AND S                 (full Repair Pass; failure/terminal rows
                                           count as False and stay in the
                                           denominator; J is never a gate for R)

Each --run joins one model's Task 2 predictions JSONL with its Task 3 result
JSONL on sample_id (fail-closed: identical id sets, no duplicates).

Metrics per group: n11/n10/n01/n00 (J,R), N, pJ, pR, PARS (= n11/N),
  phi = (PARS - pJ*pR) / sqrt(pJ*(1-pJ)*pR*(1-pR)),
  P(R=1|J=1), P(R=1|J=0), P(J=1|R=1), deltaR = P(R|J=1) - P(R|J=0).
Any zero denominator -> N/A (empty CSV cell + flag); no smoothing, no abs(),
phi stays on the raw [-1, 1] scale.

Bootstrap: N_BOOT resamples (default 5000, seed 42) jointly resampling
instance-level (J, R) pairs.  Undefined resamples are dropped (never filled
with 0); per-metric n_valid is reported and percentile CIs require >= 40
valid resamples; groups with sparse cells or degenerate bootstrap are flagged
instead of smoothed over.

Run from the repository root:

    python -m analysis.jaa \
        --run mymodel=outputs/task2/predictions/diagnosis_mymodel.jsonl:outputs/task3/results/repair_mymodel.jsonl \
        --layers data/task3/unified_gt.jsonl \
        --out-dir outputs/jaa

No API calls. Input result files are never modified. No digests are computed.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPAIR_COMPONENTS = (
    "fix_pass",
    "collateral_pass",
    "content_preservation_pass",
    "visual_scope_pass",
)
METRIC_IDS = ("phi", "p_r_given_j1", "p_r_given_j0", "p_j_given_r1", "delta_r")
DEFAULT_N_BOOT = 5000
DEFAULT_SEED = 42
CI_MIN_VALID = 40
DEGENERATE_RATIO = 0.95
POOLED_LABEL = "__pooled_all_models__"


class JAAError(ValueError):
    pass


def load_jsonl(path: Path, kind: str) -> dict[str, dict]:
    records: dict[str, dict] = {}
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            sample_id = row.get("sample_id")
            if not isinstance(sample_id, str) or not sample_id:
                raise JAAError(f"{path}:{line_number}: {kind} row lacks sample_id")
            if sample_id in records:
                raise JAAError(f"{path}:{line_number}: duplicate {kind} sample_id {sample_id!r}")
            records[sample_id] = row
    return records


def load_layers(path: Path) -> dict[str, str]:
    layers: dict[str, str] = {}
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            sample_id = row.get("sample_id")
            layer = row.get("layer")
            if not isinstance(sample_id, str) or not isinstance(layer, str) or not layer:
                raise JAAError(
                    f"{path}:{line_number}: layer rows need sample_id and layer"
                )
            layers[sample_id] = layer
    return layers


def parse_run(value: str) -> tuple[str, Path, Path]:
    label, sep, rest = value.partition("=")
    if sep != "=" or not label:
        raise JAAError("each --run must have the form LABEL=TASK2_JSONL:TASK3_JSONL")
    task2, sep, task3 = rest.rpartition(":")
    if sep != ":" or not task2 or not task3:
        raise JAAError("each --run must have the form LABEL=TASK2_JSONL:TASK3_JSONL")
    return label, Path(task2), Path(task3)


def build_per_sample(
    runs: list[tuple[str, Path, Path]],
    layers: dict[str, str] | None,
) -> tuple[list[dict], list[str]]:
    """Join every run's Task 2 / Task 3 rows into instance-level JAA records."""

    per_sample: list[dict] = []
    split_names: list[str] = []
    for label, task2_path, task3_path in runs:
        task2 = load_jsonl(task2_path, "task2")
        task3 = load_jsonl(task3_path, "task3")
        if set(task2) != set(task3):
            only2 = sorted(set(task2) - set(task3))[:5]
            only3 = sorted(set(task3) - set(task2))[:5]
            raise JAAError(
                f"{label}: Task2/Task 3 sample id sets differ "
                f"(only task2: {only2}, only task3: {only3})"
            )
        for sample_id in sorted(task2):
            row2, row3 = task2[sample_id], task3[sample_id]
            d = row2.get("det_correct") is True
            pred, gt = row2.get("attr_pred_raw"), row2.get("gt_dims")
            a = isinstance(pred, list) and isinstance(gt, list) and set(pred) == set(gt)
            components = {key: row3.get(key) is True for key in REPAIR_COMPONENTS}
            layer = None
            if layers is not None:
                layer = layers.get(sample_id)
                if layer is None:
                    raise JAAError(f"{label}/{sample_id}: missing layer label")
                if layer not in split_names:
                    split_names.append(layer)
            per_sample.append({
                "model": label,
                "sample_id": sample_id,
                "layer": layer,
                "D": d,
                "A": a,
                "J": d and a,
                "F": components["fix_pass"],
                "C": components["collateral_pass"],
                "P": components["content_preservation_pass"],
                "S": components["visual_scope_pass"],
                "R": all(components.values()),
                "task3_status": row3.get("status"),
            })
    return per_sample, split_names


def group_metrics(j: np.ndarray, r: np.ndarray) -> dict:
    """Counts, point metrics and N/A reasons for one group of (J, R) rows."""

    n = int(len(j))
    n11 = int(np.sum(j & r))
    n10 = int(np.sum(j & ~r))
    n01 = int(np.sum(~j & r))
    n00 = int(np.sum(~j & ~r))
    out = {"N": n, "n11": n11, "n10": n10, "n01": n01, "n00": n00}
    na = []

    pJ = (n11 + n10) / n if n else None
    pR = (n11 + n01) / n if n else None
    pJR = n11 / n if n else None
    out.update(pJ=pJ, pR=pR, PARS=pJR)

    if pJ is None or pR is None or pJ in (0.0, 1.0) or pR in (0.0, 1.0):
        out["phi"] = None
        na.append("phi: undefined (J or R has zero variance in this group)")
    else:
        out["phi"] = (pJR - pJ * pR) / float(np.sqrt(pJ * (1 - pJ) * pR * (1 - pR)))

    def cond(num: int, den: int, name: str) -> float | None:
        if den == 0:
            na.append(f"{name}: undefined (denominator 0)")
            return None
        return num / den

    out["p_r_given_j1"] = cond(n11, n11 + n10, "P(R=1|J=1)")
    out["p_r_given_j0"] = cond(n01, n01 + n00, "P(R=1|J=0)")
    out["p_j_given_r1"] = cond(n11, n11 + n01, "P(J=1|R=1)")
    if out["p_r_given_j1"] is None or out["p_r_given_j0"] is None:
        out["delta_r"] = None
        na.append("deltaR: undefined (a conditional probability is undefined)")
    else:
        out["delta_r"] = out["p_r_given_j1"] - out["p_r_given_j0"]
    out["na_reasons"] = na
    return out


def bootstrap(rows: list[dict], rng: np.random.Generator, n_boot: int) -> dict:
    """Per-metric CI and validity counts for one group of per-sample rows."""

    j = np.array([x["J"] for x in rows])
    r = np.array([x["R"] for x in rows])
    n = len(rows)
    idx = rng.integers(0, n, size=(n_boot, n))
    jb, rb = j[idx], r[idx]
    vals: dict[str, list] = {m: [] for m in METRIC_IDS}
    for b in range(n_boot):
        g = group_metrics(jb[b], rb[b])
        for metric in METRIC_IDS:
            vals[metric].append(g[metric])
    out = {}
    for metric in METRIC_IDS:
        arr = np.array([v for v in vals[metric] if v is not None])
        n_valid = int(arr.size)
        entry = {"n_valid": n_valid}
        if n_valid >= CI_MIN_VALID:
            entry["ci_lo"] = float(np.percentile(arr, 2.5))
            entry["ci_hi"] = float(np.percentile(arr, 97.5))
        else:
            entry["ci_lo"] = entry["ci_hi"] = None
        entry["degenerate"] = n_valid < n_boot * DEGENERATE_RATIO
        out[metric] = entry
    return out


def summarize(
    per_sample: list[dict],
    runs: list[tuple[str, Path, Path]],
    split_names: list[str],
    n_boot: int,
    seed: int,
) -> tuple[list[dict], dict]:
    rng = np.random.default_rng(seed)
    groups: list[tuple[str, str, list[dict]]] = []
    for label, _, _ in runs:
        mrows = [x for x in per_sample if x["model"] == label]
        groups.append((label, "overall", mrows))
        for split in split_names:
            groups.append((label, split, [x for x in mrows if x["layer"] == split]))
    for split in split_names:
        groups.append((POOLED_LABEL, split,
                       [x for x in per_sample if x["layer"] == split]))
    groups.append((POOLED_LABEL, "overall", per_sample))

    summary_rows, boot_detail = [], {}
    for model, split, rows in groups:
        j = np.array([x["J"] for x in rows])
        r = np.array([x["R"] for x in rows])
        g = group_metrics(j, r)
        boot = bootstrap(rows, rng, n_boot)
        boot_detail[f"{model}|{split}"] = boot
        row = {
            "model": model, "split": split,
            "N": g["N"], "n11": g["n11"], "n10": g["n10"], "n01": g["n01"], "n00": g["n00"],
            "pJ": g["pJ"], "pR": g["pR"], "PARS": g["PARS"],
            "phi": g["phi"],
            "p_r_given_j1": g["p_r_given_j1"], "p_r_given_j0": g["p_r_given_j0"],
            "p_j_given_r1": g["p_j_given_r1"], "delta_r": g["delta_r"],
        }
        for metric in METRIC_IDS:
            row[f"{metric}_ci_lo"] = boot[metric]["ci_lo"]
            row[f"{metric}_ci_hi"] = boot[metric]["ci_hi"]
            row[f"{metric}_boot_n_valid"] = boot[metric]["n_valid"]
        flags = list(g["na_reasons"])
        n_j1, n_r1 = g["n11"] + g["n10"], g["n11"] + g["n01"]
        if n_j1 < 30:
            flags.append(f"sparse J=1 successes (n_j1={n_j1}); no stability conclusion")
        if n_r1 < 30:
            flags.append(f"sparse R=1 successes (n_r1={n_r1}); no stability conclusion")
        if 0 < g["N"] and g["n11"] < 10:
            flags.append(f"n11={g['n11']} < 10; phi/CI are count artifacts, not association evidence")
        if any(boot[metric]["degenerate"] for metric in METRIC_IDS):
            flags.append("bootstrap degenerate: <95% valid resamples for >=1 metric; no stability conclusion")
        row["flags"] = " | ".join(flags)
        summary_rows.append(row)
    return summary_rows, boot_detail


def write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> list[dict]:
    runs = [parse_run(value) for value in args.run]
    labels = [label for label, _, _ in runs]
    if len(labels) != len(set(labels)):
        raise JAAError("duplicate --run labels are not allowed")
    layers = load_layers(Path(args.layers)) if args.layers else None
    per_sample, split_names = build_per_sample(runs, layers)
    if not per_sample:
        raise JAAError("no joined samples")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "per_sample.csv", per_sample)
    summary_rows, boot_detail = summarize(
        per_sample, runs, split_names, args.n_boot, args.seed
    )
    write_csv(out_dir / "summary.csv",
              [row for row in summary_rows if row["split"] == "overall"])
    write_csv(out_dir / "by_split.csv", summary_rows)
    (out_dir / "bootstrap_detail.json").write_text(
        json.dumps(
            {"n_boot": args.n_boot, "seed": args.seed, "groups": boot_detail},
            indent=1,
        ),
        encoding="utf-8",
    )

    for row in summary_rows:
        if row["split"] != "overall" or row["model"] == POOLED_LABEL:
            continue

        def fmt(value):
            if value is None:
                return "N/A"
            return f"{value:+.4f}" if isinstance(value, float) else str(value)

        print(
            f"{row['model']:32s} N={row['N']} "
            f"n11={row['n11']:3d} n10={row['n10']:3d} n01={row['n01']:3d} n00={row['n00']:3d} "
            f"phi={fmt(row['phi'])} [{fmt(row['phi_ci_lo'])},{fmt(row['phi_ci_hi'])}] "
            f"P(R|J1)={fmt(row['p_r_given_j1'])} P(R|J0)={fmt(row['p_r_given_j0'])} "
            f"P(J|R1)={fmt(row['p_j_given_r1'])} dR={fmt(row['delta_r'])}",
            flush=True,
        )
    print(f"written: {out_dir}/per_sample.csv, summary.csv, by_split.csv, bootstrap_detail.json")
    return summary_rows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        metavar="LABEL=TASK2_JSONL:TASK3_JSONL",
        help="one model's joined Task 2 predictions and Task 3 results; repeatable",
    )
    parser.add_argument(
        "--layers",
        default=None,
        help="JSONL with sample_id + layer per row (e.g. data/task3/unified_gt.jsonl); "
             "omit to compute overall rows only",
    )
    parser.add_argument("--out-dir", default="outputs/jaa")
    parser.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)
    if not args.run:
        parser.error("at least one --run is required")
    if args.n_boot < 1:
        parser.error("--n-boot must be at least 1")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        run(args)
    except Exception as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
