#!/usr/bin/env python3
"""Task 4 Calibrated Aesthetic Judge (Cal. Aes. Judge) post-hoc calibration.

This is the paper's Task 4 headline metric. GPT-5.4 (or another judge) scores
each generated UI on the Task 1 eight-dimension 1-5 rubric; per-dimension
monotonic isotonic calibration functions f_d map the raw judge scores onto the
professional-designer MOS scale:

  AesJudge_i = (1/8) * sum_d f_d(raw_{i,d})      for a valid generation
  AesJudge_i = 0                                  for invalid/failed rows
  AesJudge_m = (1/N) * sum_i AesJudge_i           over the fixed denominator N

Calibration protocol (frozen; matches the paper's Appendix E.1):

  * fitted EXCLUSIVELY on Task 1 data: judge raw scores joined with the
    cleaned professional-designer MOS over the Task 1 manifest;
  * one IsotonicRegression per dimension (increasing=True, y_min=1.0,
    y_max=5.0, out_of_bounds="clip"), evaluated on the raw grid {1,2,3,4,5};
  * validated with 5-fold cross-validation (KFold shuffle=True,
    random_state=20260915, pooled out-of-fold predictions; MAE / signed bias /
    SRCC per dimension plus the equally weighted macro row);
  * refit on all joined Task 1 samples, frozen, and only then applied to the
    Task 4 pointwise results. No Task 4 signal participates in any fitting
    decision; the optional pairwise win-rate file is a reference column only.

Inputs:

  --task1-scores     Task 1 prediction JSONL(s) from the judge model
                     (repeatable; first valid record per sample id wins across
                     files in the order given, so a supplement file rescues
                     ids the earlier file failed on). Record shape:
                     {"id": ..., "scores": {dim: {"score": int 1..5}, ...}}
  --task1-mos        cleaned human MOS CSV (output_folder == "<id>_output")
  --task1-manifest   Task 1 manifest JSONL (one {"id": ...} per line)
  --task4-pointwise  Task 4 pointwise results JSONL(s) (repeatable). Record
                     shape: {"participant", "sample_id", "status", "scores"};
                     only status == "complete" rows are scored, every other
                     row stays in the fixed denominator as zero.
  --pairwise-wr      optional Task 4 pairwise formal_metrics.json; only the
                     metrics.overall_pairwise_win_rate column is read, as a
                     reference alongside the calibrated scores.

Run from the repository root:

    python -m analysis.cal_aes \
        --task1-scores outputs/task1/predictions/scores_gpt-5.4.jsonl \
        --task1-mos data/task1/human_gt_cleaned_mos.csv \
        --task1-manifest data/task1/manifest.jsonl \
        --task4-pointwise outputs/task4/eval/<pointwise-out>/results.jsonl \
        --pairwise-wr outputs/task4/eval/<pairwise-out>/formal_metrics.json \
        --out-dir outputs/cal_aes

No API calls. Input files are never modified. No digests are computed.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold

from tasks.task1.prompts import DIM_KEYS as DIMS

DEFAULT_SEED = 20260915
DEFAULT_N_SPLITS = 5
DEFAULT_DENOMINATOR = 1395
RAW_GRID = (1, 2, 3, 4, 5)


class CalibrationError(ValueError):
    pass


def _valid_scores(scores: dict) -> bool:
    for dim in DIMS:
        value = (scores.get(dim) or {}).get("score")
        if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= 5:
            return False
    return True


def load_manifest_ids(path: Path) -> list[str]:
    ids: list[str] = []
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            sample_id = record.get("id")
            if not isinstance(sample_id, str) or not sample_id:
                raise CalibrationError(f"{path}:{line_number}: manifest row lacks id")
            ids.append(sample_id)
    if not ids:
        raise CalibrationError(f"{path}: manifest is empty")
    return ids


def load_task1(
    score_paths: list[Path], mos_path: Path, manifest_path: Path
) -> tuple[list[str], dict[str, dict[str, int]], dict[str, dict[str, float]], list[str]]:
    anomalies: list[str] = []
    manifest_ids = set(load_manifest_ids(manifest_path))

    gpt: dict[str, dict[str, int]] = {}
    for index, path in enumerate(score_paths):
        rows = 0
        added = 0
        rescued: list[str] = []
        with open(path, encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise CalibrationError(
                        f"{path}:{line_number}: invalid JSON"
                    ) from error
                rows += 1
                sample_id = record.get("id")
                if not isinstance(sample_id, str) or not sample_id:
                    raise CalibrationError(f"{path}:{line_number}: row lacks id")
                if sample_id in gpt:
                    continue
                scores = record.get("scores") or {}
                if not _valid_scores(scores):
                    continue
                gpt[sample_id] = {dim: scores[dim]["score"] for dim in DIMS}
                added += 1
                if index > 0:
                    rescued.append(sample_id)
        anomalies.append(
            f"task1 scores[{index}] {path}: rows={rows}, new valid ids={added}"
        )
        if index > 0:
            anomalies.append(f"  rescued by file[{index}]: {len(rescued)} {rescued}")

    human: dict[str, dict[str, float]] = {}
    with open(mos_path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"output_folder", *DIMS}
        missing_columns = sorted(required - set(reader.fieldnames or ()))
        if missing_columns:
            raise CalibrationError(
                f"{mos_path}: MOS CSV is missing columns: {', '.join(missing_columns)}"
            )
        for row_number, row in enumerate(reader, 2):
            output_folder = row.get("output_folder") or ""
            if not output_folder.endswith("_output"):
                raise CalibrationError(
                    f"{mos_path}:{row_number}: invalid output_folder {output_folder!r}"
                )
            sample_id = output_folder[: -len("_output")]
            if not sample_id:
                raise CalibrationError(f"{mos_path}:{row_number}: empty sample id")
            if sample_id in human:
                raise CalibrationError(f"{mos_path}:{row_number}: duplicate id {sample_id}")
            values = {}
            for dim in DIMS:
                try:
                    value = float(row[dim])
                except (TypeError, ValueError) as error:
                    raise CalibrationError(
                        f"{mos_path}:{row_number}: {dim} is not a number"
                    ) from error
                if not 1.0 <= value <= 5.0:
                    anomalies.append(f"MOS out of range: {sample_id} {dim}={value}")
                values[dim] = value
            human[sample_id] = values

    unknown_mos = sorted(set(human) - manifest_ids)
    if unknown_mos:
        anomalies.append(
            f"MOS rows outside the manifest are ignored: {len(unknown_mos)} "
            f"(first: {unknown_mos[0]})"
        )

    joined = sorted(manifest_ids & set(gpt) & set(human))
    missing_gpt = sorted(manifest_ids - set(gpt))
    missing_human = sorted(manifest_ids - set(human))
    if missing_gpt:
        anomalies.append(f"excluded, no valid judge scores: {missing_gpt}")
    if missing_human:
        anomalies.append(f"excluded, no human MOS: {missing_human}")
    anomalies.append(f"joined samples={len(joined)} of {len(manifest_ids)} manifest")
    if not joined:
        raise CalibrationError("no Task 1 samples joined; cannot fit calibration")
    return joined, gpt, human, anomalies


def metric_block(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    error = y_pred - y_true
    srcc = spearmanr(y_pred, y_true).statistic
    return {
        "mae": float(np.mean(np.abs(error))),
        "bias": float(np.mean(error)),
        "srcc": float(srcc),
    }


def cross_validate(
    joined: list[str],
    gpt: dict[str, dict[str, int]],
    human: dict[str, dict[str, float]],
    seed: int = DEFAULT_SEED,
    n_splits: int = DEFAULT_N_SPLITS,
) -> list[dict]:
    if not 2 <= n_splits <= len(joined):
        raise CalibrationError(
            f"n_splits={n_splits} must be within [2, {len(joined)} joined samples]"
        )
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    indices = np.arange(len(joined))
    results = []
    for dim in DIMS:
        x_all = np.array([gpt[sid][dim] for sid in joined], dtype=float)
        y_all = np.array([human[sid][dim] for sid in joined], dtype=float)
        raw_pred = np.empty_like(y_all)
        cal_pred = np.empty_like(y_all)
        for train_idx, test_idx in kfold.split(indices):
            iso = IsotonicRegression(increasing=True, y_min=1.0, y_max=5.0, out_of_bounds="clip")
            iso.fit(x_all[train_idx], y_all[train_idx])
            raw_pred[test_idx] = x_all[test_idx]
            cal_pred[test_idx] = iso.predict(x_all[test_idx])
        results.append(
            {
                "dim": dim,
                "raw": metric_block(y_all, raw_pred),
                "calibrated": metric_block(y_all, cal_pred),
            }
        )
    macro = {"dim": "MACRO"}
    for branch in ("raw", "calibrated"):
        macro[branch] = {
            key: float(np.mean([entry[branch][key] for entry in results]))
            for key in ("mae", "bias", "srcc")
        }
    return results + [macro]


def fit_final_mappings(
    joined: list[str],
    gpt: dict[str, dict[str, int]],
    human: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    grid = np.array([float(value) for value in RAW_GRID])
    mappings: dict[str, dict[str, float]] = {}
    for dim in DIMS:
        x_all = np.array([gpt[sid][dim] for sid in joined], dtype=float)
        y_all = np.array([human[sid][dim] for sid in joined], dtype=float)
        iso = IsotonicRegression(increasing=True, y_min=1.0, y_max=5.0, out_of_bounds="clip")
        iso.fit(x_all, y_all)
        calibrated = iso.predict(grid)
        mappings[dim] = {str(raw): float(value) for raw, value in zip(RAW_GRID, calibrated)}
    return mappings


def apply_task4(
    mappings: dict[str, dict[str, float]],
    results_paths: list[Path],
) -> tuple[dict[str, dict], dict[str, int]]:
    per_participant: dict[str, dict] = {}
    seen: set[tuple[str, str]] = set()
    total_rows = 0
    for path in results_paths:
        with open(path, encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise CalibrationError(
                        f"{path}:{line_number}: invalid JSON"
                    ) from error
                total_rows += 1
                label = record.get("participant")
                sample_id = record.get("sample_id")
                status = record.get("status")
                if not isinstance(label, str) or not label:
                    raise CalibrationError(f"{path}:{line_number}: row lacks participant")
                if not isinstance(sample_id, str) or not sample_id:
                    raise CalibrationError(f"{path}:{line_number}: row lacks sample_id")
                bucket = per_participant.setdefault(
                    label,
                    {
                        "n_valid": 0,
                        "raw_dimmean_sum": 0.0,
                        "cal_dimmean_sum": 0.0,
                        "raw_overall_sum": 0.0,
                        "status_counts": {},
                    },
                )
                bucket["status_counts"][status] = bucket["status_counts"].get(status, 0) + 1
                if status != "complete":
                    continue
                key = (label, sample_id)
                if key in seen:
                    raise CalibrationError(f"duplicate Task4 record: {key}")
                seen.add(key)
                scores = record.get("scores")
                if not isinstance(scores, dict) or not _valid_scores(scores):
                    raise CalibrationError(
                        f"{path}:{line_number}: complete row {key} lacks valid 8-dim scores"
                    )
                overall = (scores.get("overall") or {}).get("score")
                if not isinstance(overall, (int, float)) or isinstance(overall, bool):
                    raise CalibrationError(
                        f"{path}:{line_number}: complete row {key} lacks overall score"
                    )
                raw_dims = np.array([scores[dim]["score"] for dim in DIMS], dtype=float)
                cal_dims = np.array(
                    [mappings[dim][str(scores[dim]["score"])] for dim in DIMS], dtype=float
                )
                bucket["raw_dimmean_sum"] += float(np.mean(raw_dims))
                bucket["cal_dimmean_sum"] += float(np.mean(cal_dims))
                bucket["raw_overall_sum"] += float(overall)
                bucket["n_valid"] += 1
    return per_participant, {"total_rows": total_rows}


def build_leaderboard(
    per_participant: dict[str, dict],
    pairwise_wr_path: Path | None = None,
    denominator: int = DEFAULT_DENOMINATOR,
) -> list[dict]:
    win_rates: dict[str, float] | None = None
    if pairwise_wr_path is not None:
        with open(pairwise_wr_path, encoding="utf-8") as handle:
            artifact = json.load(handle)
        try:
            win_rates = artifact["metrics"]["overall_pairwise_win_rate"]
        except (KeyError, TypeError) as error:
            raise CalibrationError(
                f"{pairwise_wr_path}: missing metrics.overall_pairwise_win_rate"
            ) from error
    rows = []
    for label, bucket in per_participant.items():
        n_valid = bucket["n_valid"]
        pairwise_wr: float | None = None
        if win_rates is not None:
            if label not in win_rates:
                raise CalibrationError(
                    f"{pairwise_wr_path}: no pairwise win rate for participant {label!r}"
                )
            pairwise_wr = win_rates[label]
        rows.append(
            {
                "model": label,
                "raw_aes5": bucket["raw_dimmean_sum"] / denominator,
                "cal_aes5": bucket["cal_dimmean_sum"] / denominator,
                "valid_pages": n_valid,
                "coverage": n_valid / denominator,
                "raw_valid_only": bucket["raw_dimmean_sum"] / n_valid if n_valid else None,
                "cal_valid_only": bucket["cal_dimmean_sum"] / n_valid if n_valid else None,
                "overall_based_aes5": bucket["raw_overall_sum"] / denominator,
                "pairwise_wr": pairwise_wr,
            }
        )
    rows.sort(key=lambda row: -row["cal_aes5"])
    for rank, row in enumerate(rows, 1):
        row["rank_cal"] = rank
    raw_sorted = sorted(rows, key=lambda row: -row["raw_aes5"])
    for rank, row in enumerate(raw_sorted, 1):
        row["rank_raw"] = rank
    return rows


def fmt(value, digits=3):
    return "  NA  " if value is None else f"{value:.{digits}f}"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--task1-scores",
        type=Path,
        action="append",
        required=True,
        help="Task 1 judge prediction JSONL; repeat for supplement files "
        "(first valid record per id wins across files in order)",
    )
    parser.add_argument("--task1-mos", type=Path, required=True, help="cleaned human MOS CSV")
    parser.add_argument(
        "--task1-manifest", type=Path, required=True, help="Task 1 manifest JSONL"
    )
    parser.add_argument(
        "--task4-pointwise",
        type=Path,
        action="append",
        required=True,
        help="Task 4 pointwise results JSONL; repeatable",
    )
    parser.add_argument(
        "--pairwise-wr",
        type=Path,
        default=None,
        help="optional Task 4 pairwise formal_metrics.json (reference column only)",
    )
    parser.add_argument(
        "--denominator",
        type=int,
        default=DEFAULT_DENOMINATOR,
        help=f"fixed per-participant denominator (default: {DEFAULT_DENOMINATOR})",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="KFold seed")
    parser.add_argument("--n-splits", type=int, default=DEFAULT_N_SPLITS, help="CV folds")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/cal_aes"),
        help="output directory for calibration_report.json (default: outputs/cal_aes)",
    )
    args = parser.parse_args(argv)
    if args.denominator < 1:
        parser.error("--denominator must be at least 1")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    joined, gpt, human, anomalies = load_task1(
        args.task1_scores, args.task1_mos, args.task1_manifest
    )
    cv_table = cross_validate(joined, gpt, human, args.seed, args.n_splits)
    mappings = fit_final_mappings(joined, gpt, human)
    per_participant, task4_meta = apply_task4(mappings, args.task4_pointwise)
    leaderboard = build_leaderboard(per_participant, args.pairwise_wr, args.denominator)

    print("== DATA ALIGNMENT ==")
    print(f"Task1 scores    : {[str(path) for path in args.task1_scores]}")
    print(f"Task1 human MOS : {args.task1_mos}")
    print(f"Task1 manifest  : {args.task1_manifest}")
    print(f"Task4 pointwise : {[str(path) for path in args.task4_pointwise]} "
          f"(rows={task4_meta['total_rows']})")
    if args.pairwise_wr is not None:
        print(f"Task4 pairWR    : {args.pairwise_wr} (reference only, not fitted)")
    print("alignment key   : manifest.id == scores.id == output_folder minus '_output'")
    for note in anomalies:
        print(f"  ! {note}")

    print(f"\n== TASK1 {args.n_splits}-FOLD CV (seed={args.seed}, out-of-fold predictions pooled) ==")
    print(f"{'dim':<26}{'rawMAE':>8}{'calMAE':>8}{'rawBias':>9}{'calBias':>9}{'rawSRCC':>9}{'calSRCC':>9}")
    for entry in cv_table:
        raw, cal = entry["raw"], entry["calibrated"]
        print(
            f"{entry['dim']:<26}{raw['mae']:>8.3f}{cal['mae']:>8.3f}"
            f"{raw['bias']:>+9.3f}{cal['bias']:>+9.3f}{raw['srcc']:>9.3f}{cal['srcc']:>9.3f}"
        )

    print(f"\n== FINAL FROZEN LOOKUP TABLE (fitted on all {len(joined)} Task1 samples) ==")
    header = f"{'raw':<5}" + "".join(f"{dim[:12]:>14}" for dim in DIMS)
    print(header)
    for raw_value in ("1", "2", "3", "4", "5"):
        row = f"{raw_value:<5}"
        for dim in DIMS:
            row += f"{mappings[dim][raw_value]:>14.3f}"
        print(row)

    print(f"\n== TASK4 LEADERBOARD (denominator={args.denominator}, failures score 0) ==")
    print(
        f"{'rk':<3}{'model':<12}{'rawAES5':>8}{'calAES5':>8}{'delta':>7}"
        f"{'valid':>6}{'cov%':>7}{'rawVO':>7}{'calVO':>7}{'pairWR':>8}{'rkRaw':>6}"
    )
    for row in leaderboard:
        print(
            f"{row['rank_cal']:<3}{row['model']:<12}{row['raw_aes5']:>8.3f}{row['cal_aes5']:>8.3f}"
            f"{row['cal_aes5'] - row['raw_aes5']:>+7.3f}{row['valid_pages']:>6}"
            f"{row['coverage'] * 100:>6.1f}%{fmt(row['raw_valid_only']):>7}{fmt(row['cal_valid_only']):>7}"
            f"{fmt(row['pairwise_wr'], 4):>8}{row['rank_raw']:>6}"
        )

    artifact = {
        "protocol": "task4-cal-aes-judge-isotonic-v1",
        "seed": args.seed,
        "n_splits": args.n_splits,
        "denominator": args.denominator,
        "task1_joined_samples": len(joined),
        "anomalies": anomalies,
        "data_files": {
            "task1_scores": [str(path) for path in args.task1_scores],
            "task1_mos": str(args.task1_mos),
            "task1_manifest": str(args.task1_manifest),
            "task4_pointwise": [str(path) for path in args.task4_pointwise],
            "task4_pairwise_wr_reference_only": (
                str(args.pairwise_wr) if args.pairwise_wr is not None else None
            ),
        },
        "cv_table": cv_table,
        "frozen_mappings": mappings,
        "task4_leaderboard": leaderboard,
    }
    out_path = args.out_dir / "calibration_report.json"
    out_path.write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nreport written: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
