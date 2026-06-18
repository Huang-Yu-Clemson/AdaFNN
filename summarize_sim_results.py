#!/usr/bin/env python
"""Summarize simulation Monte Carlo MSE results into paper-style tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


BASELINE_MODELS = [
    ("raw_51", "Raw data (51) + NN"),
    ("bspline_4", "B-spline (4) + NN"),
    ("bspline_15", "B-spline (15) + NN"),
    ("fpca_0p9", "FPCA0.9 + NN"),
    ("fpca_0p99", "FPCA0.99 + NN"),
]

ADAFNN_LAMBDAS = [
    (0.0, 0.0),
    (0.0, 1.0),
    (0.0, 2.0),
    (0.5, 0.0),
    (0.5, 1.0),
    (0.5, 2.0),
    (1.0, 0.0),
    (1.0, 1.0),
    (1.0, 2.0),
]


def tag_float(value: float) -> str:
    return str(value).replace(".", "p")


def read_metric(csv_path: Path, metric: str) -> tuple[float | None, float | None, int]:
    if not csv_path.exists():
        return None, None, 0

    df = pd.read_csv(csv_path)
    if "seed" in df.columns:
        df = df.drop_duplicates(subset=["seed"], keep="last")
    if metric not in df.columns:
        raise ValueError(f"{csv_path} does not contain column {metric!r}.")

    values = pd.to_numeric(df[metric], errors="coerce").dropna()
    if values.empty:
        return None, None, 0
    return float(values.mean()), float(values.std(ddof=1)), int(values.shape[0])


def build_rows(results_root: Path, cases: list[int], metric: str) -> list[dict]:
    rows = []

    for label, model_name in BASELINE_MODELS:
        row = {"model": model_name, "kind": "baseline", "source": label}
        for case in cases:
            csv_path = results_root / f"case{case}_baselines_mc" / label / "mse.csv"
            mean, std, count = read_metric(csv_path, metric)
            row[f"case{case}_mean"] = mean
            row[f"case{case}_std"] = std
            row[f"case{case}_n"] = count
        rows.append(row)

    for lambda1, lambda2 in ADAFNN_LAMBDAS:
        label = f"adafnn_l1_{tag_float(lambda1)}_l2_{tag_float(lambda2)}"
        model_name = f"AdaFNN ({lambda1:.1f}, {lambda2:.1f})"
        row = {"model": model_name, "kind": "adafnn", "source": label}
        for case in cases:
            csv_path = results_root / f"case{case}_mc" / label / "mse.csv"
            mean, std, count = read_metric(csv_path, metric)
            row[f"case{case}_mean"] = mean
            row[f"case{case}_std"] = std
            row[f"case{case}_n"] = count
        rows.append(row)

    return rows


def wide_table(summary: pd.DataFrame, cases: list[int], suffix: str) -> pd.DataFrame:
    columns = ["model"] + [f"case{case}_{suffix}" for case in cases]
    table = summary[columns].copy()
    table.columns = ["Model"] + [f"Case {case}" for case in cases]
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--output-dir", default="summary")
    parser.add_argument("--cases", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--metric", default="test_mse")
    parser.add_argument("--digits", type=int, default=6)
    args = parser.parse_args()

    results_root = Path(args.results_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.DataFrame(build_rows(results_root, args.cases, args.metric))
    mean_table = wide_table(summary, args.cases, "mean")
    std_table = wide_table(summary, args.cases, "std")
    count_table = wide_table(summary, args.cases, "n")

    summary.to_csv(output_dir / f"{args.metric}_summary_long.csv", index=False)
    mean_table.to_csv(output_dir / f"{args.metric}_mean_table.csv", index=False)
    std_table.to_csv(output_dir / f"{args.metric}_std_table.csv", index=False)
    count_table.to_csv(output_dir / f"{args.metric}_count_table.csv", index=False)

    print(f"\nMean {args.metric} table:")
    print(mean_table.round(args.digits).to_string(index=False))
    print(f"\nCounts per model/case:")
    print(count_table.to_string(index=False))
    print(f"\nWrote summary files to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
