import argparse
import math
import sys
from pathlib import Path

import pandas as pd

TIMEOUT = 300.0


def agile_score(runtime, solved):
    if not solved or pd.isna(runtime):
        return 0.0
    if runtime < 1.0:
        return 1.0
    return max(0.0, 1.0 - math.log(runtime) / math.log(TIMEOUT))


def resolve_csv_path(path):
    path = Path(path)
    if path.is_dir():
        for name in ("metrics.csv", "dev/metrics.csv"):
            candidate = path / name
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"no hay metrics.csv en {path}")
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def rank_candidates(csv_path):
    df = pd.read_csv(
        resolve_csv_path(csv_path),
        dtype={"candidate": str},
        index_col=False,
    )

    df["solved_bool"] = df["solved"].astype(str).str.lower().eq("yes")

    for col in ["runtime", "expanded", "evaluated", "generated", "cost"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    best_costs = (
        df[df["solved_bool"]]
        .groupby("instance")["cost"]
        .min()
    )

    df["best_observed_cost"] = df["instance"].map(best_costs)
    df["relative_gap"] = float("nan")

    mask = df["solved_bool"] & df["cost"].notna()

    df.loc[mask, "relative_gap"] = (
        df.loc[mask, "cost"] - df.loc[mask, "best_observed_cost"]
    ) / df.loc[mask, "best_observed_cost"]

    df["agile_score"] = df.apply(
        lambda row: agile_score(row["runtime"], row["solved_bool"]),
        axis=1,
    )

    ranking = (
        df.groupby("candidate", as_index=False)
        .agg(
            coverage=("solved_bool", "sum"),
            total_instances=("instance", "count"),
            mean_relative_gap=("relative_gap", "mean"),
            agile_score=("agile_score", "sum"),
            mean_runtime=("runtime", "mean"),
            mean_expanded=("expanded", "mean"),
            mean_evaluated=("evaluated", "mean"),
            mean_generated=("generated", "mean"),
        )
    )

    ranking["coverage_ratio"] = ranking["coverage"] / ranking["total_instances"]
    ranking["mean_relative_gap"] = ranking["mean_relative_gap"].fillna(float("inf"))
    ranking["mean_expanded"] = ranking["mean_expanded"].fillna(float("inf"))

    ranking = ranking.sort_values(
        ["coverage", "mean_relative_gap", "agile_score", "mean_expanded"],
        ascending=[False, True, False, True],
    ).reset_index(drop=True)

    ranking.insert(0, "rank", range(1, len(ranking) + 1))
    ranking["mean_relative_gap_pct"] = ranking["mean_relative_gap"] * 100

    return ranking


def main(argv=None):
    parser = argparse.ArgumentParser(description="Rankea candidatos a partir de un metrics.csv")
    parser.add_argument(
        "path",
        nargs="?",
        default="metrics.csv",
        help="metrics.csv o carpeta de experimento (busca metrics.csv)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="CSV de salida (default: ranking.csv junto al metrics)",
    )
    args = parser.parse_args(argv)

    csv_path = resolve_csv_path(args.path)
    ranking = rank_candidates(csv_path)
    out = Path(args.output) if args.output else csv_path.parent / "ranking.csv"
    ranking.to_csv(out, index=False)
    print(ranking.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    if len(ranking):
        print(f"\nSelected candidate: {ranking.iloc[0]['candidate']}")
    print(f"escrito: {out}")
    return ranking


if __name__ == "__main__":
    main()
    sys.exit(0)
