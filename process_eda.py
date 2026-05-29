#!/usr/bin/env python3
"""Process-level EDA for manufacturing/queueing datasets.

Generates EDA outputs per process/station and a cross-process summary table
that maps directly to queueing-model input checks.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype

# Optional plotting dependency.
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except Exception:
    HAS_MATPLOTLIB = False


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(text)).strip("_") or "unknown"


def read_table(path: Path) -> pd.DataFrame:
    ext = path.suffix.lower()
    if ext == ".csv":
        return pd.read_csv(path)
    if ext in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if ext == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file type: {ext}. Use CSV, XLSX, XLS, or Parquet.")


def parse_datetime_if_present(df: pd.DataFrame, cols: Iterable[str | None]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col and col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce")
    return out


def add_time_kpis(
    df: pd.DataFrame,
    arrival_col: str | None,
    start_col: str | None,
    end_col: str | None,
) -> pd.DataFrame:
    out = df.copy()

    if arrival_col and start_col and arrival_col in out.columns and start_col in out.columns:
        out["wait_min"] = (out[start_col] - out[arrival_col]).dt.total_seconds() / 60.0

    if start_col and end_col and start_col in out.columns and end_col in out.columns:
        out["service_min"] = (out[end_col] - out[start_col]).dt.total_seconds() / 60.0

    if arrival_col and end_col and arrival_col in out.columns and end_col in out.columns:
        out["cycle_min"] = (out[end_col] - out[arrival_col]).dt.total_seconds() / 60.0

    if arrival_col and arrival_col in out.columns:
        ordered = out.sort_values(arrival_col)
        ia = ordered[arrival_col].diff().dt.total_seconds() / 60.0
        out.loc[ordered.index, "interarrival_min"] = ia

    return out


def build_numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    numeric = df.select_dtypes(include=[np.number])
    if numeric.empty:
        return pd.DataFrame(columns=["column"])

    q = numeric.quantile([0.05, 0.25, 0.5, 0.75, 0.90, 0.95]).T
    desc = numeric.describe().T

    out = pd.DataFrame(index=numeric.columns)
    out["count"] = desc["count"]
    out["mean"] = desc["mean"]
    out["std"] = desc["std"]
    out["min"] = desc["min"]
    out["p05"] = q[0.05]
    out["p25"] = q[0.25]
    out["p50"] = q[0.5]
    out["p75"] = q[0.75]
    out["p90"] = q[0.90]
    out["p95"] = q[0.95]
    out["max"] = desc["max"]

    iqr = out["p75"] - out["p25"]
    low = out["p25"] - 1.5 * iqr
    high = out["p75"] + 1.5 * iqr

    outlier_counts = {}
    for col in numeric.columns:
        s = numeric[col]
        outlier_counts[col] = int(((s < low[col]) | (s > high[col])).sum())
    out["outlier_count_iqr"] = pd.Series(outlier_counts)

    cv = (out["std"] / out["mean"]).replace([np.inf, -np.inf], np.nan)
    out["cv"] = cv

    return out.reset_index().rename(columns={"index": "column"})


def build_categorical_summary(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    cat_cols = [
        c
        for c in df.columns
        if pd.api.types.is_object_dtype(df[c])
        or isinstance(df[c].dtype, CategoricalDtype)
        or pd.api.types.is_bool_dtype(df[c])
    ]

    rows = []
    for col in cat_cols:
        vc = df[col].astype("string").fillna("<NA>").value_counts(dropna=False).head(top_n)
        total = len(df)
        for value, count in vc.items():
            rows.append(
                {
                    "column": col,
                    "value": value,
                    "count": int(count),
                    "pct": float(count) / total if total else np.nan,
                }
            )
    return pd.DataFrame(rows)


def build_missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    miss = df.isna().sum()
    return (
        pd.DataFrame({"column": miss.index, "missing_count": miss.values})
        .assign(missing_pct=lambda x: x["missing_count"] / n if n else np.nan)
        .sort_values(["missing_pct", "missing_count"], ascending=False)
    )


def series_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()


def coefficient_of_variation(series: pd.Series) -> float | None:
    s = series_numeric(series)
    if len(s) < 2:
        return None
    mean = float(s.mean())
    if np.isclose(mean, 0.0):
        return None
    std = float(s.std(ddof=1))
    return std / mean


def metric_stat_row(name: str, series: pd.Series) -> dict[str, float | int | str | None]:
    s = series_numeric(series)
    if len(s) == 0:
        return {
            "metric": name,
            "count": 0,
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "p75": None,
            "p90": None,
            "p95": None,
            "max": None,
            "cv": None,
        }

    mean = float(s.mean())
    std = float(s.std(ddof=1)) if len(s) > 1 else 0.0
    cv = None if np.isclose(mean, 0.0) else float(std / mean)

    return {
        "metric": name,
        "count": int(s.count()),
        "mean": mean,
        "median": float(s.median()),
        "std": std,
        "min": float(s.min()),
        "p75": float(s.quantile(0.75)),
        "p90": float(s.quantile(0.90)),
        "p95": float(s.quantile(0.95)),
        "max": float(s.max()),
        "cv": cv,
    }


def estimate_servers(
    proc_df: pd.DataFrame,
    servers_col: str | None,
    machine_col: str | None,
) -> int:
    for col in [servers_col, machine_col]:
        if col and col in proc_df.columns:
            n = int(proc_df[col].dropna().nunique())
            if n > 0:
                return n
    return 1


def queue_model_suggestion(
    cv_arrival: float | None,
    cv_service: float | None,
    servers: int,
    has_batch_signal: bool,
) -> str:
    c = max(1, servers)

    if has_batch_signal:
        return "Batch queue model"

    if cv_arrival is None or cv_service is None:
        return f"G/G/{c}"

    random_arrivals = 0.8 <= cv_arrival <= 1.25
    random_service = 0.8 <= cv_service <= 1.25
    scheduled_arrivals = cv_arrival < 0.6

    if random_arrivals and random_service:
        return f"M/M/{c}"

    if scheduled_arrivals:
        return f"D/G/{c}"

    return f"G/G/{c}"


def save_hist(series: pd.Series, path: Path, title: str, xlabel: str) -> None:
    if not HAS_MATPLOTLIB:
        return
    s = series_numeric(series)
    if len(s) == 0:
        return
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    ax.hist(s, bins=30)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_bar(
    x: list,
    y: list,
    path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    rotate_x: bool = False,
) -> None:
    if not HAS_MATPLOTLIB or len(x) == 0:
        return
    fig = plt.figure(figsize=(8, 4.5))
    ax = fig.add_subplot(111)
    ax.bar(x, y)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if rotate_x:
        plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_line(
    x: list,
    y: list,
    path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    if not HAS_MATPLOTLIB or len(x) == 0:
        return
    fig = plt.figure(figsize=(8, 4.5))
    ax = fig.add_subplot(111)
    ax.plot(x, y, marker="o", linewidth=1.4, markersize=3)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_boxplot(
    df: pd.DataFrame,
    value_col: str,
    by_col: str,
    path: Path,
    title: str,
    max_categories: int = 12,
) -> None:
    if not HAS_MATPLOTLIB:
        return
    if value_col not in df.columns or by_col not in df.columns:
        return

    tmp = df[[value_col, by_col]].dropna()
    if tmp.empty:
        return

    top = tmp[by_col].astype(str).value_counts().head(max_categories).index.tolist()
    tmp = tmp[tmp[by_col].astype(str).isin(top)]

    grouped = []
    labels = []
    for label in top:
        g = series_numeric(tmp[tmp[by_col].astype(str) == label][value_col])
        if len(g) > 0:
            grouped.append(g)
            labels.append(label)
    if len(grouped) == 0:
        return

    fig = plt.figure(figsize=(max(8, len(labels) * 0.8), 4.8))
    ax = fig.add_subplot(111)
    ax.boxplot(grouped, tick_labels=labels, showfliers=False)
    ax.set_title(title)
    ax.set_xlabel(by_col)
    ax.set_ylabel(value_col)
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def queue_kpi_snapshot(
    proc_df: pd.DataFrame,
    service_mean: float | None,
    wait_mean: float | None,
    wait_p90: float | None,
    arrival_rate_per_hour: float | None,
    servers: int,
    has_batch_signal: bool,
) -> dict[str, float | int | str | None]:
    cv_service = coefficient_of_variation(proc_df.get("service_min", pd.Series(dtype=float)))
    cv_interarrival = coefficient_of_variation(proc_df.get("interarrival_min", pd.Series(dtype=float)))

    utilization = None
    if service_mean is not None and arrival_rate_per_hour is not None and servers > 0:
        utilization = (arrival_rate_per_hour * (service_mean / 60.0)) / servers

    queue_model = queue_model_suggestion(
        cv_arrival=cv_interarrival,
        cv_service=cv_service,
        servers=servers,
        has_batch_signal=has_batch_signal,
    )

    snapshot: dict[str, float | int | str | None] = {
        "lots_processed": int(len(proc_df)),
        "mean_service_time_min": service_mean,
        "cv_service": cv_service,
        "mean_wait_min": wait_mean,
        "p90_wait_min": wait_p90,
        "arrivals_per_hour": arrival_rate_per_hour,
        "servers": int(servers),
        "utilization": utilization,
        "queue_model_suggestion": queue_model,
    }

    return snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description="EDA per process/station for plant data")
    parser.add_argument("--input", required=True, help="Path to CSV/XLSX/XLS/Parquet")
    parser.add_argument(
        "--output",
        default="eda_output",
        help="Output folder for reports/charts (default: eda_output)",
    )
    parser.add_argument(
        "--process-col",
        default="process",
        help="Column with process/station label (default: process)",
    )
    parser.add_argument("--lot-col", default="lot_id", help="Lot identifier column")
    parser.add_argument("--product-col", default="product_type", help="Product type column")
    parser.add_argument("--machine-col", default="machine_id", help="Machine/station id column")
    parser.add_argument("--shift-col", default="shift", help="Shift column")
    parser.add_argument("--batch-col", default="batch_size", help="Batch size column")
    parser.add_argument(
        "--servers-col",
        default=None,
        help="Optional server id column (if omitted, machine_col is used)",
    )
    parser.add_argument(
        "--arrival-col",
        default=None,
        help="Arrival timestamp column (optional)",
    )
    parser.add_argument(
        "--start-col",
        default=None,
        help="Service start timestamp column (optional)",
    )
    parser.add_argument(
        "--end-col",
        default=None,
        help="Service end/completion timestamp column (optional)",
    )
    args = parser.parse_args()

    in_path = Path(args.input).expanduser().resolve()
    out_root = Path(args.output).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    df = read_table(in_path)
    df = parse_datetime_if_present(df, [args.arrival_col, args.start_col, args.end_col])

    process_col = args.process_col
    if process_col not in df.columns:
        process_col = "__process__"
        df[process_col] = in_path.stem

    overall = {
        "source_file": str(in_path),
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "process_col": process_col,
        "process_count": int(df[process_col].nunique(dropna=True)),
        "has_plots": HAS_MATPLOTLIB,
    }
    (out_root / "overall.json").write_text(json.dumps(overall, indent=2), encoding="utf-8")

    process_rows: list[dict[str, float | int | str | None]] = []

    for proc in sorted(df[process_col].dropna().astype(str).unique()):
        proc_raw = df[df[process_col].astype(str) == proc].copy()
        proc_df = add_time_kpis(proc_raw, args.arrival_col, args.start_col, args.end_col)

        proc_dir = out_root / safe_name(proc)
        charts_dir = proc_dir / "charts"
        proc_dir.mkdir(parents=True, exist_ok=True)
        charts_dir.mkdir(parents=True, exist_ok=True)

        missing = build_missing_summary(proc_df)
        numeric = build_numeric_summary(proc_df)
        categorical = build_categorical_summary(proc_df)

        missing.to_csv(proc_dir / "missing_summary.csv", index=False)
        numeric.to_csv(proc_dir / "numeric_summary.csv", index=False)
        categorical.to_csv(proc_dir / "categorical_top_values.csv", index=False)
        proc_df.head(200).to_csv(proc_dir / "sample_head_200.csv", index=False)

        # Data-quality checks for impossible timestamp order.
        quality = {}
        if args.arrival_col and args.start_col and args.arrival_col in proc_df.columns and args.start_col in proc_df.columns:
            quality["start_before_arrival_count"] = int((proc_df[args.start_col] < proc_df[args.arrival_col]).sum())
        if args.start_col and args.end_col and args.start_col in proc_df.columns and args.end_col in proc_df.columns:
            quality["end_before_start_count"] = int((proc_df[args.end_col] < proc_df[args.start_col]).sum())
        if args.arrival_col and args.end_col and args.arrival_col in proc_df.columns and args.end_col in proc_df.columns:
            quality["end_before_arrival_count"] = int((proc_df[args.end_col] < proc_df[args.arrival_col]).sum())
        (proc_dir / "data_quality.json").write_text(json.dumps(quality, indent=2), encoding="utf-8")

        # Optional process-level mix views.
        if args.product_col in proc_df.columns:
            mix = (
                proc_df[args.product_col]
                .astype("string")
                .fillna("<NA>")
                .value_counts(dropna=False)
                .rename_axis(args.product_col)
                .reset_index(name="count")
            )
            mix["pct"] = mix["count"] / max(1, len(proc_df))
            mix.to_csv(proc_dir / "product_mix.csv", index=False)

        # Per-process queue metric stats.
        metric_rows = []
        for m in ["service_min", "wait_min", "cycle_min", "interarrival_min"]:
            if m in proc_df.columns:
                metric_rows.append(metric_stat_row(m, proc_df[m]))
        pd.DataFrame(metric_rows).to_csv(proc_dir / "queue_metric_stats.csv", index=False)

        # Arrival analysis outputs.
        if args.arrival_col and args.arrival_col in proc_df.columns:
            arr = proc_df[args.arrival_col].dropna()
            if len(arr) > 0:
                by_hod = arr.dt.hour.value_counts().sort_index()
                by_hod_df = pd.DataFrame({"hour_of_day": by_hod.index, "arrivals": by_hod.values})
                by_hod_df.to_csv(proc_dir / "arrivals_by_hour_of_day.csv", index=False)
                save_bar(
                    x=by_hod_df["hour_of_day"].tolist(),
                    y=by_hod_df["arrivals"].tolist(),
                    path=charts_dir / "arrival_by_hour_of_day.png",
                    title=f"Arrivals by Hour of Day - {proc}",
                    xlabel="Hour of day",
                    ylabel="Arrivals",
                )

                hourly_timeline = (
                    arr.sort_values()
                    .to_frame(name=args.arrival_col)
                    .set_index(args.arrival_col)
                    .resample("1h")
                    .size()
                    .rename("arrivals")
                    .reset_index()
                )
                hourly_timeline.to_csv(proc_dir / "arrivals_timeline_hourly.csv", index=False)
                save_line(
                    x=[str(t) for t in hourly_timeline[args.arrival_col].tolist()],
                    y=hourly_timeline["arrivals"].tolist(),
                    path=charts_dir / "arrivals_timeline_hourly.png",
                    title=f"Hourly Arrival Timeline - {proc}",
                    xlabel="Hour",
                    ylabel="Arrivals",
                )

                cumulative = (
                    arr.sort_values()
                    .to_frame(name=args.arrival_col)
                    .assign(cumulative_arrivals=lambda x: np.arange(1, len(x) + 1))
                )
                cumulative.to_csv(proc_dir / "cumulative_arrivals_curve.csv", index=False)
                save_line(
                    x=[str(t) for t in cumulative[args.arrival_col].tolist()],
                    y=cumulative["cumulative_arrivals"].tolist(),
                    path=charts_dir / "cumulative_arrivals_curve.png",
                    title=f"Cumulative Arrivals - {proc}",
                    xlabel="Time",
                    ylabel="Cumulative arrivals",
                )

        # Shift arrivals summary.
        if args.shift_col in proc_df.columns:
            shift_counts = (
                proc_df[args.shift_col]
                .astype("string")
                .fillna("<NA>")
                .value_counts(dropna=False)
                .rename_axis(args.shift_col)
                .reset_index(name="arrivals")
            )
            shift_counts.to_csv(proc_dir / "arrivals_by_shift.csv", index=False)
            save_bar(
                x=shift_counts[args.shift_col].tolist(),
                y=shift_counts["arrivals"].tolist(),
                path=charts_dir / "arrivals_by_shift.png",
                title=f"Arrivals by Shift - {proc}",
                xlabel="Shift",
                ylabel="Arrivals",
                rotate_x=True,
            )

        # Core histograms.
        for metric in ["service_min", "wait_min", "cycle_min", "interarrival_min"]:
            if metric in proc_df.columns:
                save_hist(
                    proc_df[metric],
                    charts_dir / f"hist_{metric}.png",
                    title=f"Histogram: {metric} - {proc}",
                    xlabel=metric,
                )

        # Service-time segment plots.
        if "service_min" in proc_df.columns:
            if args.product_col in proc_df.columns:
                save_boxplot(
                    proc_df,
                    value_col="service_min",
                    by_col=args.product_col,
                    path=charts_dir / "service_time_by_product.png",
                    title=f"Service Time by Product - {proc}",
                )
            if args.machine_col in proc_df.columns:
                save_boxplot(
                    proc_df,
                    value_col="service_min",
                    by_col=args.machine_col,
                    path=charts_dir / "service_time_by_machine.png",
                    title=f"Service Time by Machine - {proc}",
                )
            if args.shift_col in proc_df.columns:
                save_boxplot(
                    proc_df,
                    value_col="service_min",
                    by_col=args.shift_col,
                    path=charts_dir / "service_time_by_shift.png",
                    title=f"Service Time by Shift - {proc}",
                )
            if args.batch_col in proc_df.columns and HAS_MATPLOTLIB:
                tmp = proc_df[[args.batch_col, "service_min"]].dropna()
                if not tmp.empty:
                    fig = plt.figure(figsize=(6.5, 4.5))
                    ax = fig.add_subplot(111)
                    ax.scatter(tmp[args.batch_col], tmp["service_min"], alpha=0.65)
                    ax.set_title(f"Service Time vs Batch Size - {proc}")
                    ax.set_xlabel(args.batch_col)
                    ax.set_ylabel("service_min")
                    fig.tight_layout()
                    fig.savefig(charts_dir / "service_time_vs_batch_size.png", dpi=160)
                    plt.close(fig)

        # Waiting-time by product plot.
        if "wait_min" in proc_df.columns and args.product_col in proc_df.columns:
            save_boxplot(
                proc_df,
                value_col="wait_min",
                by_col=args.product_col,
                path=charts_dir / "wait_time_by_product.png",
                title=f"Wait Time by Product - {proc}",
            )

        # Build plan-style summary row.
        service_stats = metric_stat_row("service_min", proc_df.get("service_min", pd.Series(dtype=float)))
        wait_stats = metric_stat_row("wait_min", proc_df.get("wait_min", pd.Series(dtype=float)))
        interarrival_stats = metric_stat_row(
            "interarrival_min",
            proc_df.get("interarrival_min", pd.Series(dtype=float)),
        )

        service_mean = service_stats["mean"] if service_stats["count"] else None
        wait_mean = wait_stats["mean"] if wait_stats["count"] else None
        wait_p90 = wait_stats["p90"] if wait_stats["count"] else None

        arrival_rate_per_hour = None
        interarrival_mean = interarrival_stats["mean"] if interarrival_stats["count"] else None
        if interarrival_mean is not None and interarrival_mean > 0:
            arrival_rate_per_hour = float(60.0 / interarrival_mean)

        servers = estimate_servers(proc_df, args.servers_col, args.machine_col)
        has_batch_signal = args.batch_col in proc_df.columns and proc_df[args.batch_col].notna().any()

        snapshot = {
            "process": proc,
            "rows": int(len(proc_df)),
            "columns": int(proc_df.shape[1]),
            "missing_pct_overall": float(proc_df.isna().sum().sum() / (proc_df.size or 1)),
        }
        snapshot.update(
            queue_kpi_snapshot(
                proc_df=proc_df,
                service_mean=service_mean,
                wait_mean=wait_mean,
                wait_p90=wait_p90,
                arrival_rate_per_hour=arrival_rate_per_hour,
                servers=servers,
                has_batch_signal=has_batch_signal,
            )
        )

        process_rows.append(snapshot)
        (proc_dir / "kpi_snapshot.json").write_text(json.dumps(snapshot, indent=2), encoding="utf-8")

    process_compare = pd.DataFrame(process_rows).sort_values("rows", ascending=False)
    process_compare.to_csv(out_root / "process_comparison.csv", index=False)

    # Plan-style final table requested before queue model selection.
    final_cols = [
        "process",
        "lots_processed",
        "mean_service_time_min",
        "cv_service",
        "mean_wait_min",
        "p90_wait_min",
        "arrivals_per_hour",
        "servers",
        "utilization",
        "queue_model_suggestion",
    ]
    plan_table = process_compare[final_cols].sort_values("process")
    plan_table.to_csv(out_root / "queue_model_input_table.csv", index=False)

    # Cross-process waiting summaries.
    if "mean_wait_min" in process_compare.columns:
        process_compare[["process", "mean_wait_min", "p90_wait_min"]].to_csv(
            out_root / "wait_summary_by_process.csv",
            index=False,
        )

    # Optional cross-process boxplots.
    if HAS_MATPLOTLIB and process_col in df.columns:
        all_df = add_time_kpis(df.copy(), args.arrival_col, args.start_col, args.end_col)
        for metric in ["wait_min", "service_min", "cycle_min"]:
            if metric in all_df.columns:
                save_boxplot(
                    all_df,
                    value_col=metric,
                    by_col=process_col,
                    path=out_root / f"boxplot_{metric}_by_process.png",
                    title=f"{metric} by process",
                    max_categories=20,
                )

    print(f"EDA complete. Output: {out_root}")


if __name__ == "__main__":
    main()
