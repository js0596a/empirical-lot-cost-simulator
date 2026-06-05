#!/usr/bin/env python3
from __future__ import annotations

import base64
import os
import re
from datetime import datetime, timedelta
from typing import Any

import dash
import dash_mantine_components as dmc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, dash_table, dcc

import app as core

APP_TITLE = "Bottleneck Early-Warning System"
UPLOAD_DIR = os.path.join(os.getcwd(), "outputs", "bottleneck_uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

DEFAULT_PROCESSES = ["RASPADO", "BAUCE", "VACIO", "LTD", "TAIC", "AEREO", "AFLOJADO", "MEDIDO"]
GRAPH_CONFIG = {
    "displaylogo": False,
    "displayModeBar": True,
    "scrollZoom": True,
    "doubleClick": "reset",
    "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"],
    "toImageButtonOptions": {"format": "png", "filename": "bottleneck_warning", "height": 900, "width": 1400, "scale": 2},
}


def option_data(values: list[str]) -> list[dict[str, str]]:
    return [{"label": v, "value": v} for v in values]


def current_process_options() -> list[str]:
    if core.DATAFRAME.empty or "process" not in core.DATAFRAME.columns:
        return []
    return sorted(
        str(v).strip().upper()
        for v in core.DATAFRAME["process"].dropna().unique().tolist()
        if str(v).strip() and str(v).strip().upper() != "UNKNOWN"
    )


def safe_upload_filename(filename: str | None) -> str:
    raw = os.path.basename(str(filename or "uploaded_workbook.xlsx")).strip() or "uploaded_workbook.xlsx"
    raw = re.sub(r"[^A-Za-z0-9._ -]+", "_", raw)
    return raw if raw.lower().endswith((".xlsx", ".xls")) else f"{raw}.xlsx"


def save_uploaded_workbooks(contents: str | list[str] | None, filenames: str | list[str] | None) -> tuple[list[str], str | None]:
    if not contents:
        return [], "No upload contents received."
    content_list = contents if isinstance(contents, list) else [contents]
    name_list = filenames if isinstance(filenames, list) else [filenames or "uploaded_workbook.xlsx"]
    saved_paths: list[str] = []
    errors: list[str] = []
    for i, content in enumerate(content_list):
        name = name_list[i] if i < len(name_list) else f"uploaded_workbook_{i+1}.xlsx"
        try:
            _header, encoded = str(content).split(",", 1)
            decoded = base64.b64decode(encoded)
            path = os.path.join(UPLOAD_DIR, safe_upload_filename(name))
            with open(path, "wb") as fh:
                fh.write(decoded)
            saved_paths.append(path)
        except Exception as exc:
            errors.append(f"{name}: {exc}")
    return saved_paths, "; ".join(errors) if errors else None


def activate_workbooks(paths: list[str]) -> tuple[bool, str, dict[str, Any]]:
    frames: list[pd.DataFrame] = []
    notes: list[str] = []
    for path in paths:
        df, err = core.load_data(path, core.SHEET_NAME)
        if err:
            notes.append(f"{os.path.basename(path)} failed: {err}")
            continue
        if df.empty:
            notes.append(f"{os.path.basename(path)} had no usable rows")
            continue
        df["uploaded_source_file"] = os.path.basename(path)
        frames.append(df)
    if not frames:
        return False, "; ".join(notes) if notes else "No usable production rows were found.", {}

    df_all = pd.concat(frames, ignore_index=True)
    core.DATAFRAME = df_all
    core.DATA_PATH = "; ".join(paths)
    core.REFERENCE_SHEETS = core.load_reference_sheets(paths[0])
    core.PROCESS_MACHINE_CATALOG_RESOLVED = core.build_process_machine_catalog(df_all)
    core.PROCESS_SERVER_COUNT_RESOLVED = core.build_process_server_counts(core.PROCESS_MACHINE_CATALOG_RESOLVED)
    core.PROCESS_OPTIONS = current_process_options()
    core.DEFAULT_PROCESS = core.PROCESS_OPTIONS[0] if core.PROCESS_OPTIONS else None
    meta = {
        "path": core.DATA_PATH,
        "files_loaded": len(frames),
        "rows": int(len(df_all)),
        "processes": len(core.PROCESS_OPTIONS),
        "notes": notes,
    }
    return True, "Uploaded workbook data is active.", meta


def data_source_status() -> tuple[str, str]:
    if core.DATAFRAME.empty:
        return "No default workbook loaded. Upload one or more production logbooks to start.", "blue"
    return f"Data source: {core.DATA_PATH} | rows={len(core.DATAFRAME):,} | processes={len(current_process_options())}", "teal"


def build_default_wip_rows() -> list[dict[str, Any]]:
    processes = current_process_options()
    ordered = [p for p in DEFAULT_PROCESSES if p in processes]
    ordered += [p for p in processes if p not in ordered]
    rows: list[dict[str, Any]] = []
    for proc in ordered[:20]:
        rows.append(
            {
                "process": proc,
                "current_wip_lots": 0,
                "notes": "",
            }
        )
    return rows


def clean_process_observations(process_value: str) -> tuple[pd.DataFrame, dict[str, int]]:
    proc = core.normalize_process_key(process_value)
    counts = {"raw_rows": 0, "strict_removed": 0, "iqr_found": 0, "iqr_removed": 0}
    if core.DATAFRAME.empty or not proc:
        return pd.DataFrame(), counts

    df = core.DATAFRAME[core.DATAFRAME["process"].astype(str).str.upper().str.strip() == proc].copy()
    counts["raw_rows"] = int(len(df))
    if df.empty:
        return df, counts

    before = len(df)
    df = core.add_quality_flags(df, process_value=proc)
    df = df[(~df["invalid_time_row"]) & (~df["missing_time_flag"])].copy()
    service = pd.to_numeric(df.get("service_hours", pd.Series(dtype=float)), errors="coerce")
    df = df[(service > 0) & service.notna()].copy()
    counts["strict_removed"] = int(before - len(df))
    if df.empty:
        return df, counts

    before_iqr = len(df)
    df = core.apply_outlier_flags(df, method=core.FIXED_OUTLIER_METHOD, metrics=["service_min"])
    out_col = "service_min_outlier_class"
    if out_col in df.columns:
        out_mask = df[out_col].astype(str).str.lower() != "normal"
        counts["iqr_found"] = int(out_mask.sum())
        df = df[~out_mask].copy()
    counts["iqr_removed"] = int(before_iqr - len(df))
    return df, counts


def historical_arrival_rate_per_hour(df: pd.DataFrame) -> float:
    if df.empty or "arrival_time" not in df.columns:
        return 0.0
    arrivals = pd.to_datetime(df["arrival_time"], errors="coerce").dropna().sort_values()
    if arrivals.empty:
        return 0.0
    start = arrivals.min()
    end = arrivals.max()
    scheduled_h = core.scheduled_hours_between(start, end)
    if not np.isfinite(scheduled_h) or scheduled_h <= 0:
        elapsed_h = max(1.0, (end - start).total_seconds() / 3600.0)
        scheduled_h = elapsed_h
    return float(len(arrivals) / max(1e-9, scheduled_h))


def process_forecast(
    process_value: str,
    current_wip_lots: float,
    horizon_h: float,
    high_threshold: float,
    medium_threshold: float,
    rng: np.random.Generator,
    reps: int = 500,
) -> tuple[dict[str, Any], pd.DataFrame]:
    proc = core.normalize_process_key(process_value)
    clean_df, counts = clean_process_observations(proc)
    servers = int(max(1, core.get_process_server_count(proc)))
    current_wip = max(0.0, float(current_wip_lots or 0.0))
    horizon = max(0.25, float(horizon_h or 8.0))

    if clean_df.empty:
        row = {
            "process": proc,
            "current_wip_lots": round(current_wip, 2),
            "expected_arrivals": 0.0,
            "servers": servers,
            "clean_n": 0,
            "mean_service_h": None,
            "p90_service_h": None,
            "expected_utilization_pct": None,
            "p90_utilization_pct": None,
            "overload_probability_pct": None,
            "expected_delay_h": None,
            "p90_delay_h": None,
            "queue_peak_lots": None,
            "queue_peak_time": "No clean data",
            "risk": "Review data",
            "suggested_action": "Clean or replace timing data before using this process for warning decisions.",
            **counts,
        }
        return row, pd.DataFrame()

    service_pool = pd.to_numeric(clean_df["service_hours"], errors="coerce").dropna().to_numpy(dtype=float)
    service_pool = service_pool[np.isfinite(service_pool) & (service_pool > 0)]
    if service_pool.size == 0:
        return process_forecast(proc, current_wip, horizon, high_threshold, medium_threshold, rng, reps=0)

    mean_service = float(np.mean(service_pool))
    p90_service = float(np.quantile(service_pool, 0.90))
    arrival_rate_h = historical_arrival_rate_per_hour(clean_df)
    expected_arrivals = float(arrival_rate_h * horizon)
    capacity_work_h = float(servers * horizon)
    service_rate_lots_h = float(servers / max(mean_service, 1e-9))

    reps_n = int(max(25, min(2000, reps)))
    util_samples = np.empty(reps_n, dtype=float)
    delay_samples = np.empty(reps_n, dtype=float)
    arrivals_samples = np.empty(reps_n, dtype=float)
    for i in range(reps_n):
        new_arrivals = int(rng.poisson(max(0.0, expected_arrivals)))
        total_lots = int(max(0, round(current_wip))) + new_arrivals
        arrivals_samples[i] = new_arrivals
        if total_lots <= 0:
            workload = 0.0
        else:
            workload = float(np.sum(rng.choice(service_pool, size=total_lots, replace=True)))
        util_samples[i] = workload / max(1e-9, capacity_work_h)
        delay_samples[i] = max(0.0, workload - capacity_work_h) / max(1, servers)

    expected_util = float(np.mean(util_samples))
    p90_util = float(np.quantile(util_samples, 0.90))
    overload_prob = float(np.mean(util_samples >= high_threshold))
    expected_delay = float(np.mean(delay_samples))
    p90_delay = float(np.quantile(delay_samples, 0.90))

    if expected_util >= high_threshold or overload_prob >= 0.35 or p90_util >= 1.0:
        risk = "High"
    elif expected_util >= medium_threshold or overload_prob >= 0.15 or p90_util >= high_threshold:
        risk = "Medium"
    else:
        risk = "Low"

    net_queue_rate = arrival_rate_h - service_rate_lots_h
    if net_queue_rate > 0:
        queue_peak_lots = current_wip + net_queue_rate * horizon
        peak_offset_h = horizon
    else:
        queue_peak_lots = current_wip
        peak_offset_h = 0.0 if current_wip > 0 else 0.0
    queue_peak_lots = max(0.0, float(queue_peak_lots))

    if risk == "High":
        action = f"Watch {proc}; expected utilization is near/above threshold. Consider adding capacity, staggering releases, or moving work before queue grows."
    elif risk == "Medium":
        action = f"Monitor {proc}; queue risk is manageable but sensitive to arrivals and long service times."
    else:
        action = f"{proc} looks stable under this horizon and WIP assumption."

    row = {
        "process": proc,
        "current_wip_lots": round(current_wip, 2),
        "expected_arrivals": round(float(np.mean(arrivals_samples)), 2),
        "servers": servers,
        "clean_n": int(service_pool.size),
        "mean_service_h": round(mean_service, 3),
        "p90_service_h": round(p90_service, 3),
        "expected_utilization_pct": round(expected_util * 100.0, 2),
        "p90_utilization_pct": round(p90_util * 100.0, 2),
        "overload_probability_pct": round(overload_prob * 100.0, 2),
        "expected_delay_h": round(expected_delay, 3),
        "p90_delay_h": round(p90_delay, 3),
        "queue_peak_lots": round(queue_peak_lots, 2),
        "queue_peak_offset_h": round(float(peak_offset_h), 3),
        "risk": risk,
        "suggested_action": action,
        **counts,
    }

    time_grid = np.linspace(0.0, horizon, 25)
    queue_projection = np.maximum(0.0, current_wip + (arrival_rate_h - service_rate_lots_h) * time_grid)
    projection = pd.DataFrame({"process": proc, "hour": time_grid, "projected_queue_lots": queue_projection})
    return row, projection


def clock_label(start_hour: int, start_minute: int, offset_h: float) -> str:
    base = datetime(2026, 1, 1, int(start_hour) % 24, int(start_minute) % 60)
    ts = base + timedelta(hours=float(offset_h or 0.0))
    return ts.strftime("%I:%M %p").lstrip("0")


def risk_color(risk: str) -> str:
    if risk == "High":
        return "red"
    if risk == "Medium":
        return "orange"
    if risk == "Low":
        return "teal"
    return "gray"


def build_summary_cards(rows: list[dict[str, Any]]) -> list[Any]:
    if not rows:
        return [dmc.Alert("Enter WIP and run the forecast.", color="blue", variant="light")]
    ranked = sorted(rows, key=lambda r: {"High": 3, "Medium": 2, "Low": 1}.get(str(r.get("risk")), 0), reverse=True)
    top = ranked[0]
    high_count = sum(1 for r in rows if r.get("risk") == "High")
    medium_count = sum(1 for r in rows if r.get("risk") == "Medium")
    return [
        dmc.SimpleGrid(
            cols={"base": 1, "sm": 3},
            spacing="md",
            children=[
                dmc.Paper(withBorder=True, radius="lg", p="md", children=dmc.Stack(gap=2, children=[dmc.Text("Top risk process", c="dimmed", fz="sm"), dmc.Title(str(top.get("process", "N/A")), order=3), dmc.Badge(str(top.get("risk", "Review")), color=risk_color(str(top.get("risk"))), variant="light")])),
                dmc.Paper(withBorder=True, radius="lg", p="md", children=dmc.Stack(gap=2, children=[dmc.Text("High-risk processes", c="dimmed", fz="sm"), dmc.Title(str(high_count), order=3), dmc.Text(f"Medium risk: {medium_count}", c="dimmed", fz="xs")])),
                dmc.Paper(withBorder=True, radius="lg", p="md", children=dmc.Stack(gap=2, children=[dmc.Text("Manager question", c="dimmed", fz="sm"), dmc.Title("What should we watch today?", order=4), dmc.Text(str(top.get("suggested_action", "Run forecast.")), c="dimmed", fz="xs")])),
            ],
        )
    ]


def build_alert_messages(rows: list[dict[str, Any]], start_hour: int, start_minute: int) -> list[Any]:
    if not rows:
        return [dmc.Text("No alerts yet. Run the forecast.", c="dimmed")]
    children: list[Any] = []
    ordered = sorted(rows, key=lambda r: {"High": 3, "Medium": 2, "Low": 1}.get(str(r.get("risk")), 0), reverse=True)
    for row in ordered[:8]:
        proc = row.get("process", "N/A")
        risk = str(row.get("risk", "Review"))
        offset = row.get("queue_peak_offset_h", 0.0) or 0.0
        peak_time = clock_label(start_hour, start_minute, float(offset)) if isinstance(offset, (int, float, np.floating)) else "N/A"
        util = row.get("expected_utilization_pct")
        delay = row.get("p90_delay_h")
        if risk == "High":
            msg = f"{proc} expected to exceed warning levels near {peak_time}. Expected utilization {util}%, P90 delay {delay}h."
        elif risk == "Medium":
            msg = f"{proc} should be monitored. Expected utilization {util}% with moderate overload risk."
        elif risk == "Low":
            msg = f"{proc} queue likely stays below warning level. Expected utilization {util}%."
        else:
            msg = f"{proc} needs data review before reliable warning decisions."
        children.append(dmc.Alert(msg, title=f"{risk} risk", color=risk_color(risk), variant="light"))
    return children


def build_utilization_figure(rows: list[dict[str, Any]], medium_threshold: float, high_threshold: float) -> go.Figure:
    if not rows:
        return core.empty_figure("Run bottleneck forecast")
    df = pd.DataFrame(rows)
    df = df[pd.to_numeric(df["expected_utilization_pct"], errors="coerce").notna()].copy()
    if df.empty:
        return core.empty_figure("No clean process data")
    df["risk_rank"] = df["risk"].map({"High": 3, "Medium": 2, "Low": 1}).fillna(0)
    df = df.sort_values(["risk_rank", "expected_utilization_pct"], ascending=[False, False])
    colors = df["risk"].map({"High": "#dc2626", "Medium": "#f97316", "Low": "#0f766e"}).fillna("#64748b")
    fig = go.Figure()
    fig.add_bar(
        x=df["process"],
        y=df["expected_utilization_pct"],
        name="Expected utilization",
        marker_color=colors,
        hovertemplate="%{x}<br>Expected util=%{y:.1f}%<extra></extra>",
    )
    fig.add_scatter(
        x=df["process"],
        y=df["p90_utilization_pct"],
        mode="markers",
        name="P90 utilization",
        marker={"size": 11, "color": "#111827", "symbol": "diamond"},
        hovertemplate="%{x}<br>P90 util=%{y:.1f}%<extra></extra>",
    )
    fig.add_hline(y=medium_threshold * 100.0, line_dash="dash", line_color="#f97316", annotation_text="Medium threshold")
    fig.add_hline(y=high_threshold * 100.0, line_dash="dash", line_color="#dc2626", annotation_text="High threshold")
    fig.update_layout(height=430, yaxis_title="Utilization (%)", margin={"l": 30, "r": 25, "t": 65, "b": 80}, legend={"orientation": "h", "y": 1.03})
    return core.style_figure(fig, "Expected utilization by process")


def build_queue_figure(projection: pd.DataFrame, start_hour: int, start_minute: int) -> go.Figure:
    if projection.empty:
        return core.empty_figure("Run bottleneck forecast")
    plot_df = projection.copy()
    plot_df["clock"] = plot_df["hour"].apply(lambda h: clock_label(start_hour, start_minute, float(h)))
    fig = go.Figure()
    for proc, g in plot_df.groupby("process"):
        fig.add_trace(
            go.Scatter(
                x=g["hour"],
                y=g["projected_queue_lots"],
                mode="lines",
                name=str(proc),
                customdata=g[["clock"]],
                hovertemplate="%{fullData.name}<br>forecast hour=%{x:.1f}<br>clock=%{customdata[0]}<br>queue=%{y:.2f} lots<extra></extra>",
            )
        )
    fig.update_layout(height=420, xaxis_title="Hours from shift start", yaxis_title="Projected queue (lots)", margin={"l": 30, "r": 25, "t": 65, "b": 65}, legend={"orientation": "h", "y": 1.03})
    return core.style_figure(fig, "Projected queue path")


def forecast_all_processes(
    wip_rows: list[dict[str, Any]] | None,
    horizon_h: float,
    start_hour: int,
    start_minute: int,
    medium_threshold: float,
    high_threshold: float,
    reps: int,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    rng = np.random.default_rng()
    rows: list[dict[str, Any]] = []
    projections: list[pd.DataFrame] = []
    source_rows = wip_rows if isinstance(wip_rows, list) and wip_rows else build_default_wip_rows()
    seen: set[str] = set()
    for raw in source_rows:
        proc = core.normalize_process_key(raw.get("process", ""))
        if not proc or proc in seen:
            continue
        seen.add(proc)
        wip = pd.to_numeric(pd.Series([raw.get("current_wip_lots", 0)]), errors="coerce").iloc[0]
        wip = float(wip) if pd.notna(wip) else 0.0
        row, projection = process_forecast(proc, wip, horizon_h, high_threshold, medium_threshold, rng, reps=int(reps or 500))
        row["queue_peak_time"] = clock_label(start_hour, start_minute, row.get("queue_peak_offset_h", 0.0) or 0.0) if row.get("queue_peak_offset_h") is not None else "N/A"
        rows.append(row)
        if not projection.empty:
            projections.append(projection)
    risk_order = {"High": 0, "Medium": 1, "Low": 2, "Review data": 3}
    rows = sorted(rows, key=lambda r: (risk_order.get(str(r.get("risk")), 9), -(r.get("expected_utilization_pct") or -1)))
    projection_df = pd.concat(projections, ignore_index=True) if projections else pd.DataFrame()
    return rows, projection_df


source_msg, source_color = data_source_status()

warning_app = dash.Dash(__name__)
warning_app.title = APP_TITLE
warning_app.layout = dmc.MantineProvider(
    forceColorScheme="light",
    theme={
        "primaryColor": "teal",
        "fontFamily": "IBM Plex Sans, Inter, sans-serif",
        "headings": {"fontFamily": "Space Grotesk, IBM Plex Sans, sans-serif"},
        "defaultRadius": "md",
    },
    children=dmc.Container(
        size="xl",
        py="xl",
        className="app-shell",
        children=dmc.Stack(
            gap="md",
            children=[
                dcc.Store(id="forecast-store", data=[]),
                dmc.Paper(
                    withBorder=True,
                    radius="lg",
                    p="lg",
                    className="hero-panel app-card",
                    children=dmc.Stack(
                        gap="sm",
                        children=[
                            dmc.Badge("Operations research module", color="indigo", variant="light"),
                            dmc.Title("Bottleneck Early-Warning System", order=1),
                            dmc.Text(
                                "Predict which process managers should watch before queues become expensive. The app reads current WIP, cleaned historical service times, and process capacity to estimate utilization risk under uncertainty.",
                                c="dimmed",
                            ),
                            dmc.Alert(id="source-alert", children=source_msg, color=source_color, variant="light"),
                        ],
                    ),
                ),
                dmc.Paper(
                    withBorder=True,
                    radius="lg",
                    p="md",
                    className="upload-panel app-card",
                    children=dmc.Stack(
                        gap="xs",
                        children=[
                            dmc.Group(
                                justify="space-between",
                                children=[
                                    dmc.Stack(gap=2, children=[dmc.Text("Upload production Excel", fw=700), dmc.Text("Optional. Use this when managers want a morning warning model from a fresh logbook.", c="dimmed", fz="sm")]),
                                    dmc.Badge("Excel logbook", color="teal", variant="light"),
                                ],
                            ),
                            dcc.Upload(
                                id="excel-upload",
                                accept=".xlsx,.xls",
                                multiple=True,
                                className="upload-shell",
                                children=dmc.Paper(
                                    withBorder=True,
                                    radius="md",
                                    p="md",
                                    className="upload-dropzone",
                                    style={"borderStyle": "dashed", "cursor": "pointer"},
                                    children=dmc.Stack(gap=2, align="center", children=[dmc.Text("Drag and drop one or more Excel files here, or click to browse.", fw=600), dmc.Text("Expected format: production timing logbooks with process, timestamps, machine, and pieces.", c="dimmed", fz="xs")]),
                                ),
                            ),
                            dmc.Alert(id="upload-status", color="gray", variant="light", children="No uploaded workbook yet. Using the default data source if available."),
                        ],
                    ),
                ),
                dmc.Paper(
                    withBorder=True,
                    radius="lg",
                    p="lg",
                    className="app-card",
                    children=dmc.Stack(
                        gap="md",
                        children=[
                            dmc.Group(
                                justify="space-between",
                                align="center",
                                children=[
                                    dmc.Stack(gap=2, children=[dmc.Text("Morning WIP input", fw=800), dmc.Text("Enter current lots waiting or already queued at each process. The model also forecasts likely arrivals from historical logbook rates.", c="dimmed", fz="sm")]),
                                    dmc.Button("Reset process list", id="reset-wip", color="gray", variant="light"),
                                ],
                            ),
                            dash_table.DataTable(
                                id="wip-table",
                                columns=[
                                    {"name": "process", "id": "process", "editable": False},
                                    {"name": "current_wip_lots", "id": "current_wip_lots", "type": "numeric", "editable": True},
                                    {"name": "notes", "id": "notes", "editable": True},
                                ],
                                data=build_default_wip_rows(),
                                editable=True,
                                page_size=12,
                                sort_action="native",
                                filter_action="native",
                                style_as_list_view=True,
                                style_table={"overflowX": "auto"},
                                style_header={"fontWeight": "700", "backgroundColor": "#f8fafc"},
                                style_cell={"padding": "8px", "fontFamily": "IBM Plex Sans, sans-serif", "fontSize": "13px"},
                            ),
                            dmc.Divider(label="Forecast assumptions", labelPosition="center"),
                            dmc.Group(
                                grow=True,
                                children=[
                                    dmc.NumberInput(id="forecast-horizon-h", label="Forecast horizon (hours)", value=8, min=1, max=72, step=1, decimalScale=2),
                                    dmc.NumberInput(id="shift-start-hour", label="Shift start hour", value=8, min=0, max=23, step=1, allowDecimal=False),
                                    dmc.NumberInput(id="shift-start-minute", label="Shift start minute", value=0, min=0, max=59, step=1, allowDecimal=False),
                                ],
                            ),
                            dmc.Group(
                                grow=True,
                                children=[
                                    dmc.NumberInput(id="medium-threshold", label="Medium utilization threshold", value=0.70, min=0.1, max=2.0, step=0.05, decimalScale=3),
                                    dmc.NumberInput(id="high-threshold", label="High utilization threshold", value=0.90, min=0.1, max=2.0, step=0.05, decimalScale=3),
                                    dmc.NumberInput(id="monte-carlo-reps", label="Monte Carlo runs", value=500, min=50, max=2000, step=50, allowDecimal=False),
                                ],
                            ),
                            dmc.Button("Forecast bottlenecks", id="run-forecast", color="indigo", size="md"),
                        ],
                    ),
                ),
                dmc.Paper(
                    withBorder=True,
                    radius="lg",
                    p="lg",
                    className="results-panel app-card",
                    children=dmc.Stack(
                        gap="md",
                        children=[
                            dmc.Text("Early-warning output", fw=900, fz="xl"),
                            dmc.Stack(id="summary-cards", gap="md", children=build_summary_cards([])),
                            dmc.Stack(id="alert-list", gap="xs", children=build_alert_messages([], 8, 0)),
                            dcc.Graph(id="utilization-fig", figure=core.empty_figure("Run bottleneck forecast"), config=GRAPH_CONFIG, className="plot-card"),
                            dcc.Graph(id="queue-fig", figure=core.empty_figure("Run bottleneck forecast"), config=GRAPH_CONFIG, className="plot-card"),
                            dmc.Text("Risk table", fw=800),
                            dash_table.DataTable(
                                id="risk-table",
                                columns=[
                                    {"name": "process", "id": "process"},
                                    {"name": "current_wip_lots", "id": "current_wip_lots"},
                                    {"name": "expected_arrivals", "id": "expected_arrivals"},
                                    {"name": "servers", "id": "servers"},
                                    {"name": "expected_utilization_pct", "id": "expected_utilization_pct"},
                                    {"name": "p90_utilization_pct", "id": "p90_utilization_pct"},
                                    {"name": "overload_probability_pct", "id": "overload_probability_pct"},
                                    {"name": "expected_delay_h", "id": "expected_delay_h"},
                                    {"name": "p90_delay_h", "id": "p90_delay_h"},
                                    {"name": "queue_peak_time", "id": "queue_peak_time"},
                                    {"name": "risk", "id": "risk"},
                                    {"name": "suggested_action", "id": "suggested_action"},
                                ],
                                data=[],
                                page_size=12,
                                sort_action="native",
                                filter_action="native",
                                style_as_list_view=True,
                                style_table={"overflowX": "auto"},
                                style_header={"fontWeight": "700", "backgroundColor": "#f8fafc"},
                                style_cell={"padding": "8px", "fontFamily": "IBM Plex Sans, sans-serif", "fontSize": "13px", "whiteSpace": "normal", "height": "auto"},
                                style_data_conditional=[
                                    {"if": {"filter_query": "{risk} = High"}, "backgroundColor": "#fee2e2"},
                                    {"if": {"filter_query": "{risk} = Medium"}, "backgroundColor": "#ffedd5"},
                                    {"if": {"filter_query": "{risk} = Low"}, "backgroundColor": "#dcfce7"},
                                ],
                            ),
                            dmc.Text("Data quality and model inputs", fw=800),
                            dash_table.DataTable(
                                id="quality-table",
                                columns=[
                                    {"name": "process", "id": "process"},
                                    {"name": "raw_rows", "id": "raw_rows"},
                                    {"name": "strict_removed", "id": "strict_removed"},
                                    {"name": "iqr_found", "id": "iqr_found"},
                                    {"name": "iqr_removed", "id": "iqr_removed"},
                                    {"name": "clean_n", "id": "clean_n"},
                                    {"name": "mean_service_h", "id": "mean_service_h"},
                                    {"name": "p90_service_h", "id": "p90_service_h"},
                                    {"name": "servers", "id": "servers"},
                                ],
                                data=[],
                                page_size=12,
                                sort_action="native",
                                filter_action="native",
                                style_as_list_view=True,
                                style_table={"overflowX": "auto"},
                                style_header={"fontWeight": "700", "backgroundColor": "#f8fafc"},
                                style_cell={"padding": "8px", "fontFamily": "IBM Plex Sans, sans-serif", "fontSize": "13px"},
                            ),
                        ],
                    ),
                ),
            ],
        ),
    ),
)


@warning_app.callback(
    Output("upload-status", "children"),
    Output("upload-status", "color"),
    Output("source-alert", "children"),
    Output("source-alert", "color"),
    Output("wip-table", "data", allow_duplicate=True),
    Input("excel-upload", "contents"),
    State("excel-upload", "filename"),
    prevent_initial_call=True,
)
def upload_workbook(contents, filename):
    saved_paths, save_error = save_uploaded_workbooks(contents, filename)
    if not saved_paths:
        msg, color = data_source_status()
        return f"Upload failed: {save_error}", "red", msg, color, dash.no_update
    ok, message, meta = activate_workbooks(saved_paths)
    msg, color = data_source_status()
    if not ok:
        return f"Upload saved, but validation failed: {message}", "red", msg, color, dash.no_update
    warning = f" Warnings: {save_error}." if save_error else ""
    notes = f" Notes: {'; '.join(meta.get('notes', []))}" if meta.get("notes") else ""
    upload_msg = f"Workbook active: files={meta.get('files_loaded')} | rows={meta.get('rows'):,} | processes={meta.get('processes')}.{warning}{notes}"
    return upload_msg, "teal", msg, color, build_default_wip_rows()


@warning_app.callback(
    Output("wip-table", "data"),
    Input("reset-wip", "n_clicks"),
    prevent_initial_call=True,
)
def reset_wip_table(_clicks):
    return build_default_wip_rows()


@warning_app.callback(
    Output("forecast-store", "data"),
    Output("summary-cards", "children"),
    Output("alert-list", "children"),
    Output("risk-table", "data"),
    Output("quality-table", "data"),
    Output("utilization-fig", "figure"),
    Output("queue-fig", "figure"),
    Input("run-forecast", "n_clicks"),
    State("wip-table", "data"),
    State("forecast-horizon-h", "value"),
    State("shift-start-hour", "value"),
    State("shift-start-minute", "value"),
    State("medium-threshold", "value"),
    State("high-threshold", "value"),
    State("monte-carlo-reps", "value"),
    prevent_initial_call=False,
)
def run_forecast(n_clicks, wip_rows, horizon_h, start_hour, start_minute, medium_threshold, high_threshold, reps):
    if not n_clicks:
        return [], build_summary_cards([]), build_alert_messages([], 8, 0), [], [], core.empty_figure("Run bottleneck forecast"), core.empty_figure("Run bottleneck forecast")
    if core.DATAFRAME.empty:
        msg = [dmc.Alert("No data loaded. Upload a production workbook first.", color="red", variant="light")]
        return [], msg, msg, [], [], core.empty_figure("No data loaded"), core.empty_figure("No data loaded")

    medium = float(medium_threshold or 0.70)
    high = float(high_threshold or 0.90)
    if medium > high:
        medium, high = high, medium
    start_h = int(start_hour or 8)
    start_m = int(start_minute or 0)
    rows, projection = forecast_all_processes(wip_rows, float(horizon_h or 8.0), start_h, start_m, medium, high, int(reps or 500))
    quality_rows = [
        {
            "process": r.get("process"),
            "raw_rows": r.get("raw_rows"),
            "strict_removed": r.get("strict_removed"),
            "iqr_found": r.get("iqr_found"),
            "iqr_removed": r.get("iqr_removed"),
            "clean_n": r.get("clean_n"),
            "mean_service_h": r.get("mean_service_h"),
            "p90_service_h": r.get("p90_service_h"),
            "servers": r.get("servers"),
        }
        for r in rows
    ]
    return (
        rows,
        build_summary_cards(rows),
        build_alert_messages(rows, start_h, start_m),
        rows,
        quality_rows,
        build_utilization_figure(rows, medium, high),
        build_queue_figure(projection, start_h, start_m),
    )


if __name__ == "__main__":
    host = os.getenv("BOTTLENECK_APP_HOST", os.getenv("SIM_APP_HOST", "127.0.0.1"))
    port = int(os.getenv("BOTTLENECK_APP_PORT", "8055"))
    warning_app.run(host=host, port=port, debug=False)
