#!/usr/bin/env python3
from __future__ import annotations

import base64
import os
import re
from typing import Any

import dash
import dash_mantine_components as dmc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, dash_table, dcc

import app as core

APP_TITLE = "Lot-Splitting Optimization Studio"
UPLOAD_DIR = os.path.join(os.getcwd(), "outputs", "lot_split_uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

DEFAULT_ROUTE_CANDIDATES = ["RASPADO", "BAUCE", "VACIO", "LTD", "AFLOJADO", "MEDIDO"]
SECADO_RATE_IQR_FILTER_PROCESSES = {"LTD", "LTB", "TAIC", "TAIK", "AEREO", "AERO"}
SECADO_NO_PIECE_SCALING_PROCESSES = {"LTD", "LTB", "TAIC", "TAIK", "AEREO", "AERO"}
SERVICE_HOUR_CAPS = {"TAIC": 4.0, "TAIK": 4.0, "AEREO": 6.0, "AERO": 6.0, "LTD": 10.0, "LTB": 10.0, "BAUCE": 4.0}
TRANSFER_GAP_MIN = 20.0
TRANSFER_GAP_MAX = 30.0
GRAPH_CONFIG = {
    "displaylogo": False,
    "displayModeBar": True,
    "scrollZoom": True,
    "doubleClick": "reset",
    "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"],
    "toImageButtonOptions": {"format": "png", "filename": "lot_splitting_optimization", "height": 900, "width": 1400, "scale": 2},
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


def default_route() -> list[str]:
    opts = current_process_options()
    route = [p for p in DEFAULT_ROUTE_CANDIDATES if p in opts]
    return route[:4] if route else opts[: min(4, len(opts))]


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
    return True, "Uploaded workbook data is active.", {"path": core.DATA_PATH, "files_loaded": len(frames), "rows": len(df_all), "processes": len(core.PROCESS_OPTIONS), "notes": notes}


def data_source_status() -> tuple[str, str]:
    if core.DATAFRAME.empty:
        return "No default workbook loaded. Upload one or more production logbooks to start.", "blue"
    return f"Data source: {core.DATA_PATH} | rows={len(core.DATAFRAME):,} | processes={len(current_process_options())}", "teal"


def parse_split_counts(raw: str | None, total_pieces: float) -> list[int]:
    values: list[int] = []
    for token in re.split(r"[,;\s]+", str(raw or "1,2,3")):
        if not token.strip():
            continue
        try:
            n = int(float(token))
        except Exception:
            continue
        if n >= 1 and n <= max(1, int(total_pieces)) and n not in values:
            values.append(n)
    return sorted(values) or [1]


def normalize_route(route_values: list[str] | None) -> list[str]:
    options = set(current_process_options())
    return [core.normalize_process_key(v) for v in (route_values or []) if core.normalize_process_key(v) in options]


def build_stage_catalog(route: list[str], rng: np.random.Generator) -> tuple[dict[str, dict[str, Any]], list[str]]:
    return core.build_stage_catalog_for_processes(
        base_df=core.DATAFRAME.copy(),
        process_list=route,
        strict_cleaning=True,
        queue_use_downtime=True,
        rng=rng,
        rate_iqr_filter=True,
        rate_iqr_processes=SECADO_RATE_IQR_FILTER_PROCESSES,
        rate_tail_guardrail=True,
        rate_tail_guardrail_processes=SECADO_RATE_IQR_FILTER_PROCESSES,
        no_piece_scaling_processes=SECADO_NO_PIECE_SCALING_PROCESSES,
        service_hour_caps=SERVICE_HOUR_CAPS,
        service_range_guardrail=True,
    )


def kwh_for_process(proc: str) -> float:
    if core.normalize_process_key(proc) == "RASPADO":
        return 44.0
    return float(core.get_energy_kwh_reference_for_process(proc))


def make_lot_plan(total_pieces: float, split_count: int, route: list[str], lot_name: str) -> list[dict[str, Any]]:
    split_count = int(max(1, split_count))
    base_pieces = float(total_pieces) / split_count
    lots = []
    remaining = float(total_pieces)
    for i in range(1, split_count + 1):
        pieces = base_pieces if i < split_count else remaining
        pieces = max(1e-6, pieces)
        remaining -= pieces
        lots.append({"lot_name": f"{lot_name}_{i}", "pieces": float(pieces), "route": list(route)})
    return lots


def simulate_strategy_once(
    stage_catalog: dict[str, dict[str, Any]],
    total_pieces: float,
    split_count: int,
    route: list[str],
    lot_name: str,
    release_mode: str,
    stagger_min: float,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    lot_plan = make_lot_plan(total_pieces, split_count, route, lot_name)
    n = len(lot_plan)
    if str(release_mode).lower() == "staggered" and n > 1:
        gap_h = max(1e-6, float(stagger_min) / 60.0)
        interarrival = np.full(n, gap_h, dtype=float)
    else:
        interarrival = np.full(n, 1e-6, dtype=float)
    between_steps_gap_sampler = lambda: float(rng.uniform(TRANSFER_GAP_MIN / 60.0, TRANSFER_GAP_MAX / 60.0))
    sim = core.simulate_lot_plan_flow(
        stage_catalog=stage_catalog,
        lot_plan=lot_plan,
        interarrival_h=interarrival,
        between_steps_gap_sampler=between_steps_gap_sampler,
        use_resource_queue=True,
    )
    return pd.DataFrame(sim.get("stage_rows", [])), pd.DataFrame(sim.get("lot_rows", []))


def summarize_strategy_run(
    stage_events: pd.DataFrame,
    lot_events: pd.DataFrame,
    total_pieces: float,
    split_count: int,
    setup_cost_per_sublot: float,
    labor_cost_per_h: float,
    energy_cost_per_kwh: float,
) -> dict[str, Any]:
    if stage_events.empty or lot_events.empty:
        return {}
    service_total_h = float(pd.to_numeric(stage_events.get("service_h"), errors="coerce").fillna(0.0).sum())
    wait_total_h = float(pd.to_numeric(stage_events.get("wait_h"), errors="coerce").fillna(0.0).sum())
    downtime_total_h = float(pd.to_numeric(stage_events.get("downtime_h"), errors="coerce").fillna(0.0).sum())
    makespan_h = float(pd.to_numeric(lot_events.get("system_finish_h"), errors="coerce").max())
    lead_mean_h = float(pd.to_numeric(lot_events.get("system_time_h"), errors="coerce").mean())
    lead_p90_h = float(np.nanquantile(pd.to_numeric(lot_events.get("system_time_h"), errors="coerce").dropna().to_numpy(dtype=float), 0.90))

    energy_cost = 0.0
    for proc, g in stage_events.groupby("process"):
        machine_h = float(pd.to_numeric(g.get("service_h"), errors="coerce").fillna(0.0).sum() + pd.to_numeric(g.get("downtime_h"), errors="coerce").fillna(0.0).sum())
        energy_cost += machine_h * kwh_for_process(str(proc)) * float(energy_cost_per_kwh)
    labor_cost = service_total_h * float(labor_cost_per_h)
    handling_cost = float(split_count) * float(setup_cost_per_sublot)
    total_cost = energy_cost + labor_cost + handling_cost

    wait_by_proc = stage_events.groupby("process")["wait_h"].sum().sort_values(ascending=False) if "wait_h" in stage_events.columns else pd.Series(dtype=float)
    bottleneck = str(wait_by_proc.index[0]) if not wait_by_proc.empty and float(wait_by_proc.iloc[0]) > 1e-9 else ""
    return {
        "split_count": int(split_count),
        "sublot_pieces": float(total_pieces) / max(1, int(split_count)),
        "makespan_h": makespan_h,
        "mean_sublot_lead_h": lead_mean_h,
        "p90_sublot_lead_h": lead_p90_h,
        "service_total_h": service_total_h,
        "wait_total_h": wait_total_h,
        "downtime_total_h": downtime_total_h,
        "energy_cost": energy_cost,
        "labor_cost": labor_cost,
        "handling_cost": handling_cost,
        "total_cost": total_cost,
        "cost_per_piece": total_cost / max(1e-9, float(total_pieces)),
        "bottleneck_wait": bottleneck,
        "throughput_pieces_per_h": float(total_pieces) / max(1e-9, makespan_h),
    }


def run_strategy_batch(
    stage_catalog: dict[str, dict[str, Any]],
    total_pieces: float,
    split_count: int,
    route: list[str],
    lot_name: str,
    reps: int,
    release_mode: str,
    stagger_min: float,
    setup_cost_per_sublot: float,
    labor_cost_per_h: float,
    energy_cost_per_kwh: float,
    rng: np.random.Generator,
) -> dict[str, Any]:
    metrics: list[dict[str, Any]] = []
    for _ in range(int(max(1, reps))):
        stage_events, lot_events = simulate_strategy_once(stage_catalog, total_pieces, split_count, route, lot_name, release_mode, stagger_min, rng)
        one = summarize_strategy_run(stage_events, lot_events, total_pieces, split_count, setup_cost_per_sublot, labor_cost_per_h, energy_cost_per_kwh)
        if one:
            metrics.append(one)
    if not metrics:
        return {"strategy": f"{split_count} x {total_pieces / max(1, split_count):.0f}", "recommendation": "No valid data"}
    df = pd.DataFrame(metrics)
    bottlenecks = df["bottleneck_wait"].dropna().astype(str)
    bottleneck_mode = bottlenecks.mode().iloc[0] if not bottlenecks.empty else ""
    return {
        "strategy": f"{split_count} lot{'s' if split_count != 1 else ''} x {total_pieces / max(1, split_count):.0f}",
        "split_count": int(split_count),
        "sublot_pieces": round(float(total_pieces) / max(1, int(split_count)), 2),
        "reps": int(len(df)),
        "mean_completion_h": round(float(df["makespan_h"].mean()), 3),
        "p90_completion_h": round(float(np.quantile(df["makespan_h"], 0.90)), 3),
        "mean_sublot_lead_h": round(float(df["mean_sublot_lead_h"].mean()), 3),
        "p90_sublot_lead_h": round(float(df["p90_sublot_lead_h"].mean()), 3),
        "mean_wait_h": round(float(df["wait_total_h"].mean()), 3),
        "mean_service_h": round(float(df["service_total_h"].mean()), 3),
        "expected_cost": round(float(df["total_cost"].mean()), 2),
        "p90_cost": round(float(np.quantile(df["total_cost"], 0.90)), 2),
        "cost_per_piece": round(float(df["cost_per_piece"].mean()), 4),
        "expected_handling_cost": round(float(df["handling_cost"].mean()), 2),
        "throughput_pieces_per_h": round(float(df["throughput_pieces_per_h"].mean()), 3),
        "bottleneck_wait_mode": bottleneck_mode,
    }


def add_recommendations(rows: list[dict[str, Any]], max_cost_increase_pct: float) -> list[dict[str, Any]]:
    if not rows:
        return []
    clean = [r for r in rows if r.get("expected_cost") is not None and r.get("mean_completion_h") is not None]
    if not clean:
        return rows
    baseline = next((r for r in clean if int(r.get("split_count", 0)) == 1), clean[0])
    baseline_cost = float(baseline["expected_cost"])
    baseline_lead = float(baseline["mean_completion_h"])
    allowed_cost = baseline_cost * (1.0 + max(0.0, float(max_cost_increase_pct)) / 100.0)
    fastest = min(clean, key=lambda r: float(r["mean_completion_h"]))
    cheapest = min(clean, key=lambda r: float(r["expected_cost"]))
    feasible = [r for r in clean if float(r["expected_cost"]) <= allowed_cost]
    recommended = min(feasible, key=lambda r: float(r["p90_completion_h"])) if feasible else cheapest
    for r in rows:
        if r not in clean:
            r["cost_increase_pct"] = None
            r["lead_time_savings_pct"] = None
            continue
        cost = float(r["expected_cost"])
        lead = float(r["mean_completion_h"])
        r["cost_increase_pct"] = round(((cost / baseline_cost) - 1.0) * 100.0, 2) if baseline_cost > 0 else None
        r["lead_time_savings_pct"] = round((1.0 - lead / baseline_lead) * 100.0, 2) if baseline_lead > 0 else None
        tags: list[str] = []
        if r is recommended:
            tags.append("Recommended")
        if r is fastest:
            tags.append("Fastest")
        if r is cheapest:
            tags.append("Lowest cost")
        if not tags:
            tags.append("Not best")
        if float(r["expected_cost"]) > allowed_cost and r is not cheapest:
            tags.append("Above cost limit")
        r["recommendation"] = ", ".join(tags)
    return rows


def build_summary(rows: list[dict[str, Any]]) -> list[Any]:
    if not rows:
        return [dmc.Alert("Run the optimizer to compare lot-splitting strategies.", color="blue", variant="light")]
    rec = next((r for r in rows if "Recommended" in str(r.get("recommendation", ""))), rows[0])
    fastest = next((r for r in rows if "Fastest" in str(r.get("recommendation", ""))), rec)
    return [
        dmc.SimpleGrid(
            cols={"base": 1, "sm": 3},
            spacing="md",
            children=[
                dmc.Paper(withBorder=True, radius="lg", p="md", children=dmc.Stack(gap=2, children=[dmc.Text("Recommended strategy", c="dimmed", fz="sm"), dmc.Title(str(rec.get("strategy")), order=3), dmc.Text(f"P90 completion: {rec.get('p90_completion_h')} h", c="dimmed", fz="xs")])),
                dmc.Paper(withBorder=True, radius="lg", p="md", children=dmc.Stack(gap=2, children=[dmc.Text("Expected cost", c="dimmed", fz="sm"), dmc.Title(f"${float(rec.get('expected_cost', 0)):,.0f}", order=3), dmc.Text(f"Cost change vs baseline: {rec.get('cost_increase_pct')}%", c="dimmed", fz="xs")])),
                dmc.Paper(withBorder=True, radius="lg", p="md", children=dmc.Stack(gap=2, children=[dmc.Text("Fastest option", c="dimmed", fz="sm"), dmc.Title(str(fastest.get("strategy")), order=3), dmc.Text(f"Mean completion: {fastest.get('mean_completion_h')} h", c="dimmed", fz="xs")])),
            ],
        ),
        dmc.Alert(
            f"Recommendation: process as {rec.get('strategy')} based on P90 completion time while respecting the selected cost-increase limit. This is a planning recommendation, not a hard production rule.",
            color="teal",
            variant="light",
            title="Decision summary",
        ),
    ]


def build_tradeoff_figure(rows: list[dict[str, Any]]) -> go.Figure:
    if not rows:
        return core.empty_figure("Run optimizer")
    df = pd.DataFrame(rows)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df["strategy"], y=df["mean_completion_h"], name="Mean completion (h)", marker_color="#2563eb"))
    fig.add_trace(go.Bar(x=df["strategy"], y=df["p90_completion_h"], name="P90 completion (h)", marker_color="#0f766e"))
    fig.add_trace(go.Scatter(x=df["strategy"], y=df["expected_cost"], name="Expected cost", mode="lines+markers", yaxis="y2", marker={"color": "#f97316", "size": 10}, line={"width": 3}))
    fig.update_layout(
        height=430,
        barmode="group",
        yaxis={"title": "Completion time (hours)"},
        yaxis2={"title": "Expected cost", "overlaying": "y", "side": "right"},
        legend={"orientation": "h", "y": 1.05},
        margin={"l": 30, "r": 30, "t": 65, "b": 90},
    )
    fig.update_xaxes(tickangle=-20)
    return core.style_figure(fig, "Lead-time and cost tradeoff")


def build_pareto_figure(rows: list[dict[str, Any]]) -> go.Figure:
    if not rows:
        return core.empty_figure("Run optimizer")
    df = pd.DataFrame(rows)
    colors = ["#16a34a" if "Recommended" in str(x) else "#2563eb" for x in df["recommendation"]]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["expected_cost"],
        y=df["p90_completion_h"],
        mode="markers+text",
        text=df["strategy"],
        textposition="top center",
        marker={"size": 14, "color": colors, "line": {"width": 1, "color": "#0f172a"}},
        customdata=df[["cost_increase_pct", "lead_time_savings_pct", "recommendation"]],
        hovertemplate="%{text}<br>Cost=$%{x:,.2f}<br>P90 completion=%{y:.2f} h<br>Cost change=%{customdata[0]}%<br>Lead savings=%{customdata[1]}%<br>%{customdata[2]}<extra></extra>",
    ))
    fig.update_layout(height=430, xaxis_title="Expected cost", yaxis_title="P90 completion time (hours)", margin={"l": 30, "r": 30, "t": 65, "b": 65})
    return core.style_figure(fig, "Pareto view: lower-left is better")


source_msg, source_color = data_source_status()

split_app = dash.Dash(__name__)
split_app.title = APP_TITLE
split_app.layout = dmc.MantineProvider(
    forceColorScheme="light",
    theme={"primaryColor": "teal", "fontFamily": "IBM Plex Sans, Inter, sans-serif", "headings": {"fontFamily": "Space Grotesk, IBM Plex Sans, sans-serif"}, "defaultRadius": "md"},
    children=dmc.Container(
        size="xl",
        py="xl",
        className="app-shell",
        children=dmc.Stack(
            gap="md",
            children=[
                dcc.Store(id="optimizer-store", data=[]),
                dmc.Paper(withBorder=True, radius="lg", p="lg", className="hero-panel app-card", children=dmc.Stack(gap="sm", children=[
                    dmc.Badge("Operations research module", color="grape", variant="light"),
                    dmc.Title("Lot-Splitting Optimization Studio", order=1),
                    dmc.Text("Compare whether one large lot or several smaller sublots creates the best lead-time/cost tradeoff under stochastic process times.", c="dimmed"),
                    dmc.Alert(id="source-alert", children=source_msg, color=source_color, variant="light"),
                ])),
                dmc.Paper(withBorder=True, radius="lg", p="md", className="upload-panel app-card", children=dmc.Stack(gap="xs", children=[
                    dmc.Group(justify="space-between", children=[dmc.Stack(gap=2, children=[dmc.Text("Upload production Excel", fw=700), dmc.Text("Optional. Use fresh plant timing data before optimizing lot size.", c="dimmed", fz="sm")]), dmc.Badge("Excel logbook", color="teal", variant="light")]),
                    dcc.Upload(id="excel-upload", accept=".xlsx,.xls", multiple=True, className="upload-shell", children=dmc.Paper(withBorder=True, radius="md", p="md", className="upload-dropzone", style={"borderStyle": "dashed", "cursor": "pointer"}, children=dmc.Stack(gap=2, align="center", children=[dmc.Text("Drag and drop one or more Excel files here, or click to browse.", fw=600), dmc.Text("Expected format: production timing logbooks with process, timestamps, machine, and pieces.", c="dimmed", fz="xs")]))),
                    dmc.Alert(id="upload-status", color="gray", variant="light", children="No uploaded workbook yet. Using the default data source if available."),
                ])),
                dmc.Paper(withBorder=True, radius="lg", p="lg", className="app-card", children=dmc.Stack(gap="md", children=[
                    dmc.Text("Lot-sizing input", fw=900, fz="xl"),
                    dmc.Group(grow=True, children=[
                        dmc.TextInput(id="lot-name", label="Lot name", value="Lote"),
                        dmc.NumberInput(id="total-pieces", label="Total pieces", value=600, min=1, max=100000, step=10, allowDecimal=False),
                        dmc.TextInput(id="split-counts", label="Candidate split counts", value="1,2,3,4,6", description="Example: 1,2,3 means one lot, two sublots, three sublots"),
                    ]),
                    dmc.MultiSelect(id="route-select", label="Process route", data=option_data(current_process_options()), value=default_route(), searchable=True, clearable=True, placeholder="Select route in order"),
                    dmc.Divider(label="Cost and simulation assumptions", labelPosition="center"),
                    dmc.Group(grow=True, children=[
                        dmc.NumberInput(id="labor-cost", label="Labor cost ($/hour)", value=60.0, min=0, step=1, decimalScale=2),
                        dmc.NumberInput(id="energy-cost", label="Energy cost ($/kWh)", value=0.12, min=0, step=0.01, decimalScale=4),
                        dmc.NumberInput(id="setup-cost", label="Setup/handling cost per sublot", value=30.0, min=0, step=5, decimalScale=2),
                    ]),
                    dmc.Group(grow=True, children=[
                        dmc.NumberInput(id="max-cost-increase", label="Max cost increase allowed vs no split (%)", value=15.0, min=0, max=500, step=1, decimalScale=2),
                        dmc.NumberInput(id="monte-carlo-reps", label="Monte Carlo runs", value=75, min=10, max=500, step=5, allowDecimal=False),
                        dmc.Select(id="release-mode", label="Sublot release mode", data=option_data(["Parallel", "Staggered"]), value="Parallel", allowDeselect=False),
                    ]),
                    dmc.Group(grow=True, children=[
                        dmc.NumberInput(id="stagger-minutes", label="Stagger gap between sublots (minutes)", value=25.0, min=0, max=240, step=5, decimalScale=2),
                    ]),
                    dmc.Button("Optimize lot split", id="run-optimizer", color="grape", size="md"),
                ])),
                dmc.Paper(withBorder=True, radius="lg", p="lg", className="results-panel app-card", children=dmc.Stack(gap="md", children=[
                    dmc.Text("Optimization output", fw=900, fz="xl"),
                    dmc.Stack(id="summary-cards", gap="md", children=build_summary([])),
                    dcc.Graph(id="tradeoff-fig", figure=core.empty_figure("Run optimizer"), config=GRAPH_CONFIG, className="plot-card"),
                    dcc.Graph(id="pareto-fig", figure=core.empty_figure("Run optimizer"), config=GRAPH_CONFIG, className="plot-card"),
                    dmc.Text("Strategy comparison", fw=800),
                    dash_table.DataTable(id="strategy-table", columns=[{"name": c, "id": c} for c in ["strategy", "split_count", "sublot_pieces", "mean_completion_h", "p90_completion_h", "expected_cost", "p90_cost", "cost_increase_pct", "lead_time_savings_pct", "cost_per_piece", "throughput_pieces_per_h", "bottleneck_wait_mode", "recommendation"]], data=[], page_size=12, sort_action="native", filter_action="native", style_as_list_view=True, style_table={"overflowX": "auto"}, style_header={"fontWeight": "700", "backgroundColor": "#f8fafc"}, style_cell={"padding": "8px", "fontFamily": "IBM Plex Sans, sans-serif", "fontSize": "13px", "whiteSpace": "normal", "height": "auto"}, style_data_conditional=[{"if": {"filter_query": '{recommendation} contains "Recommended"'}, "backgroundColor": "#dcfce7"}, {"if": {"filter_query": '{recommendation} contains "Fastest"'}, "backgroundColor": "#e0f2fe"}, {"if": {"filter_query": '{recommendation} contains "Above cost limit"'}, "backgroundColor": "#fee2e2"}]),
                    dmc.Text("Model note", fw=800),
                    dmc.Alert(id="model-note", color="blue", variant="light", children="Run optimizer to generate model note."),
                ])),
            ],
        ),
    ),
)


@split_app.callback(
    Output("upload-status", "children"),
    Output("upload-status", "color"),
    Output("source-alert", "children"),
    Output("source-alert", "color"),
    Output("route-select", "data"),
    Output("route-select", "value"),
    Input("excel-upload", "contents"),
    State("excel-upload", "filename"),
    prevent_initial_call=True,
)
def upload_workbook(contents, filename):
    saved_paths, save_error = save_uploaded_workbooks(contents, filename)
    if not saved_paths:
        msg, color = data_source_status()
        return f"Upload failed: {save_error}", "red", msg, color, dash.no_update, dash.no_update
    ok, message, meta = activate_workbooks(saved_paths)
    msg, color = data_source_status()
    if not ok:
        return f"Upload saved, but validation failed: {message}", "red", msg, color, dash.no_update, dash.no_update
    warning = f" Warnings: {save_error}." if save_error else ""
    notes = f" Notes: {'; '.join(meta.get('notes', []))}" if meta.get("notes") else ""
    upload_msg = f"Workbook active: files={meta.get('files_loaded')} | rows={meta.get('rows'):,} | processes={meta.get('processes')}.{warning}{notes}"
    return upload_msg, "teal", msg, color, option_data(current_process_options()), default_route()


@split_app.callback(
    Output("optimizer-store", "data"),
    Output("summary-cards", "children"),
    Output("strategy-table", "data"),
    Output("tradeoff-fig", "figure"),
    Output("pareto-fig", "figure"),
    Output("model-note", "children"),
    Input("run-optimizer", "n_clicks"),
    State("lot-name", "value"),
    State("total-pieces", "value"),
    State("split-counts", "value"),
    State("route-select", "value"),
    State("labor-cost", "value"),
    State("energy-cost", "value"),
    State("setup-cost", "value"),
    State("max-cost-increase", "value"),
    State("monte-carlo-reps", "value"),
    State("release-mode", "value"),
    State("stagger-minutes", "value"),
    prevent_initial_call=False,
)
def optimize_lot_split(n_clicks, lot_name, total_pieces, split_counts, route_values, labor_cost, energy_cost, setup_cost, max_cost_increase, reps, release_mode, stagger_minutes):
    if not n_clicks:
        return [], build_summary([]), [], core.empty_figure("Run optimizer"), core.empty_figure("Run optimizer"), "Run optimizer to generate model note."
    if core.DATAFRAME.empty:
        msg = [dmc.Alert("No data loaded. Upload a production workbook first.", color="red", variant="light")]
        return [], msg, [], core.empty_figure("No data loaded"), core.empty_figure("No data loaded"), "No data loaded."
    route = normalize_route(route_values)
    pieces = float(pd.to_numeric(pd.Series([total_pieces]), errors="coerce").iloc[0]) if total_pieces is not None else 0.0
    if pieces <= 0 or not route:
        msg = [dmc.Alert("Enter total pieces and choose at least one process in the route.", color="red", variant="light")]
        return [], msg, [], core.empty_figure("Missing input"), core.empty_figure("Missing input"), "Missing lot size or route."
    split_values = parse_split_counts(split_counts, pieces)
    rng = np.random.default_rng()
    stage_catalog, missing = build_stage_catalog(route, rng)
    valid_route = [p for p in route if p in stage_catalog]
    if not valid_route:
        msg = [dmc.Alert("No valid process data for selected route after cleaning.", color="red", variant="light")]
        return [], msg, [], core.empty_figure("No valid process data"), core.empty_figure("No valid process data"), f"Missing processes: {', '.join(missing)}"
    rows: list[dict[str, Any]] = []
    for split_count in split_values:
        row = run_strategy_batch(
            stage_catalog=stage_catalog,
            total_pieces=pieces,
            split_count=int(split_count),
            route=valid_route,
            lot_name=str(lot_name or "Lote"),
            reps=int(reps or 75),
            release_mode=str(release_mode or "Parallel"),
            stagger_min=float(stagger_minutes or 0),
            setup_cost_per_sublot=float(setup_cost or 0),
            labor_cost_per_h=float(labor_cost or 0),
            energy_cost_per_kwh=float(energy_cost or 0),
            rng=rng,
        )
        rows.append(row)
    rows = add_recommendations(rows, float(max_cost_increase or 0))
    missing_note = f" Missing route processes after cleaning: {', '.join(missing)}." if missing else ""
    note = (
        f"Simulation uses SimPy resources, cleaned empirical service-time samples, process server counts, and random {TRANSFER_GAP_MIN:.0f}-{TRANSFER_GAP_MAX:.0f} minute transfer gaps. "
        f"Recommendation is the lowest P90 completion time among strategies within {float(max_cost_increase or 0):.1f}% expected cost increase versus the no-split baseline.{missing_note}"
    )
    return rows, build_summary(rows), rows, build_tradeoff_figure(rows), build_pareto_figure(rows), note


if __name__ == "__main__":
    host = os.getenv("LOT_SPLIT_APP_HOST", os.getenv("SIM_APP_HOST", "127.0.0.1"))
    port = int(os.getenv("LOT_SPLIT_APP_PORT", "8057"))
    split_app.run(host=host, port=port, debug=False)
