#!/usr/bin/env python3
from __future__ import annotations

import os
import numpy as np
import pandas as pd
import dash
import dash_mantine_components as dmc
import plotly.graph_objects as go
from dash import Input, Output, State, dcc, dash_table

import app as core


def option_data(values: list[str]) -> list[dict[str, str]]:
    return [{"label": str(v), "value": str(v)} for v in values]


def build_lot_rows(lot_store: list[dict]) -> list[dict]:
    rows = []
    for i, lot in enumerate(lot_store, start=1):
        route = [str(p).strip() for p in lot.get("route", []) if str(p).strip()]
        pieces = pd.to_numeric(pd.Series([lot.get("pieces")]), errors="coerce").iloc[0]
        rows.append(
            {
                "lot_id": i,
                "lot_name": str(lot.get("lot_name", f"Lote_{i}")),
                "pieces": round(float(pieces), 2) if pd.notna(pieces) else None,
                "steps": int(len(route)),
                "route": " > ".join(route),
            }
        )
    return rows


def empty_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(height=340, margin={"l": 20, "r": 20, "t": 50, "b": 20})
    return fig


PROCESS_OPTIONS = core.PROCESS_OPTIONS
DEFAULT_ROUTE = [x for x in ["RASPADO", "BAUCE", "VACIO"] if x in PROCESS_OPTIONS]
ENERGY_REF_BY_PROCESS = core.ENERGY_REFERENCE.get("by_process", {}) if isinstance(core.ENERGY_REFERENCE, dict) else {}
ENERGY_REF_SOURCE = core.ENERGY_REFERENCE.get("source_path") if isinstance(core.ENERGY_REFERENCE, dict) else None
ENERGY_REF_ROWS = int(core.ENERGY_REFERENCE.get("rows_parsed", 0)) if isinstance(core.ENERGY_REFERENCE, dict) else 0
DEFAULT_ENERGY_FALLBACK = float(core.DEFAULT_ENERGY_KWH_PER_MACHINE_HOUR)
# One-time manual override requested by user.
PROCESS_KWH_OVERRIDE = {"RASPADO": 44.0}


def get_process_kwh(process_value: str) -> tuple[float, str]:
    key = core.normalize_process_key(process_value)
    if key in PROCESS_KWH_OVERRIDE:
        return float(PROCESS_KWH_OVERRIDE[key]), "manual_override"
    hit = ENERGY_REF_BY_PROCESS.get(key) if isinstance(ENERGY_REF_BY_PROCESS, dict) else None
    if hit and "kwh_per_machine_hour" in hit:
        return float(hit["kwh_per_machine_hour"]), "excel_fixed"
    return float(core.get_energy_kwh_reference_for_process(process_value)), "fallback_default"


def build_energy_reference_rows(processes: list[str]) -> list[dict]:
    rows: list[dict] = []
    for proc in sorted(processes):
        kwh, source = get_process_kwh(proc)
        rows.append(
            {
                "process": proc,
                "kwh_per_machine_hour": round(kwh, 4),
                "source": source,
            }
        )
    return rows

if core.DATAFRAME.empty:
    MIN_DATE = None
    MAX_DATE = None
else:
    dts = pd.to_datetime(core.DATAFRAME["arrival_time"], errors="coerce").dropna()
    MIN_DATE = dts.min().date().isoformat() if not dts.empty else None
    MAX_DATE = dts.max().date().isoformat() if not dts.empty else None

mini_app = dash.Dash(__name__)
mini_app.title = "Empirical Lot Simulator"

mini_app.layout = dmc.MantineProvider(
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
        children=dmc.Stack(
            gap="md",
            children=[
                dcc.Store(id="lot-store", data=[]),
                dmc.Alert(
                    core.DATA_ERROR if core.DATA_ERROR else f"Data source: {core.DATA_PATH}",
                    color="red" if core.DATA_ERROR else "teal",
                    variant="light",
                ),
                dmc.Paper(
                    withBorder=True,
                    radius="lg",
                    p="lg",
                    children=dmc.Stack(
                        gap="sm",
                        children=[
                            dmc.Title("Empirical Lot Cost Simulator", order=2),
                            dmc.Text("Only new flow: add lots, simulate, and get cost.", c="dimmed"),
                            dmc.DatePickerInput(
                                id="date-range",
                                label="Arrival date range",
                                type="range",
                                value=[MIN_DATE, MAX_DATE],
                                clearable=False,
                                valueFormat="YYYY-MM-DD",
                            ),
                            dmc.Group(
                                grow=True,
                                children=[
                                    dmc.TextInput(id="lot-name", label="Lot name", value="Lote_1"),
                                    dmc.NumberInput(id="lot-pieces", label="Pieces", value=200, min=1, max=100000, step=10, allowDecimal=False),
                                    dmc.NumberInput(id="lot-repeat", label="Repeats", value=1, min=1, max=500, step=1, allowDecimal=False),
                                ],
                            ),
                            dmc.MultiSelect(
                                id="lot-processes",
                                label="Processes (ordered)",
                                data=option_data(PROCESS_OPTIONS),
                                value=DEFAULT_ROUTE,
                                searchable=True,
                                clearable=True,
                                placeholder="Select process sequence",
                            ),
                            dmc.Group(
                                gap="xs",
                                children=[
                                    dmc.Button("Add lot", id="add-lot", color="teal"),
                                    dmc.Button("Clear lots", id="clear-lots", color="gray", variant="light"),
                                    dmc.Button("Simulate", id="run-sim", color="indigo"),
                                ],
                            ),
                            dmc.Divider(label="Cost Inputs", labelPosition="center"),
                            dmc.Group(
                                grow=True,
                                children=[
                                    dmc.NumberInput(id="energy-cost", label="Energy cost ($/kWh)", value=0.12, min=0, step=0.01, decimalScale=4),
                                    dmc.NumberInput(id="labor-cost", label="Labor cost ($/hour)", value=60.0, min=0, step=0.1, decimalScale=3),
                                ],
                            ),
                            dmc.Text(
                                f"Fixed energy consumption per process loaded from Excel: {ENERGY_REF_SOURCE or 'N/A'} ({ENERGY_REF_ROWS} rows).",
                                c="dimmed",
                                fz="xs",
                            ),
                            dmc.Alert(
                                id="sim-output",
                                title="Simulation Output",
                                color="indigo",
                                variant="light",
                                children="Add lots and click Simulate.",
                            ),
                        ],
                    ),
                ),
                dmc.Paper(
                    withBorder=True,
                    radius="lg",
                    p="lg",
                    children=dmc.Stack(
                        gap="sm",
                        children=[
                            dmc.Text("Configured lots", fw=700),
                            dash_table.DataTable(
                                id="lot-table",
                                columns=[
                                    {"name": "lot_id", "id": "lot_id"},
                                    {"name": "lot_name", "id": "lot_name"},
                                    {"name": "pieces", "id": "pieces"},
                                    {"name": "steps", "id": "steps"},
                                    {"name": "route", "id": "route"},
                                ],
                                data=[],
                                page_size=8,
                                sort_action="native",
                                filter_action="native",
                                style_as_list_view=True,
                                style_table={"overflowX": "auto"},
                                style_header={"fontWeight": "700", "backgroundColor": "#f8fafc"},
                                style_cell={"padding": "8px", "fontFamily": "IBM Plex Sans, sans-serif", "fontSize": "13px"},
                            ),
                            dmc.Text("Cost by process", fw=700),
                            dcc.Graph(id="cost-fig", figure=empty_figure("Run simulation")),
                            dash_table.DataTable(
                                id="cost-table",
                                columns=[
                                    {"name": "process", "id": "process"},
                                    {"name": "kwh_per_machine_hour", "id": "kwh_per_machine_hour"},
                                    {"name": "kwh_source", "id": "kwh_source"},
                                    {"name": "visits", "id": "visits"},
                                    {"name": "service_total_h", "id": "service_total_h"},
                                    {"name": "downtime_total_h", "id": "downtime_total_h"},
                                    {"name": "energy_cost", "id": "energy_cost"},
                                    {"name": "labor_cost", "id": "labor_cost"},
                                    {"name": "total_cost", "id": "total_cost"},
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
                            dmc.Text("Fixed energy reference by process", fw=700),
                            dash_table.DataTable(
                                id="energy-ref-table",
                                columns=[
                                    {"name": "process", "id": "process"},
                                    {"name": "kwh_per_machine_hour", "id": "kwh_per_machine_hour"},
                                    {"name": "source", "id": "source"},
                                ],
                                data=build_energy_reference_rows(PROCESS_OPTIONS),
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


@mini_app.callback(
    Output("lot-store", "data"),
    Output("lot-table", "data"),
    Output("lot-name", "value"),
    Output("lot-repeat", "value"),
    Input("add-lot", "n_clicks"),
    Input("clear-lots", "n_clicks"),
    State("lot-name", "value"),
    State("lot-pieces", "value"),
    State("lot-processes", "value"),
    State("lot-repeat", "value"),
    State("lot-store", "data"),
    prevent_initial_call=False,
)
def manage_lots(add_clicks, clear_clicks, lot_name, lot_pieces, lot_processes, lot_repeat, lot_store):
    _ = (add_clicks, clear_clicks)
    store = list(lot_store) if isinstance(lot_store, list) else []
    trigger = dash.ctx.triggered_id

    if trigger == "clear-lots":
        return [], [], "Lote_1", 1

    if trigger == "add-lot":
        route = [p for p in (lot_processes or []) if p in PROCESS_OPTIONS]
        pieces = pd.to_numeric(pd.Series([lot_pieces]), errors="coerce").iloc[0]
        repeat_n = pd.to_numeric(pd.Series([lot_repeat]), errors="coerce").iloc[0]
        repeat_n = int(max(1, min(500, int(repeat_n if pd.notna(repeat_n) else 1))))

        if route and pd.notna(pieces) and float(pieces) > 0:
            base_name = str(lot_name).strip() if str(lot_name or "").strip() else f"Lote_{len(store)+1}"
            for i in range(repeat_n):
                final_name = base_name if repeat_n == 1 else f"{base_name}#{i+1}"
                store.append({"lot_name": final_name, "pieces": float(pieces), "route": list(route)})

    rows = build_lot_rows(store)
    next_name = f"Lote_{len(store)+1}" if store else "Lote_1"
    return store, rows, next_name, 1


@mini_app.callback(
    Output("sim-output", "children"),
    Output("cost-table", "data"),
    Output("cost-fig", "figure"),
    Input("run-sim", "n_clicks"),
    State("lot-store", "data"),
    State("date-range", "value"),
    State("energy-cost", "value"),
    State("labor-cost", "value"),
)
def run_simulation(n_clicks, lot_store, date_range, energy_cost, labor_cost):
    if not n_clicks:
        return "Add lots and click Simulate.", [], empty_figure("Waiting for simulation")
    if core.DATAFRAME.empty:
        return "No data loaded.", [], empty_figure("No data")

    lots_raw = list(lot_store) if isinstance(lot_store, list) else []
    if not lots_raw:
        return "No lots configured.", [], empty_figure("No lots configured")

    lots = []
    for i, lot in enumerate(lots_raw, start=1):
        route = [p for p in lot.get("route", []) if p in PROCESS_OPTIONS]
        pieces = pd.to_numeric(pd.Series([lot.get("pieces")]), errors="coerce").iloc[0]
        if route and pd.notna(pieces) and float(pieces) > 0:
            lots.append({"lot_name": str(lot.get("lot_name", f"Lote_{i}")), "pieces": float(pieces), "route": list(route)})

    if not lots:
        return "Configured lots are invalid.", [], empty_figure("Invalid lots")

    base = core.DATAFRAME.copy()
    if date_range and len(date_range) == 2 and date_range[0] and date_range[1]:
        start = pd.to_datetime(date_range[0], errors="coerce")
        end = pd.to_datetime(date_range[1], errors="coerce")
        if pd.notna(start) and pd.notna(end):
            base = base[(base["arrival_time"] >= start) & (base["arrival_time"] <= end + pd.Timedelta(days=1))].copy()

    if base.empty:
        return "No rows in selected date range.", [], empty_figure("No rows in scope")

    rng = np.random.default_rng(20260529)
    ordered_processes = []
    for lot in lots:
        for proc in lot["route"]:
            if proc not in ordered_processes:
                ordered_processes.append(proc)

    stage_catalog, missing_processes = core.build_stage_catalog_for_processes(
        base_df=base,
        process_list=ordered_processes,
        strict_cleaning=True,
        queue_use_downtime=True,
        rng=rng,
    )

    filtered_lots = []
    for lot in lots:
        route = [p for p in lot["route"] if p in stage_catalog]
        if route:
            filtered_lots.append({"lot_name": lot["lot_name"], "pieces": lot["pieces"], "route": route})

    if not filtered_lots:
        return "No usable stages for selected lots.", [], empty_figure("No usable stage data")

    arrival_obs = pd.to_datetime(base.get("arrival_time", pd.Series(dtype="datetime64[ns]")), errors="coerce").dropna().sort_values()
    ia_obs = arrival_obs.diff().dt.total_seconds() / 3600.0 if len(arrival_obs) > 1 else pd.Series(dtype=float)
    ia_obs = pd.to_numeric(ia_obs, errors="coerce").dropna()
    ia_obs = ia_obs[(ia_obs > 0) & np.isfinite(ia_obs)]
    if ia_obs.empty:
        ia_samples = np.full(len(filtered_lots), 1.0, dtype=float)
    else:
        ia_samples = np.asarray(rng.choice(ia_obs.to_numpy(dtype=float), size=len(filtered_lots), replace=True), dtype=float)
        ia_samples = np.where(np.isfinite(ia_samples) & (ia_samples > 0), ia_samples, 1e-6)

    sim = core.simulate_lot_plan_flow(stage_catalog=stage_catalog, lot_plan=filtered_lots, interarrival_h=ia_samples)
    stage_events = pd.DataFrame(sim.get("stage_rows", []))
    lot_events = pd.DataFrame(sim.get("lot_rows", []))
    if stage_events.empty or lot_events.empty:
        return "Simulation produced no events.", [], empty_figure("No simulation events")

    energy_cost = float(energy_cost) if energy_cost is not None else 0.12
    labor_cost = float(labor_cost) if labor_cost is not None else 60.0

    rows = []
    total_energy_cost = 0.0
    total_labor_cost = 0.0
    fallback_energy_processes: list[str] = []

    for proc, g in stage_events.groupby("process"):
        srv = pd.to_numeric(g["service_h"], errors="coerce").dropna().to_numpy(dtype=float)
        dt = pd.to_numeric(g["downtime_h"], errors="coerce").dropna().to_numpy(dtype=float)
        service_total_h = float(np.nansum(srv)) if srv.size > 0 else 0.0
        downtime_total_h = float(np.nansum(dt)) if dt.size > 0 else 0.0
        machine_hours = service_total_h + downtime_total_h
        proc_kwh, proc_kwh_source = get_process_kwh(str(proc))
        if proc_kwh_source == "fallback_default":
            fallback_energy_processes.append(str(proc))

        e_cost = machine_hours * proc_kwh * energy_cost
        l_cost = service_total_h * labor_cost
        t_cost = e_cost + l_cost

        total_energy_cost += e_cost
        total_labor_cost += l_cost

        rows.append(
            {
                "process": str(proc),
                "kwh_per_machine_hour": round(proc_kwh, 4),
                "kwh_source": proc_kwh_source,
                "visits": int(len(g)),
                "service_total_h": round(service_total_h, 4),
                "downtime_total_h": round(downtime_total_h, 4),
                "energy_cost": round(e_cost, 2),
                "labor_cost": round(l_cost, 2),
                "total_cost": round(t_cost, 2),
            }
        )

    rows = sorted(rows, key=lambda x: x["total_cost"], reverse=True)

    fig = go.Figure()
    fig.add_bar(x=[r["process"] for r in rows], y=[r["energy_cost"] for r in rows], name="Energy cost")
    fig.add_bar(x=[r["process"] for r in rows], y=[r["labor_cost"] for r in rows], name="Labor cost")
    fig.update_layout(barmode="stack", yaxis={"title": "Cost"}, margin={"l": 20, "r": 20, "t": 50, "b": 20}, height=360)

    total_cost = total_energy_cost + total_labor_cost
    total_pieces = float(pd.to_numeric(lot_events.get("pieces", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
    cost_per_piece = (total_cost / total_pieces) if total_pieces > 0 else np.nan
    lead_mean = float(pd.to_numeric(lot_events.get("system_time_h", pd.Series(dtype=float)), errors="coerce").dropna().mean())
    most_expensive = rows[0] if rows else None
    cheapest = rows[-1] if rows else None

    missing_txt = f" Missing stage data: {', '.join(sorted(set(missing_processes)))}." if missing_processes else ""
    fallback_txt = (
        f" Energy fallback used for: {', '.join(sorted(set(fallback_energy_processes)))} "
        f"(default {DEFAULT_ENERGY_FALLBACK} kWh/h)."
        if fallback_energy_processes
        else ""
    )
    compare_txt = (
        f" Most expensive process={most_expensive['process']} (${core.safe_number(most_expensive['total_cost'], 2)}), "
        f"cheapest={cheapest['process']} (${core.safe_number(cheapest['total_cost'], 2)})."
        if most_expensive and cheapest
        else ""
    )
    summary = (
        f"Empirical simulation complete | lots={int(sim.get('n', 0))}, pieces={core.safe_number(total_pieces, 0)}, "
        f"lead_time_mean={core.safe_number(lead_mean, 4)} h | "
        f"energy=${core.safe_number(total_energy_cost, 2)}, labor=${core.safe_number(total_labor_cost, 2)}, "
        f"total=${core.safe_number(total_cost, 2)}, cost_per_piece=${core.safe_number(cost_per_piece, 4)}."
        + compare_txt
        + missing_txt
        + fallback_txt
    )

    return summary, rows, fig


if __name__ == "__main__":
    port = int(os.getenv("SIM_APP_PORT", "8051"))
    mini_app.run(debug=False, host="127.0.0.1", port=port)
