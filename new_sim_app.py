#!/usr/bin/env python3
from __future__ import annotations

import base64
import os
from datetime import datetime
import numpy as np
import pandas as pd
import dash
import dash_mantine_components as dmc
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, State, dcc, dash_table, no_update
from plotly.subplots import make_subplots

import app as core
import bayes_classifier_app as bayes_core


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


def business_metric_card(label: str, value: str, detail: str) -> dmc.Paper:
    return dmc.Paper(
        withBorder=True,
        radius="md",
        p="sm",
        children=dmc.Stack(
            gap=2,
            children=[
                dmc.Text(label, c="dimmed", fz="xs"),
                dmc.Text(value, fw=800, fz="lg"),
                dmc.Text(detail, c="dimmed", fz="xs"),
            ],
        ),
    )


def build_business_summary_children(
    message: str | None = None,
    *,
    lots: int | None = None,
    pieces: float | None = None,
    total_cost: float | None = None,
    cost_per_piece: float | None = None,
    lead_mean_h: float | None = None,
    top_process: str | None = None,
    top_process_cost: float | None = None,
    queue_wait_max_min: float | None = None,
) -> list:
    if message:
        return [
            dmc.Text(message, fw=600),
            dmc.Text(
                "Business workflow: enter lots and process routes, click Simulate, then review cost, timing, bottlenecks, and process risk.",
                c="dimmed",
                fz="sm",
            ),
        ]

    top_process_txt = str(top_process or "N/A")
    top_cost_txt = core.safe_number(top_process_cost, 2)
    return [
        dmc.Text(
            "Decision view: this run estimates the production cost, timeline, and main cost driver for the lot mix entered by the user.",
            fw=600,
        ),
        dmc.Group(
            grow=True,
            align="stretch",
            children=[
                business_metric_card("Configured work", f"{int(lots or 0)} lots", f"{core.safe_number(pieces, 0)} pieces entered"),
                business_metric_card("Estimated total cost", f"${core.safe_number(total_cost, 2)}", f"${core.safe_number(cost_per_piece, 4)} per piece"),
                business_metric_card("Average lead time", f"{core.safe_number(lead_mean_h, 2)} h", f"Max queue wait {core.safe_number(queue_wait_max_min, 2)} min"),
                business_metric_card("Main cost driver", top_process_txt, f"${top_cost_txt} estimated cost"),
            ],
        ),
        dmc.Text(
            "Business interpretation: use this to compare lot mixes, routes, machine capacity, and cost assumptions before committing production resources.",
            c="dimmed",
            fz="sm",
        ),
    ]


def build_gantt_figure(
    stage_events: pd.DataFrame,
    date_range: list[str] | None,
    start_hour: float | int | None,
    start_minute: float | int | None,
) -> go.Figure:
    if stage_events.empty:
        return empty_figure("No stage events for Gantt")
    required = {"lot_name", "process", "route_step", "stage_start_h", "stage_finish_h"}
    if not required.issubset(set(stage_events.columns)):
        return empty_figure("Gantt timing columns are missing")

    hour = int(max(0, min(23, int(start_hour if start_hour is not None else 11))))
    minute = int(max(0, min(59, int(start_minute if start_minute is not None else 0))))

    base_date = pd.Timestamp.today().normalize()
    if date_range and len(date_range) >= 1 and date_range[0]:
        parsed = pd.to_datetime(date_range[0], errors="coerce")
        if pd.notna(parsed):
            base_date = parsed.normalize()
    base_dt = base_date + pd.Timedelta(hours=hour, minutes=minute)

    g = stage_events.copy()
    g["stage_arrive_h"] = pd.to_numeric(g.get("stage_arrive_h", pd.Series(index=g.index, dtype=float)), errors="coerce")
    g["stage_start_h"] = pd.to_numeric(g["stage_start_h"], errors="coerce")
    g["stage_finish_h"] = pd.to_numeric(g["stage_finish_h"], errors="coerce")
    # Plot actual machine work. Queue wait stays visible as empty timeline
    # before the bar and is shown in hover.
    g["stage_plot_start_h"] = g["stage_start_h"]
    g = g[g["stage_finish_h"] >= g["stage_plot_start_h"]].copy()
    if g.empty:
        return empty_figure("Gantt events are invalid")

    g["start_ts"] = base_dt + pd.to_timedelta(g["stage_plot_start_h"], unit="h")
    g["finish_ts"] = base_dt + pd.to_timedelta(g["stage_finish_h"], unit="h")
    g["stage_arrive_min"] = g["stage_arrive_h"] * 60.0
    g["stage_start_min"] = g["stage_start_h"] * 60.0
    g["stage_finish_min"] = g["stage_finish_h"] * 60.0
    if "wait_h" in g.columns:
        g["wait_min"] = pd.to_numeric(g["wait_h"], errors="coerce") * 60.0
    if "service_h" in g.columns:
        g["service_min"] = pd.to_numeric(g["service_h"], errors="coerce") * 60.0
    if "downtime_h" in g.columns:
        g["downtime_min"] = pd.to_numeric(g["downtime_h"], errors="coerce") * 60.0
    if "between_steps_gap_h" in g.columns:
        g["between_steps_gap_min"] = pd.to_numeric(g["between_steps_gap_h"], errors="coerce") * 60.0
    if "lot_arrival_h" in g.columns:
        g["lot_arrival_min"] = pd.to_numeric(g["lot_arrival_h"], errors="coerce") * 60.0
    if "interarrival_h" in g.columns:
        g["interarrival_min"] = pd.to_numeric(g["interarrival_h"], errors="coerce") * 60.0
    if "machine_id" in g.columns:
        g["machine_id"] = g["machine_id"].fillna("").astype(str)
    if "machine_label" in g.columns:
        g["machine_label"] = g["machine_label"].fillna("").astype(str)
    g["lot_name"] = g["lot_name"].astype(str)
    g["process"] = g["process"].astype(str)
    g = g.sort_values(["lot_name", "stage_start_h", "route_step"], ascending=[True, True, True])

    hover_data = {
        "process": True,
        "route_step": True,
        "start_ts": True,
        "finish_ts": True,
        "stage_arrive_min": ":.2f",
        "stage_start_min": ":.2f",
        "stage_finish_min": ":.2f",
    }
    if "machine_id" in g.columns:
        hover_data["machine_id"] = True
    if "machine_label" in g.columns:
        hover_data["machine_label"] = True
    if "wait_min" in g.columns:
        hover_data["wait_min"] = ":.2f"
    if "service_min" in g.columns:
        hover_data["service_min"] = ":.2f"
    if "downtime_min" in g.columns:
        hover_data["downtime_min"] = ":.2f"
    if "between_steps_gap_min" in g.columns:
        hover_data["between_steps_gap_min"] = ":.2f"
    if "lot_arrival_min" in g.columns:
        hover_data["lot_arrival_min"] = ":.2f"
    if "interarrival_min" in g.columns:
        hover_data["interarrival_min"] = ":.2f"

    fig = px.timeline(
        g,
        x_start="start_ts",
        x_end="finish_ts",
        y="lot_name",
        color="process",
        hover_data=hover_data,
    )
    fig.update_yaxes(autorange="reversed", title="Lot")
    fig.update_xaxes(title="Timeline")
    fig.update_layout(height=420, margin={"l": 20, "r": 20, "t": 50, "b": 20}, legend_title_text="Process")
    return fig


def build_history_options(history: list[dict]) -> list[dict[str, str]]:
    options: list[dict[str, str]] = []
    for item in history:
        run_id = str(item.get("run_id", "")).strip()
        label = str(item.get("label", run_id)).strip() or run_id
        if run_id:
            options.append({"label": label, "value": run_id})
    return options


def history_item_by_id(history: list[dict], run_id: str | None) -> dict | None:
    rid = str(run_id or "").strip()
    if not rid:
        return None
    for item in history:
        if str(item.get("run_id", "")).strip() == rid:
            return item
    return None


def ensure_total_cost_row(rows: list[dict]) -> list[dict]:
    out = list(rows) if isinstance(rows, list) else []
    if not out:
        return out

    for r in out:
        if str(r.get("process", "")).strip().upper() == "TOTAL":
            return out

    def n(v: object) -> float:
        try:
            x = float(v)
            return x if np.isfinite(x) else 0.0
        except Exception:
            return 0.0

    total_energy_cost = float(sum(n(r.get("energy_cost")) for r in out))
    total_labor_cost = float(sum(n(r.get("labor_cost")) for r in out))
    total_gas_cost = float(sum(n(r.get("gas_cost")) for r in out))
    total_pieces = float(sum(n(r.get("pieces_total")) for r in out))

    total_row = {
        "process": "TOTAL",
        "kwh_per_machine_hour": None,
        "kwh_source": "summary",
        "visits": int(round(sum(n(r.get("visits")) for r in out))),
        "pieces_total": round(total_pieces, 2),
        "clean_service_min_h": None,
        "clean_service_max_h": None,
        "sim_service_min_h": None,
        "sim_service_max_h": None,
        "service_total_h": round(float(sum(n(r.get("service_total_h")) for r in out)), 4),
        "downtime_total_h": round(float(sum(n(r.get("downtime_total_h")) for r in out)), 4),
        "gas_temp_c": None,
        "gas_cost_per_cuero": round((total_gas_cost / total_pieces), 6) if total_pieces > 0 else None,
        "energy_cost": round(total_energy_cost, 2),
        "labor_cost": round(total_labor_cost, 2),
        "gas_cost": round(total_gas_cost, 2),
        "total_cost": round(total_energy_cost + total_labor_cost + total_gas_cost, 2),
    }
    return out + [total_row]


RECURTIDO_PROCESS = "RECURTIDO"
RECURTIDO_SERVERS = 12
PROCESS_OPTIONS = sorted(set(core.PROCESS_OPTIONS + [RECURTIDO_PROCESS]))
EMPIRICAL_PROCESS_OPTIONS = [p for p in PROCESS_OPTIONS if p != RECURTIDO_PROCESS]
DEFAULT_ROUTE = [x for x in ["RASPADO", "BAUCE", "VACIO"] if x in PROCESS_OPTIONS]
ENERGY_REF_BY_PROCESS = core.ENERGY_REFERENCE.get("by_process", {}) if isinstance(core.ENERGY_REFERENCE, dict) else {}
ENERGY_REF_SOURCE = core.ENERGY_REFERENCE.get("source_path") if isinstance(core.ENERGY_REFERENCE, dict) else None
ENERGY_REF_ROWS = int(core.ENERGY_REFERENCE.get("rows_parsed", 0)) if isinstance(core.ENERGY_REFERENCE, dict) else 0
DEFAULT_ENERGY_FALLBACK = float(core.DEFAULT_ENERGY_KWH_PER_MACHINE_HOUR)
UPLOAD_DIR = os.path.join(os.getcwd(), "outputs", "uploads")
SPC_METRIC_OPTIONS = {
    "service_hours": "Service time (hours)",
    "wait_hours": "Queue wait (hours)",
    "cycle_hours": "Cycle time (hours)",
}
SPC_METRIC_TO_MIN = {
    "service_hours": "service_min",
    "wait_hours": "wait_min",
    "cycle_hours": "cycle_min",
}
# One-time manual override requested by user.
PROCESS_KWH_OVERRIDE = {"RASPADO": 44.0}

# Drying gas model applies only to these processes.
GAS_DRYING_PROCESSES = {"LTD", "TAIC", "AEREO"}
SECADO_RATE_IQR_FILTER_PROCESSES = {"LTD", "LTB", "TAIC", "TAIK", "AEREO", "AERO"}
SECADO_NO_PIECE_SCALING_PROCESSES = {"LTD", "LTB", "TAIC", "TAIK", "AEREO", "AERO"}
SECADO_SERVICE_HOUR_CAPS = {
    "BAUCE": 4.0,
    "TAIC": 4.0,
    "TAIK": 4.0,
    "LTD": 10.0,
    "LTB": 10.0,
    "AEREO": 6.0,
    "AERO": 6.0,
}
SECADO_EXPECTED_MEAN_H = {"LTD": "8-9"}
GAS_REFERENCE_TEMP_C = 50.0
DRYING_TEMPERATURE_C_FIXED = {
    "LTD": 50.0,
    "LTB": 50.0,
    "TAIC": 60.0,
    "TAIK": 60.0,
    "AEREO": 45.0,
    "AERO": 45.0,
}


def current_process_options() -> list[str]:
    if core.DATAFRAME.empty or "process" not in core.DATAFRAME.columns:
        return []
    return sorted(
        {
            str(v).strip().upper()
            for v in core.DATAFRAME["process"].dropna().tolist()
            if str(v).strip() and str(v).strip().upper() != "UNKNOWN"
        }
    )


def current_date_range() -> list[str | None]:
    if core.DATAFRAME.empty or "arrival_time" not in core.DATAFRAME.columns:
        return [None, None]
    dts = pd.to_datetime(core.DATAFRAME["arrival_time"], errors="coerce").dropna()
    if dts.empty:
        return [None, None]
    return [dts.min().date().isoformat(), dts.max().date().isoformat()]


def default_route_for_options(options: list[str]) -> list[str]:
    preferred = [x for x in ["RASPADO", "BAUCE", "VACIO"] if x in options]
    if preferred:
        return preferred
    return options[: min(3, len(options))]


def default_process_for_options(options: list[str]) -> str | None:
    if "RASPADO" in options:
        return "RASPADO"
    return options[0] if options else None


def safe_upload_filename(filename: str | None) -> str:
    raw = os.path.basename(str(filename or "uploaded_workbook.xlsx")).strip() or "uploaded_workbook.xlsx"
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in raw)
    if not safe.lower().endswith((".xlsx", ".xls")):
        safe += ".xlsx"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{stamp}_{safe}"


def refresh_active_data_context(paths: str | list[str]) -> tuple[bool, str, dict]:
    path_list = [paths] if isinstance(paths, str) else [str(p) for p in (paths or []) if str(p).strip()]
    loaded_frames: list[pd.DataFrame] = []
    load_notes: list[str] = []
    valid_paths: list[str] = []

    for path in path_list:
        df, error = core.load_data(path, core.SHEET_NAME)
        if error:
            load_notes.append(f"{os.path.basename(path)} failed: {error}")
            continue
        if df.empty:
            load_notes.append(f"{os.path.basename(path)} had no usable production rows")
            continue
        df = df.copy()
        df["uploaded_source_file"] = os.path.basename(path)
        loaded_frames.append(df)
        valid_paths.append(path)

    if not loaded_frames:
        details = "; ".join(load_notes) if load_notes else "No usable production rows were found."
        return False, details, {}

    df = pd.concat(loaded_frames, ignore_index=True)
    source_label = " + ".join(os.path.basename(p) for p in valid_paths)

    core.DATA_PATH = source_label
    core.DATAFRAME = df
    core.DATA_ERROR = None
    core.REFERENCE_SHEETS = core.load_reference_sheets(valid_paths[0])
    core.PROCESS_MACHINE_CATALOG_RESOLVED = core.build_process_machine_catalog(df)
    core.PROCESS_SERVER_COUNT_RESOLVED = core.build_process_server_counts(core.PROCESS_MACHINE_CATALOG_RESOLVED)
    core.PROCESS_OPTIONS = current_process_options()
    core.DEFAULT_PROCESS = default_process_for_options(core.PROCESS_OPTIONS)
    core.DEFAULT_ENERGY_REFERENCE_KWH = core.get_energy_kwh_reference_for_process(core.DEFAULT_PROCESS)
    core.DEFAULT_DATE_RANGE = current_date_range()

    global PROCESS_OPTIONS, EMPIRICAL_PROCESS_OPTIONS, DEFAULT_ROUTE, MIN_DATE, MAX_DATE
    PROCESS_OPTIONS = sorted(set(core.PROCESS_OPTIONS + [RECURTIDO_PROCESS]))
    EMPIRICAL_PROCESS_OPTIONS = [p for p in PROCESS_OPTIONS if p != RECURTIDO_PROCESS]
    DEFAULT_ROUTE = default_route_for_options(PROCESS_OPTIONS)
    MIN_DATE, MAX_DATE = current_date_range()

    bayes_core.PROCESS_OPTIONS = [str(p).strip().upper() for p in core.PROCESS_OPTIONS if str(p).strip()]

    meta = {
        "path": source_label,
        "paths": valid_paths,
        "files_loaded": len(valid_paths),
        "rows": int(len(df)),
        "processes": len(core.PROCESS_OPTIONS),
        "date_range": current_date_range(),
        "notes": load_notes,
        "process_options": PROCESS_OPTIONS,
        "empirical_process_options": EMPIRICAL_PROCESS_OPTIONS,
    }
    return True, "Uploaded workbook data is active.", meta


def save_uploaded_workbook(contents: str, filename: str | None) -> tuple[str | None, str | None]:
    if not contents:
        return None, "No upload contents received."
    try:
        _content_type, encoded = contents.split(",", 1)
        decoded = base64.b64decode(encoded)
    except Exception as exc:
        return None, f"Could not decode uploaded file: {exc}"

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    path = os.path.join(UPLOAD_DIR, safe_upload_filename(filename))
    try:
        with open(path, "wb") as f:
            f.write(decoded)
    except Exception as exc:
        return None, f"Could not save uploaded file: {exc}"
    return path, None


def save_uploaded_workbooks(contents: str | list[str], filenames: str | list[str] | None) -> tuple[list[str], str | None]:
    content_list = contents if isinstance(contents, list) else [contents]
    if isinstance(filenames, list):
        name_list = filenames
    else:
        name_list = [filenames] * len(content_list)

    saved_paths: list[str] = []
    errors: list[str] = []
    for i, content in enumerate(content_list):
        name = name_list[i] if i < len(name_list) else f"uploaded_workbook_{i+1}.xlsx"
        path, err = save_uploaded_workbook(content, name)
        if err:
            errors.append(f"{name}: {err}")
        elif path:
            saved_paths.append(path)

    if not saved_paths:
        return [], "; ".join(errors) if errors else "No upload contents received."
    return saved_paths, "; ".join(errors) if errors else None


def normalize_sim_route(route_values: list[str] | None) -> list[str]:
    # Preserve the user's chosen order exactly. RECURTIDO is optional and is
    # only simulated where the user places it in the route.
    return [str(p).strip().upper() for p in (route_values or []) if str(p).strip().upper() in PROCESS_OPTIONS]


def build_recurtido_stage_spec(hours_value: float | int | None) -> dict[str, object]:
    try:
        hours = float(hours_value)
    except Exception:
        hours = 1.0
    hours = float(max(1e-6, min(240.0, hours)))

    return {
        "process": RECURTIDO_PROCESS,
        "servers": RECURTIDO_SERVERS,
        "machine_labels": [str(i) for i in range(1, RECURTIDO_SERVERS + 1)],
        "rows_used": 0,
        "service_cap_h": hours,
        "service_rows_before_cap": 0,
        "service_rows_after_cap": 0,
        "service_rows_cap_removed": 0,
        "rate_rows_before": 0,
        "rate_rows_after": 0,
        "rate_guardrail_on": False,
        "rate_guardrail_low": None,
        "rate_guardrail_high": None,
        "no_piece_scaling": True,
        "service_range_guardrail_on": True,
        "service_min_h": hours,
        "service_max_h": hours,
        "mean_service_h_empirical": hours,
        "draw_service_h": lambda _pieces, h=hours: float(h),
        "draw_downtime_h": lambda: 0.0,
    }


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


def get_drying_temperature_c(process_value: str) -> float | None:
    key = core.normalize_process_key(process_value)
    return DRYING_TEMPERATURE_C_FIXED.get(key)


def build_drying_temperature_rows() -> list[dict]:
    rows: list[dict] = []
    for proc in sorted(GAS_DRYING_PROCESSES):
        temp_c = get_drying_temperature_c(proc)
        rows.append({"process": proc, "temperature_c": temp_c})
    return rows


def _finite_float(value: object) -> float:
    try:
        x = float(value)
        return x if np.isfinite(x) else np.nan
    except Exception:
        return np.nan


def _rounded(value: object, ndigits: int = 4) -> float | None:
    x = _finite_float(value)
    return round(float(x), ndigits) if np.isfinite(x) else None


def clean_process_metric_df(
    process_value: str,
    metric_col: str,
    date_range: list[str] | None,
    strict_cleaning: bool = True,
    exclude_iqr: bool = True,
) -> tuple[pd.DataFrame, dict[str, int]]:
    metric_col = metric_col if metric_col in SPC_METRIC_OPTIONS else "service_hours"
    metric_min = SPC_METRIC_TO_MIN.get(metric_col, "service_min")
    proc = str(process_value or "").strip().upper()
    counts = {"raw_rows": 0, "strict_removed": 0, "outliers_found": 0, "iqr_removed": 0}

    if core.DATAFRAME.empty or not proc:
        return pd.DataFrame(), counts

    df = core.DATAFRAME[core.DATAFRAME["process"].astype(str).str.upper().str.strip() == proc].copy()
    counts["raw_rows"] = int(len(df))

    if date_range and len(date_range) == 2 and date_range[0] and date_range[1]:
        start = pd.to_datetime(date_range[0], errors="coerce")
        end = pd.to_datetime(date_range[1], errors="coerce")
        if pd.notna(start) and pd.notna(end):
            df = df[(df["arrival_time"] >= start) & (df["arrival_time"] <= end + pd.Timedelta(days=1))].copy()

    if df.empty:
        return df, counts

    before_strict = int(len(df))
    df = core.add_quality_flags(df, process_value=proc)
    if strict_cleaning:
        df = df[(~df["invalid_time_row"]) & (~df["missing_time_flag"])].copy()
    counts["strict_removed"] = int(before_strict - len(df))

    if df.empty:
        return df, counts

    df = core.apply_outlier_flags(df, method=core.FIXED_OUTLIER_METHOD, metrics=[metric_min])
    class_col = f"{metric_min}_outlier_class"
    if class_col in df.columns:
        is_outlier = df[class_col].astype(str).isin(["mild", "extreme"])
        counts["outliers_found"] = int(is_outlier.sum())
        if exclude_iqr:
            before_iqr = int(len(df))
            df = df[~is_outlier].copy()
            counts["iqr_removed"] = int(before_iqr - len(df))

    df[metric_col] = pd.to_numeric(df.get(metric_col), errors="coerce")
    df = df[np.isfinite(df[metric_col]) & (df[metric_col] > 0)].copy()
    if "arrival_time" in df.columns:
        df = df.sort_values("arrival_time").copy()
    df["obs_order"] = np.arange(1, len(df) + 1)
    return df, counts


def capability_metrics(values: pd.Series, lsl: object, usl: object) -> dict[str, object]:
    x = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    x = x[np.isfinite(x) & (x > 0)]
    n = int(len(x))
    lsl_v = _finite_float(lsl)
    usl_v = _finite_float(usl)
    mean = float(x.mean()) if n else np.nan
    std_overall = float(x.std(ddof=1)) if n > 1 else np.nan
    mr = np.abs(np.diff(x.to_numpy(dtype=float))) if n > 1 else np.array([], dtype=float)
    mrbar = float(np.nanmean(mr)) if mr.size else np.nan
    sigma_within = float(mrbar / 1.128) if np.isfinite(mrbar) and mrbar > 0 else np.nan

    def capability_pair(sigma: float) -> tuple[float, float]:
        if not np.isfinite(sigma) or sigma <= 0 or not np.isfinite(mean):
            return np.nan, np.nan
        cp = (usl_v - lsl_v) / (6.0 * sigma) if np.isfinite(lsl_v) and np.isfinite(usl_v) and usl_v > lsl_v else np.nan
        side_vals = []
        if np.isfinite(usl_v):
            side_vals.append((usl_v - mean) / (3.0 * sigma))
        if np.isfinite(lsl_v):
            side_vals.append((mean - lsl_v) / (3.0 * sigma))
        cpk = float(min(side_vals)) if side_vals else np.nan
        return float(cp), float(cpk)

    cp, cpk = capability_pair(sigma_within)
    pp, ppk = capability_pair(std_overall)
    out_spec = np.zeros(n, dtype=bool)
    if n:
        if np.isfinite(lsl_v):
            out_spec |= x.to_numpy(dtype=float) < lsl_v
        if np.isfinite(usl_v):
            out_spec |= x.to_numpy(dtype=float) > usl_v

    return {
        "n": n,
        "mean": mean,
        "median": float(x.median()) if n else np.nan,
        "std_overall": std_overall,
        "mrbar": mrbar,
        "sigma_within": sigma_within,
        "lsl": lsl_v if np.isfinite(lsl_v) else np.nan,
        "usl": usl_v if np.isfinite(usl_v) else np.nan,
        "cp": cp,
        "cpk": cpk,
        "pp": pp,
        "ppk": ppk,
        "out_of_spec": int(out_spec.sum()) if n else 0,
        "out_of_spec_pct": float(out_spec.mean() * 100.0) if n else np.nan,
        "min": float(x.min()) if n else np.nan,
        "max": float(x.max()) if n else np.nan,
    }


def capability_table_rows(
    process_value: str,
    metric_col: str,
    cap: dict[str, object],
    counts: dict[str, int],
    i_violations: int,
    mr_violations: int,
) -> list[dict]:
    if int(cap.get("n", 0) or 0) == 0:
        return []
    return [
        {
            "process": process_value,
            "metric": SPC_METRIC_OPTIONS.get(metric_col, metric_col),
            "n": int(cap["n"]),
            "mean": _rounded(cap.get("mean"), 4),
            "median": _rounded(cap.get("median"), 4),
            "min": _rounded(cap.get("min"), 4),
            "max": _rounded(cap.get("max"), 4),
            "std_overall": _rounded(cap.get("std_overall"), 4),
            "sigma_within_mr": _rounded(cap.get("sigma_within"), 4),
            "lsl": _rounded(cap.get("lsl"), 4),
            "usl": _rounded(cap.get("usl"), 4),
            "cp": _rounded(cap.get("cp"), 4),
            "cpk": _rounded(cap.get("cpk"), 4),
            "pp": _rounded(cap.get("pp"), 4),
            "ppk": _rounded(cap.get("ppk"), 4),
            "out_of_spec": int(cap.get("out_of_spec", 0) or 0),
            "out_of_spec_pct": _rounded(cap.get("out_of_spec_pct"), 2),
            "i_chart_violations": int(i_violations),
            "mr_chart_violations": int(mr_violations),
            "raw_rows": int(counts.get("raw_rows", 0)),
            "strict_removed": int(counts.get("strict_removed", 0)),
            "iqr_found": int(counts.get("outliers_found", 0)),
            "iqr_removed": int(counts.get("iqr_removed", 0)),
        }
    ]


def build_control_chart(df: pd.DataFrame, metric_col: str, cap: dict[str, object]) -> tuple[go.Figure, int, int]:
    if df.empty or int(cap.get("n", 0) or 0) < 2:
        return empty_figure("Need at least 2 clean observations for I-MR chart"), 0, 0

    y = pd.to_numeric(df[metric_col], errors="coerce").to_numpy(dtype=float)
    x = df["obs_order"].to_numpy(dtype=int)
    mean = float(cap.get("mean", np.nan))
    sigma_within = float(cap.get("sigma_within", np.nan))
    mr = np.abs(np.diff(y))
    mrbar = float(cap.get("mrbar", np.nan))

    i_ucl = mean + 3.0 * sigma_within if np.isfinite(sigma_within) else np.nan
    i_lcl = mean - 3.0 * sigma_within if np.isfinite(sigma_within) else np.nan
    mr_ucl = 3.267 * mrbar if np.isfinite(mrbar) else np.nan
    mr_lcl = 0.0
    i_viol = int(((y > i_ucl) | (y < i_lcl)).sum()) if np.isfinite(i_ucl) and np.isfinite(i_lcl) else 0
    mr_viol = int(((mr > mr_ucl) | (mr < mr_lcl)).sum()) if np.isfinite(mr_ucl) else 0

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        subplot_titles=("Individuals chart", "Moving range chart"),
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines+markers",
            name="Observation",
            marker={"color": "#2563eb", "size": 7},
            line={"color": "#2563eb", "width": 1.5},
            hovertemplate="obs=%{x}<br>value=%{y:.4f} h<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x[1:],
            y=mr,
            mode="lines+markers",
            name="Moving range",
            marker={"color": "#0f766e", "size": 7},
            line={"color": "#0f766e", "width": 1.5},
            hovertemplate="obs=%{x}<br>MR=%{y:.4f} h<extra></extra>",
        ),
        row=2,
        col=1,
    )

    for value, label, color in [
        (mean, "Mean", "#111827"),
        (i_ucl, "I UCL", "#dc2626"),
        (i_lcl, "I LCL", "#dc2626"),
    ]:
        if np.isfinite(value):
            fig.add_hline(y=value, line_dash="dash", line_color=color, annotation_text=f"{label}={value:.3f}", row=1, col=1)
    for value, label, color in [
        (mrbar, "MRbar", "#111827"),
        (mr_ucl, "MR UCL", "#dc2626"),
        (mr_lcl, "MR LCL", "#dc2626"),
    ]:
        if np.isfinite(value):
            fig.add_hline(y=value, line_dash="dash", line_color=color, annotation_text=f"{label}={value:.3f}", row=2, col=1)

    fig.update_xaxes(title="Observation order", row=2, col=1)
    fig.update_yaxes(title=SPC_METRIC_OPTIONS.get(metric_col, metric_col), row=1, col=1)
    fig.update_yaxes(title="Moving range", row=2, col=1)
    fig.update_layout(height=560, margin={"l": 30, "r": 20, "t": 70, "b": 40}, showlegend=False)
    return core.style_figure(fig, "I-MR control chart"), i_viol, mr_viol


def build_capability_histogram(df: pd.DataFrame, metric_col: str, cap: dict[str, object]) -> go.Figure:
    if df.empty:
        return empty_figure("No clean data for capability histogram")
    x = pd.to_numeric(df[metric_col], errors="coerce").dropna()
    x = x[np.isfinite(x) & (x > 0)]
    if x.empty:
        return empty_figure("No positive metric values")

    def normal_pdf(grid: np.ndarray, mu: float, sigma: float) -> np.ndarray:
        if not np.isfinite(mu) or not np.isfinite(sigma) or sigma <= 0:
            return np.full_like(grid, np.nan, dtype=float)
        z = (grid - mu) / sigma
        return np.exp(-0.5 * z * z) / (sigma * np.sqrt(2.0 * np.pi))

    mean = _finite_float(cap.get("mean"))
    sigma_within = _finite_float(cap.get("sigma_within"))
    sigma_overall = _finite_float(cap.get("std_overall"))
    lsl = _finite_float(cap.get("lsl"))
    usl = _finite_float(cap.get("usl"))

    spread_candidates = [s for s in [sigma_within, sigma_overall, float(x.std(ddof=1))] if np.isfinite(s) and s > 0]
    spread = max(spread_candidates) if spread_candidates else max(float(x.max() - x.min()), 1.0)
    plot_values = [float(x.min()), float(x.max())]
    if np.isfinite(mean):
        plot_values.extend([mean - 4.0 * spread, mean + 4.0 * spread])
    if np.isfinite(lsl):
        plot_values.append(lsl)
    if np.isfinite(usl):
        plot_values.append(usl)
    x_min = max(0.0, min(plot_values) - 0.12 * spread)
    x_max = max(plot_values) + 0.12 * spread
    if not np.isfinite(x_min) or not np.isfinite(x_max) or x_max <= x_min:
        x_min, x_max = float(x.min()), float(x.max())
    grid = np.linspace(x_min, x_max, 400)
    overall_pdf = normal_pdf(grid, mean, sigma_overall)
    within_pdf = normal_pdf(grid, mean, sigma_within)

    fig = go.Figure()
    fig.add_histogram(
        x=x,
        nbinsx=min(35, max(6, int(np.sqrt(len(x)) * 2))),
        histnorm="probability density",
        marker_color="#99f6e4",
        marker_line_color="#0f766e",
        opacity=0.78,
        name="Observed density",
    )

    if np.isfinite(overall_pdf).any():
        fig.add_trace(
            go.Scatter(
                x=grid,
                y=overall_pdf,
                mode="lines",
                name="Overall spread (Pp/Ppk)",
                line={"color": "#ea580c", "width": 3},
                hovertemplate="x=%{x:.4f}<br>density=%{y:.4f}<extra>Overall</extra>",
            )
        )
    if np.isfinite(within_pdf).any():
        fig.add_trace(
            go.Scatter(
                x=grid,
                y=within_pdf,
                mode="lines",
                name="Within spread (Cp/Cpk)",
                line={"color": "#2563eb", "width": 3, "dash": "dash"},
                hovertemplate="x=%{x:.4f}<br>density=%{y:.4f}<extra>Within</extra>",
            )
        )

    shade_pdf = overall_pdf if np.isfinite(overall_pdf).any() else within_pdf
    if np.isfinite(shade_pdf).any():
        if np.isfinite(lsl):
            mask = grid < lsl
            if mask.any():
                fig.add_trace(
                    go.Scatter(
                        x=grid[mask],
                        y=shade_pdf[mask],
                        mode="lines",
                        fill="tozeroy",
                        name="Below LSL risk",
                        line={"color": "rgba(220, 38, 38, 0.05)"},
                        fillcolor="rgba(220, 38, 38, 0.18)",
                        hoverinfo="skip",
                    )
                )
        if np.isfinite(usl):
            mask = grid > usl
            if mask.any():
                fig.add_trace(
                    go.Scatter(
                        x=grid[mask],
                        y=shade_pdf[mask],
                        mode="lines",
                        fill="tozeroy",
                        name="Above USL risk",
                        line={"color": "rgba(220, 38, 38, 0.05)"},
                        fillcolor="rgba(220, 38, 38, 0.18)",
                        hoverinfo="skip",
                    )
                )

    if np.isfinite(lsl) and np.isfinite(usl) and usl > lsl:
        fig.add_vrect(
            x0=lsl,
            x1=usl,
            fillcolor="rgba(16, 185, 129, 0.08)",
            line_width=0,
            layer="below",
            annotation_text="Spec window",
            annotation_position="top left",
        )

    for value, label, color in [(mean, "Mean", "#111827"), (lsl, "LSL", "#dc2626"), (usl, "USL", "#dc2626")]:
        if np.isfinite(value):
            dash = "solid" if label == "Mean" else "dash"
            fig.add_vline(x=value, line_dash=dash, line_color=color, annotation_text=f"{label}={value:.3f}")

    cp = cap.get("cp")
    cpk = cap.get("cpk")
    pp = cap.get("pp")
    ppk = cap.get("ppk")
    spec_text = (
        f"LSL={core.safe_number(lsl, 3)} | USL={core.safe_number(usl, 3)}"
        if np.isfinite(lsl) or np.isfinite(usl)
        else "Enter LSL/USL to calculate capability"
    )
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.99,
        y=0.98,
        showarrow=False,
        align="right",
        bgcolor="rgba(255,255,255,0.86)",
        bordercolor="#cbd5e1",
        borderwidth=1,
        text=(
            f"<b>Capability summary</b><br>"
            f"{spec_text}<br>"
            f"Mean={core.safe_number(mean, 3)} | n={int(cap.get('n', 0) or 0)}<br>"
            f"Cp={core.safe_number(cp, 3)} | Cpk={core.safe_number(cpk, 3)}<br>"
            f"Pp={core.safe_number(pp, 3)} | Ppk={core.safe_number(ppk, 3)}"
        ),
    )

    fig.update_xaxes(title=SPC_METRIC_OPTIONS.get(metric_col, metric_col))
    fig.update_yaxes(title="Density")
    fig.update_layout(
        height=520,
        margin={"l": 30, "r": 30, "t": 70, "b": 45},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
    )
    return core.style_figure(fig, "Capability curve with LSL/USL")


def build_process_comparison_histogram(process_values: list[str] | None, metric_col: str, date_range: list[str] | None) -> go.Figure:
    selected = [str(p).strip().upper() for p in (process_values or []) if str(p).strip()]
    if not selected:
        return empty_figure("Select one or more processes")
    rows = []
    for proc in selected:
        df, _counts = clean_process_metric_df(proc, metric_col, date_range, strict_cleaning=True, exclude_iqr=True)
        if df.empty:
            continue
        tmp = df[["process", metric_col]].copy()
        tmp["metric_value"] = pd.to_numeric(tmp[metric_col], errors="coerce")
        rows.append(tmp[["process", "metric_value"]])
    if not rows:
        return empty_figure("No clean data for selected processes")
    comp = pd.concat(rows, ignore_index=True)
    fig = px.histogram(
        comp,
        x="metric_value",
        color="process",
        histnorm="probability density",
        barmode="overlay",
        opacity=0.55,
        nbins=30,
        labels={"metric_value": SPC_METRIC_OPTIONS.get(metric_col, metric_col), "process": "Process"},
    )
    fig.update_yaxes(title="Density")
    return core.style_figure(fig, "Clean process histogram comparison")

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
        className="app-shell",
        children=dmc.Stack(
            gap="md",
            children=[
                dcc.Store(id="lot-store", data=[]),
                dcc.Store(
                    id="data-source-store",
                    data={
                        "path": core.DATA_PATH,
                        "rows": int(len(core.DATAFRAME)),
                        "processes": len(current_process_options()),
                        "date_range": current_date_range(),
                    },
                ),
                # Keep simulation history in-memory to avoid localStorage quota
                # failures when figure payloads get large.
                dcc.Store(id="sim-history-store", data=[], storage_type="memory"),
                dmc.Alert(
                    id="data-source-alert",
                    children=core.DATA_ERROR if core.DATA_ERROR else f"Data source: {core.DATA_PATH}",
                    color="red" if core.DATA_ERROR else "teal",
                    variant="light",
                    className="source-alert",
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
                                align="center",
                                children=[
                                    dmc.Stack(
                                        gap=2,
                                        children=[
                                            dmc.Text("Upload production Excel", fw=700),
                                            dmc.Text(
                                                "User imports a plant workbook; the app validates it and uses it for simulation, SPC, capability, and classifier analysis.",
                                                c="dimmed",
                                                fz="sm",
                                            ),
                                        ],
                                    ),
                                    dmc.Badge("Product input", color="teal", variant="light"),
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
                                    children=dmc.Stack(
                                        gap=2,
                                        align="center",
                                        children=[
                                            dmc.Text("Drag and drop one or more Excel files here, or click to browse.", fw=600),
                                            dmc.Text("Expected format: production time records / production logbook sheets, e.g. TIEMPOS or BITACORA.", c="dimmed", fz="xs"),
                                        ],
                                    ),
                                ),
                            ),
                            dmc.Alert(
                                id="upload-status",
                                color="gray",
                                variant="light",
                                className="status-alert",
                                children="No uploaded workbook yet. The app is using the default data source above. You can upload one file or combine multiple Excel logbooks.",
                            ),
                        ],
                    ),
                ),
                dmc.Paper(
                    withBorder=True,
                    radius="lg",
                    p="lg",
                    className="hero-panel app-card",
                    children=dmc.Stack(
                        gap="sm",
                        children=[
                            dmc.Group(
                                justify="space-between",
                                align="flex-start",
                                children=[
                                    dmc.Stack(
                                        gap=4,
                                        children=[
                                            dmc.Badge("Business demo mode", color="indigo", variant="light"),
                                            dmc.Title("Production Flow Decision Studio", order=2),
                                            dmc.Text(
                                                "A user enters lots, pieces, and process routes; the app simulates how the plant behaves and turns the result into cost, timeline, bottleneck, and quality insights.",
                                                c="dimmed",
                                            ),
                                        ],
                                    ),
                                    dmc.Paper(
                                        withBorder=True,
                                        radius="md",
                                        p="sm",
                                        className="workflow-card",
                                        children=dmc.Stack(
                                            gap=3,
                                            children=[
                                                dmc.Text("Demo workflow", fw=700, fz="sm"),
                                                dmc.Text("1. Add lot mix and route", fz="xs", c="dimmed"),
                                                dmc.Text("2. Simulate plant flow", fz="xs", c="dimmed"),
                                                dmc.Text("3. Compare cost, time, capacity, and risk", fz="xs", c="dimmed"),
                                            ],
                                        ),
                                    ),
                                ],
                            ),
                            dmc.Alert(
                                id="business-output",
                                title="Executive decision summary",
                                color="blue",
                                variant="light",
                                className="executive-summary",
                                children=build_business_summary_children("Run a simulation to generate the business summary."),
                            ),
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
                                label="Processes (ordered; place RECURTIDO wherever you want)",
                                data=option_data(PROCESS_OPTIONS),
                                value=DEFAULT_ROUTE,
                                searchable=True,
                                clearable=True,
                                placeholder="Select process sequence",
                            ),
                            dmc.NumberInput(
                                id="recurtido-hours",
                                label="RECURTIDO hours (fixed service time, c=12)",
                                value=1.0,
                                min=0.01,
                                max=240,
                                step=0.25,
                                decimalScale=3,
                            ),
                            dmc.Group(
                                gap="xs",
                                className="action-row",
                                children=[
                                    dmc.Button("Add lot", id="add-lot", color="teal"),
                                    dmc.Button("Clear lots", id="clear-lots", color="gray", variant="light"),
                                    dmc.Button("Simulate", id="run-sim", color="indigo"),
                                    dmc.Button("Estimate gas (drying only)", id="run-gas", color="orange", variant="light"),
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
                            dmc.Group(
                                grow=True,
                                children=[
                                    dmc.NumberInput(id="gantt-start-hour", label="Gantt start hour", value=11, min=0, max=23, step=1, allowDecimal=False),
                                    dmc.NumberInput(id="gantt-start-minute", label="Gantt start minute", value=0, min=0, max=59, step=1, allowDecimal=False),
                                ],
                            ),
                            dmc.Group(
                                grow=True,
                                children=[
                                    dmc.NumberInput(
                                        id="gas-cost-per-cuero-50c",
                                        label="Estimated gas cost at 50 C ($ / cuero)",
                                        value=0.12,
                                        min=0,
                                        step=0.001,
                                        decimalScale=6,
                                    ),
                                ],
                            ),
                            dmc.Text(
                                f"Fixed energy consumption per process loaded from Excel: {ENERGY_REF_SOURCE or 'N/A'} ({ENERGY_REF_ROWS} rows).",
                                c="dimmed",
                                fz="xs",
                            ),
                            dmc.Text(
                                "Drying-only gas model (LTD, TAIC, AEREO): estimated gas cost per cuero scales by fixed process temperature.",
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
                            dmc.Alert(
                                id="gas-output",
                                title="Gas Output (Drying)",
                                color="orange",
                                variant="light",
                                children="Click Estimate gas (drying only) to include gas KPIs.",
                            ),
                        ],
                    ),
                ),
                dmc.Paper(
                    withBorder=True,
                    radius="lg",
                    p="lg",
                    className="results-panel app-card",
                    children=dmc.Stack(
                        gap="sm",
                        children=[
                            dmc.Group(
                                justify="space-between",
                                align="center",
                                children=[
                                    dmc.Text("Configured lots", fw=700),
                                    dmc.Group(
                                        gap="xs",
                                        children=[
                                            dmc.Button("Clear lots", id="clear-lots-bottom", color="gray", variant="light", size="xs"),
                                            dmc.Button("Clear simulations", id="clear-sim-history", color="gray", variant="outline", size="xs"),
                                        ],
                                    ),
                                ],
                            ),
                            dmc.Select(
                                id="sim-toggle-select",
                                label="Simulation view",
                                data=[],
                                value=None,
                                placeholder="Run simulation to create history",
                                clearable=True,
                                searchable=True,
                            ),
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
                            dcc.Graph(id="cost-fig", figure=empty_figure("Run simulation"), className="plot-card"),
                            dmc.Text("Gantt by timeline", fw=700),
                            dcc.Graph(id="gantt-fig", figure=empty_figure("Run simulation"), className="plot-card"),
                            dash_table.DataTable(
                                id="cost-table",
                                columns=[
                                    {"name": "process", "id": "process"},
                                    {"name": "kwh_per_machine_hour", "id": "kwh_per_machine_hour"},
                                    {"name": "kwh_source", "id": "kwh_source"},
                                    {"name": "visits", "id": "visits"},
                                    {"name": "pieces_total", "id": "pieces_total"},
                                    {"name": "clean_service_min_h", "id": "clean_service_min_h"},
                                    {"name": "clean_service_max_h", "id": "clean_service_max_h"},
                                    {"name": "sim_service_min_h", "id": "sim_service_min_h"},
                                    {"name": "sim_service_max_h", "id": "sim_service_max_h"},
                                    {"name": "service_total_h", "id": "service_total_h"},
                                    {"name": "downtime_total_h", "id": "downtime_total_h"},
                                    {"name": "gas_temp_c", "id": "gas_temp_c"},
                                    {"name": "gas_cost_per_cuero", "id": "gas_cost_per_cuero"},
                                    {"name": "energy_cost", "id": "energy_cost"},
                                    {"name": "labor_cost", "id": "labor_cost"},
                                    {"name": "gas_cost", "id": "gas_cost"},
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
                            dmc.Divider(label="SPC + Capability", labelPosition="center"),
                            dmc.Text("Control charts and process capability", fw=700),
                            dmc.Text(
                                "Uses clean historical process observations in the selected date range. Cp/Cpk use within sigma from moving range; Pp/Ppk use overall process sigma.",
                                c="dimmed",
                                fz="sm",
                            ),
                            dmc.Group(
                                grow=True,
                                children=[
                                    dmc.Select(
                                        id="spc-process",
                                        label="Process for control chart",
                                        data=option_data(EMPIRICAL_PROCESS_OPTIONS),
                                        value="RASPADO" if "RASPADO" in EMPIRICAL_PROCESS_OPTIONS else (EMPIRICAL_PROCESS_OPTIONS[0] if EMPIRICAL_PROCESS_OPTIONS else None),
                                        searchable=True,
                                        allowDeselect=False,
                                    ),
                                    dmc.Select(
                                        id="spc-metric",
                                        label="Metric",
                                        data=[{"label": label, "value": value} for value, label in SPC_METRIC_OPTIONS.items()],
                                        value="service_hours",
                                        allowDeselect=False,
                                    ),
                                    dmc.MultiSelect(
                                        id="spc-compare-processes",
                                        label="Histogram comparison processes",
                                        data=option_data(EMPIRICAL_PROCESS_OPTIONS),
                                        value=[p for p in ["RASPADO", "BAUCE"] if p in EMPIRICAL_PROCESS_OPTIONS],
                                        searchable=True,
                                        clearable=True,
                                    ),
                                ],
                            ),
                            dmc.Group(
                                grow=True,
                                children=[
                                    dmc.NumberInput(id="spc-lsl", label="LSL (hours, optional)", value=None, min=0, step=0.1, decimalScale=4),
                                    dmc.NumberInput(id="spc-usl", label="USL (hours, optional)", value=None, min=0, step=0.1, decimalScale=4),
                                    dmc.Switch(id="spc-exclude-iqr", label="Exclude IQR outliers", checked=True),
                                ],
                            ),
                            dmc.Alert(
                                color="blue",
                                variant="light",
                                title="Capability logic for Cp, Cpk, Pp, Ppk",
                                children=[
                                    dmc.Text("LSL/USL are specification limits from engineering, customer requirements, or the SBD when the metric matches. They are not the same as control-chart limits.", fz="sm"),
                                    dmc.Text("Cp = (USL - LSL) / (6 * within sigma). Cpk = min((USL - mean) / (3 * within sigma), (mean - LSL) / (3 * within sigma)).", fz="sm"),
                                    dmc.Text("Pp/Ppk use the same formulas, but with overall sigma instead of within sigma. If only USL is known, the app reports one-sided Cpk/Ppk and leaves Cp/Pp blank.", fz="sm"),
                                ],
                            ),
                            dmc.Alert(
                                id="spc-notes",
                                color="teal",
                                variant="light",
                                children="Select a process and enter LSL/USL if you want Cp/Cpk/Pp/Ppk.",
                            ),
                            dcc.Graph(id="spc-control-fig", figure=empty_figure("Select process"), className="plot-card"),
                            dcc.Graph(id="spc-hist-fig", figure=empty_figure("Select process"), className="plot-card"),
                            dcc.Graph(id="spc-compare-hist-fig", figure=empty_figure("Select processes"), className="plot-card"),
                            dash_table.DataTable(
                                id="spc-capability-table",
                                columns=[
                                    {"name": "process", "id": "process"},
                                    {"name": "metric", "id": "metric"},
                                    {"name": "n", "id": "n"},
                                    {"name": "mean", "id": "mean"},
                                    {"name": "median", "id": "median"},
                                    {"name": "min", "id": "min"},
                                    {"name": "max", "id": "max"},
                                    {"name": "std_overall", "id": "std_overall"},
                                    {"name": "sigma_within_mr", "id": "sigma_within_mr"},
                                    {"name": "lsl", "id": "lsl"},
                                    {"name": "usl", "id": "usl"},
                                    {"name": "cp", "id": "cp"},
                                    {"name": "cpk", "id": "cpk"},
                                    {"name": "pp", "id": "pp"},
                                    {"name": "ppk", "id": "ppk"},
                                    {"name": "out_of_spec", "id": "out_of_spec"},
                                    {"name": "out_of_spec_pct", "id": "out_of_spec_pct"},
                                    {"name": "i_chart_violations", "id": "i_chart_violations"},
                                    {"name": "mr_chart_violations", "id": "mr_chart_violations"},
                                    {"name": "raw_rows", "id": "raw_rows"},
                                    {"name": "strict_removed", "id": "strict_removed"},
                                    {"name": "iqr_found", "id": "iqr_found"},
                                    {"name": "iqr_removed", "id": "iqr_removed"},
                                ],
                                data=[],
                                page_size=5,
                                sort_action="native",
                                filter_action="native",
                                style_as_list_view=True,
                                style_table={"overflowX": "auto"},
                                style_header={"fontWeight": "700", "backgroundColor": "#f8fafc"},
                                style_cell={"padding": "8px", "fontFamily": "IBM Plex Sans, sans-serif", "fontSize": "13px"},
                            ),
                            dmc.Divider(label="Bayesian Classifier", labelPosition="center"),
                            dmc.Text("Bayesian time classifier", fw=700),
                            dmc.Text(
                                "Choose two processes and classify a service time using Bayes. This is the same logic as the standalone 8054 app, embedded here for the simulator.",
                                c="dimmed",
                                fz="sm",
                            ),
                            dmc.Group(
                                grow=True,
                                children=[
                                    dmc.Select(
                                        id="sim-bayes-class-a",
                                        label="Process A",
                                        data=option_data(bayes_core.PROCESS_OPTIONS),
                                        value="MEDIDO" if "MEDIDO" in bayes_core.PROCESS_OPTIONS else (bayes_core.PROCESS_OPTIONS[0] if bayes_core.PROCESS_OPTIONS else None),
                                        searchable=True,
                                        allowDeselect=False,
                                    ),
                                    dmc.Select(
                                        id="sim-bayes-class-b",
                                        label="Process B",
                                        data=option_data(bayes_core.PROCESS_OPTIONS),
                                        value="TAIC" if "TAIC" in bayes_core.PROCESS_OPTIONS else (bayes_core.PROCESS_OPTIONS[1] if len(bayes_core.PROCESS_OPTIONS) > 1 else None),
                                        searchable=True,
                                        allowDeselect=False,
                                    ),
                                    dmc.DatePickerInput(
                                        id="sim-bayes-date-range",
                                        label="Arrival date range",
                                        type="range",
                                        value=[bayes_core.MIN_DATE, bayes_core.MAX_DATE],
                                        clearable=False,
                                        valueFormat="YYYY-MM-DD",
                                    ),
                                ],
                            ),
                            dmc.Group(
                                grow=True,
                                children=[
                                    dmc.Select(
                                        id="sim-bayes-model-kind",
                                        label="Likelihood model",
                                        data=option_data(["kde", "normal"]),
                                        value="kde",
                                        allowDeselect=False,
                                    ),
                                    dmc.NumberInput(
                                        id="sim-bayes-input-time-h",
                                        label="Time to classify (hours)",
                                        value=3.0,
                                        min=0.01,
                                        max=72,
                                        step=0.1,
                                        decimalScale=4,
                                    ),
                                    dmc.Select(
                                        id="sim-bayes-cap-process",
                                        label="Operational cap process",
                                        data=option_data(["NONE"] + bayes_core.PROCESS_OPTIONS),
                                        value="TAIC" if "TAIC" in bayes_core.PROCESS_OPTIONS else "NONE",
                                        searchable=True,
                                        allowDeselect=False,
                                    ),
                                ],
                            ),
                            dmc.Group(
                                grow=True,
                                children=[
                                    dmc.Switch(id="sim-bayes-exclude-iqr", label="Exclude IQR outliers", checked=True),
                                    dmc.Switch(id="sim-bayes-cap-on", label="Apply operational cap", checked=True),
                                    dmc.NumberInput(
                                        id="sim-bayes-cap-h",
                                        label="Max hours for capped process",
                                        value=4.0,
                                        min=0.1,
                                        max=72,
                                        step=0.25,
                                        decimalScale=3,
                                    ),
                                ],
                            ),
                            dmc.Group(
                                grow=True,
                                children=[
                                    dmc.Switch(id="sim-bayes-empirical-prior", label="Use empirical prior", checked=True),
                                    dmc.NumberInput(
                                        id="sim-bayes-manual-prior-a",
                                        label="Manual prior P(Process A)",
                                        value=0.5,
                                        min=0.01,
                                        max=0.99,
                                        step=0.01,
                                        decimalScale=3,
                                    ),
                                ],
                            ),
                            dmc.Alert(
                                id="sim-bayes-output",
                                color="indigo",
                                variant="light",
                                children="Bayesian classifier output will appear here.",
                            ),
                            dmc.Group(
                                grow=True,
                                align="stretch",
                                children=[
                                    dcc.Graph(id="sim-bayes-density-fig", figure=empty_figure("Waiting for classifier"), className="plot-card"),
                                    dcc.Graph(id="sim-bayes-posterior-fig", figure=empty_figure("Waiting for classifier"), className="plot-card"),
                                ],
                            ),
                            dmc.Text("Bayesian classifier summary", fw=700),
                            dash_table.DataTable(
                                id="sim-bayes-summary-table",
                                columns=[{"name": c, "id": c} for c in ["class", "raw_rows", "strict_removed", "iqr_found", "iqr_removed", "cap_removed", "clean_n", "min_h", "max_h", "mean_h", "median_h"]],
                                data=[],
                                page_size=5,
                                sort_action="native",
                                filter_action="native",
                                style_as_list_view=True,
                                style_table={"overflowX": "auto"},
                                style_header={"fontWeight": "700", "backgroundColor": "#f8fafc"},
                                style_cell={"padding": "8px", "fontFamily": "IBM Plex Sans, sans-serif", "fontSize": "13px"},
                            ),
                            dash_table.DataTable(
                                id="sim-bayes-validation-table",
                                columns=[{"name": c, "id": c} for c in ["metric", "value"]],
                                data=[],
                                page_size=5,
                                sort_action="native",
                                style_as_list_view=True,
                                style_table={"overflowX": "auto"},
                                style_header={"fontWeight": "700", "backgroundColor": "#f8fafc"},
                                style_cell={"padding": "8px", "fontFamily": "IBM Plex Sans, sans-serif", "fontSize": "13px"},
                            ),
                            dash_table.DataTable(
                                id="sim-bayes-confusion-table",
                                columns=[{"name": c, "id": c} for c in ["actual", "predicted", "count"]],
                                data=[],
                                page_size=8,
                                sort_action="native",
                                style_as_list_view=True,
                                style_table={"overflowX": "auto"},
                                style_header={"fontWeight": "700", "backgroundColor": "#f8fafc"},
                                style_cell={"padding": "8px", "fontFamily": "IBM Plex Sans, sans-serif", "fontSize": "13px"},
                            ),
                            dmc.Text("Drying gas detail (LTD, TAIC, AEREO only)", fw=700),
                            dash_table.DataTable(
                                id="gas-table",
                                columns=[
                                    {"name": "process", "id": "process"},
                                    {"name": "temperature_c", "id": "temperature_c"},
                                    {"name": "pieces_total", "id": "pieces_total"},
                                    {"name": "estimated_gas_cost_per_cuero", "id": "gas_cost_per_cuero"},
                                    {"name": "gas_cost_total", "id": "gas_cost_total"},
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
                            dmc.Text("Fixed drying temperatures", fw=700),
                            dash_table.DataTable(
                                id="drying-temp-table",
                                columns=[
                                    {"name": "process", "id": "process"},
                                    {"name": "temperature_c", "id": "temperature_c"},
                                ],
                                data=build_drying_temperature_rows(),
                                page_size=8,
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
    Output("data-source-alert", "children"),
    Output("data-source-alert", "color"),
    Output("upload-status", "children"),
    Output("upload-status", "color"),
    Output("data-source-store", "data"),
    Output("date-range", "value"),
    Output("lot-processes", "data"),
    Output("lot-processes", "value"),
    Output("spc-process", "data"),
    Output("spc-process", "value"),
    Output("spc-compare-processes", "data"),
    Output("spc-compare-processes", "value"),
    Output("sim-bayes-class-a", "data"),
    Output("sim-bayes-class-a", "value"),
    Output("sim-bayes-class-b", "data"),
    Output("sim-bayes-class-b", "value"),
    Output("sim-bayes-cap-process", "data"),
    Output("sim-bayes-cap-process", "value"),
    Output("sim-bayes-date-range", "value"),
    Input("excel-upload", "contents"),
    State("excel-upload", "filename"),
    prevent_initial_call=True,
)
def upload_workbook(contents, filename):
    saved_paths, save_error = save_uploaded_workbooks(contents, filename)
    if not saved_paths:
        return (
            no_update,
            no_update,
            f"Upload failed: {save_error}",
            "red",
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
        )

    ok, message, meta = refresh_active_data_context(saved_paths)
    if not ok:
        return (
            no_update,
            no_update,
            f"Upload saved, but validation failed: {message}",
            "red",
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
        )

    process_options = option_data(PROCESS_OPTIONS)
    empirical_options = option_data(EMPIRICAL_PROCESS_OPTIONS)
    date_range = meta.get("date_range") or current_date_range()
    spc_default = default_process_for_options(EMPIRICAL_PROCESS_OPTIONS)
    compare_default = [p for p in ["RASPADO", "BAUCE"] if p in EMPIRICAL_PROCESS_OPTIONS]
    if not compare_default:
        compare_default = EMPIRICAL_PROCESS_OPTIONS[: min(2, len(EMPIRICAL_PROCESS_OPTIONS))]

    bayes_options = option_data(bayes_core.PROCESS_OPTIONS)
    bayes_a = "MEDIDO" if "MEDIDO" in bayes_core.PROCESS_OPTIONS else (bayes_core.PROCESS_OPTIONS[0] if bayes_core.PROCESS_OPTIONS else None)
    bayes_b = "TAIC" if "TAIC" in bayes_core.PROCESS_OPTIONS else None
    if bayes_b is None or bayes_b == bayes_a:
        bayes_b = next((p for p in bayes_core.PROCESS_OPTIONS if p != bayes_a), bayes_a)
    cap_options = option_data(["NONE"] + bayes_core.PROCESS_OPTIONS)
    cap_value = "TAIC" if "TAIC" in bayes_core.PROCESS_OPTIONS else "NONE"

    source_msg = f"Active uploaded data source: {meta.get('path')}"
    warning_txt = f" Partial upload warnings: {save_error}." if save_error else ""
    validation_notes = meta.get("notes") if isinstance(meta.get("notes"), list) else []
    validation_txt = f" Validation notes: {'; '.join(validation_notes)}." if validation_notes else ""
    status_msg = (
        f"Workbook data active: files={meta.get('files_loaded', len(saved_paths))} | "
        f"rows={meta.get('rows')} | processes={meta.get('processes')} | "
        f"date range={date_range[0]} to {date_range[1]}. "
        "You can now enter lots and simulate with this uploaded dataset."
        f"{warning_txt}{validation_txt}"
    )
    return (
        source_msg,
        "teal",
        status_msg,
        "teal",
        meta,
        date_range,
        process_options,
        DEFAULT_ROUTE,
        empirical_options,
        spc_default,
        empirical_options,
        compare_default,
        bayes_options,
        bayes_a,
        bayes_options,
        bayes_b,
        cap_options,
        cap_value,
        date_range,
    )


@mini_app.callback(
    Output("lot-store", "data"),
    Output("lot-table", "data"),
    Output("lot-name", "value"),
    Output("lot-repeat", "value"),
    Input("add-lot", "n_clicks"),
    Input("clear-lots", "n_clicks"),
    Input("clear-lots-bottom", "n_clicks"),
    Input("sim-toggle-select", "value"),
    State("lot-name", "value"),
    State("lot-pieces", "value"),
    State("lot-processes", "value"),
    State("lot-repeat", "value"),
    State("lot-store", "data"),
    State("sim-history-store", "data"),
    prevent_initial_call=False,
)
def manage_lots(
    add_clicks,
    clear_clicks,
    clear_clicks_bottom,
    selected_run_id,
    lot_name,
    lot_pieces,
    lot_processes,
    lot_repeat,
    lot_store,
    sim_history_store,
):
    _ = (add_clicks, clear_clicks, clear_clicks_bottom)
    store = list(lot_store) if isinstance(lot_store, list) else []
    trigger = dash.ctx.triggered_id

    if trigger in {"clear-lots", "clear-lots-bottom"}:
        return [], [], "Lote_1", 1

    if trigger == "sim-toggle-select":
        history = list(sim_history_store) if isinstance(sim_history_store, list) else []
        item = history_item_by_id(history, selected_run_id)
        if item is None:
            rows = build_lot_rows(store)
            next_name = f"Lote_{len(store)+1}" if store else "Lote_1"
            return store, rows, next_name, 1
        loaded_store = list(item.get("lot_store", [])) if isinstance(item.get("lot_store", []), list) else []
        rows = build_lot_rows(loaded_store)
        next_name = f"Lote_{len(loaded_store)+1}" if loaded_store else "Lote_1"
        return loaded_store, rows, next_name, 1

    if trigger == "add-lot":
        route = normalize_sim_route(lot_processes or [])
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
    Output("gas-output", "children"),
    Output("business-output", "children"),
    Output("cost-table", "data"),
    Output("gas-table", "data"),
    Output("cost-fig", "figure"),
    Output("gantt-fig", "figure"),
    Output("sim-history-store", "data"),
    Input("clear-sim-history", "n_clicks"),
    Input("clear-lots", "n_clicks"),
    Input("clear-lots-bottom", "n_clicks"),
    Input("run-sim", "n_clicks"),
    Input("run-gas", "n_clicks"),
    Input("sim-toggle-select", "value"),
    State("lot-store", "data"),
    State("date-range", "value"),
    State("energy-cost", "value"),
    State("labor-cost", "value"),
    State("gantt-start-hour", "value"),
    State("gantt-start-minute", "value"),
    State("gas-cost-per-cuero-50c", "value"),
    State("recurtido-hours", "value"),
    State("sim-history-store", "data"),
)
def run_simulation(
    clear_sim_history_clicks,
    clear_clicks_top,
    clear_clicks_bottom,
    n_clicks,
    gas_clicks,
    selected_run_id,
    lot_store,
    date_range,
    energy_cost,
    labor_cost,
    gantt_start_hour,
    gantt_start_minute,
    gas_cost_per_cuero_50c,
    recurtido_hours,
    sim_history_store,
):
    _ = (clear_sim_history_clicks, clear_clicks_top, clear_clicks_bottom)
    history = list(sim_history_store) if isinstance(sim_history_store, list) else []
    trigger = dash.ctx.triggered_id

    if trigger == "clear-sim-history":
        return no_update, no_update, no_update, no_update, no_update, no_update, no_update, []

    if trigger in {"clear-lots", "clear-lots-bottom"}:
        return (
            "Lots cleared. Add lots and click Simulate.",
            "Click Estimate gas (drying only) to include gas KPIs.",
            build_business_summary_children("Lots cleared. Add a lot mix to generate the business decision summary."),
            [],
            [],
            empty_figure("Waiting for simulation"),
            empty_figure("Waiting for simulation"),
            history,
        )

    if trigger == "sim-toggle-select":
        item = history_item_by_id(history, selected_run_id)
        if item is None and history:
            item = history[0]
        if item is None:
            return (
                "Add lots and click Simulate.",
                "Click Estimate gas (drying only) to include gas KPIs.",
                build_business_summary_children("Run a simulation to generate the business decision summary."),
                [],
                [],
                empty_figure("Waiting for simulation"),
                empty_figure("Waiting for simulation"),
                history,
            )
        fig_payload = item.get("fig")
        gantt_payload = item.get("gantt_fig")
        try:
            fig = go.Figure(fig_payload) if isinstance(fig_payload, dict) else empty_figure("Saved simulation has no chart")
        except Exception:
            fig = empty_figure("Saved simulation has invalid chart payload")
        try:
            gantt_fig = go.Figure(gantt_payload) if isinstance(gantt_payload, dict) else empty_figure("Saved simulation has no Gantt")
        except Exception:
            gantt_fig = empty_figure("Saved simulation has invalid Gantt payload")
        business_metrics = item.get("business_metrics", {}) if isinstance(item.get("business_metrics", {}), dict) else {}
        business_children = (
            build_business_summary_children(
                lots=business_metrics.get("lots"),
                pieces=business_metrics.get("pieces"),
                total_cost=business_metrics.get("total_cost"),
                cost_per_piece=business_metrics.get("cost_per_piece"),
                lead_mean_h=business_metrics.get("lead_mean_h"),
                top_process=business_metrics.get("top_process"),
                top_process_cost=business_metrics.get("top_process_cost"),
                queue_wait_max_min=business_metrics.get("queue_wait_max_min"),
            )
            if business_metrics
            else build_business_summary_children("Saved simulation loaded. Review the charts and tables for the business decision summary.")
        )
        return (
            str(item.get("summary", "Saved simulation")),
            str(item.get("gas_summary", "No gas summary in this saved simulation.")),
            business_children,
            ensure_total_cost_row(list(item.get("cost_rows", [])) if isinstance(item.get("cost_rows", []), list) else []),
            list(item.get("gas_rows", [])) if isinstance(item.get("gas_rows", []), list) else [],
            fig,
            gantt_fig,
            history,
        )

    if not n_clicks and not gas_clicks:
        return (
            "Add lots and click Simulate.",
            "Click Estimate gas (drying only) to include gas KPIs.",
            build_business_summary_children("Run a simulation to generate the business decision summary."),
            [],
            [],
            empty_figure("Waiting for simulation"),
            empty_figure("Waiting for simulation"),
            history,
        )
    if core.DATAFRAME.empty:
        return "No data loaded.", "No data loaded.", build_business_summary_children("No data loaded."), [], [], empty_figure("No data"), empty_figure("No data"), history

    lots_raw = list(lot_store) if isinstance(lot_store, list) else []
    if not lots_raw:
        return "No lots configured.", "No lots configured.", build_business_summary_children("No lots configured yet."), [], [], empty_figure("No lots configured"), empty_figure("No lots configured"), history

    lots = []
    for i, lot in enumerate(lots_raw, start=1):
        route = normalize_sim_route(lot.get("route", []))
        pieces = pd.to_numeric(pd.Series([lot.get("pieces")]), errors="coerce").iloc[0]
        if route and pd.notna(pieces) and float(pieces) > 0:
            lots.append({"lot_name": str(lot.get("lot_name", f"Lote_{i}")), "pieces": float(pieces), "route": list(route)})

    if not lots:
        return "Configured lots are invalid.", "Configured lots are invalid.", build_business_summary_children("Configured lots are invalid."), [], [], empty_figure("Invalid lots"), empty_figure("Invalid lots"), history

    base = core.DATAFRAME.copy()
    if date_range and len(date_range) == 2 and date_range[0] and date_range[1]:
        start = pd.to_datetime(date_range[0], errors="coerce")
        end = pd.to_datetime(date_range[1], errors="coerce")
        if pd.notna(start) and pd.notna(end):
            base = base[(base["arrival_time"] >= start) & (base["arrival_time"] <= end + pd.Timedelta(days=1))].copy()

    if base.empty:
        return "No rows in selected date range.", "No rows in selected date range.", build_business_summary_children("No historical rows are available for the selected date range."), [], [], empty_figure("No rows in scope"), empty_figure("No rows in scope"), history

    # Fresh RNG each simulation run so between-process gaps are truly random
    # (Uniform 20-30 min), not the same sequence every time.
    rng = np.random.default_rng()
    ordered_processes = []
    for lot in lots:
        for proc in lot["route"]:
            if proc not in ordered_processes:
                ordered_processes.append(proc)
    ordered_empirical_processes = [p for p in ordered_processes if p != RECURTIDO_PROCESS]

    stage_catalog, missing_processes = core.build_stage_catalog_for_processes(
        base_df=base,
        process_list=ordered_empirical_processes,
        strict_cleaning=True,
        queue_use_downtime=True,
        rng=rng,
        rate_iqr_filter=False,
        rate_iqr_processes=SECADO_RATE_IQR_FILTER_PROCESSES,
        rate_tail_guardrail=False,
        rate_tail_guardrail_processes=SECADO_RATE_IQR_FILTER_PROCESSES,
        no_piece_scaling_processes=SECADO_NO_PIECE_SCALING_PROCESSES,
        service_hour_caps=SECADO_SERVICE_HOUR_CAPS,
        service_range_guardrail=True,
    )
    if RECURTIDO_PROCESS in ordered_processes:
        stage_catalog[RECURTIDO_PROCESS] = build_recurtido_stage_spec(recurtido_hours)
    secado_iqr_notes: list[str] = []
    secado_rate_filter_keys = {core.normalize_process_key(x) for x in SECADO_RATE_IQR_FILTER_PROCESSES}
    secado_cap_notes: list[str] = []
    for proc_name, spec in stage_catalog.items():
        proc_key = core.normalize_process_key(proc_name)
        if proc_key not in secado_rate_filter_keys:
            continue
        cap_h = spec.get("service_cap_h")
        before_cap = int(spec.get("service_rows_before_cap", 0) or 0)
        after_cap = int(spec.get("service_rows_after_cap", 0) or 0)
        if cap_h is not None and before_cap > 0:
            mean_after = spec.get("mean_service_h_empirical")
            expected = SECADO_EXPECTED_MEAN_H.get(proc_key)
            expected_txt = f", expected mean {expected}h" if expected else ""
            secado_cap_notes.append(
                f"{proc_name}: cap<={core.safe_number(float(cap_h), 2)}h, rows {before_cap}->{after_cap}, mean={core.safe_number(mean_after, 2)}h{expected_txt}"
            )
        before_n = int(spec.get("rate_rows_before", 0) or 0)
        after_n = int(spec.get("rate_rows_after", 0) or 0)
        if before_n > 0:
            removed_n = max(0, before_n - after_n)
            secado_iqr_notes.append(f"{proc_name}:{before_n}->{after_n} (removed {removed_n})")

    filtered_lots = []
    for lot in lots:
        route = [p for p in lot["route"] if p in stage_catalog]
        if route:
            filtered_lots.append({"lot_name": lot["lot_name"], "pieces": lot["pieces"], "route": route})

    if not filtered_lots:
        return "No usable stages for selected lots.", "No usable stages for selected lots.", build_business_summary_children("No usable process stages were found for the selected lots."), [], [], empty_figure("No usable stage data"), empty_figure("No usable stage data"), history

    # User-planned lots are released together so different lots can work at
    # the same time across available machines/processes.
    ia_samples = np.full(len(filtered_lots), 1e-6, dtype=float)

    # User rule: add 20-30 min random wait between process steps in the same lot.
    between_steps_gap_sampler = lambda: float(rng.uniform(low=(20.0 / 60.0), high=(30.0 / 60.0)))
    sim = core.simulate_lot_plan_flow(
        stage_catalog=stage_catalog,
        lot_plan=filtered_lots,
        interarrival_h=ia_samples,
        between_steps_gap_sampler=between_steps_gap_sampler,
        use_resource_queue=True,
    )
    stage_events = pd.DataFrame(sim.get("stage_rows", []))
    lot_events = pd.DataFrame(sim.get("lot_rows", []))
    arrival_h_vals = np.asarray(sim.get("arrival_h", np.array([], dtype=float)), dtype=float)
    lot_arrival_h_map: dict[str, float] = {}
    interarrival_h_map: dict[str, float] = {}
    prev_arrival_h = np.nan
    for i, lot in enumerate(filtered_lots):
        lot_name = str(lot.get("lot_name", f"lot_{i+1}"))
        arr_h = float(arrival_h_vals[i]) if i < arrival_h_vals.size and np.isfinite(arrival_h_vals[i]) else np.nan
        lot_arrival_h_map[lot_name] = arr_h
        if i == 0 or not np.isfinite(prev_arrival_h) or not np.isfinite(arr_h):
            interarrival_h_map[lot_name] = np.nan
        else:
            interarrival_h_map[lot_name] = float(arr_h - prev_arrival_h)
        if np.isfinite(arr_h):
            prev_arrival_h = arr_h

    if not stage_events.empty:
        lot_key = stage_events["lot_name"].astype(str)
        stage_events["lot_arrival_h"] = lot_key.map(lot_arrival_h_map)
        stage_events["interarrival_h"] = lot_key.map(interarrival_h_map)

    if stage_events.empty or lot_events.empty:
        return "Simulation produced no events.", "Simulation produced no events.", build_business_summary_children("Simulation produced no events."), [], [], empty_figure("No simulation events"), empty_figure("No simulation events"), history

    energy_cost = float(energy_cost) if energy_cost is not None else 0.12
    labor_cost = float(labor_cost) if labor_cost is not None else 60.0
    gas_cost_per_cuero_50c = float(gas_cost_per_cuero_50c) if gas_cost_per_cuero_50c is not None else 0.12
    gas_cost_per_cuero_50c = max(0.0, gas_cost_per_cuero_50c)

    rows = []
    gas_rows = []
    total_energy_cost = 0.0
    total_labor_cost = 0.0
    total_gas_cost = 0.0
    fallback_energy_processes: list[str] = []

    for proc, g in stage_events.groupby("process"):
        srv = pd.to_numeric(g["service_h"], errors="coerce").dropna().to_numpy(dtype=float)
        dt = pd.to_numeric(g["downtime_h"], errors="coerce").dropna().to_numpy(dtype=float)
        pieces_series = pd.to_numeric(g["pieces"], errors="coerce").fillna(0.0)
        pieces_total = float(pieces_series.sum())
        service_total_h = float(np.nansum(srv)) if srv.size > 0 else 0.0
        downtime_total_h = float(np.nansum(dt)) if dt.size > 0 else 0.0
        machine_hours = service_total_h + downtime_total_h
        proc_kwh, proc_kwh_source = get_process_kwh(str(proc))
        if proc_kwh_source == "fallback_default":
            fallback_energy_processes.append(str(proc))

        temp_c = get_drying_temperature_c(str(proc))
        if temp_c is not None:
            temp_factor = float(max(0.0, temp_c) / max(1e-6, GAS_REFERENCE_TEMP_C))
            gas_cost_per_cuero = float(gas_cost_per_cuero_50c * temp_factor)
            gas_proc_cost = float(max(0.0, pieces_total) * gas_cost_per_cuero)
            total_gas_cost += gas_proc_cost
            gas_rows.append(
                {
                    "process": str(proc),
                    "temperature_c": round(float(temp_c), 3),
                    "pieces_total": round(pieces_total, 2),
                    "gas_cost_per_cuero": round(gas_cost_per_cuero, 6) if np.isfinite(gas_cost_per_cuero) else None,
                    "gas_cost_total": round(gas_proc_cost, 2),
                }
            )
        else:
            gas_cost_per_cuero = np.nan
            gas_proc_cost = 0.0

        e_cost = machine_hours * proc_kwh * energy_cost
        l_cost = service_total_h * labor_cost
        t_cost = e_cost + l_cost + gas_proc_cost

        total_energy_cost += e_cost
        total_labor_cost += l_cost
        spec = stage_catalog.get(str(proc), {})
        clean_service_min_h = spec.get("service_min_h")
        clean_service_max_h = spec.get("service_max_h")

        rows.append(
            {
                "process": str(proc),
                "kwh_per_machine_hour": round(proc_kwh, 4),
                "kwh_source": proc_kwh_source,
                "visits": int(len(g)),
                "pieces_total": round(pieces_total, 2),
                "clean_service_min_h": round(float(clean_service_min_h), 4) if clean_service_min_h is not None else None,
                "clean_service_max_h": round(float(clean_service_max_h), 4) if clean_service_max_h is not None else None,
                "sim_service_min_h": round(float(np.nanmin(srv)), 4) if srv.size > 0 else None,
                "sim_service_max_h": round(float(np.nanmax(srv)), 4) if srv.size > 0 else None,
                "service_total_h": round(service_total_h, 4),
                "downtime_total_h": round(downtime_total_h, 4),
                "gas_temp_c": round(float(temp_c), 3) if temp_c is not None else None,
                "gas_cost_per_cuero": round(gas_cost_per_cuero, 6) if np.isfinite(gas_cost_per_cuero) else None,
                "energy_cost": round(e_cost, 2),
                "labor_cost": round(l_cost, 2),
                "gas_cost": round(gas_proc_cost, 2),
                "total_cost": round(t_cost, 2),
            }
        )

    rows = sorted(rows, key=lambda x: x["total_cost"], reverse=True)
    gas_rows = sorted(gas_rows, key=lambda x: x["gas_cost_total"], reverse=True)
    total_cost = total_energy_cost + total_labor_cost + total_gas_cost
    total_pieces = float(pd.to_numeric(lot_events.get("pieces", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
    gas_cost_per_cuero_total = (total_gas_cost / total_pieces) if total_pieces > 0 else np.nan
    cost_per_piece = (total_cost / total_pieces) if total_pieces > 0 else np.nan

    total_row = {
        "process": "TOTAL",
        "kwh_per_machine_hour": None,
        "kwh_source": "summary",
        "visits": int(sum(float(r.get("visits", 0) or 0) for r in rows)),
        "pieces_total": round(float(sum(float(r.get("pieces_total", 0) or 0.0) for r in rows)), 2),
        "clean_service_min_h": None,
        "clean_service_max_h": None,
        "sim_service_min_h": None,
        "sim_service_max_h": None,
        "service_total_h": round(float(sum(float(r.get("service_total_h", 0) or 0.0) for r in rows)), 4),
        "downtime_total_h": round(float(sum(float(r.get("downtime_total_h", 0) or 0.0) for r in rows)), 4),
        "gas_temp_c": None,
        "gas_cost_per_cuero": round(gas_cost_per_cuero_total, 6) if np.isfinite(gas_cost_per_cuero_total) else None,
        "energy_cost": round(float(total_energy_cost), 2),
        "labor_cost": round(float(total_labor_cost), 2),
        "gas_cost": round(float(total_gas_cost), 2),
        "total_cost": round(float(total_cost), 2),
    }
    rows_with_total = list(rows) + [total_row]

    fig = go.Figure()
    fig.add_bar(x=[r["process"] for r in rows], y=[r["energy_cost"] for r in rows], name="Energy cost")
    fig.add_bar(x=[r["process"] for r in rows], y=[r["labor_cost"] for r in rows], name="Labor cost")
    fig.add_bar(x=[r["process"] for r in rows], y=[r["gas_cost"] for r in rows], name="Gas cost")
    fig.update_layout(barmode="stack", yaxis={"title": "Cost"}, margin={"l": 20, "r": 20, "t": 50, "b": 20}, height=360)
    gantt_fig = build_gantt_figure(
        stage_events=stage_events,
        date_range=date_range,
        start_hour=gantt_start_hour,
        start_minute=gantt_start_minute,
    )

    step_gap_min_obs = pd.to_numeric(stage_events.get("between_steps_gap_h", pd.Series(dtype=float)), errors="coerce").dropna()
    step_gap_min_obs = step_gap_min_obs[step_gap_min_obs > 0] * 60.0
    step_gap_sample_min = float(step_gap_min_obs.min()) if not step_gap_min_obs.empty else np.nan
    step_gap_sample_max = float(step_gap_min_obs.max()) if not step_gap_min_obs.empty else np.nan
    wait_min_obs = pd.to_numeric(stage_events.get("wait_h", pd.Series(dtype=float)), errors="coerce").dropna() * 60.0
    wait_mean_min = float(wait_min_obs.mean()) if not wait_min_obs.empty else np.nan
    wait_max_min = float(wait_min_obs.max()) if not wait_min_obs.empty else np.nan
    lead_mean = float(pd.to_numeric(lot_events.get("system_time_h", pd.Series(dtype=float)), errors="coerce").dropna().mean())
    most_expensive = rows[0] if rows else None
    cheapest = rows[-1] if rows else None
    capacity_txt = "; ".join(
        f"{proc}={int(spec.get('servers', 1))}"
        for proc, spec in stage_catalog.items()
    )

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
        f"gas=${core.safe_number(total_gas_cost, 2)}, total=${core.safe_number(total_cost, 2)}, "
        f"between_process_wait=Uniform(20,30) min (sampled {core.safe_number(step_gap_sample_min, 2)}-{core.safe_number(step_gap_sample_max, 2)} min), "
        f"queue_wait_mean={core.safe_number(wait_mean_min, 2)} min, queue_wait_max={core.safe_number(wait_max_min, 2)} min, "
        f"cost_per_piece=${core.safe_number(cost_per_piece, 4)}. "
        f"service_times_bounded_by_clean_observed_min_max."
        f" secado_mode=empirical_direct_service_with_operational_caps(no_piece_scaling)."
        f" flow_timeline=parallel_machine_flow_simpy; lot_release=parallel_at_t0; machines={capacity_txt}."
        + compare_txt
        + missing_txt
        + fallback_txt
    )
    service_bound_notes = []
    for proc_name, spec in stage_catalog.items():
        if not bool(spec.get("service_range_guardrail_on", False)):
            continue
        lo_h = spec.get("service_min_h")
        hi_h = spec.get("service_max_h")
        if lo_h is not None and hi_h is not None:
            service_bound_notes.append(
                f"{proc_name} [{core.safe_number(float(lo_h), 2)}-{core.safe_number(float(hi_h), 2)}h]"
            )
    if service_bound_notes:
        summary = summary + " Clean observed service bounds: " + "; ".join(service_bound_notes) + "."
    if secado_cap_notes:
        summary = summary + " Secado operational caps: " + "; ".join(secado_cap_notes) + "."
    if secado_iqr_notes:
        summary = summary + " Secado pools (after base preprocessing): " + "; ".join(secado_iqr_notes) + "."
    secado_mode_notes: list[str] = []
    for proc_name, spec in stage_catalog.items():
        proc_key = core.normalize_process_key(proc_name)
        if proc_key not in secado_rate_filter_keys:
            continue
        if bool(spec.get("no_piece_scaling", False)):
            secado_mode_notes.append(f"{proc_name}: direct empirical service bootstrap")
    if secado_mode_notes:
        summary = summary + " Secado sampling: " + "; ".join(secado_mode_notes) + "."
    if gas_rows:
        gas_expensive = gas_rows[0]
        gas_cheapest = gas_rows[-1]
        gas_summary = (
            f"Drying gas estimate | total_gas_cost=${core.safe_number(total_gas_cost, 2)} "
            f"| estimated_gas_cost_per_cuero=${core.safe_number(gas_cost_per_cuero_total, 6)} "
            f"| base_gas_cost_50C=${core.safe_number(gas_cost_per_cuero_50c, 6)}/cuero "
            f"| highest gas cost={gas_expensive['process']} (${core.safe_number(gas_expensive['gas_cost_total'], 2)}), "
            f"lowest={gas_cheapest['process']} (${core.safe_number(gas_cheapest['gas_cost_total'], 2)})."
        )
    else:
        gas_summary = "No drying stages (LTD, TAIC, AEREO) were found in this simulation route."

    business_metrics = {
        "lots": int(sim.get("n", 0) or 0),
        "pieces": float(total_pieces),
        "total_cost": float(total_cost),
        "cost_per_piece": float(cost_per_piece) if np.isfinite(cost_per_piece) else None,
        "lead_mean_h": float(lead_mean) if np.isfinite(lead_mean) else None,
        "top_process": most_expensive["process"] if most_expensive else None,
        "top_process_cost": float(most_expensive["total_cost"]) if most_expensive else None,
        "queue_wait_max_min": float(wait_max_min) if np.isfinite(wait_max_min) else None,
    }
    business_children = build_business_summary_children(
        lots=business_metrics["lots"],
        pieces=business_metrics["pieces"],
        total_cost=business_metrics["total_cost"],
        cost_per_piece=business_metrics["cost_per_piece"],
        lead_mean_h=business_metrics["lead_mean_h"],
        top_process=business_metrics["top_process"],
        top_process_cost=business_metrics["top_process_cost"],
        queue_wait_max_min=business_metrics["queue_wait_max_min"],
    )

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    run_id = datetime.now().strftime("sim_%Y%m%d_%H%M%S_%f")
    label = (
        f"{ts} | lots={int(sim.get('n', 0))} | "
        f"pieces={core.safe_number(total_pieces, 0)} | total=${core.safe_number(total_cost, 2)}"
    )
    current_payload = {
        "run_id": run_id,
        "label": label,
        "timestamp": ts,
        "summary": summary,
        "gas_summary": gas_summary,
        "business_metrics": business_metrics,
        "lot_store": lots_raw,
        "cost_rows": rows_with_total,
        "gas_rows": gas_rows,
        "fig": fig.to_dict(),
        "gantt_fig": gantt_fig.to_dict(),
    }

    history.insert(0, current_payload)
    history = history[:50]
    return summary, gas_summary, business_children, rows_with_total, gas_rows, fig, gantt_fig, history


@mini_app.callback(
    Output("spc-notes", "children"),
    Output("spc-control-fig", "figure"),
    Output("spc-hist-fig", "figure"),
    Output("spc-compare-hist-fig", "figure"),
    Output("spc-capability-table", "data"),
    Input("spc-process", "value"),
    Input("spc-metric", "value"),
    Input("spc-lsl", "value"),
    Input("spc-usl", "value"),
    Input("spc-exclude-iqr", "checked"),
    Input("spc-compare-processes", "value"),
    Input("date-range", "value"),
)
def update_spc_capability(
    process_value,
    metric_col,
    lsl,
    usl,
    exclude_iqr,
    compare_processes,
    date_range,
):
    proc = str(process_value or "").strip().upper()
    metric_col = metric_col if metric_col in SPC_METRIC_OPTIONS else "service_hours"
    if not proc:
        msg = "Select a process to build control charts and capability."
        return msg, empty_figure(msg), empty_figure(msg), empty_figure(msg), []

    df, counts = clean_process_metric_df(
        process_value=proc,
        metric_col=metric_col,
        date_range=date_range,
        strict_cleaning=True,
        exclude_iqr=bool(exclude_iqr),
    )
    compare_fig = build_process_comparison_histogram(compare_processes, metric_col, date_range)
    if df.empty:
        msg = f"No clean {SPC_METRIC_OPTIONS.get(metric_col, metric_col).lower()} observations for {proc} in the selected date range."
        return msg, empty_figure(msg), empty_figure(msg), compare_fig, []

    cap = capability_metrics(df[metric_col], lsl=lsl, usl=usl)
    control_fig, i_viol, mr_viol = build_control_chart(df, metric_col, cap)
    hist_fig = build_capability_histogram(df, metric_col, cap)
    table_rows = capability_table_rows(proc, metric_col, cap, counts, i_viol, mr_viol)

    lsl_v = _finite_float(lsl)
    usl_v = _finite_float(usl)
    spec_note = (
        f"Specs: LSL={core.safe_number(lsl_v, 4)}, USL={core.safe_number(usl_v, 4)}."
        if np.isfinite(lsl_v) or np.isfinite(usl_v)
        else "No LSL/USL entered yet, so Cp/Cpk/Pp/Ppk are left blank; charts still show process behavior."
    )
    cap_note = ""
    if table_rows:
        row = table_rows[0]
        cap_note = (
            f" Cp={row.get('cp')}, Cpk={row.get('cpk')}, "
            f"Pp={row.get('pp')}, Ppk={row.get('ppk')}, out_of_spec={row.get('out_of_spec')}."
        )
    sample_note = " Small sample: interpret capability cautiously." if int(cap.get("n", 0) or 0) < 25 else ""
    notes = (
        f"{proc} SPC using {SPC_METRIC_OPTIONS.get(metric_col, metric_col).lower()}: "
        f"n={int(cap.get('n', 0) or 0)}, mean={core.safe_number(cap.get('mean'), 4)} h, "
        f"range={core.safe_number(cap.get('min'), 4)}-{core.safe_number(cap.get('max'), 4)} h. "
        f"IQR outliers found={counts.get('outliers_found', 0)} "
        f"{'and removed' if exclude_iqr else '(included)'}. "
        f"I-chart violations={i_viol}, MR violations={mr_viol}. "
        f"{spec_note}{cap_note}{sample_note}"
    )
    return notes, control_fig, hist_fig, compare_fig, table_rows


@mini_app.callback(
    Output("sim-bayes-output", "children"),
    Output("sim-bayes-density-fig", "figure"),
    Output("sim-bayes-posterior-fig", "figure"),
    Output("sim-bayes-summary-table", "data"),
    Output("sim-bayes-validation-table", "data"),
    Output("sim-bayes-confusion-table", "data"),
    Input("sim-bayes-class-a", "value"),
    Input("sim-bayes-class-b", "value"),
    Input("sim-bayes-date-range", "value"),
    Input("sim-bayes-model-kind", "value"),
    Input("sim-bayes-input-time-h", "value"),
    Input("sim-bayes-exclude-iqr", "checked"),
    Input("sim-bayes-cap-on", "checked"),
    Input("sim-bayes-cap-process", "value"),
    Input("sim-bayes-cap-h", "value"),
    Input("sim-bayes-empirical-prior", "checked"),
    Input("sim-bayes-manual-prior-a", "value"),
)
def update_embedded_bayes_classifier(
    class_a,
    class_b,
    date_range,
    model_kind,
    input_time_h,
    exclude_iqr,
    cap_on,
    cap_process,
    cap_h,
    empirical_prior,
    manual_prior_a,
):
    classes = bayes_core.normalize_pair(class_a, class_b)
    model_kind = model_kind if model_kind in {"kde", "normal"} else "kde"
    selected_time = max(0.001, bayes_core._num(input_time_h, 3.0))
    data, summary_rows, _counts = bayes_core.clean_bayes_data(
        classes=classes,
        date_range=date_range,
        exclude_iqr=bool(exclude_iqr),
        cap_on=bool(cap_on),
        cap_process=cap_process,
        cap_h=bayes_core._num(cap_h, 4.0),
    )

    if data.empty or data["class"].nunique() < 2:
        msg = f"Need clean observations for both {classes[0]} and {classes[1]}. Try disabling IQR exclusion or operational cap."
        return msg, empty_figure(msg), empty_figure(msg), summary_rows, [], []

    priors = bayes_core.class_priors(data, classes, bool(empirical_prior), bayes_core._num(manual_prior_a, 0.5))
    pred, post = bayes_core.predict_class(selected_time, data, classes, model_kind, priors)
    density_fig = bayes_core.build_density_figure(data, classes, model_kind, selected_time)
    posterior_fig = bayes_core.build_posterior_figure(data, classes, model_kind, priors, selected_time)
    cm_rows, metric_rows = bayes_core.loocv_rows(data, classes, model_kind, bool(empirical_prior), bayes_core._num(manual_prior_a, 0.5))

    class_counts = data["class"].value_counts().to_dict()
    model_txt = "empirical KDE" if model_kind == "kde" else "normal likelihood"
    cap_key = str(cap_process or "NONE").strip().upper()
    cap_txt = f"cap {cap_key} <= {bayes_core._num(cap_h, 4.0):.2f}h ON" if cap_on and cap_key in classes else "cap OFF/not applied to selected pair"
    output = (
        f"Pair={classes[0]} vs {classes[1]}. Input time={selected_time:.3f} h ({selected_time*60:.1f} min). Prediction={pred}. "
        f"Posterior: P({classes[0]}|time)={post[classes[0]]:.3f}, P({classes[1]}|time)={post[classes[1]]:.3f}. "
        f"Model={model_txt}; priors: {classes[0]}={priors[classes[0]]:.3f}, {classes[1]}={priors[classes[1]]:.3f}; {cap_txt}. "
        f"Clean n: {classes[0]}={class_counts.get(classes[0], 0)}, {classes[1]}={class_counts.get(classes[1], 0)}. "
        f"This is a 1-feature Bayesian classifier, so use it as decision support, not as a final process label without context."
    )
    return output, density_fig, posterior_fig, summary_rows, metric_rows, cm_rows


@mini_app.callback(
    Output("sim-toggle-select", "data"),
    Output("sim-toggle-select", "value"),
    Input("sim-history-store", "data"),
    State("sim-toggle-select", "value"),
    prevent_initial_call=False,
)
def sync_sim_selector(
    history_store,
    selected_run_id,
):
    history = list(history_store) if isinstance(history_store, list) else []
    options = build_history_options(history)
    selected = selected_run_id if history_item_by_id(history, selected_run_id) else (options[0]["value"] if options else None)
    return options, selected


if __name__ == "__main__":
    port = int(os.getenv("SIM_APP_PORT", "8051"))
    host = os.getenv("SIM_APP_HOST", "127.0.0.1")
    mini_app.run(debug=False, host=host, port=port)
