#!/usr/bin/env python3
from __future__ import annotations

import os
import re
import unicodedata
from glob import glob
from typing import Any

import dash
import dash_mantine_components as dmc
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, State, dash_table, dcc

try:
    from scipy import stats

    SCIPY_AVAILABLE = True
except Exception:
    stats = None
    SCIPY_AVAILABLE = False

try:
    import simpy

    SIMPY_AVAILABLE = True
except Exception:
    simpy = None
    SIMPY_AVAILABLE = False

APP_TITLE = "Plant Process EDA Studio"


def resolve_default_data_path() -> str:
    env_path = os.getenv("RASPADO_XLSX_PATH", "").strip()
    if env_path:
        return env_path

    # The file name may contain accented characters; use wildcard discovery.
    wildcard_candidates = sorted(glob("/Users/jeslgdo/Downloads/BIT*CORA*TIEMPOS*.xlsx"))
    if wildcard_candidates:
        return wildcard_candidates[0]

    static_candidates = [
        "/Users/jeslgdo/Downloads/datos_raspado.xlsx",
    ]
    for c in static_candidates:
        if os.path.exists(c):
            return c

    # Last-resort fallback to preserve previous behavior.
    return "/Users/jeslgdo/Downloads/datos_raspado.xlsx"


DATA_PATH = resolve_default_data_path()
SHEET_NAME = os.getenv("RASPADO_SHEET", "TIEMPOS")
ENERGY_REF_PATH = os.getenv("ENERGY_REF_XLSX_PATH", "/Users/jeslgdo/Downloads/datos_energia (2).xlsx")
ENERGY_REF_SHEET = os.getenv("ENERGY_REF_SHEET", "Hoja1")

PALETTE = {
    "ink": "#0f172a",
    "muted": "#475569",
    "accent": "#0f766e",
    "accent_soft": "#99f6e4",
    "warm": "#b45309",
    "danger": "#b91c1c",
}

OUTLIER_CLASS_ORDER = ["normal", "mild", "extreme"]
OUTLIER_COLOR_MAP = {
    "normal": "#0f766e",
    "mild": "#d97706",
    "extreme": "#b91c1c",
}

# Process machine catalog overrides (manual business rules).
PROCESS_MACHINE_CATALOG: dict[str, set[str]] = {
    "RASPADO": {"2", "3", "4", "5"},  # Boss-confirmed: 4 machines only
}

# Optional manual c overrides per process. If absent, c is inferred from process machine count.
PROCESS_SERVER_COUNT_OVERRIDES: dict[str, int] = {
    "RASPADO": 4,
}

# Hard business rules requested by user.
FIXED_OUTLIER_METHOD = "iqr"
FIXED_OUTLIER_VIEW = "exclude"
FIXED_TIME_UNIT = "hours"
FALLBACK_SERVERS_IF_UNKNOWN = 1
DEFAULT_ENERGY_KWH_PER_MACHINE_HOUR = 4.0
DEFAULT_LABOR_COST_PER_HOUR = 60.0
WORKING_HOURS_BY_WEEKDAY = {
    0: 10.0,  # Monday
    1: 10.0,  # Tuesday
    2: 10.0,  # Wednesday
    3: 10.0,  # Thursday
    4: 10.0,  # Friday
    5: 3.0,   # Saturday
    6: 0.0,   # Sunday
}
WEEKLY_SCHEDULE_HOURS = float(sum(WORKING_HOURS_BY_WEEKDAY.values()))

# Parametric families for AIC model comparison.
DIST_FAMILY_SPECS = [
    {"key": "weibull_min", "name": "Weibull", "k_params": 2, "force_loc0": True},
    {"key": "gamma", "name": "Gamma", "k_params": 2, "force_loc0": True},
    {"key": "lognorm", "name": "Lognormal", "k_params": 2, "force_loc0": True},
    {"key": "invgauss", "name": "Inverse Gaussian", "k_params": 2, "force_loc0": True},
    {"key": "expon", "name": "Exponential", "k_params": 1, "force_loc0": True},
    {"key": "norm", "name": "Normal", "k_params": 2, "force_loc0": False},
]

GRAPH_CONFIG = {
    "displaylogo": False,
    "displayModeBar": True,
    "scrollZoom": True,
    "doubleClick": "reset",
    "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"],
    "toImageButtonOptions": {"format": "png", "filename": "raspado_eda_chart", "height": 900, "width": 1400, "scale": 2},
}


def to_numeric_series(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce")


def scheduled_hours_between(start_ts: Any, end_ts: Any) -> float:
    start = pd.to_datetime(start_ts, errors="coerce")
    end = pd.to_datetime(end_ts, errors="coerce")
    if pd.isna(start) or pd.isna(end):
        return float("nan")
    start_day = start.normalize()
    end_day = end.normalize()
    if end_day < start_day:
        return 0.0

    days = pd.date_range(start=start_day, end=end_day, freq="D")
    return float(sum(WORKING_HOURS_BY_WEEKDAY.get(int(day.weekday()), 0.0) for day in days))


def class_rank(value: str) -> int:
    mapping = {"normal": 0, "mild": 1, "extreme": 2}
    return mapping.get(value, 0)


def rank_to_class(rank: int) -> str:
    mapping = {0: "normal", 1: "mild", 2: "extreme"}
    return mapping.get(rank, "normal")


def classify_outliers(series: pd.Series, method: str) -> tuple[pd.Series, pd.Series]:
    s = pd.to_numeric(series, errors="coerce")
    out_class = pd.Series("normal", index=s.index, dtype="object")
    out_score = pd.Series(np.nan, index=s.index, dtype="float64")
    valid = s.dropna()
    if valid.empty:
        return out_class, out_score

    if method == "zscore":
        mu = float(valid.mean())
        sigma = float(valid.std(ddof=0))
        if np.isclose(sigma, 0.0):
            return out_class, out_score
        z = (s - mu) / sigma
        out_score = z.abs()
        out_class.loc[z.abs() >= 3.0] = "mild"
        out_class.loc[z.abs() >= 4.5] = "extreme"
        return out_class, out_score

    if method == "quantile":
        q01, q025, q975, q99 = (
            valid.quantile(0.01),
            valid.quantile(0.025),
            valid.quantile(0.975),
            valid.quantile(0.99),
        )
        iqr = max(valid.quantile(0.75) - valid.quantile(0.25), 1e-9)
        out_score = ((s - valid.median()) / iqr).abs()
        mild_mask = (s < q025) | (s > q975)
        extreme_mask = (s < q01) | (s > q99)
        out_class.loc[mild_mask.fillna(False)] = "mild"
        out_class.loc[extreme_mask.fillna(False)] = "extreme"
        return out_class, out_score

    # default: IQR method
    q1, q3 = valid.quantile(0.25), valid.quantile(0.75)
    iqr = q3 - q1
    if np.isclose(iqr, 0.0):
        return out_class, out_score
    mild_lo, mild_hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    ext_lo, ext_hi = q1 - 3.0 * iqr, q3 + 3.0 * iqr
    out_score = ((s - valid.median()) / iqr).abs()
    mild_mask = (s < mild_lo) | (s > mild_hi)
    extreme_mask = (s < ext_lo) | (s > ext_hi)
    out_class.loc[mild_mask.fillna(False)] = "mild"
    out_class.loc[extreme_mask.fillna(False)] = "extreme"
    return out_class, out_score


def apply_outlier_flags(df: pd.DataFrame, method: str, metrics: list[str]) -> pd.DataFrame:
    out = df.copy()
    rank_cols = []
    for metric in metrics:
        class_col = f"{metric}_outlier_class"
        score_col = f"{metric}_outlier_score"
        rank_col = f"{metric}_outlier_rank"
        classes, score = classify_outliers(out[metric], method=method)
        out[class_col] = classes
        out[score_col] = score
        out[rank_col] = out[class_col].map(class_rank).fillna(0).astype(int)
        rank_cols.append(rank_col)

    if rank_cols:
        out["outlier_rank_any"] = out[rank_cols].max(axis=1)
    else:
        out["outlier_rank_any"] = 0
    out["outlier_class_any"] = out["outlier_rank_any"].map(rank_to_class)
    out["is_outlier_any"] = out["outlier_rank_any"] > 0
    return out


def option_data(values: list[str]) -> list[dict[str, str]]:
    return [{"label": v, "value": v} for v in values]


def clean_str(series: pd.Series) -> pd.Series:
    return series.fillna("UNKNOWN").astype(str).str.strip()


def canonicalize_operator(value: Any) -> str:
    if pd.isna(value):
        return "UNKNOWN"

    s = str(value).upper().strip()
    s = re.sub(r"[.]", "", s)
    s = re.sub(r"\s+", " ", s)

    # Common cleanup for this dataset
    s = s.replace("FEELIPE", "FELIPE")

    if "ALBERTO" in s and "H" in s:
        return "ALBERTO H"
    if "FELIPE" in s and "M" in s:
        return "FELIPE M"
    if "MANUEL" in s and "M" in s:
        return "MANUEL M"
    if "JAIRO" in s and "R" in s:
        return "JAIRO R"

    return s


def machine_label(value: Any) -> str:
    if pd.isna(value):
        return "UNKNOWN"

    try:
        f = float(value)
        if f.is_integer():
            return str(int(f))
        return f"{f:.2f}"
    except Exception:
        return str(value).strip()


def normalize_process_key(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    s = str(value).strip().upper()
    s = "".join(
        c
        for c in unicodedata.normalize("NFKD", s)
        if not unicodedata.combining(c)
    )
    s = re.sub(r"[^A-Z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_header_key(value: Any) -> str:
    return normalize_process_key(value)


def select_sheet_name(path: str, preferred_sheet: str) -> str:
    try:
        xls = pd.ExcelFile(path)
    except Exception:
        return preferred_sheet

    available = xls.sheet_names
    if not available:
        return preferred_sheet

    by_norm = {normalize_header_key(name): name for name in available}
    preferred_norm = normalize_header_key(preferred_sheet)
    if preferred_norm in by_norm:
        return by_norm[preferred_norm]

    for candidate in ["TIEMPOS", "BITACORA", "BITACORA RASPADO", "DATA"]:
        c_norm = normalize_header_key(candidate)
        if c_norm in by_norm:
            return by_norm[c_norm]

    return available[0]


def load_data(path: str, sheet_name: str) -> tuple[pd.DataFrame, str | None]:
    try:
        selected_sheet = select_sheet_name(path, sheet_name)
        raw = pd.read_excel(path, sheet_name=selected_sheet)
    except Exception as exc:
        return pd.DataFrame(), f"Could not load Excel file: {exc}"

    raw_cols_norm = {c: normalize_header_key(c) for c in raw.columns}
    alias_targets = {
        "arrival_time": {"LLEGADA", "FECHA LLEGADA"},
        "start_time": {"FECHA INICIAL", "HORA INICIO", "FECHA INICIO", "INICIO"},
        "end_time": {"FECHA FINAL", "HORA FIN", "FECHA FIN", "FIN"},
        "date": {"FECHA"},
        "lot_id": {"LOTE", "LOTE CARGA", "LOTE CARGA"},
        "client": {"CLIENTE"},
        "item": {"ITEM"},
        "leather": {"CUERO"},
        "pieces": {"PIEZAS", "NUMERO DE CUEROS", "NUMERO CUEROS"},
        "status": {"ESTATUS", "STATUS"},
        "process": {"PROCESO"},
        "machine": {"MAQUINA", "MACHINE"},
        "operator": {"OPERADOR"},
        "downtime": {"TIEMPOS MUERTOS"},
        "service_min_source": {"TIEMPO", "TIEMPO DE PROCESO"},
        "effective_min": {"TIMEPO EFECTIVO", "TIEMPO EFECTIVO"},
        "hours": {"HORAS"},
        "factor": {"FACTOR", "FACTOR MIN CUERO"},
    }
    target_by_norm: dict[str, str] = {}
    for target, aliases in alias_targets.items():
        for a in aliases:
            target_by_norm[normalize_header_key(a)] = target

    rename_map: dict[str, str] = {}
    seen_targets: set[str] = set()
    for col, norm in raw_cols_norm.items():
        target = target_by_norm.get(norm)
        if target and target not in seen_targets:
            rename_map[col] = target
            seen_targets.add(target)

    df = raw.rename(columns=rename_map).copy()

    # If downtime is not provided as a single column, aggregate all TM* columns.
    tm_cols = [c for c, n in raw_cols_norm.items() if n.startswith("TM ")]
    if tm_cols:
        tm_sum = raw[tm_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1, min_count=1)
        if "downtime" not in df.columns:
            df["downtime"] = tm_sum
        else:
            existing = pd.to_numeric(df["downtime"], errors="coerce")
            df["downtime"] = existing.where(existing.notna(), tm_sum)

    required_cols = [
        "arrival_time",
        "start_time",
        "end_time",
        "date",
        "lot_id",
        "client",
        "pieces",
        "status",
        "process",
        "machine",
        "operator",
        "service_min_source",
        "effective_min",
        "hours",
        "factor",
    ]
    for col in required_cols:
        if col not in df.columns:
            df[col] = np.nan

    # Parse date/time fields and compose robust datetimes when only FECHA + HORA columns exist.
    date_base = pd.to_datetime(df["date"], errors="coerce")
    start_parsed = pd.to_datetime(df["start_time"], errors="coerce")
    end_parsed = pd.to_datetime(df["end_time"], errors="coerce")
    arrival_parsed = pd.to_datetime(df["arrival_time"], errors="coerce")

    if not date_base.isna().all():
        start_time_only = start_parsed.notna() & (start_parsed.dt.year <= 1900)
        end_time_only = end_parsed.notna() & (end_parsed.dt.year <= 1900)
        if start_time_only.any():
            start_offset = start_parsed - start_parsed.dt.normalize()
            start_parsed.loc[start_time_only] = date_base.loc[start_time_only] + start_offset.loc[start_time_only]
        if end_time_only.any():
            end_offset = end_parsed - end_parsed.dt.normalize()
            end_parsed.loc[end_time_only] = date_base.loc[end_time_only] + end_offset.loc[end_time_only]

    # Arrival fallback for TIEMPOS-like sheets where explicit arrival is not logged.
    arrival_parsed = arrival_parsed.where(arrival_parsed.notna(), start_parsed)

    # Handle overnight records where end timestamp rolls past midnight.
    overnight_mask = start_parsed.notna() & end_parsed.notna() & (end_parsed < start_parsed)
    if overnight_mask.any():
        end_parsed.loc[overnight_mask] = end_parsed.loc[overnight_mask] + pd.Timedelta(days=1)

    df["arrival_time"] = arrival_parsed
    df["start_time"] = start_parsed
    df["end_time"] = end_parsed

    for col in ["pieces", "machine", "service_min_source", "effective_min", "hours", "factor", "item", "downtime"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["process"] = clean_str(df["process"]).apply(normalize_process_key).replace("", "UNKNOWN")
    df["client"] = clean_str(df["client"])
    df["status"] = clean_str(df["status"])
    df["lot_id"] = clean_str(df["lot_id"])
    df["operator_raw"] = clean_str(df["operator"])
    df["operator_std"] = df["operator_raw"].apply(canonicalize_operator)
    df["machine_label"] = df["machine"].apply(machine_label)

    # Canonical process/service time: end_time - start_time
    df["service_hours"] = (df["end_time"] - df["start_time"]).dt.total_seconds() / 3600.0
    df["wait_hours"] = (df["start_time"] - df["arrival_time"]).dt.total_seconds() / 3600.0
    df["cycle_hours"] = (df["end_time"] - df["arrival_time"]).dt.total_seconds() / 3600.0

    # Keep minute versions for legacy chart/table code paths.
    df["service_min"] = df["service_hours"] * 60.0
    df["wait_min"] = df["wait_hours"] * 60.0
    df["cycle_min"] = df["cycle_hours"] * 60.0

    # Source references from workbook
    df["effective_hours_source"] = pd.to_numeric(df["effective_min"], errors="coerce") / 60.0
    df["service_hours_source"] = pd.to_numeric(df["service_min_source"], errors="coerce") / 60.0
    df["downtime_hours_source"] = pd.to_numeric(df["downtime"], errors="coerce") / 60.0

    df["arrival_date"] = df["arrival_time"].dt.date
    df["arrival_hour"] = df["arrival_time"].dt.hour

    return df, None


def load_reference_sheets(path: str) -> dict[str, float | int | None]:
    out: dict[str, float | int | None] = {
        "horas_lote_count": None,
        "horas_lote_mean": None,
        "interarribos_count": None,
        "interarribos_mean": None,
    }
    try:
        horas_lote = pd.read_excel(path, sheet_name="HORAS_LOTE")
        interarribos = pd.read_excel(path, sheet_name="INTERARRIBOS")
    except Exception:
        return out

    if not horas_lote.empty:
        s = pd.to_numeric(horas_lote.iloc[:, 0], errors="coerce").dropna()
        if not s.empty:
            out["horas_lote_count"] = int(s.shape[0])
            out["horas_lote_mean"] = float(s.mean())

    if not interarribos.empty:
        s = pd.to_numeric(interarribos.iloc[:, 0], errors="coerce").dropna()
        if not s.empty:
            out["interarribos_count"] = int(s.shape[0])
            out["interarribos_mean"] = float(s.mean())

    return out


def load_energy_reference(path: str, sheet_name: str) -> dict[str, Any]:
    out: dict[str, Any] = {
        "loaded": False,
        "source_path": path,
        "sheet_name": sheet_name,
        "rows_parsed": 0,
        "by_process": {},
        "error": None,
    }
    if not path:
        out["error"] = "Missing energy reference path"
        return out

    try:
        raw = pd.read_excel(path, sheet_name=sheet_name, header=None)
    except Exception as exc:
        out["error"] = f"Could not read energy reference: {exc}"
        return out

    by_process: dict[str, dict[str, Any]] = {}
    for row in raw.itertuples(index=False):
        op = row[0] if len(row) > 0 else None
        kwh = row[1] if len(row) > 1 else None
        if pd.isna(op):
            continue
        op_raw = str(op).strip()
        if not op_raw:
            continue

        op_key = normalize_process_key(op_raw)
        if not op_key:
            continue
        if any(token in op_key for token in ["OPERACION", "MAQUINA", "AREA PROMEDIO", "TRANSFER VALUES", "ENERGIA", "AGUA"]):
            continue

        kwh_value = pd.to_numeric(pd.Series([kwh]), errors="coerce").iloc[0]
        if pd.isna(kwh_value):
            continue

        kwh_float = float(kwh_value)
        if kwh_float <= 0:
            continue

        by_process[op_key] = {
            "operation": op_raw,
            "kwh_per_machine_hour": kwh_float,
        }

    out["by_process"] = by_process
    out["rows_parsed"] = len(by_process)
    out["loaded"] = len(by_process) > 0
    return out


def get_energy_kwh_reference_for_process(process_value: str | None) -> float:
    process_key = normalize_process_key(process_value)
    entry = ENERGY_REFERENCE.get("by_process", {}).get(process_key, None)
    if entry and "kwh_per_machine_hour" in entry:
        return float(entry["kwh_per_machine_hour"])
    return float(DEFAULT_ENERGY_KWH_PER_MACHINE_HOUR)


def build_process_machine_catalog(df: pd.DataFrame) -> dict[str, set[str]]:
    catalog: dict[str, set[str]] = {}
    if df.empty or "process" not in df.columns or "machine_label" not in df.columns:
        return dict(PROCESS_MACHINE_CATALOG)

    base = df.copy()
    base["process_key"] = base["process"].apply(normalize_process_key)
    base = base[base["process_key"].astype(str).str.len() > 0]
    for process_key, g in base.groupby("process_key"):
        machines = {str(m).strip() for m in g["machine_label"].dropna().tolist() if str(m).strip() and str(m).strip().upper() != "UNKNOWN"}
        if machines:
            catalog[process_key] = machines

    # Manual catalog rules override inferred values.
    for k, v in PROCESS_MACHINE_CATALOG.items():
        catalog[k] = set(v)

    return catalog


def build_process_server_counts(catalog: dict[str, set[str]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for process_key, machines in catalog.items():
        valid = [m for m in machines if str(m).strip().upper() != "UNKNOWN"]
        inferred = len(valid)
        counts[process_key] = int(max(1, inferred))

    for process_key, c in PROCESS_SERVER_COUNT_OVERRIDES.items():
        if c is None:
            continue
        counts[process_key] = int(max(1, int(c)))

    return counts


def get_process_server_count(process_value: str | None) -> int:
    process_key = normalize_process_key(process_value)
    c = PROCESS_SERVER_COUNT_RESOLVED.get(process_key)
    if c is None:
        return int(FALLBACK_SERVERS_IF_UNKNOWN)
    return int(max(1, int(c)))


def empty_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font={"size": 16, "color": PALETTE["muted"]},
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin={"l": 10, "r": 10, "t": 40, "b": 10},
    )
    return fig


def style_figure(fig: go.Figure, title: str) -> go.Figure:
    fig.update_layout(
        title=title,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ffffff",
        margin={"l": 30, "r": 20, "t": 60, "b": 40},
        font={"family": "IBM Plex Sans, Inter, sans-serif", "color": PALETTE["ink"]},
        title_font={"family": "Space Grotesk, IBM Plex Sans, sans-serif", "size": 18},
        hovermode="x unified",
        transition={"duration": 260, "easing": "cubic-in-out"},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1.0,
            "bgcolor": "rgba(255,255,255,0.75)",
            "bordercolor": "rgba(15,23,42,0.10)",
            "borderwidth": 1,
        },
    )
    fig.update_xaxes(
        showline=True,
        linecolor="#cbd5e1",
        mirror=False,
        gridcolor="rgba(148, 163, 184, 0.20)",
        showspikes=True,
        spikecolor="rgba(15,118,110,0.40)",
        spikethickness=1,
        spikesnap="cursor",
    )
    fig.update_yaxes(
        showline=True,
        linecolor="#cbd5e1",
        mirror=False,
        gridcolor="rgba(148, 163, 184, 0.20)",
        zerolinecolor="rgba(148,163,184,0.25)",
    )
    return fig


def safe_number(value: Any, ndigits: int = 2) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "N/A"
    return f"{value:,.{ndigits}f}"


def _format_dist_params(family_key: str, params: tuple[Any, ...]) -> str:
    try:
        if family_key == "weibull_min":
            k, _loc, scale = params
            return f"k={k:.4f}, lambda={scale:.4f}"
        if family_key == "gamma":
            shape, _loc, scale = params
            return f"shape={shape:.4f}, scale={scale:.4f}"
        if family_key == "lognorm":
            sigma, _loc, scale = params
            return f"sigma={sigma:.4f}, exp(mu)={scale:.4f}"
        if family_key == "invgauss":
            mu, _loc, scale = params
            return f"mu={mu:.4f}, scale={scale:.4f}"
        if family_key == "expon":
            _loc, scale = params
            return f"rate={1.0 / scale:.4f}, scale={scale:.4f}"
        if family_key == "norm":
            mu, sigma = params
            return f"mu={mu:.4f}, sigma={sigma:.4f}"
    except Exception:
        pass
    return str(tuple(params))


def fit_distribution_candidates(values: pd.Series, dataset_label: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    notes: list[dict[str, Any]] = []
    x = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    x = x[x > 0]
    n = int(len(x))

    if n < 25:
        notes.append(
            {
                "dataset": dataset_label,
                "rank": None,
                "family": "N/A",
                "k_params": None,
                "log_likelihood": None,
                "aic": None,
                "delta_aic": None,
                "bic": None,
                "ks_d": None,
                "ks_p": None,
                "n": n,
                "params": None,
                "aic_formula": None,
                "note": "Not enough positive samples for robust fit",
            }
        )
        return rows, notes

    if not SCIPY_AVAILABLE or stats is None:
        notes.append(
            {
                "dataset": dataset_label,
                "rank": None,
                "family": "N/A",
                "k_params": None,
                "log_likelihood": None,
                "aic": None,
                "delta_aic": None,
                "bic": None,
                "ks_d": None,
                "ks_p": None,
                "n": n,
                "params": None,
                "aic_formula": None,
                "note": "SciPy unavailable; install scipy to enable AIC fit",
            }
        )
        return rows, notes

    for spec in DIST_FAMILY_SPECS:
        dist = getattr(stats, spec["key"], None)
        if dist is None:
            continue
        try:
            fit_kwargs = {"floc": 0} if spec.get("force_loc0", False) else {}
            params = dist.fit(x, **fit_kwargs)
            logpdf = dist.logpdf(x, *params)
            if not np.isfinite(logpdf).all():
                continue
            loglik = float(np.sum(logpdf))
            k = int(spec["k_params"])
            aic = float(2.0 * k - 2.0 * loglik)
            bic = float(k * np.log(n) - 2.0 * loglik)
            ks_d, ks_p = stats.kstest(x, lambda z: dist.cdf(z, *params))
            rows.append(
                {
                    "dataset": dataset_label,
                    "family_key": spec["key"],
                    "family": spec["name"],
                    "k_params": k,
                    "log_likelihood": loglik,
                    "aic": aic,
                    "bic": bic,
                    "ks_d": float(ks_d),
                    "ks_p": float(ks_p),
                    "n": n,
                    "params_tuple": tuple(float(v) for v in params),
                    "params": _format_dist_params(spec["key"], tuple(float(v) for v in params)),
                    "aic_formula": f"AIC = 2*{k} - 2*({loglik:.6f}) = {aic:.6f}",
                }
            )
        except Exception:
            continue

    if not rows:
        notes.append(
            {
                "dataset": dataset_label,
                "rank": None,
                "family": "N/A",
                "k_params": None,
                "log_likelihood": None,
                "aic": None,
                "delta_aic": None,
                "bic": None,
                "ks_d": None,
                "ks_p": None,
                "n": n,
                "params": None,
                "aic_formula": None,
                "note": "Fit failed for all configured families",
            }
        )
        return rows, notes

    rows = sorted(rows, key=lambda r: r["aic"])
    best_aic = rows[0]["aic"]
    for i, row in enumerate(rows, start=1):
        row["rank"] = i
        row["delta_aic"] = float(row["aic"] - best_aic)
    return rows, notes


def fit_rows_for_table(fits: list[dict[str, Any]], top_n: int = 5) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in fits[:top_n]:
        out.append(
            {
                "dataset": row.get("dataset"),
                "rank": row.get("rank"),
                "family": row.get("family"),
                "k_params": row.get("k_params"),
                "log_likelihood": round(float(row.get("log_likelihood")), 6) if row.get("log_likelihood") is not None else None,
                "aic": round(float(row.get("aic")), 3) if row.get("aic") is not None else None,
                "delta_aic": round(float(row.get("delta_aic")), 3) if row.get("delta_aic") is not None else None,
                "bic": round(float(row.get("bic")), 3) if row.get("bic") is not None else None,
                "ks_d": round(float(row.get("ks_d")), 4) if row.get("ks_d") is not None else None,
                "ks_p": round(float(row.get("ks_p")), 4) if row.get("ks_p") is not None else None,
                "n": row.get("n"),
                "params": row.get("params"),
                "aic_formula": row.get("aic_formula"),
                "note": "",
            }
        )
    return out


def pick_fit_by_family_key(fits: list[dict[str, Any]], family_key: str) -> dict[str, Any] | None:
    for row in fits:
        if str(row.get("family_key")) == family_key:
            return row
    return None


def histogram_with_fits(
    values: pd.Series,
    fits: list[dict[str, Any]],
    x_label: str,
    title: str,
    nbins: int = 40,
) -> go.Figure:
    x = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    x = x[x > 0]
    if x.empty:
        return empty_figure(f"No data for {title}")

    fig = go.Figure()
    fig.add_histogram(
        x=x,
        histnorm="probability density",
        nbinsx=nbins,
        marker_color=PALETTE["accent_soft"],
        marker_line_color=PALETTE["accent"],
        marker_line_width=1,
        opacity=0.7,
        name="Observed density",
    )

    if SCIPY_AVAILABLE and stats is not None and fits:
        x_min = float(np.quantile(x, 0.01))
        x_max = float(np.quantile(x, 0.99))
        if np.isclose(x_min, x_max):
            x_min = float(x.min())
            x_max = float(x.max())
        if np.isclose(x_min, x_max):
            x_max = x_min + 1.0

        x_grid = np.linspace(x_min, x_max, 300)
        line_colors = ["#0f766e", "#d97706", "#b91c1c"]
        for i, fit in enumerate(fits[:3]):
            dist = getattr(stats, str(fit["family_key"]), None)
            if dist is None:
                continue
            try:
                y = dist.pdf(x_grid, *fit["params_tuple"])
                if not np.isfinite(y).all():
                    continue
                fig.add_scatter(
                    x=x_grid,
                    y=y,
                    mode="lines",
                    line={"width": 2.3, "color": line_colors[i % len(line_colors)]},
                    name=f"{fit['family']} (AIC={fit['aic']:.1f})",
                )
            except Exception:
                continue

    fig.update_xaxes(title=x_label)
    fig.update_yaxes(title="Density")
    fig = style_figure(fig, title)
    return fig


def sample_from_distribution_fit(
    fit_row: dict[str, Any] | None,
    observed_values: pd.Series,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    n_samples = int(max(0, n_samples))
    if n_samples == 0:
        return np.array([], dtype=float)

    observed = pd.to_numeric(observed_values, errors="coerce").dropna().astype(float)
    observed = observed[observed > 0]

    sample = np.array([], dtype=float)
    if SCIPY_AVAILABLE and stats is not None and fit_row:
        family_key = fit_row.get("family_key")
        params = fit_row.get("params_tuple")
        dist = getattr(stats, str(family_key), None) if family_key else None
        if dist is not None and params is not None:
            try:
                sample = np.asarray(dist.rvs(*params, size=n_samples, random_state=rng), dtype=float)
            except Exception:
                try:
                    sample = np.asarray(dist.rvs(*params, size=n_samples), dtype=float)
                except Exception:
                    sample = np.array([], dtype=float)
            sample = sample[np.isfinite(sample)]
            sample = sample[sample > 0]

    if sample.shape[0] < n_samples:
        needed = n_samples - sample.shape[0]
        if not observed.empty:
            refill = rng.choice(observed.to_numpy(), size=needed, replace=True)
        else:
            refill = np.full(needed, 1e-6, dtype=float)
        sample = np.concatenate([sample, refill.astype(float)])

    return sample[:n_samples]


def split_train_holdout_by_arrival(
    df: pd.DataFrame,
    holdout_ratio: float = 0.20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty or "arrival_time" not in df.columns:
        return df.copy(), df.iloc[0:0].copy()

    ordered = df.dropna(subset=["arrival_time"]).sort_values("arrival_time").copy()
    n = int(len(ordered))
    if n < 50:
        return ordered.copy(), ordered.iloc[0:0].copy()

    holdout_n = int(round(n * float(max(0.05, min(0.40, holdout_ratio)))))
    holdout_n = max(20, min(n - 20, holdout_n))
    train = ordered.iloc[: n - holdout_n].copy()
    holdout = ordered.iloc[n - holdout_n :].copy()
    return train, holdout


def bootstrap_positive_series(values: pd.Series, n_samples: int, rng: np.random.Generator) -> np.ndarray:
    n_samples = int(max(0, n_samples))
    if n_samples == 0:
        return np.array([], dtype=float)

    x = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    x = x[np.isfinite(x) & (x > 0)]
    if x.empty:
        return np.full(n_samples, 1e-6, dtype=float)
    return np.asarray(rng.choice(x.to_numpy(), size=n_samples, replace=True), dtype=float)


def generate_time_varying_interarrival(
    arrival_time_series: pd.Series,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    n_samples = int(max(0, n_samples))
    if n_samples == 0:
        return np.array([], dtype=float)

    arr = pd.to_datetime(arrival_time_series, errors="coerce").dropna()
    if arr.empty:
        return np.full(n_samples, 1e-6, dtype=float)

    # Template-day bootstrap: preserve intraday arrival shape and daily volume variability.
    by_day = (
        pd.DataFrame({"arrival_time": arr})
        .assign(day=lambda x: x["arrival_time"].dt.date)
        .sort_values("arrival_time")
        .groupby("day")["arrival_time"]
    )

    day_templates: list[np.ndarray] = []
    for _day, timestamps in by_day:
        offsets = (
            timestamps.dt.hour.astype(float)
            + timestamps.dt.minute.astype(float) / 60.0
            + timestamps.dt.second.astype(float) / 3600.0
        ).to_numpy(dtype=float)
        offsets = offsets[np.isfinite(offsets)]
        offsets = offsets[(offsets >= 0.0) & (offsets < 24.0)]
        if offsets.size > 0:
            day_templates.append(np.sort(offsets))

    if not day_templates:
        arr_sorted = np.sort(arr.view("int64").astype(float) / 1e9 / 3600.0)
        if arr_sorted.size < 2:
            return np.full(n_samples, 1e-6, dtype=float)
        ia_obs = np.diff(arr_sorted)
        ia_obs = ia_obs[np.isfinite(ia_obs) & (ia_obs > 0)]
        if ia_obs.size == 0:
            return np.full(n_samples, 1e-6, dtype=float)
        return np.asarray(rng.choice(ia_obs, size=n_samples, replace=True), dtype=float)

    simulated_arrivals: list[float] = []
    day_idx = 0
    while len(simulated_arrivals) < n_samples:
        template = day_templates[int(rng.integers(0, len(day_templates)))]
        jitter = rng.normal(loc=0.0, scale=0.03, size=template.size)  # about +/- 2 minutes
        offsets = np.clip(template + jitter, 0.0, 23.9999)
        offsets.sort()
        base = float(day_idx * 24.0)
        simulated_arrivals.extend((base + offsets).tolist())
        day_idx += 1
        if day_idx > 3650:  # hard stop guard
            break

    arr_h = np.asarray(simulated_arrivals[:n_samples], dtype=float)
    arr_h = arr_h[np.isfinite(arr_h)]
    if arr_h.size == 0:
        return np.full(n_samples, 1e-6, dtype=float)
    arr_h.sort()

    ia = np.empty(arr_h.size, dtype=float)
    ia[0] = max(float(arr_h[0]), 1e-6)
    if arr_h.size > 1:
        ia[1:] = np.diff(arr_h)
    ia = np.where(np.isfinite(ia) & (ia > 0), ia, 1e-6)
    if ia.size < n_samples:
        refill = np.full(n_samples - ia.size, max(float(np.nanmean(ia)) if ia.size > 0 else 1.0, 1e-6), dtype=float)
        ia = np.concatenate([ia, refill])
    return ia[:n_samples]


def build_piece_service_profile(
    pieces: pd.Series,
    service_h: pd.Series,
    max_bins: int = 4,
) -> dict[str, Any]:
    df = pd.DataFrame(
        {
            "pieces": pd.to_numeric(pieces, errors="coerce"),
            "service_h": pd.to_numeric(service_h, errors="coerce"),
        }
    ).dropna()
    df = df[(df["pieces"] > 0) & (df["service_h"] > 0)]

    profile: dict[str, Any] = {
        "valid": False,
        "pieces_boot": np.array([], dtype=float),
        "service_global": np.array([], dtype=float),
        "edges": np.array([], dtype=float),
        "service_by_bin": [],
    }
    if df.empty:
        return profile

    pieces_arr = df["pieces"].to_numpy(dtype=float)
    service_arr = df["service_h"].to_numpy(dtype=float)
    profile["pieces_boot"] = pieces_arr
    profile["service_global"] = service_arr

    if len(df) < 25 or np.unique(pieces_arr).size < 3:
        return profile

    q = int(max(2, min(int(max_bins), int(np.unique(pieces_arr).size))))
    try:
        _, edges = pd.qcut(pieces_arr, q=q, retbins=True, duplicates="drop")
        edges = np.asarray(edges, dtype=float)
    except Exception:
        return profile

    if edges.size < 3:
        return profile

    # Guarantee strictly increasing edges for digitize.
    edges = np.unique(edges)
    if edges.size < 3:
        return profile

    service_by_bin: list[np.ndarray] = []
    for i in range(edges.size - 1):
        if i == edges.size - 2:
            mask = (pieces_arr >= edges[i]) & (pieces_arr <= edges[i + 1])
        else:
            mask = (pieces_arr >= edges[i]) & (pieces_arr < edges[i + 1])
        s = service_arr[mask]
        s = s[np.isfinite(s) & (s > 0)]
        if s.size == 0:
            s = service_arr
        service_by_bin.append(np.asarray(s, dtype=float))

    profile["valid"] = True
    profile["edges"] = edges
    profile["service_by_bin"] = service_by_bin
    return profile


def sample_piece_dependent_service(
    profile: dict[str, Any],
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    n_samples = int(max(0, n_samples))
    if n_samples == 0:
        return np.array([], dtype=float)

    global_service = np.asarray(profile.get("service_global", np.array([], dtype=float)), dtype=float)
    global_service = global_service[np.isfinite(global_service) & (global_service > 0)]
    if global_service.size == 0:
        return np.full(n_samples, 1e-6, dtype=float)

    if not bool(profile.get("valid")):
        return np.asarray(rng.choice(global_service, size=n_samples, replace=True), dtype=float)

    pieces_boot = np.asarray(profile.get("pieces_boot", np.array([], dtype=float)), dtype=float)
    edges = np.asarray(profile.get("edges", np.array([], dtype=float)), dtype=float)
    service_by_bin: list[np.ndarray] = profile.get("service_by_bin", [])
    if pieces_boot.size == 0 or edges.size < 3 or not service_by_bin:
        return np.asarray(rng.choice(global_service, size=n_samples, replace=True), dtype=float)

    sampled_pieces = np.asarray(rng.choice(pieces_boot, size=n_samples, replace=True), dtype=float)
    bin_idx = np.digitize(sampled_pieces, edges[1:-1], right=False)

    out = np.empty(n_samples, dtype=float)
    for i in range(n_samples):
        idx = int(max(0, min(len(service_by_bin) - 1, int(bin_idx[i]))))
        pool = np.asarray(service_by_bin[idx], dtype=float)
        pool = pool[np.isfinite(pool) & (pool > 0)]
        if pool.size == 0:
            pool = global_service
        out[i] = float(rng.choice(pool))

    out = np.where(np.isfinite(out) & (out > 0), out, 1e-6)
    return out


def make_downtime_sampler(
    downtime_hours: pd.Series,
    rng: np.random.Generator,
) -> tuple[dict[str, float], Any]:
    dt = pd.to_numeric(downtime_hours, errors="coerce").dropna().astype(float)
    dt = dt[dt >= 0]
    positive = dt[dt > 0]
    total_n = int(len(dt))
    positive_n = int(len(positive))
    p_event = float(positive_n / total_n) if total_n > 0 else 0.0

    stats_out = {
        "downtime_rows": float(total_n),
        "downtime_positive_rows": float(positive_n),
        "p_downtime": float(p_event),
        "downtime_mean_h": float(positive.mean()) if positive_n > 0 else 0.0,
    }

    if positive_n == 0 or p_event <= 0:
        return stats_out, (lambda: 0.0)

    pool = positive.to_numpy(dtype=float)

    def _draw() -> float:
        if float(rng.random()) < p_event:
            return float(rng.choice(pool))
        return 0.0

    return stats_out, _draw


def simulate_gigc_queue(
    interarrival_h: np.ndarray,
    service_h: np.ndarray,
    servers: int,
    downtime_draw: Any | None = None,
) -> dict[str, Any]:
    c = int(max(1, servers))
    n = int(min(len(interarrival_h), len(service_h)))
    if n <= 0:
        return {
            "n": 0,
            "wait_h": np.array([], dtype=float),
            "sojourn_h": np.array([], dtype=float),
            "arrival_h": np.array([], dtype=float),
            "service_h": np.array([], dtype=float),
            "downtime_h": np.array([], dtype=float),
            "lambda_h": np.nan,
            "mu_h": np.nan,
            "rho": np.nan,
            "mean_wait_h": np.nan,
            "p_wait": np.nan,
            "p_wait_1h": np.nan,
            "p_wait_2h": np.nan,
            "p90_wait_h": np.nan,
            "mean_sojourn_h": np.nan,
            "lq": np.nan,
            "ls": np.nan,
            "total_downtime_h": np.nan,
            "mean_downtime_h": np.nan,
            "p_downtime_event": np.nan,
        }

    ia = np.asarray(interarrival_h[:n], dtype=float)
    sv = np.asarray(service_h[:n], dtype=float)
    ia = np.where(np.isfinite(ia) & (ia > 0), ia, 1e-6)
    sv = np.where(np.isfinite(sv) & (sv > 0), sv, 1e-6)
    arrivals = np.cumsum(ia)

    wait = np.zeros(n, dtype=float)
    sojourn = np.zeros(n, dtype=float)
    finish = np.zeros(n, dtype=float)
    downtime_used = np.zeros(n, dtype=float)

    if not SIMPY_AVAILABLE or simpy is None:
        return {
            "n": n,
            "wait_h": wait,
            "sojourn_h": sojourn,
            "arrival_h": arrivals,
            "service_h": sv,
            "downtime_h": downtime_used,
            "lambda_h": np.nan,
            "mu_h": np.nan,
            "rho": np.nan,
            "mean_wait_h": np.nan,
            "p_wait": np.nan,
            "p_wait_1h": np.nan,
            "p_wait_2h": np.nan,
            "p90_wait_h": np.nan,
            "mean_sojourn_h": np.nan,
            "lq": np.nan,
            "ls": np.nan,
            "total_downtime_h": np.nan,
            "mean_downtime_h": np.nan,
            "p_downtime_event": np.nan,
            "sim_engine": "missing_simpy",
        }

    env = simpy.Environment()
    resource = simpy.Resource(env, capacity=c)

    def lot_process(i: int):
        yield env.timeout(max(0.0, float(arrivals[i]) - float(env.now)))
        t_arrive = float(env.now)
        with resource.request() as req:
            yield req
            t_start = float(env.now)
            wait[i] = t_start - t_arrive
            if callable(downtime_draw):
                dt_h = float(max(0.0, float(downtime_draw())))
                downtime_used[i] = dt_h
                if dt_h > 0:
                    # Server is unavailable while resolving the stop event.
                    yield env.timeout(dt_h)
            yield env.timeout(float(sv[i]))
            t_finish = float(env.now)
            sojourn[i] = t_finish - t_arrive
            finish[i] = t_finish

    for i in range(n):
        env.process(lot_process(i))
    env.run()

    horizon_arrival = float(arrivals[-1]) if n > 0 else np.nan
    horizon_system = float(max(float(finish.max()), horizon_arrival)) if n > 0 else np.nan
    lambda_h = float(n / horizon_arrival) if horizon_arrival > 0 else np.nan
    mean_service = float(np.mean(sv)) if n > 0 else np.nan
    mu_h = float(1.0 / mean_service) if mean_service > 0 else np.nan
    rho = float(lambda_h / (c * mu_h)) if np.isfinite(lambda_h) and np.isfinite(mu_h) and mu_h > 0 else np.nan
    mean_wait = float(np.mean(wait)) if n > 0 else np.nan
    mean_sojourn = float(np.mean(sojourn)) if n > 0 else np.nan
    lq = float(lambda_h * mean_wait) if np.isfinite(lambda_h) else np.nan
    ls = float(lambda_h * mean_sojourn) if np.isfinite(lambda_h) else np.nan
    util_with_dt = float(np.sum(sv + downtime_used) / (c * horizon_system)) if horizon_system > 0 else np.nan
    util_base = float(np.sum(sv) / (c * horizon_system)) if horizon_system > 0 else np.nan
    if np.isfinite(util_with_dt):
        rho = util_with_dt
    elif np.isfinite(util_base):
        rho = util_base

    total_downtime_h = float(np.sum(downtime_used))
    mean_downtime_h = float(np.mean(downtime_used[downtime_used > 0])) if np.any(downtime_used > 0) else 0.0
    p_downtime_event = float(np.mean(downtime_used > 0)) if n > 0 else np.nan

    return {
        "n": n,
        "wait_h": wait,
        "sojourn_h": sojourn,
        "arrival_h": arrivals,
        "service_h": sv,
        "downtime_h": downtime_used,
        "lambda_h": lambda_h,
        "mu_h": mu_h,
        "rho": rho,
        "mean_wait_h": mean_wait,
        "p_wait": float(np.mean(wait > 1e-9)),
        "p_wait_1h": float(np.mean(wait > 1.0)),
        "p_wait_2h": float(np.mean(wait > 2.0)),
        "p90_wait_h": float(np.quantile(wait, 0.90)) if n > 0 else np.nan,
        "mean_sojourn_h": mean_sojourn,
        "lq": lq,
        "ls": ls,
        "total_downtime_h": total_downtime_h,
        "mean_downtime_h": mean_downtime_h,
        "p_downtime_event": p_downtime_event,
        "sim_engine": "simpy",
    }


def parse_flow_sequence(flow_text: str | None, available_processes: list[str]) -> list[str]:
    if not flow_text:
        return []
    by_key = {normalize_process_key(p): p for p in available_processes}
    tokens = re.split(r"[>,;|\n]+", str(flow_text))
    out: list[str] = []
    for tok in tokens:
        key = normalize_process_key(tok)
        if not key:
            continue
        canonical = by_key.get(key)
        if canonical:
            out.append(canonical)
    return out


def parse_flow_sequence_specs(flow_text: str | None, available_processes: list[str]) -> list[dict[str, Any]]:
    if not flow_text:
        return []
    by_key = {normalize_process_key(p): p for p in available_processes}
    tokens = re.split(r"[>,;|\n]+", str(flow_text))
    specs: list[dict[str, Any]] = []
    for tok in tokens:
        raw = str(tok).strip()
        if not raw:
            continue
        m = re.match(r"^(.*?)(?:\s*(?:\(|:)\s*(\d+)\s*\)?)?$", raw)
        name_part = (m.group(1) if m else raw).strip()
        c_part = m.group(2) if m else None
        key = normalize_process_key(name_part)
        if not key:
            continue
        canonical = by_key.get(key)
        if not canonical:
            continue
        c_override = int(c_part) if c_part and str(c_part).isdigit() else None
        specs.append({"process": canonical, "c_override": c_override})
    return specs


def extract_first_number(value: Any) -> float | None:
    if value is None:
        return None
    m = re.search(r"[-+]?\d*\.?\d+", str(value))
    if not m:
        return None
    try:
        return float(m.group(0))
    except Exception:
        return None


def parse_user_lot_plan(
    lot_plan_text: str | None,
    available_processes: list[str],
    default_pieces: float,
) -> tuple[list[dict[str, Any]], list[str]]:
    if not lot_plan_text:
        return [], []

    lines = [str(x).strip() for x in str(lot_plan_text).splitlines() if str(x).strip()]
    if not lines:
        return [], []

    lots: list[dict[str, Any]] = []
    notes: list[str] = []

    for idx, line in enumerate(lines, start=1):
        line_clean = line.strip()
        lot_name = f"Lote_{idx}"
        pieces_value: float | None = None
        repeat_lots = 1
        route: list[str] = []

        if "|" in line_clean:
            parts = [p.strip() for p in line_clean.split("|")]
            if len(parts) >= 1 and parts[0]:
                lot_name = parts[0]
            if len(parts) >= 2:
                pieces_value = extract_first_number(parts[1])
            if len(parts) >= 3:
                route = parse_flow_sequence(parts[2], available_processes)
            if len(parts) >= 4:
                repeat_candidate = extract_first_number(parts[3])
                if repeat_candidate is not None and repeat_candidate > 0:
                    repeat_lots = int(max(1, min(500, round(repeat_candidate))))
        else:
            parts = [p.strip() for p in line_clean.split(",")]
            if len(parts) >= 1:
                first_as_route = parse_flow_sequence(parts[0], available_processes)
                if first_as_route:
                    route = parse_flow_sequence(line_clean, available_processes)
                else:
                    lot_name = parts[0] if parts[0] else lot_name
                    if len(parts) >= 2:
                        pieces_value = extract_first_number(parts[1])
                    route = parse_flow_sequence(",".join(parts[2:]), available_processes) if len(parts) >= 3 else []
            if not route:
                route = parse_flow_sequence(line_clean, available_processes)

        if not route:
            notes.append(f"Skipped line {idx}: no valid process names found.")
            continue

        if pieces_value is None or not np.isfinite(pieces_value) or pieces_value <= 0:
            pieces_value = float(default_pieces)
            notes.append(f"Line {idx}: pieces missing/invalid, defaulted to {safe_number(default_pieces, 1)}.")

        for rep in range(repeat_lots):
            final_name = lot_name if repeat_lots == 1 else f"{lot_name}#{rep + 1}"
            lots.append(
                {
                    "lot_name": final_name,
                    "pieces": float(pieces_value),
                    "route": list(route),
                }
            )

    return lots, notes


def build_stage_catalog_for_processes(
    base_df: pd.DataFrame,
    process_list: list[str],
    strict_cleaning: bool,
    queue_use_downtime: bool,
    rng: np.random.Generator,
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    stage_catalog: dict[str, dict[str, Any]] = {}
    missing: list[str] = []

    for proc in process_list:
        stage_df = build_stage_clean_data(base_df, proc, strict_cleaning=bool(strict_cleaning))
        service_pool = pd.to_numeric(stage_df.get("service_hours", pd.Series(dtype=float)), errors="coerce").dropna()
        service_pool = service_pool[(service_pool > 0) & np.isfinite(service_pool)]
        if service_pool.empty:
            missing.append(proc)
            continue

        pieces_col = pd.to_numeric(stage_df.get("pieces", pd.Series(dtype=float)), errors="coerce")
        rate_df = pd.DataFrame({"service_h": service_pool.reindex(stage_df.index), "pieces": pieces_col})
        rate_df = rate_df[(rate_df["service_h"] > 0) & (rate_df["pieces"] > 0)]
        rate_df["service_h_per_piece"] = rate_df["service_h"] / rate_df["pieces"]
        rate_pool = pd.to_numeric(rate_df["service_h_per_piece"], errors="coerce").dropna()
        rate_pool = rate_pool[(rate_pool > 0) & np.isfinite(rate_pool)]

        service_values = service_pool.to_numpy(dtype=float)
        rate_values = rate_pool.to_numpy(dtype=float)

        def draw_service_h(
            lot_pieces: float,
            sv_pool: np.ndarray = service_values,
            rate_pool_local: np.ndarray = rate_values,
            rg: np.random.Generator = rng,
        ) -> float:
            if np.isfinite(lot_pieces) and lot_pieces > 0 and rate_pool_local.size > 0:
                return float(max(1e-6, float(rg.choice(rate_pool_local)) * float(lot_pieces)))
            return float(max(1e-6, float(rg.choice(sv_pool))))

        draw_downtime_h = lambda: 0.0
        if bool(queue_use_downtime):
            _dt_stats, dt_sampler = make_downtime_sampler(stage_df.get("downtime_hours_source", pd.Series(dtype=float)), rng)
            draw_downtime_h = dt_sampler

        stage_catalog[proc] = {
            "process": proc,
            "servers": int(get_process_server_count(proc)),
            "rows_used": int(len(stage_df)),
            "mean_service_h_empirical": float(service_pool.mean()),
            "draw_service_h": draw_service_h,
            "draw_downtime_h": draw_downtime_h,
        }

    return stage_catalog, missing


def simulate_lot_plan_flow(
    stage_catalog: dict[str, dict[str, Any]],
    lot_plan: list[dict[str, Any]],
    interarrival_h: np.ndarray,
) -> dict[str, Any]:
    n = int(len(lot_plan))
    if n == 0:
        return {
            "n": 0,
            "lot_rows": [],
            "stage_rows": [],
            "sim_engine": "simpy" if SIMPY_AVAILABLE else "missing_simpy",
        }

    ia = np.asarray(interarrival_h, dtype=float)
    if ia.size < n:
        fill = float(np.nanmean(ia)) if ia.size > 0 and np.isfinite(np.nanmean(ia)) else 1.0
        ia = np.concatenate([ia, np.full(n - ia.size, max(fill, 1e-6), dtype=float)])
    ia = ia[:n]
    ia = np.where(np.isfinite(ia) & (ia > 0), ia, 1e-6)
    arrivals = np.cumsum(ia)

    lot_rows: list[dict[str, Any]] = []
    stage_rows: list[dict[str, Any]] = []

    if not SIMPY_AVAILABLE or simpy is None:
        return {
            "n": n,
            "lot_rows": lot_rows,
            "stage_rows": stage_rows,
            "arrival_h": arrivals,
            "sim_engine": "missing_simpy",
        }

    env = simpy.Environment()
    resources = {
        proc: simpy.Resource(env, capacity=int(max(1, spec["servers"])))
        for proc, spec in stage_catalog.items()
    }

    def lot_process(i: int, lot: dict[str, Any]):
        yield env.timeout(max(0.0, float(arrivals[i]) - float(env.now)))
        t_system_arrive = float(env.now)
        valid_steps = 0

        for step_idx, proc in enumerate(lot["route"], start=1):
            spec = stage_catalog.get(proc)
            if spec is None:
                continue
            valid_steps += 1
            t_stage_arrive = float(env.now)
            with resources[proc].request() as req:
                yield req
                t_stage_start = float(env.now)
                wait_h = t_stage_start - t_stage_arrive
                service_h = float(max(1e-6, spec["draw_service_h"](float(lot["pieces"]))))
                dt_h = float(max(0.0, spec["draw_downtime_h"]()))
                if dt_h > 0:
                    yield env.timeout(dt_h)
                yield env.timeout(service_h)
                t_stage_finish = float(env.now)
                stage_rows.append(
                    {
                        "lot_name": lot["lot_name"],
                        "pieces": float(lot["pieces"]),
                        "process": proc,
                        "route_step": int(step_idx),
                        "wait_h": float(wait_h),
                        "service_h": float(service_h),
                        "downtime_h": float(dt_h),
                        "stage_time_h": float(t_stage_finish - t_stage_arrive),
                    }
                )

        total_system_h = float(env.now) - t_system_arrive
        lot_rows.append(
            {
                "lot_name": lot["lot_name"],
                "pieces": float(lot["pieces"]),
                "route_len": int(len(lot["route"])),
                "valid_steps": int(valid_steps),
                "system_time_h": float(total_system_h),
            }
        )

    for i, lot in enumerate(lot_plan):
        env.process(lot_process(i, lot))
    env.run()

    return {
        "n": n,
        "lot_rows": lot_rows,
        "stage_rows": stage_rows,
        "arrival_h": arrivals,
        "sim_engine": "simpy",
    }


def lot_plan_table_rows(lot_plan: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i, lot in enumerate(lot_plan, start=1):
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


def build_stage_clean_data(
    df: pd.DataFrame,
    process_value: str,
    strict_cleaning: bool = True,
) -> pd.DataFrame:
    stage = df[df["process"] == process_value].copy()
    if stage.empty:
        return stage
    stage = add_quality_flags(stage, process_value=process_value)
    if strict_cleaning:
        stage = stage[(~stage["invalid_time_row"]) & (~stage["missing_time_flag"])].copy()
    stage = apply_outlier_flags(stage, method=FIXED_OUTLIER_METHOD, metrics=["service_min", "wait_min", "cycle_min"])
    stage = stage[~stage["is_outlier_any"]].copy()
    return stage


def simulate_connected_flow(
    stage_specs: list[dict[str, Any]],
    interarrival_h: np.ndarray,
    n_jobs: int,
) -> dict[str, Any]:
    m = int(len(stage_specs))
    n = int(max(0, n_jobs))
    if m == 0 or n == 0:
        return {
            "n": 0,
            "m": m,
            "stage_wait_h": np.empty((m, 0), dtype=float),
            "stage_service_h": np.empty((m, 0), dtype=float),
            "stage_downtime_h": np.empty((m, 0), dtype=float),
            "stage_sojourn_h": np.empty((m, 0), dtype=float),
            "system_sojourn_h": np.array([], dtype=float),
            "sim_engine": "simpy" if SIMPY_AVAILABLE else "missing_simpy",
        }

    ia = np.asarray(interarrival_h, dtype=float)
    if ia.size < n:
        fill = float(np.nanmean(ia)) if ia.size > 0 and np.isfinite(np.nanmean(ia)) else 1.0
        ia = np.concatenate([ia, np.full(n - ia.size, max(fill, 1e-6), dtype=float)])
    ia = ia[:n]
    ia = np.where(np.isfinite(ia) & (ia > 0), ia, 1e-6)
    arrivals = np.cumsum(ia)

    stage_wait = np.zeros((m, n), dtype=float)
    stage_service = np.zeros((m, n), dtype=float)
    stage_downtime = np.zeros((m, n), dtype=float)
    stage_sojourn = np.zeros((m, n), dtype=float)
    system_sojourn = np.zeros(n, dtype=float)

    if not SIMPY_AVAILABLE or simpy is None:
        return {
            "n": n,
            "m": m,
            "stage_wait_h": stage_wait,
            "stage_service_h": stage_service,
            "stage_downtime_h": stage_downtime,
            "stage_sojourn_h": stage_sojourn,
            "system_sojourn_h": system_sojourn,
            "sim_engine": "missing_simpy",
        }

    env = simpy.Environment()
    resources = [simpy.Resource(env, capacity=int(max(1, s["servers"]))) for s in stage_specs]

    def lot_process(i: int):
        yield env.timeout(max(0.0, float(arrivals[i]) - float(env.now)))
        t_system_arrive = float(env.now)
        for s_idx, spec in enumerate(stage_specs):
            t_stage_arrive = float(env.now)
            with resources[s_idx].request() as req:
                yield req
                t_stage_start = float(env.now)
                stage_wait[s_idx, i] = t_stage_start - t_stage_arrive
                service_h = float(max(1e-6, spec["draw_service_h"]()))
                stage_service[s_idx, i] = service_h
                dt_h = float(max(0.0, spec["draw_downtime_h"]())) if callable(spec.get("draw_downtime_h")) else 0.0
                stage_downtime[s_idx, i] = dt_h
                if dt_h > 0:
                    yield env.timeout(dt_h)
                yield env.timeout(service_h)
                t_stage_finish = float(env.now)
                stage_sojourn[s_idx, i] = t_stage_finish - t_stage_arrive
        system_sojourn[i] = float(env.now) - t_system_arrive

    for i in range(n):
        env.process(lot_process(i))
    env.run()

    return {
        "n": n,
        "m": m,
        "stage_wait_h": stage_wait,
        "stage_service_h": stage_service,
        "stage_downtime_h": stage_downtime,
        "stage_sojourn_h": stage_sojourn,
        "system_sojourn_h": system_sojourn,
        "sim_engine": "simpy",
    }


def allen_cunneen_approx(
    interarrival_h: pd.Series,
    service_h: pd.Series,
    servers: int,
) -> dict[str, Any]:
    c = int(max(1, servers))
    ia = pd.to_numeric(interarrival_h, errors="coerce").dropna().astype(float)
    ia = ia[ia > 0]
    sv = pd.to_numeric(service_h, errors="coerce").dropna().astype(float)
    sv = sv[sv > 0]

    if ia.empty or sv.empty:
        return {
            "valid": False,
            "n_ia": int(len(ia)),
            "n_sv": int(len(sv)),
            "lambda_h": np.nan,
            "mu_h": np.nan,
            "rho": np.nan,
            "ca2": np.nan,
            "cs2": np.nan,
            "wq_h": np.nan,
            "ws_h": np.nan,
            "lq": np.nan,
            "ls": np.nan,
        }

    mean_ia = float(ia.mean())
    mean_sv = float(sv.mean())
    std_ia = float(ia.std(ddof=1)) if len(ia) > 1 else 0.0
    std_sv = float(sv.std(ddof=1)) if len(sv) > 1 else 0.0

    lambda_h = float(1.0 / mean_ia) if mean_ia > 0 else np.nan
    mu_h = float(1.0 / mean_sv) if mean_sv > 0 else np.nan
    rho = float(lambda_h / (c * mu_h)) if np.isfinite(lambda_h) and np.isfinite(mu_h) and mu_h > 0 else np.nan
    ca2 = float((std_ia / mean_ia) ** 2) if mean_ia > 0 else np.nan
    cs2 = float((std_sv / mean_sv) ** 2) if mean_sv > 0 else np.nan

    if not np.isfinite(rho) or not np.isfinite(ca2) or not np.isfinite(cs2) or not np.isfinite(mean_sv):
        return {
            "valid": False,
            "n_ia": int(len(ia)),
            "n_sv": int(len(sv)),
            "lambda_h": lambda_h,
            "mu_h": mu_h,
            "rho": rho,
            "ca2": ca2,
            "cs2": cs2,
            "wq_h": np.nan,
            "ws_h": np.nan,
            "lq": np.nan,
            "ls": np.nan,
        }

    if rho >= 1.0:
        wq_h = np.inf
    else:
        exponent = float(np.sqrt(2.0 * (c + 1.0)) - 1.0)
        wq_h = float(
            ((ca2 + cs2) / 2.0)
            * (rho**exponent)
            * (mean_sv / (c * (1.0 - rho)))
        )
    ws_h = float(wq_h + mean_sv) if np.isfinite(wq_h) else np.inf
    lq = float(lambda_h * wq_h) if np.isfinite(lambda_h) and np.isfinite(wq_h) else np.inf
    ls = float(lambda_h * ws_h) if np.isfinite(lambda_h) and np.isfinite(ws_h) else np.inf

    return {
        "valid": True,
        "n_ia": int(len(ia)),
        "n_sv": int(len(sv)),
        "lambda_h": lambda_h,
        "mu_h": mu_h,
        "rho": rho,
        "ca2": ca2,
        "cs2": cs2,
        "wq_h": wq_h,
        "ws_h": ws_h,
        "lq": lq,
        "ls": ls,
    }


def filter_frame(
    df: pd.DataFrame,
    process_value: str | None,
    date_range: list[str] | None,
    machine_values: list[str] | None,
    operator_values: list[str] | None,
    client_values: list[str] | None,
    operator_mode: str,
) -> pd.DataFrame:
    out = df.copy()

    if process_value:
        out = out[out["process"] == process_value]
        process_key = str(process_value).upper().strip()
        official_catalog = PROCESS_MACHINE_CATALOG_RESOLVED.get(process_key, set())
        if official_catalog:
            # Hard process rule: keep only official machines for this process.
            out = out[out["machine_label"].isin(sorted(official_catalog))]

    if date_range and len(date_range) == 2 and date_range[0] and date_range[1]:
        start = pd.to_datetime(date_range[0], errors="coerce")
        end = pd.to_datetime(date_range[1], errors="coerce")
        if pd.notna(start) and pd.notna(end):
            out = out[(out["arrival_time"] >= start) & (out["arrival_time"] <= end + pd.Timedelta(days=1))]

    if machine_values:
        out = out[out["machine_label"].isin(machine_values)]

    if client_values:
        out = out[out["client"].isin(client_values)]

    op_col = "operator_std" if operator_mode == "normalized" else "operator_raw"
    if operator_values:
        out = out[out[op_col].isin(operator_values)]

    out["operator_view"] = out[op_col]
    return out


def add_quality_flags(df: pd.DataFrame, process_value: str | None = None) -> pd.DataFrame:
    out = df.copy()
    out["missing_time_flag"] = out[["arrival_time", "start_time", "end_time"]].isna().any(axis=1)
    out["start_before_arrival_flag"] = (out["start_time"] < out["arrival_time"]).fillna(False)
    out["end_before_start_flag"] = (out["end_time"] < out["start_time"]).fillna(False)
    out["service_nonpositive_flag"] = (to_numeric_series(out, "service_min") <= 0).fillna(False)
    out["wait_negative_flag"] = (to_numeric_series(out, "wait_min") < 0).fillna(False)
    out["cycle_negative_flag"] = (to_numeric_series(out, "cycle_min") < 0).fillna(False)
    out["time_logic_invalid_flag"] = out[
        [
            "start_before_arrival_flag",
            "end_before_start_flag",
            "service_nonpositive_flag",
            "wait_negative_flag",
            "cycle_negative_flag",
        ]
    ].any(axis=1)

    process_key = (process_value or "").upper().strip()
    official_catalog = PROCESS_MACHINE_CATALOG_RESOLVED.get(process_key, set())
    if official_catalog:
        out["machine_catalog_invalid_flag"] = (~out["machine_label"].isin(sorted(official_catalog))).fillna(True)
    else:
        out["machine_catalog_invalid_flag"] = False

    out["invalid_time_row"] = out["time_logic_invalid_flag"] | out["machine_catalog_invalid_flag"]

    out["row_quality"] = np.where(
        out["missing_time_flag"],
        "missing_time",
        np.where(out["machine_catalog_invalid_flag"], "invalid_machine_catalog", np.where(out["time_logic_invalid_flag"], "invalid_time_logic", "valid")),
    )
    return out


DATAFRAME, DATA_ERROR = load_data(DATA_PATH, SHEET_NAME)
REFERENCE_SHEETS = load_reference_sheets(DATA_PATH)
ENERGY_REFERENCE = load_energy_reference(ENERGY_REF_PATH, ENERGY_REF_SHEET)
PROCESS_MACHINE_CATALOG_RESOLVED = build_process_machine_catalog(DATAFRAME)
PROCESS_SERVER_COUNT_RESOLVED = build_process_server_counts(PROCESS_MACHINE_CATALOG_RESOLVED)
PROCESS_OPTIONS = sorted(v for v in DATAFRAME["process"].dropna().unique().tolist() if str(v).strip() and str(v).strip().upper() != "UNKNOWN")
DEFAULT_PROCESS = "RASPADO" if "RASPADO" in PROCESS_OPTIONS else (PROCESS_OPTIONS[0] if PROCESS_OPTIONS else None)
DEFAULT_ENERGY_REFERENCE_KWH = get_energy_kwh_reference_for_process(DEFAULT_PROCESS)

if not DATAFRAME.empty and DEFAULT_PROCESS:
    proc_df = DATAFRAME[DATAFRAME["process"] == DEFAULT_PROCESS]
    default_start = proc_df["arrival_time"].min().date().isoformat() if proc_df["arrival_time"].notna().any() else None
    default_end = proc_df["arrival_time"].max().date().isoformat() if proc_df["arrival_time"].notna().any() else None
    DEFAULT_DATE_RANGE = [default_start, default_end]
else:
    DEFAULT_DATE_RANGE = [None, None]

app = dash.Dash(__name__)
app.title = APP_TITLE
server = app.server


def metric_card(title: str, metric_id: str, note: str) -> dmc.Paper:
    return dmc.Paper(
        withBorder=True,
        shadow="xs",
        radius="lg",
        p="md",
        children=[
            dmc.Text(title, c="dimmed", fw=600, fz="sm"),
            dmc.Text("--", id=metric_id, fw=800, fz="1.7rem"),
            dmc.Text(note, c="dimmed", fz="xs"),
        ],
    )


controls_panel = dmc.Paper(
    withBorder=True,
    radius="lg",
    p="lg",
    shadow="xs",
    children=dmc.Stack(
        gap="sm",
        children=[
            dmc.Group(
                justify="space-between",
                children=[
                    dmc.Text("Filters", fw=800, fz="lg"),
                    dmc.Button("Reset", id="reset-filters", variant="light", color="teal", size="xs"),
                ],
            ),
            dmc.Select(
                id="process-select",
                label="Process",
                value=DEFAULT_PROCESS,
                data=option_data(PROCESS_OPTIONS),
                clearable=False,
                searchable=True,
            ),
            dmc.DatePickerInput(
                id="date-range",
                type="range",
                value=DEFAULT_DATE_RANGE,
                label="Arrival date range",
                clearable=False,
                valueFormat="YYYY-MM-DD",
            ),
            dmc.MultiSelect(
                id="machine-filter",
                label="Machines",
                placeholder="All machines",
                data=[],
                value=[],
                searchable=True,
                clearable=True,
            ),
            dmc.MultiSelect(
                id="operator-filter",
                label="Operators",
                placeholder="All operators",
                data=[],
                value=[],
                searchable=True,
                clearable=True,
            ),
            dmc.MultiSelect(
                id="client-filter",
                label="Clients",
                placeholder="All clients",
                data=[],
                value=[],
                searchable=True,
                clearable=True,
            ),
            dmc.SegmentedControl(
                id="operator-mode",
                value="normalized",
                data=[
                    {"label": "Operator (clean)", "value": "normalized"},
                    {"label": "Operator (raw)", "value": "raw"},
                ],
                fullWidth=True,
                color="teal",
            ),
            dmc.SegmentedControl(
                id="metric-select",
                value="service_min",
                data=[
                    {"label": "Service", "value": "service_min"},
                    {"label": "Wait", "value": "wait_min"},
                    {"label": "Cycle", "value": "cycle_min"},
                ],
                fullWidth=True,
                color="orange",
            ),
            dmc.Divider(label="Outlier Distinction", labelPosition="center"),
            dmc.SegmentedControl(
                id="outlier-method",
                value=FIXED_OUTLIER_METHOD,
                data=[
                    {"label": "IQR (fixed)", "value": FIXED_OUTLIER_METHOD},
                ],
                fullWidth=True,
                color="teal",
                disabled=True,
            ),
            dmc.SegmentedControl(
                id="outlier-view",
                value=FIXED_OUTLIER_VIEW,
                data=[
                    {"label": "Exclude outliers (fixed)", "value": FIXED_OUTLIER_VIEW},
                ],
                fullWidth=True,
                color="orange",
                disabled=True,
            ),
            dmc.Switch(
                id="strict-cleaning",
                checked=True,
                label="Strict preprocessing (remove invalid time rows)",
                color="teal",
            ),
            dmc.SegmentedControl(
                id="time-unit",
                value=FIXED_TIME_UNIT,
                data=[
                    {"label": "Hours (fixed)", "value": FIXED_TIME_UNIT},
                ],
                fullWidth=True,
                color="teal",
                disabled=True,
            ),
            dmc.NumberInput(
                id="servers-count",
                label="Servers for selected process (fixed by process c)",
                value=get_process_server_count(DEFAULT_PROCESS),
                min=1,
                max=20,
                step=1,
                allowDecimal=False,
                disabled=False,
            ),
            dmc.SegmentedControl(
                id="queue-model-mode",
                value="real_empirical",
                data=[
                    {"label": "Simulation with real (empirical) [fixed]", "value": "real_empirical"},
                ],
                fullWidth=True,
                color="teal",
                disabled=True,
            ),
            dmc.Divider(label="Connected Flow (SimPy)", labelPosition="center"),
            dmc.TextInput(
                id="flow-lot-name",
                label="Lot name",
                value="Lote_1",
            ),
            dmc.NumberInput(
                id="flow-lot-pieces",
                label="Pieces",
                value=200,
                min=1,
                max=100000,
                step=10,
                allowDecimal=False,
            ),
            dmc.NumberInput(
                id="flow-lot-repeat",
                label="How many lots with this config",
                value=1,
                min=1,
                max=500,
                step=1,
                allowDecimal=False,
            ),
            dmc.MultiSelect(
                id="flow-lot-processes",
                label="Processes (select in desired order)",
                placeholder="Example: RASPADO, BAUCE, VACIO",
                data=option_data(PROCESS_OPTIONS),
                value=["RASPADO", "BAUCE", "VACIO"] if "RASPADO" in PROCESS_OPTIONS and "BAUCE" in PROCESS_OPTIONS and "VACIO" in PROCESS_OPTIONS else [],
                searchable=True,
                clearable=True,
            ),
            dmc.Group(
                gap="xs",
                children=[
                    dmc.Button("Add lot", id="flow-add-lot-btn", color="teal", variant="filled"),
                    dmc.Button("Clear lots", id="flow-clear-lots-btn", color="gray", variant="light"),
                    dmc.Button("Simulate", id="flow-run-sim-btn", color="indigo", variant="filled"),
                ],
            ),
            dmc.Alert(
                id="flow-cost-output",
                color="indigo",
                variant="light",
                title="Simulation Output",
                children="Simulation output will appear here after you click Simulate.",
            ),
            dmc.Text(
                "Empirical simulation only: arrivals, service, and downtime are sampled from real historical data.",
                c="dimmed",
                fz="xs",
            ),
            dmc.Stack(
                style={"display": "none"},
                children=[
                    dmc.Divider(label="Chart Interactivity", labelPosition="center"),
                    dmc.NumberInput(
                        id="histogram-bins",
                        label="Histogram bins",
                        value=45,
                        min=20,
                        max=100,
                        step=5,
                        allowDecimal=False,
                    ),
                    dmc.NumberInput(
                        id="throughput-roll-days",
                        label="Throughput rolling window (days)",
                        value=3,
                        min=1,
                        max=14,
                        step=1,
                        allowDecimal=False,
                    ),
                    dmc.Switch(
                        id="show-trendline",
                        checked=True,
                        label="Show trendline in pieces chart",
                        color="teal",
                    ),
                    dmc.Switch(
                        id="queue-use-timevarying-arrivals",
                        checked=True,
                        label="Use time-varying arrivals (hour/day profile)",
                        color="teal",
                    ),
                    dmc.Switch(
                        id="queue-use-piece-service",
                        checked=True,
                        label="Use piece-dependent service time",
                        color="teal",
                    ),
                    dmc.Switch(
                        id="queue-use-downtime",
                        checked=True,
                        label="Use machine downtime in SimPy",
                        color="teal",
                    ),
                    dmc.NumberInput(
                        id="holdout-ratio",
                        label="Holdout ratio for validation",
                        value=0.20,
                        min=0.1,
                        max=0.4,
                        step=0.05,
                        decimalScale=2,
                    ),
                ],
            ),
            dmc.Divider(label="Energy Costing", labelPosition="center"),
            dmc.NumberInput(
                id="energy-cost-per-kwh",
                label="Energy cost ($ / kWh)",
                value=0.12,
                min=0.0,
                step=0.01,
                decimalScale=4,
            ),
            dmc.NumberInput(
                id="energy-kwh-per-machine-hour",
                label="Energy per machine-hour (kWh)",
                value=DEFAULT_ENERGY_REFERENCE_KWH,
                min=0.0,
                step=0.1,
                decimalScale=3,
            ),
            dmc.NumberInput(
                id="labor-cost-per-hour",
                label="Labor cost ($ / labor-hour)",
                value=DEFAULT_LABOR_COST_PER_HOUR,
                min=0.0,
                step=0.1,
                decimalScale=3,
            ),
            dmc.Text(
                "Operating schedule fixed: Mon-Fri 10h, Sat 3h, Sun 0h.",
                c="dimmed",
                fz="xs",
            ),
            dmc.Text(
                "Tip: switch to raw operator mode to audit spelling variations in manual logs.",
                c="dimmed",
                fz="xs",
            ),
        ],
    ),
)

main_panel = dmc.Stack(
    gap="lg",
    children=[
        dmc.Paper(
            radius="lg",
            withBorder=True,
            p="lg",
            shadow="xs",
            children=dmc.Stack(
                gap=2,
                children=[
                    dmc.Group(
                        justify="space-between",
                        children=[
                            dmc.Title(APP_TITLE, order=2),
                            dmc.Badge("Multi-process", color="teal", variant="light", size="lg"),
                        ],
                    ),
                    dmc.Text(
                        "Interactive EDA for arrivals, service behavior, waiting risk, and queue-model signals.",
                        c="dimmed",
                    ),
                    dmc.Text(id="scope-text", fz="sm", c="dimmed"),
                ],
            ),
        ),
        dmc.SimpleGrid(
            cols={"base": 1, "sm": 2, "lg": 8},
            spacing="md",
            children=[
                metric_card("Lots", "kpi-lots", "Filtered records"),
                metric_card("Mean Service", "kpi-service", "Selected time unit"),
                metric_card("P90 Wait", "kpi-wait", "Tail waiting risk"),
                metric_card("Arrival Rate", "kpi-rate", "Estimated lots/hour"),
                metric_card("Utilization ρ", "kpi-utilization", "λ·E[S]/c"),
                metric_card("Outlier Rate", "kpi-outliers", "Mild + extreme share"),
                metric_card("Energy Cost", "kpi-energy-cost", "Estimated total in scope"),
                metric_card("Labor Cost", "kpi-labor-cost", "Estimated total in scope"),
            ],
        ),
        dmc.Tabs(
            value="flow",
            color="teal",
            radius="md",
            keepMounted=False,
            children=[
                dmc.TabsList(
                    grow=True,
                    children=[
                        dmc.TabsTab("Flow", value="flow"),
                        dmc.TabsTab("Time Behavior", value="time"),
                        dmc.TabsTab("Capacity", value="capacity"),
                        dmc.TabsTab("Queue Modeling", value="queue"),
                        dmc.TabsTab("Data Quality", value="quality"),
                    ],
                ),
                dmc.TabsPanel(
                    value="flow",
                    pt="md",
                    children=[
                        dmc.Grid(
                            gutter="md",
                            children=[
                                dmc.GridCol(span={"base": 12, "lg": 6}, children=dcc.Graph(id="fig-arrivals-hour", config=GRAPH_CONFIG)),
                                dmc.GridCol(span={"base": 12, "lg": 6}, children=dcc.Graph(id="fig-cumulative", config=GRAPH_CONFIG)),
                                dmc.GridCol(span=12, children=dcc.Graph(id="fig-interarrival", config=GRAPH_CONFIG)),
                            ],
                        )
                    ],
                ),
                dmc.TabsPanel(
                    value="time",
                    pt="md",
                    children=[
                        dmc.Grid(
                            gutter="md",
                            children=[
                                dmc.GridCol(span={"base": 12, "lg": 6}, children=dcc.Graph(id="fig-duration", config=GRAPH_CONFIG)),
                                dmc.GridCol(span={"base": 12, "lg": 6}, children=dcc.Graph(id="fig-machine", config=GRAPH_CONFIG)),
                                dmc.GridCol(span=12, children=dcc.Graph(id="fig-operator", config=GRAPH_CONFIG)),
                            ],
                        )
                    ],
                ),
                dmc.TabsPanel(
                    value="capacity",
                    pt="md",
                    children=[
                        dmc.Grid(
                            gutter="md",
                            children=[
                                dmc.GridCol(span={"base": 12, "lg": 6}, children=dcc.Graph(id="fig-pieces", config=GRAPH_CONFIG)),
                                dmc.GridCol(span={"base": 12, "lg": 6}, children=dcc.Graph(id="fig-throughput", config=GRAPH_CONFIG)),
                                dmc.GridCol(
                                    span=12,
                                    children=dmc.Paper(
                                        withBorder=True,
                                        radius="md",
                                        p="sm",
                                        children=[
                                            dmc.Text("Top delayed lots", fw=700, mb=8),
                                            dash_table.DataTable(
                                                id="top-lots-table",
                                                page_size=8,
                                                sort_action="native",
                                                filter_action="native",
                                                export_format="csv",
                                                style_as_list_view=True,
                                                style_table={"overflowX": "auto"},
                                                style_header={"fontWeight": "700", "backgroundColor": "#f8fafc"},
                                                style_cell={
                                                    "padding": "8px",
                                                    "fontFamily": "IBM Plex Sans, sans-serif",
                                                    "fontSize": "13px",
                                                },
                                            ),
                                        ],
                                    ),
                                ),
                            ],
                        )
                    ],
                ),
                dmc.TabsPanel(
                    value="queue",
                    pt="md",
                    children=[
                        dmc.Stack(
                            gap="md",
                            children=[
                                dmc.SimpleGrid(
                                    cols={"base": 1, "sm": 2, "lg": 4},
                                    spacing="md",
                                    children=[
                                        metric_card("Sim Model", "kpi-queue-model", "Best-fit families"),
                                        metric_card("Sim Mean Wait Wq", "kpi-queue-wq", "Queue waiting (hours)"),
                                        metric_card("Sim P(wait>0)", "kpi-queue-pwait", "Fraction of lots that wait"),
                                        metric_card("Sim Lq", "kpi-queue-lq", "Avg lots in queue"),
                                    ],
                                ),
                                dmc.Alert(
                                    id="queue-recommendation",
                                    color="teal",
                                    variant="light",
                                    title="Recommended Kendall Notation",
                                    children="Waiting for model fit...",
                                ),
                                dmc.Grid(
                                    gutter="md",
                                    children=[
                                        dmc.GridCol(span={"base": 12, "lg": 6}, children=dcc.Graph(id="fig-queue-wait", config=GRAPH_CONFIG)),
                                        dmc.GridCol(span={"base": 12, "lg": 6}, children=dcc.Graph(id="fig-queue-survival", config=GRAPH_CONFIG)),
                                        dmc.GridCol(
                                            span=12,
                                            children=dmc.Paper(
                                                withBorder=True,
                                                radius="md",
                                                p="sm",
                                                children=[
                                                    dmc.Text("Queue simulation summary", fw=700, mb=8),
                                                    dash_table.DataTable(
                                                        id="queue-sim-table",
                                                        sort_action="native",
                                                        filter_action="native",
                                                        page_size=12,
                                                        export_format="csv",
                                                        style_as_list_view=True,
                                                        style_table={"overflowX": "auto"},
                                                        style_header={"fontWeight": "700", "backgroundColor": "#f8fafc"},
                                                        style_cell={
                                                            "padding": "8px",
                                                            "fontFamily": "IBM Plex Sans, sans-serif",
                                                            "fontSize": "13px",
                                                        },
                                                    ),
                                                    dmc.Divider(my="sm", label="Baseline + what-if around process c", labelPosition="left"),
                                                    dash_table.DataTable(
                                                        id="queue-scenario-table",
                                                        sort_action="native",
                                                        filter_action="native",
                                                        page_size=6,
                                                        export_format="csv",
                                                        style_as_list_view=True,
                                                        style_table={"overflowX": "auto"},
                                                        style_header={"fontWeight": "700", "backgroundColor": "#f8fafc"},
                                                        style_cell={
                                                            "padding": "8px",
                                                            "fontFamily": "IBM Plex Sans, sans-serif",
                                                            "fontSize": "13px",
                                                        },
                                                    ),
                                                ],
                                            ),
                                        ),
                                    ],
                                ),
                                dmc.Paper(
                                    withBorder=True,
                                    radius="md",
                                    p="sm",
                                    children=[
                                        dmc.Text("Connected process simulation (user-defined lots)", fw=700, mb=8),
                                        dmc.Text("Configured lots", fw=600, mb=6),
                                        dash_table.DataTable(
                                            id="flow-lot-table",
                                            sort_action="native",
                                            filter_action="native",
                                            page_size=8,
                                            export_format="csv",
                                            style_as_list_view=True,
                                            style_table={"overflowX": "auto"},
                                            style_header={"fontWeight": "700", "backgroundColor": "#f8fafc"},
                                            style_cell={
                                                "padding": "8px",
                                                "fontFamily": "IBM Plex Sans, sans-serif",
                                                "fontSize": "13px",
                                            },
                                            columns=[
                                                {"name": "lot_id", "id": "lot_id"},
                                                {"name": "lot_name", "id": "lot_name"},
                                                {"name": "pieces", "id": "pieces"},
                                                {"name": "steps", "id": "steps"},
                                                {"name": "route", "id": "route"},
                                            ],
                                            data=[],
                                        ),
                                        dmc.Divider(my="sm"),
                                        dmc.Alert(
                                            id="flow-sim-summary",
                                            color="indigo",
                                            variant="light",
                                            title="Flow Simulation Summary",
                                            children="Add lots and click Simulate.",
                                        ),
                                        dcc.Graph(id="fig-flow-stage-wait", config=GRAPH_CONFIG),
                                        dash_table.DataTable(
                                            id="flow-stage-table",
                                            sort_action="native",
                                            filter_action="native",
                                            page_size=20,
                                            export_format="csv",
                                            style_as_list_view=True,
                                            style_table={"overflowX": "auto"},
                                            style_header={"fontWeight": "700", "backgroundColor": "#f8fafc"},
                                            style_cell={
                                                "padding": "8px",
                                                "fontFamily": "IBM Plex Sans, sans-serif",
                                                "fontSize": "13px",
                                            },
                                        ),
                                    ],
                                ),
                            ],
                        )
                    ],
                ),
                dmc.TabsPanel(
                    value="quality",
                    pt="md",
                    children=dmc.Paper(
                        withBorder=True,
                        radius="md",
                        p="sm",
                        children=[
                            dmc.Text("Data-quality diagnostics", fw=700, mb=8),
                            dash_table.DataTable(
                                id="quality-table",
                                sort_action="native",
                                filter_action="native",
                                page_size=16,
                                export_format="csv",
                                style_as_list_view=True,
                                style_table={"overflowX": "auto"},
                                style_header={"fontWeight": "700", "backgroundColor": "#f8fafc"},
                                style_cell={
                                    "padding": "8px",
                                    "fontFamily": "IBM Plex Sans, sans-serif",
                                    "fontSize": "13px",
                                },
                            ),
                            dmc.Divider(my="sm", label="Outlier audit", labelPosition="left"),
                            dash_table.DataTable(
                                id="outlier-table",
                                sort_action="native",
                                filter_action="native",
                                page_size=12,
                                export_format="csv",
                                style_as_list_view=True,
                                style_table={"overflowX": "auto"},
                                style_header={"fontWeight": "700", "backgroundColor": "#f8fafc"},
                                style_cell={
                                    "padding": "8px",
                                    "fontFamily": "IBM Plex Sans, sans-serif",
                                    "fontSize": "13px",
                                },
                            ),
                            dmc.Divider(my="sm", label="Distribution fit audit (AIC/BIC/KS)", labelPosition="left"),
                            dash_table.DataTable(
                                id="dist-fit-table",
                                sort_action="native",
                                filter_action="native",
                                page_size=12,
                                export_format="csv",
                                style_as_list_view=True,
                                style_table={"overflowX": "auto"},
                                style_header={"fontWeight": "700", "backgroundColor": "#f8fafc"},
                                style_cell={
                                    "padding": "8px",
                                    "fontFamily": "IBM Plex Sans, sans-serif",
                                    "fontSize": "13px",
                                },
                            ),
                            dmc.Divider(my="sm", label="Process Line Summary (separated, no mixing)", labelPosition="left"),
                            dash_table.DataTable(
                                id="process-line-table",
                                sort_action="native",
                                filter_action="native",
                                page_size=20,
                                export_format="csv",
                                style_as_list_view=True,
                                style_table={"overflowX": "auto"},
                                style_header={"fontWeight": "700", "backgroundColor": "#f8fafc"},
                                style_cell={
                                    "padding": "8px",
                                    "fontFamily": "IBM Plex Sans, sans-serif",
                                    "fontSize": "13px",
                                },
                            ),
                        ],
                    ),
                ),
            ],
        ),
    ],
)

app.layout = dmc.MantineProvider(
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
                dcc.Store(id="flow-lot-store", data=[]),
                dmc.Alert(
                    DATA_ERROR if DATA_ERROR else f"Data source: {DATA_PATH}",
                    color="red" if DATA_ERROR else "teal",
                    variant="light",
                ),
                dmc.Grid(
                    gutter="md",
                    children=[
                        dmc.GridCol(span={"base": 12, "lg": 3}, children=controls_panel),
                        dmc.GridCol(span={"base": 12, "lg": 9}, children=main_panel),
                    ],
                ),
            ],
        ),
    ),
)


@app.callback(
    Output("machine-filter", "data"),
    Output("machine-filter", "value"),
    Output("operator-filter", "data"),
    Output("operator-filter", "value"),
    Output("client-filter", "data"),
    Output("client-filter", "value"),
    Output("date-range", "value"),
    Output("servers-count", "value"),
    Output("servers-count", "disabled"),
    Output("energy-kwh-per-machine-hour", "value"),
    Input("process-select", "value"),
    Input("operator-mode", "value"),
    Input("reset-filters", "n_clicks"),
    prevent_initial_call=False,
)
def reset_or_refresh_filters(process_value: str, operator_mode: str, _: int | None):
    if DATAFRAME.empty or not process_value:
        return [], [], [], [], [], [], [None, None], 1, False, DEFAULT_ENERGY_KWH_PER_MACHINE_HOUR

    base = DATAFRAME[DATAFRAME["process"] == process_value].copy()
    if base.empty:
        return [], [], [], [], [], [], [None, None], 1, False, DEFAULT_ENERGY_KWH_PER_MACHINE_HOUR

    process_key = (process_value or "").upper().strip()
    official_catalog = PROCESS_MACHINE_CATALOG_RESOLVED.get(process_key, set())
    if official_catalog:
        machine_values = sorted(official_catalog)
    else:
        machine_values = sorted(base["machine_label"].dropna().astype(str).unique().tolist())
    op_col = "operator_std" if operator_mode == "normalized" else "operator_raw"
    operator_values = sorted(base[op_col].dropna().astype(str).unique().tolist())
    client_values = sorted(base["client"].dropna().astype(str).unique().tolist())

    start = base["arrival_time"].min().date().isoformat() if base["arrival_time"].notna().any() else None
    end = base["arrival_time"].max().date().isoformat() if base["arrival_time"].notna().any() else None

    default_servers = get_process_server_count(process_value)
    default_energy_kwh = get_energy_kwh_reference_for_process(process_value)

    return (
        option_data(machine_values),
        [],
        option_data(operator_values),
        [],
        option_data(client_values),
        [],
        [start, end],
        default_servers,
        True,
        default_energy_kwh,
    )


@app.callback(
    Output("scope-text", "children"),
    Output("kpi-lots", "children"),
    Output("kpi-service", "children"),
    Output("kpi-wait", "children"),
    Output("kpi-rate", "children"),
    Output("kpi-utilization", "children"),
    Output("kpi-outliers", "children"),
    Output("kpi-energy-cost", "children"),
    Output("kpi-labor-cost", "children"),
    Output("fig-arrivals-hour", "figure"),
    Output("fig-cumulative", "figure"),
    Output("fig-interarrival", "figure"),
    Output("fig-duration", "figure"),
    Output("fig-machine", "figure"),
    Output("fig-operator", "figure"),
    Output("fig-pieces", "figure"),
    Output("fig-throughput", "figure"),
    Output("kpi-queue-model", "children"),
    Output("kpi-queue-wq", "children"),
    Output("kpi-queue-pwait", "children"),
    Output("kpi-queue-lq", "children"),
    Output("queue-recommendation", "children"),
    Output("fig-queue-wait", "figure"),
    Output("fig-queue-survival", "figure"),
    Output("queue-sim-table", "data"),
    Output("queue-sim-table", "columns"),
    Output("queue-scenario-table", "data"),
    Output("queue-scenario-table", "columns"),
    Output("quality-table", "data"),
    Output("quality-table", "columns"),
    Output("outlier-table", "data"),
    Output("outlier-table", "columns"),
    Output("dist-fit-table", "data"),
    Output("dist-fit-table", "columns"),
    Output("process-line-table", "data"),
    Output("process-line-table", "columns"),
    Output("top-lots-table", "data"),
    Output("top-lots-table", "columns"),
    Input("process-select", "value"),
    Input("date-range", "value"),
    Input("machine-filter", "value"),
    Input("operator-filter", "value"),
    Input("client-filter", "value"),
    Input("operator-mode", "value"),
    Input("metric-select", "value"),
    Input("outlier-method", "value"),
    Input("outlier-view", "value"),
    Input("strict-cleaning", "checked"),
    Input("time-unit", "value"),
    Input("servers-count", "value"),
    Input("queue-model-mode", "value"),
    Input("queue-use-timevarying-arrivals", "checked"),
    Input("queue-use-piece-service", "checked"),
    Input("queue-use-downtime", "checked"),
    Input("holdout-ratio", "value"),
    Input("histogram-bins", "value"),
    Input("throughput-roll-days", "value"),
    Input("show-trendline", "checked"),
    Input("energy-cost-per-kwh", "value"),
    Input("energy-kwh-per-machine-hour", "value"),
    Input("labor-cost-per-hour", "value"),
)
def update_dashboard(
    process_value: str,
    date_range: list[str] | None,
    machine_values: list[str] | None,
    operator_values: list[str] | None,
    client_values: list[str] | None,
    operator_mode: str,
    metric_select: str,
    outlier_method: str,
    outlier_view: str,
    strict_cleaning: bool,
    time_unit: str,
    servers_count: int | float | None,
    queue_model_mode: str | None,
    queue_use_timevarying_arrivals: bool,
    queue_use_piece_service: bool,
    queue_use_downtime: bool,
    holdout_ratio: float | int | None,
    histogram_bins: int | float | None,
    throughput_roll_days: int | float | None,
    show_trendline: bool,
    energy_cost_per_kwh: float | int | None,
    energy_kwh_per_machine_hour: float | int | None,
    labor_cost_per_hour: float | int | None,
):
    # Enforced settings per user request.
    outlier_method = FIXED_OUTLIER_METHOD
    outlier_view = FIXED_OUTLIER_VIEW
    time_unit = FIXED_TIME_UNIT
    queue_model_mode = "real_empirical"
    hist_bins = int(max(20, min(100, int(histogram_bins or 45))))
    roll_days = int(max(1, min(14, int(throughput_roll_days or 3))))
    show_trendline = bool(show_trendline)

    if DATAFRAME.empty:
        fig = empty_figure("No data loaded")
        return (
            "No data scope",
            "0",
            "N/A",
            "N/A",
            "N/A",
            "N/A",
            "N/A",
            "N/A",
            "N/A",
            fig,
            fig,
            fig,
            fig,
            fig,
            fig,
            fig,
            fig,
            "N/A",
            "N/A",
            "N/A",
            "N/A",
            "N/A",
            fig,
            fig,
            [],
            [],
            [],
            [],
            [{"check": "Data source", "value": "Not loaded"}],
            [{"name": "check", "id": "check"}, {"name": "value", "id": "value"}],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )

    dff_raw = filter_frame(
        DATAFRAME,
        process_value,
        date_range,
        machine_values,
        operator_values,
        client_values,
        operator_mode,
    )
    dff_raw = add_quality_flags(dff_raw, process_value=process_value)

    base_count = int(len(dff_raw))
    removed_invalid = int(dff_raw["time_logic_invalid_flag"].sum()) if "time_logic_invalid_flag" in dff_raw.columns else 0
    removed_missing = int(dff_raw["missing_time_flag"].sum()) if "missing_time_flag" in dff_raw.columns else 0

    dff = dff_raw.copy()
    if strict_cleaning:
        dff = dff[(~dff["invalid_time_row"]) & (~dff["missing_time_flag"])].copy()

    dff = apply_outlier_flags(dff, method=outlier_method, metrics=["service_min", "wait_min", "cycle_min"])
    dff_outlier_pool = dff.copy()

    # Fixed behavior: always exclude outliers from analytical outputs.
    dff = dff[~dff["is_outlier_any"]].copy()

    process_key = (process_value or "").upper().strip()
    official_catalog = PROCESS_MACHINE_CATALOG_RESOLVED.get(process_key, set())
    catalog_note = f", machines={','.join(sorted(official_catalog))}" if official_catalog else ""
    scope = (
        (
            f"{process_value} | {len(dff):,}/{base_count:,} rows "
            f"(strict={'on' if strict_cleaning else 'off'}, outliers=exclude-fixed, method=IQR-fixed{catalog_note})"
        )
        if process_value
        else f"All processes | {len(dff):,}/{base_count:,} rows"
    )

    # Boss request: concatenate all processes in separated lines (no mixing).
    process_line_scope = DATAFRAME.copy()
    if date_range and len(date_range) == 2 and date_range[0] and date_range[1]:
        s = pd.to_datetime(date_range[0], errors="coerce")
        e = pd.to_datetime(date_range[1], errors="coerce")
        if pd.notna(s) and pd.notna(e):
            process_line_scope = process_line_scope[
                (process_line_scope["arrival_time"] >= s)
                & (process_line_scope["arrival_time"] <= e + pd.Timedelta(days=1))
            ].copy()

    process_line_rows: list[dict[str, Any]] = []
    if not process_line_scope.empty:
        for proc in sorted(process_line_scope["process"].dropna().unique().tolist()):
            if str(proc).strip().upper() == "UNKNOWN":
                continue
            g = process_line_scope[process_line_scope["process"] == proc].copy()
            if g.empty:
                continue
            machines = sorted(
                {
                    str(m).strip()
                    for m in g["machine_label"].dropna().tolist()
                    if str(m).strip() and str(m).strip().upper() != "UNKNOWN"
                }
            )
            process_line_rows.append(
                {
                    "process": proc,
                    "c_servers": int(get_process_server_count(proc)),
                    "machines_in_process": ",".join(machines) if machines else "UNKNOWN",
                    "rows": int(len(g)),
                    "arrival_start": g["arrival_time"].min().isoformat() if g["arrival_time"].notna().any() else None,
                    "arrival_end": g["arrival_time"].max().isoformat() if g["arrival_time"].notna().any() else None,
                    "mean_service_h": round(float(pd.to_numeric(g["service_hours"], errors="coerce").dropna().mean()), 4)
                    if pd.to_numeric(g["service_hours"], errors="coerce").dropna().shape[0] > 0
                    else None,
                    "mean_wait_h": round(float(pd.to_numeric(g["wait_hours"], errors="coerce").dropna().mean()), 4)
                    if pd.to_numeric(g["wait_hours"], errors="coerce").dropna().shape[0] > 0
                    else None,
                }
            )
    process_line_columns = [
        {"name": c_name, "id": c_name}
        for c_name in [
            "process",
            "c_servers",
            "machines_in_process",
            "rows",
            "arrival_start",
            "arrival_end",
            "mean_service_h",
            "mean_wait_h",
        ]
    ]

    if dff.empty:
        fig = empty_figure("No records for current filters")
        return (
            scope,
            "0",
            "N/A",
            "N/A",
            "N/A",
            "N/A",
            "N/A",
            "N/A",
            "N/A",
            fig,
            fig,
            fig,
            fig,
            fig,
            fig,
            fig,
            fig,
            "N/A",
            "N/A",
            "N/A",
            "N/A",
            "N/A",
            fig,
            fig,
            [],
            [],
            [],
            [],
            [{"check": "Filtered rows", "value": 0}],
            [{"name": "check", "id": "check"}, {"name": "value", "id": "value"}],
            [],
            [],
            [],
            [],
            process_line_rows,
            process_line_columns,
            [],
            [],
        )

    time_divisor = 60.0 if time_unit == "hours" else 1.0
    unit_label = "hours" if time_unit == "hours" else "minutes"

    service_mean_min = float(pd.to_numeric(dff["service_min"], errors="coerce").dropna().mean())
    wait_p90_min = float(pd.to_numeric(dff["wait_min"], errors="coerce").dropna().quantile(0.90))

    arrivals_sorted = dff["arrival_time"].dropna().sort_values()
    arrivals_daily = (
        dff[["arrival_time"]]
        .dropna(subset=["arrival_time"])
        .sort_values("arrival_time")
        .assign(arrival_day=lambda x: x["arrival_time"].dt.date)
    )
    arrivals_daily["interarrival_hours_day"] = (
        arrivals_daily.groupby("arrival_day")["arrival_time"].diff().dt.total_seconds() / 3600.0
    )
    interarrival_hours_day = pd.to_numeric(arrivals_daily["interarrival_hours_day"], errors="coerce").dropna()
    arrival_rate = (
        float(1.0 / interarrival_hours_day.mean())
        if len(interarrival_hours_day) > 0 and interarrival_hours_day.mean() > 0
        else None
    )

    # Pre-outlier-exclusion daily interarrival reference (for workbook reconciliation)
    arrivals_daily_pool = (
        dff_outlier_pool[["arrival_time"]]
        .dropna(subset=["arrival_time"])
        .sort_values("arrival_time")
        .assign(arrival_day=lambda x: x["arrival_time"].dt.date)
    )
    arrivals_daily_pool["interarrival_hours_day"] = (
        arrivals_daily_pool.groupby("arrival_day")["arrival_time"].diff().dt.total_seconds() / 3600.0
    )
    interarrival_hours_day_pool = pd.to_numeric(arrivals_daily_pool["interarrival_hours_day"], errors="coerce").dropna()

    c = get_process_server_count(process_value)
    utilization = None
    if arrival_rate is not None and not np.isnan(service_mean_min):
        utilization = (arrival_rate * (service_mean_min / 60.0)) / c

    process_energy_ref_kwh = get_energy_kwh_reference_for_process(process_value)
    process_energy_ref_key = normalize_process_key(process_value)
    process_energy_ref_loaded = process_energy_ref_key in ENERGY_REFERENCE.get("by_process", {})
    energy_cost_rate = float(max(0.0, float(energy_cost_per_kwh or 0.0)))
    if energy_kwh_per_machine_hour is None:
        energy_kwh_rate = float(max(0.0, process_energy_ref_kwh))
    else:
        energy_kwh_rate = float(max(0.0, float(energy_kwh_per_machine_hour)))
    labor_cost_rate = float(max(0.0, float(labor_cost_per_hour or 0.0)))
    total_machine_hours = float(pd.to_numeric(dff["service_hours"], errors="coerce").dropna().sum())
    total_energy_kwh = float(total_machine_hours * energy_kwh_rate)
    total_energy_cost = float(total_energy_kwh * energy_cost_rate)
    total_labor_hours = float(total_machine_hours)
    total_labor_cost = float(total_labor_hours * labor_cost_rate)
    total_operating_cost = float(total_energy_cost + total_labor_cost)
    scope_start_ts = dff["arrival_time"].min()
    scope_end_ts = dff["arrival_time"].max()
    scheduled_hours_scope = scheduled_hours_between(scope_start_ts, scope_end_ts)
    scheduled_machine_hours_scope = float(scheduled_hours_scope * c) if np.isfinite(scheduled_hours_scope) else float("nan")
    schedule_load_ratio = (
        float(total_machine_hours / scheduled_machine_hours_scope)
        if np.isfinite(scheduled_machine_hours_scope) and scheduled_machine_hours_scope > 0
        else float("nan")
    )

    outlier_rate_pct = float(dff_outlier_pool["is_outlier_any"].mean() * 100.0) if len(dff_outlier_pool) > 0 else np.nan
    mild_count = int((dff_outlier_pool["outlier_class_any"] == "mild").sum())
    extreme_count = int((dff_outlier_pool["outlier_class_any"] == "extreme").sum())

    metric_map = {
        "service_min": "service_min",
        "wait_min": "wait_min",
        "cycle_min": "cycle_min",
    }
    metric_col = metric_map.get(metric_select, "service_min")

    # Flow charts
    by_hour = dff.dropna(subset=["arrival_hour"]).groupby("arrival_hour").size().reset_index(name="lots")
    fig_arrivals_hour = px.bar(by_hour, x="arrival_hour", y="lots", color_discrete_sequence=[PALETTE["accent"]])
    fig_arrivals_hour.update_xaxes(dtick=1)
    fig_arrivals_hour = style_figure(fig_arrivals_hour, "Arrivals by hour of day")

    cumulative = arrivals_sorted.to_frame(name="arrival_time")
    cumulative["cum_lots"] = np.arange(1, len(cumulative) + 1)
    fig_cumulative = px.line(cumulative, x="arrival_time", y="cum_lots", color_discrete_sequence=[PALETTE["warm"]])
    fig_cumulative = style_figure(fig_cumulative, "Cumulative arrivals")

    interarrival_fit_rows, interarrival_fit_notes = fit_distribution_candidates(
        interarrival_hours_day,
        dataset_label="interarrival_hours_day",
    )
    interarrival_title = "Interarrival distribution (density + fitted families)"
    if interarrival_fit_rows:
        interarrival_title = f"{interarrival_title} | best={interarrival_fit_rows[0]['family']}"
    fig_interarrival = histogram_with_fits(
        interarrival_hours_day,
        interarrival_fit_rows,
        x_label="Interarrival (hours, grouped by day)",
        title=interarrival_title,
        nbins=hist_bins,
    )

    # Time behavior charts
    metric_series = pd.to_numeric(dff[metric_col], errors="coerce") / time_divisor
    metric_name_map = {
        "service_min": "Service",
        "wait_min": "Wait",
        "cycle_min": "Cycle",
    }
    metric_label = metric_name_map.get(metric_col, metric_col)
    metric_fit_rows, metric_fit_notes = fit_distribution_candidates(
        metric_series,
        dataset_label=f"{metric_col}_{unit_label}",
    )
    duration_title = f"{metric_label} distribution (density + fitted families)"
    if metric_fit_rows:
        duration_title = f"{duration_title} | best={metric_fit_rows[0]['family']}"
    fig_duration = histogram_with_fits(
        metric_series,
        metric_fit_rows,
        x_label=f"{metric_label} ({unit_label})",
        title=duration_title,
        nbins=hist_bins,
    )

    box_machine_df = dff.copy()
    box_machine_df["service_unit"] = pd.to_numeric(box_machine_df["service_min"], errors="coerce") / time_divisor
    fig_machine = px.box(
        box_machine_df,
        x="machine_label",
        y="service_unit",
        points="outliers",
        color="service_min_outlier_class",
        category_orders={"service_min_outlier_class": OUTLIER_CLASS_ORDER},
        color_discrete_map=OUTLIER_COLOR_MAP,
    )
    fig_machine.update_layout(showlegend=True)
    fig_machine.update_yaxes(title=f"Service ({unit_label})")
    fig_machine = style_figure(fig_machine, "Service by machine (outlier class)")

    op_summary = dff.copy()
    top_ops = op_summary["operator_view"].value_counts().head(10).index
    op_summary = op_summary[op_summary["operator_view"].isin(top_ops)].copy()
    op_summary["wait_unit"] = pd.to_numeric(op_summary["wait_min"], errors="coerce") / time_divisor
    fig_operator = px.box(
        op_summary,
        x="operator_view",
        y="wait_unit",
        points="outliers",
        color="wait_min_outlier_class",
        category_orders={"wait_min_outlier_class": OUTLIER_CLASS_ORDER},
        color_discrete_map=OUTLIER_COLOR_MAP,
    )
    fig_operator.update_layout(showlegend=True)
    fig_operator.update_yaxes(title=f"Wait ({unit_label})")
    fig_operator = style_figure(fig_operator, "Wait by operator (outlier class)")

    # Capacity charts
    scatter_df = dff.copy()
    scatter_df["service_unit"] = pd.to_numeric(scatter_df["service_min"], errors="coerce") / time_divisor
    fig_pieces = px.scatter(
        scatter_df,
        x="pieces",
        y="service_unit",
        color="service_min_outlier_class",
        trendline="ols" if show_trendline else None,
        labels={"service_unit": f"Service ({unit_label})", "pieces": "Pieces"},
        category_orders={"service_min_outlier_class": OUTLIER_CLASS_ORDER},
        color_discrete_map=OUTLIER_COLOR_MAP,
    )
    fig_pieces = style_figure(fig_pieces, "Pieces vs service (outlier class)")

    daily = (
        dff.dropna(subset=["arrival_time"])
        .assign(day=lambda x: x["arrival_time"].dt.date)
        .groupby("day")
        .agg(lots=("lot_id", "count"), total_service_hours=("service_hours", "sum"))
        .reset_index()
    )
    daily = daily.sort_values("day")
    daily["lots_roll"] = daily["lots"].rolling(roll_days, min_periods=1).mean()
    daily["service_roll"] = daily["total_service_hours"].rolling(roll_days, min_periods=1).mean()

    fig_throughput = go.Figure()
    fig_throughput.add_bar(x=daily["day"], y=daily["lots"], name="Lots/day", marker_color="#0891b2")
    fig_throughput.add_scatter(
        x=daily["day"],
        y=daily["total_service_hours"],
        name="Total service hours/day",
        mode="lines+markers",
        marker_color="#d97706",
        yaxis="y2",
    )
    fig_throughput.add_scatter(
        x=daily["day"],
        y=daily["lots_roll"],
        name=f"Rolling lots ({roll_days}d)",
        mode="lines",
        line={"color": "#0f172a", "dash": "dash", "width": 2.0},
    )
    fig_throughput.add_scatter(
        x=daily["day"],
        y=daily["service_roll"],
        name=f"Rolling service ({roll_days}d)",
        mode="lines",
        line={"color": "#be123c", "dash": "dot", "width": 1.8},
        yaxis="y2",
    )
    fig_throughput.update_layout(
        yaxis={"title": "Lots"},
        yaxis2={"title": "Service hours", "overlaying": "y", "side": "right"},
    )
    fig_throughput = style_figure(fig_throughput, "Daily throughput and workload")

    # Queue modeling (SimPy + data-engineering calibrations)
    holdout_ratio_value = float(holdout_ratio) if holdout_ratio is not None else 0.20
    holdout_ratio_value = float(max(0.10, min(0.40, holdout_ratio_value)))
    train_df, holdout_df = split_train_holdout_by_arrival(dff, holdout_ratio=holdout_ratio_value)
    if train_df.empty:
        train_df = dff.copy()
        holdout_df = dff.iloc[0:0].copy()

    train_arrivals_daily = (
        train_df[["arrival_time"]]
        .dropna(subset=["arrival_time"])
        .sort_values("arrival_time")
        .assign(arrival_day=lambda x: x["arrival_time"].dt.date)
    )
    train_arrivals_daily["interarrival_hours_day"] = (
        train_arrivals_daily.groupby("arrival_day")["arrival_time"].diff().dt.total_seconds() / 3600.0
    )
    train_interarrival_hours_day = pd.to_numeric(train_arrivals_daily["interarrival_hours_day"], errors="coerce").dropna()
    if train_interarrival_hours_day.empty:
        train_interarrival_hours_day = interarrival_hours_day.copy()

    service_hours_series_train = pd.to_numeric(train_df["service_hours"], errors="coerce")
    service_hours_series_train = service_hours_series_train[service_hours_series_train > 0]
    if service_hours_series_train.empty:
        service_hours_series_train = pd.to_numeric(dff["service_hours"], errors="coerce")
        service_hours_series_train = service_hours_series_train[service_hours_series_train > 0]

    interarrival_fit_rows_queue, interarrival_fit_notes_queue = fit_distribution_candidates(
        train_interarrival_hours_day,
        dataset_label="interarrival_hours_day_train",
    )
    service_fit_rows_queue, service_fit_notes_queue = fit_distribution_candidates(
        service_hours_series_train,
        dataset_label="service_hours_train",
    )

    queue_mode = (queue_model_mode or "best_fit").strip().lower()
    if queue_mode == "mg_exp_norm":
        selected_arrival_fit = pick_fit_by_family_key(interarrival_fit_rows_queue, "expon")
        selected_service_fit = pick_fit_by_family_key(service_fit_rows_queue, "norm")
    elif queue_mode == "real_empirical":
        selected_arrival_fit = None
        selected_service_fit = None
    else:
        selected_arrival_fit = interarrival_fit_rows_queue[0] if interarrival_fit_rows_queue else None
        selected_service_fit = service_fit_rows_queue[0] if service_fit_rows_queue else None

    if selected_arrival_fit is None and queue_mode != "real_empirical":
        selected_arrival_fit = interarrival_fit_rows_queue[0] if interarrival_fit_rows_queue else None
    if selected_service_fit is None and queue_mode != "real_empirical":
        selected_service_fit = service_fit_rows_queue[0] if service_fit_rows_queue else None

    sim_n = int(max(2000, min(10000, len(train_df) * 12)))
    rng = np.random.default_rng(20260528)

    if bool(queue_use_timevarying_arrivals):
        sim_interarrival = generate_time_varying_interarrival(train_df["arrival_time"], sim_n, rng)
        arrival_sampler_label = "Time-varying empirical (hour/day profile)"
    else:
        sim_interarrival = sample_from_distribution_fit(selected_arrival_fit, train_interarrival_hours_day, sim_n, rng)
        arrival_sampler_label = selected_arrival_fit["family"] if selected_arrival_fit else "Empirical bootstrap"

    piece_profile = build_piece_service_profile(train_df["pieces"], train_df["service_hours"])
    if bool(queue_use_piece_service):
        sim_service = sample_piece_dependent_service(piece_profile, sim_n, rng)
        service_sampler_label = "Piece-dependent empirical"
    else:
        sim_service = sample_from_distribution_fit(selected_service_fit, service_hours_series_train, sim_n, rng)
        service_sampler_label = selected_service_fit["family"] if selected_service_fit else "Empirical bootstrap"

    downtime_stats: dict[str, float] = {
        "downtime_rows": 0.0,
        "downtime_positive_rows": 0.0,
        "p_downtime": 0.0,
        "downtime_mean_h": 0.0,
    }
    downtime_draw = None
    if bool(queue_use_downtime):
        downtime_stats, downtime_draw = make_downtime_sampler(train_df.get("downtime_hours_source", pd.Series(dtype=float)), rng)

    queue_sim = simulate_gigc_queue(sim_interarrival, sim_service, c, downtime_draw=downtime_draw)
    observed_wait_h = pd.to_numeric(dff["wait_hours"], errors="coerce").dropna()
    observed_wait_h = observed_wait_h[observed_wait_h >= 0]
    observed_mean_wait_h = float(observed_wait_h.mean()) if not observed_wait_h.empty else np.nan

    allen = allen_cunneen_approx(
        train_interarrival_hours_day,
        service_hours_series_train,
        c,
    )
    sim_wq = float(queue_sim["mean_wait_h"]) if np.isfinite(queue_sim["mean_wait_h"]) else np.nan
    ac_wq = float(allen["wq_h"]) if np.isfinite(allen["wq_h"]) else np.nan
    sim_vs_ac_wq_gap = sim_wq - ac_wq if np.isfinite(sim_wq) and np.isfinite(ac_wq) else np.nan
    sim_vs_ac_wq_rel = (
        abs(sim_vs_ac_wq_gap) / abs(sim_wq)
        if np.isfinite(sim_vs_ac_wq_gap) and np.isfinite(sim_wq) and not np.isclose(sim_wq, 0.0)
        else np.nan
    )
    sim_vs_obs_wq_gap = sim_wq - observed_mean_wait_h if np.isfinite(sim_wq) and np.isfinite(observed_mean_wait_h) else np.nan
    ac_vs_obs_wq_gap = ac_wq - observed_mean_wait_h if np.isfinite(ac_wq) and np.isfinite(observed_mean_wait_h) else np.nan

    arrival_family_name = selected_arrival_fit["family"] if selected_arrival_fit else "Empirical"
    service_family_name = selected_service_fit["family"] if selected_service_fit else "Empirical"
    if queue_mode == "mg_exp_norm":
        queue_model_label = f"M/G/{c} (Exp+Normal, SimPy)"
    elif queue_mode == "real_empirical":
        queue_model_label = f"G/G/{c} (Empirical arrivals+service, SimPy)"
    else:
        queue_model_label = f"{arrival_family_name}/{service_family_name}/{c} (SimPy)"

    # Holdout validation (time split): train earliest data, validate on latest data.
    holdout_wait_h = pd.to_numeric(holdout_df.get("wait_hours", pd.Series(dtype=float)), errors="coerce").dropna()
    holdout_wait_h = holdout_wait_h[holdout_wait_h >= 0]
    holdout_n = int(len(holdout_wait_h))
    holdout_metrics = {
        "n": holdout_n,
        "obs_mean_wait_h": float(holdout_wait_h.mean()) if holdout_n > 0 else np.nan,
        "obs_p90_wait_h": float(np.quantile(holdout_wait_h, 0.90)) if holdout_n > 0 else np.nan,
        "obs_p_wait_2h": float(np.mean(holdout_wait_h > 2.0)) if holdout_n > 0 else np.nan,
        "sim_mean_wait_h": np.nan,
        "sim_p90_wait_h": np.nan,
        "sim_p_wait_2h": np.nan,
        "err_mean_wait_h": np.nan,
        "err_p90_wait_h": np.nan,
        "err_p_wait_2h_pp": np.nan,
    }
    if holdout_n > 0:
        holdout_ia = generate_time_varying_interarrival(holdout_df["arrival_time"], holdout_n, rng)
        holdout_service = sample_piece_dependent_service(piece_profile, holdout_n, rng)
        holdout_downtime_draw = None
        if bool(queue_use_downtime):
            _holdout_dt_stats, holdout_downtime_draw = make_downtime_sampler(
                train_df.get("downtime_hours_source", pd.Series(dtype=float)),
                rng,
            )
        holdout_sim = simulate_gigc_queue(holdout_ia, holdout_service, c, downtime_draw=holdout_downtime_draw)
        holdout_metrics["sim_mean_wait_h"] = float(holdout_sim["mean_wait_h"]) if np.isfinite(holdout_sim["mean_wait_h"]) else np.nan
        holdout_metrics["sim_p90_wait_h"] = float(holdout_sim["p90_wait_h"]) if np.isfinite(holdout_sim["p90_wait_h"]) else np.nan
        holdout_metrics["sim_p_wait_2h"] = float(holdout_sim["p_wait_2h"]) if np.isfinite(holdout_sim["p_wait_2h"]) else np.nan
        holdout_metrics["err_mean_wait_h"] = holdout_metrics["sim_mean_wait_h"] - holdout_metrics["obs_mean_wait_h"]
        holdout_metrics["err_p90_wait_h"] = holdout_metrics["sim_p90_wait_h"] - holdout_metrics["obs_p90_wait_h"]
        holdout_metrics["err_p_wait_2h_pp"] = (holdout_metrics["sim_p_wait_2h"] - holdout_metrics["obs_p_wait_2h"]) * 100.0

    # Baseline + what-if server scenarios around the process-specific c.
    scenario_rows: list[dict[str, Any]] = []
    scenario_candidates = sorted({max(1, c - 1), c, c + 1})
    for c_s in scenario_candidates:
        sc_rng = np.random.default_rng(20260528 + c_s)
        sc_ia = (
            generate_time_varying_interarrival(train_df["arrival_time"], sim_n, sc_rng)
            if bool(queue_use_timevarying_arrivals)
            else sample_from_distribution_fit(selected_arrival_fit, train_interarrival_hours_day, sim_n, sc_rng)
        )
        sc_service = (
            sample_piece_dependent_service(piece_profile, sim_n, sc_rng)
            if bool(queue_use_piece_service)
            else sample_from_distribution_fit(selected_service_fit, service_hours_series_train, sim_n, sc_rng)
        )
        sc_dt_stats, sc_downtime_draw = ({**downtime_stats}, None)
        if bool(queue_use_downtime):
            sc_dt_stats, sc_downtime_draw = make_downtime_sampler(train_df.get("downtime_hours_source", pd.Series(dtype=float)), sc_rng)
        sc_sim = simulate_gigc_queue(sc_ia, sc_service, c_s, downtime_draw=sc_downtime_draw)
        scenario_rows.append(
            {
                "scenario": "Baseline (production)" if c_s == c else f"What-if c={c_s}",
                "servers": c_s,
                "rho": round(float(sc_sim["rho"]), 4) if np.isfinite(sc_sim["rho"]) else None,
                "wq_h": round(float(sc_sim["mean_wait_h"]), 4) if np.isfinite(sc_sim["mean_wait_h"]) else None,
                "p_wait_gt_2h_pct": round(float(sc_sim["p_wait_2h"]) * 100.0, 2) if np.isfinite(sc_sim["p_wait_2h"]) else None,
                "lq": round(float(sc_sim["lq"]), 4) if np.isfinite(sc_sim["lq"]) else None,
                "ws_h": round(float(sc_sim["mean_sojourn_h"]), 4) if np.isfinite(sc_sim["mean_sojourn_h"]) else None,
                "downtime_event_pct": round(float(sc_sim["p_downtime_event"]) * 100.0, 2) if np.isfinite(sc_sim["p_downtime_event"]) else None,
            }
        )
    queue_scenario_columns = [{"name": c_name, "id": c_name} for c_name in ["scenario", "servers", "rho", "wq_h", "p_wait_gt_2h_pct", "lq", "ws_h", "downtime_event_pct"]]

    # Boss-ready recommendation text based on train-only AIC ranking.
    a_best = interarrival_fit_rows_queue[0] if interarrival_fit_rows_queue else None
    s_best = service_fit_rows_queue[0] if service_fit_rows_queue else None
    s_second = service_fit_rows_queue[1] if len(service_fit_rows_queue) > 1 else None
    a_exp = pick_fit_by_family_key(interarrival_fit_rows_queue, "expon")
    if a_best and s_best:
        k_arr = "M" if a_best.get("family_key") == "expon" else "G"
        k_srv = "M" if s_best.get("family_key") == "expon" else "G"
        recommended_kendall = f"{k_arr}/{k_srv}/{c}/∞/∞/FIFO"
        explicit_model = f"{a_best['family']}/{s_best['family']}/{c}"
        arrival_reason = (
            f"Arrivals(train): best AIC is {a_best['family']} ({a_best['aic']:.2f}); "
            + (
                f"Exponential AIC={a_exp['aic']:.2f}, ΔAIC={a_exp.get('delta_aic', np.nan):.2f}. "
                if a_exp is not None and a_exp.get("aic") is not None
                else ""
            )
        )
        service_reason = (
            f"Service(train): best AIC is {s_best['family']} ({s_best['aic']:.2f}); "
            + (f"next best is {s_second['family']} (ΔAIC={s_second.get('delta_aic', np.nan):.2f}). " if s_second is not None else "")
        )
        queue_recommendation = (
            f"Recommended Kendall: {recommended_kendall} "
            f"(explicit fit: {explicit_model}). "
            f"{arrival_reason}{service_reason}"
        )
    else:
        queue_recommendation = "Recommended Kendall unavailable: missing distribution fit results."

    if np.isfinite(sim_vs_ac_wq_gap):
        queue_recommendation += (
            f" Allen-Cunneen check (Wq): AC={safe_number(ac_wq, 3)} h vs "
            f"SimPy={safe_number(sim_wq, 3)} h, gap={safe_number(sim_vs_ac_wq_gap, 3)} h."
        )
    if holdout_metrics["n"] > 0:
        queue_recommendation += (
            f" Holdout validation ({holdout_metrics['n']} latest rows): "
            f"ΔmeanWq={safe_number(holdout_metrics['err_mean_wait_h'], 3)} h, "
            f"ΔP90={safe_number(holdout_metrics['err_p90_wait_h'], 3)} h, "
            f"ΔP(wait>2h)={safe_number(holdout_metrics['err_p_wait_2h_pp'], 2)} pp."
        )
    if scenario_rows:
        best_row = min(
            [r for r in scenario_rows if r.get("wq_h") is not None],
            key=lambda r: float(r["wq_h"]),
            default=None,
        )
        if best_row is not None:
            queue_recommendation += (
                f" Scenario check: lowest Wq at c={best_row['servers']} "
                f"(Wq={safe_number(best_row['wq_h'], 3)} h, rho={safe_number(best_row['rho'], 3)})."
            )

    kpi_queue_model = queue_model_label
    kpi_queue_wq = f"{safe_number(queue_sim['mean_wait_h'], 3)} h"
    kpi_queue_pwait = f"{safe_number(queue_sim['p_wait'] * 100.0, 1)}%"
    kpi_queue_lq = f"{safe_number(queue_sim['lq'], 3)}"

    fig_queue_wait = go.Figure()
    fig_queue_wait.add_histogram(
        x=queue_sim["wait_h"],
        name="Simulated wait",
        histnorm="probability density",
        nbinsx=hist_bins,
        marker_color=PALETTE["accent"],
        opacity=0.72,
    )
    if not observed_wait_h.empty:
        fig_queue_wait.add_histogram(
            x=observed_wait_h,
            name="Observed wait",
            histnorm="probability density",
            nbinsx=hist_bins,
            marker_color=PALETTE["warm"],
            opacity=0.45,
        )
    fig_queue_wait.update_layout(barmode="overlay")
    fig_queue_wait.update_xaxes(title="Wait (hours)")
    fig_queue_wait.update_yaxes(title="Density")
    fig_queue_wait = style_figure(fig_queue_wait, "Queue waiting distribution: simulated vs observed")

    sim_wait_sorted = np.sort(queue_sim["wait_h"])
    sim_survival = 1.0 - (np.arange(1, len(sim_wait_sorted) + 1) / len(sim_wait_sorted)) if len(sim_wait_sorted) > 0 else np.array([])
    fig_queue_survival = go.Figure()
    if len(sim_wait_sorted) > 0:
        fig_queue_survival.add_scatter(
            x=sim_wait_sorted,
            y=sim_survival,
            mode="lines",
            name="Simulated survival",
            line={"color": PALETTE["accent"], "width": 2.4},
        )
    if not observed_wait_h.empty:
        obs_wait_sorted = np.sort(observed_wait_h.to_numpy())
        obs_survival = 1.0 - (np.arange(1, len(obs_wait_sorted) + 1) / len(obs_wait_sorted))
        fig_queue_survival.add_scatter(
            x=obs_wait_sorted,
            y=obs_survival,
            mode="lines",
            name="Observed survival",
            line={"color": PALETTE["warm"], "width": 2.1, "dash": "dash"},
        )
    fig_queue_survival.update_xaxes(title="Wait threshold (hours)")
    fig_queue_survival.update_yaxes(title="P(Wait > x)", rangemode="tozero")
    fig_queue_survival = style_figure(fig_queue_survival, "Queue tail risk curve")

    queue_sim_rows = [
        {"metric": "Model used", "value": queue_model_label},
        {"metric": "Simulation engine", "value": queue_sim.get("sim_engine", "unknown")},
        {"metric": "Arrival generator", "value": arrival_sampler_label},
        {"metric": "Service generator", "value": service_sampler_label},
        {"metric": "Downtime enabled", "value": bool(queue_use_downtime)},
        {"metric": "Energy reference loaded for process", "value": bool(process_energy_ref_loaded)},
        {"metric": "Energy reference kWh/machine-hour", "value": safe_number(process_energy_ref_kwh, 4)},
        {"metric": "Energy cost rate ($/kWh)", "value": safe_number(energy_cost_rate, 4)},
        {"metric": "Energy rate (kWh per machine-hour)", "value": safe_number(energy_kwh_rate, 4)},
        {"metric": "Total machine-hours (filtered)", "value": safe_number(total_machine_hours, 4)},
        {"metric": "Plant schedule (h/week)", "value": safe_number(WEEKLY_SCHEDULE_HOURS, 2)},
        {"metric": "Scheduled open hours in scope", "value": safe_number(scheduled_hours_scope, 4)},
        {"metric": "Scheduled machine-hours in scope (open_hours*c)", "value": safe_number(scheduled_machine_hours_scope, 4)},
        {"metric": "Observed load vs schedule", "value": f"{safe_number(schedule_load_ratio * 100.0, 2)}%" if np.isfinite(schedule_load_ratio) else "N/A"},
        {"metric": "Estimated total energy (kWh)", "value": safe_number(total_energy_kwh, 4)},
        {"metric": "Estimated total energy cost ($)", "value": safe_number(total_energy_cost, 2)},
        {"metric": "Labor cost rate ($/labor-hour)", "value": safe_number(labor_cost_rate, 4)},
        {"metric": "Total labor-hours (filtered)", "value": safe_number(total_labor_hours, 4)},
        {"metric": "Estimated total labor cost ($)", "value": safe_number(total_labor_cost, 2)},
        {"metric": "Estimated total operating cost ($)", "value": safe_number(total_operating_cost, 2)},
        {"metric": "Downtime event rate in data", "value": f"{safe_number(downtime_stats.get('p_downtime', np.nan) * 100.0, 2)}%"},
        {"metric": "Downtime mean when event (h)", "value": safe_number(downtime_stats.get("downtime_mean_h", np.nan), 4)},
        {"metric": "Simulated jobs", "value": int(queue_sim["n"])},
        {"metric": "Servers c (process fixed)", "value": c},
        {"metric": "Arrival rate λ (lots/h)", "value": safe_number(queue_sim["lambda_h"], 4)},
        {"metric": "Service rate μ per server (lots/h)", "value": safe_number(queue_sim["mu_h"], 4)},
        {"metric": "Utilization ρ (sim)", "value": safe_number(queue_sim["rho"], 4)},
        {"metric": "Mean wait Wq (h)", "value": safe_number(queue_sim["mean_wait_h"], 4)},
        {"metric": "P(wait > 0)", "value": f"{safe_number(queue_sim['p_wait'] * 100.0, 2)}%"},
        {"metric": "P(wait > 1h)", "value": f"{safe_number(queue_sim['p_wait_1h'] * 100.0, 2)}%"},
        {"metric": "P(wait > 2h)", "value": f"{safe_number(queue_sim['p_wait_2h'] * 100.0, 2)}%"},
        {"metric": "P90 wait (h)", "value": safe_number(queue_sim["p90_wait_h"], 4)},
        {"metric": "Mean system time Ws (h)", "value": safe_number(queue_sim["mean_sojourn_h"], 4)},
        {"metric": "Lq (avg lots in queue)", "value": safe_number(queue_sim["lq"], 4)},
        {"metric": "Ls (avg lots in system)", "value": safe_number(queue_sim["ls"], 4)},
        {"metric": "Total simulated downtime (h)", "value": safe_number(queue_sim["total_downtime_h"], 4)},
        {"metric": "Sim downtime event rate", "value": f"{safe_number(queue_sim['p_downtime_event'] * 100.0, 2)}%" if np.isfinite(queue_sim.get("p_downtime_event", np.nan)) else "N/A"},
        {"metric": "Allen-Cunneen Wq (h)", "value": safe_number(allen["wq_h"], 4)},
        {"metric": "Allen-Cunneen Ws (h)", "value": safe_number(allen["ws_h"], 4)},
        {"metric": "Allen-Cunneen Lq", "value": safe_number(allen["lq"], 4)},
        {"metric": "Allen-Cunneen Ls", "value": safe_number(allen["ls"], 4)},
        {"metric": "Allen-Cunneen ρ", "value": safe_number(allen["rho"], 4)},
        {"metric": "Allen-Cunneen Ca²", "value": safe_number(allen["ca2"], 4)},
        {"metric": "Allen-Cunneen Cs²", "value": safe_number(allen["cs2"], 4)},
        {"metric": "Gap (SimPy Wq - Allen Wq) h", "value": safe_number(sim_vs_ac_wq_gap, 4)},
        {"metric": "Relative gap |Sim-Allen| / Sim", "value": f"{safe_number(sim_vs_ac_wq_rel * 100.0, 2)}%" if np.isfinite(sim_vs_ac_wq_rel) else "N/A"},
        {
            "metric": "Observed mean wait (h)",
            "value": safe_number(observed_mean_wait_h, 4) if np.isfinite(observed_mean_wait_h) else "N/A",
        },
        {"metric": "Gap (SimPy Wq - Observed mean) h", "value": safe_number(sim_vs_obs_wq_gap, 4)},
        {"metric": "Gap (Allen Wq - Observed mean) h", "value": safe_number(ac_vs_obs_wq_gap, 4)},
        {"metric": "Holdout rows", "value": holdout_metrics["n"]},
        {"metric": "Holdout obs mean Wq (h)", "value": safe_number(holdout_metrics["obs_mean_wait_h"], 4)},
        {"metric": "Holdout sim mean Wq (h)", "value": safe_number(holdout_metrics["sim_mean_wait_h"], 4)},
        {"metric": "Holdout error mean Wq (h)", "value": safe_number(holdout_metrics["err_mean_wait_h"], 4)},
        {"metric": "Holdout obs P90 Wq (h)", "value": safe_number(holdout_metrics["obs_p90_wait_h"], 4)},
        {"metric": "Holdout sim P90 Wq (h)", "value": safe_number(holdout_metrics["sim_p90_wait_h"], 4)},
        {"metric": "Holdout error P90 Wq (h)", "value": safe_number(holdout_metrics["err_p90_wait_h"], 4)},
        {"metric": "Holdout obs P(wait>2h)", "value": f"{safe_number(holdout_metrics['obs_p_wait_2h'] * 100.0, 2)}%" if np.isfinite(holdout_metrics["obs_p_wait_2h"]) else "N/A"},
        {"metric": "Holdout sim P(wait>2h)", "value": f"{safe_number(holdout_metrics['sim_p_wait_2h'] * 100.0, 2)}%" if np.isfinite(holdout_metrics["sim_p_wait_2h"]) else "N/A"},
        {"metric": "Holdout error P(wait>2h) pp", "value": safe_number(holdout_metrics["err_p_wait_2h_pp"], 3)},
    ]
    queue_sim_columns = [{"name": c_name, "id": c_name} for c_name in ["metric", "value"]]

    # Tables
    duplicate_visit_count = int(
        dff_raw.duplicated(subset=["lot_id", "arrival_time", "start_time", "end_time", "machine_label"], keep=False).sum()
    )

    quality_rows = [
        {"check": "Rows before preprocessing", "value": base_count},
        {"check": "Rows after preprocessing + outlier view", "value": int(len(dff))},
        {"check": "Strict preprocessing enabled", "value": bool(strict_cleaning)},
        {"check": "Rows with missing time fields", "value": removed_missing},
        {"check": "Rows with invalid time logic", "value": removed_invalid},
        {
            "check": "Rows outside official machine catalog",
            "value": int(dff_raw["machine_catalog_invalid_flag"].sum()),
        },
        {
            "check": "Process c (servers, fixed by process)",
            "value": int(c),
        },
        {
            "check": "Process machine catalog",
            "value": ",".join(sorted(official_catalog)) if official_catalog else "N/A",
        },
        {"check": "Potential duplicate visits", "value": duplicate_visit_count},
        {
            "check": "Start before arrival",
            "value": int(dff_raw["start_before_arrival_flag"].sum()),
        },
        {
            "check": "End before start",
            "value": int(dff_raw["end_before_start_flag"].sum()),
        },
        {
            "check": "Service <= 0 (FECHA FINAL - FECHA INICIAL)",
            "value": int(dff_raw["service_nonpositive_flag"].sum()),
        },
        {
            "check": "Outlier method",
            "value": outlier_method,
        },
        {
            "check": "Outlier view mode",
            "value": outlier_view,
        },
        {
            "check": "Outlier mild count",
            "value": mild_count,
        },
        {
            "check": "Outlier extreme count",
            "value": extreme_count,
        },
        {
            "check": "Service mean hours (app, pre-outlier)",
            "value": round(float(pd.to_numeric(dff_outlier_pool["service_hours"], errors="coerce").dropna().mean()), 6)
            if len(dff_outlier_pool) > 0
            else None,
        },
        {
            "check": "Effective mean hours source (BITACORA)",
            "value": round(float(pd.to_numeric(dff_outlier_pool["effective_hours_source"], errors="coerce").dropna().mean()), 6)
            if len(dff_outlier_pool) > 0
            else None,
        },
        {
            "check": "HORAS_LOTE mean hours (sheet)",
            "value": round(float(REFERENCE_SHEETS["horas_lote_mean"]), 6) if REFERENCE_SHEETS["horas_lote_mean"] is not None else None,
        },
        {
            "check": "Interarrival mean hours/day (app, pre-outlier)",
            "value": round(float(interarrival_hours_day_pool.mean()), 6) if len(interarrival_hours_day_pool) > 0 else None,
        },
        {
            "check": "INTERARRIBOS mean hours (sheet)",
            "value": round(float(REFERENCE_SHEETS["interarribos_mean"]), 6) if REFERENCE_SHEETS["interarribos_mean"] is not None else None,
        },
        {"check": "Plant schedule Mon-Fri hours/day", "value": 10.0},
        {"check": "Plant schedule Saturday hours/day", "value": 3.0},
        {"check": "Plant schedule Sunday hours/day", "value": 0.0},
        {"check": "Plant schedule hours/week", "value": round(float(WEEKLY_SCHEDULE_HOURS), 6)},
        {"check": "Scheduled open hours in scope", "value": round(float(scheduled_hours_scope), 6) if np.isfinite(scheduled_hours_scope) else None},
        {
            "check": "Scheduled machine-hours in scope (open_hours*c)",
            "value": round(float(scheduled_machine_hours_scope), 6) if np.isfinite(scheduled_machine_hours_scope) else None,
        },
        {
            "check": "Observed load vs schedule (%)",
            "value": round(float(schedule_load_ratio * 100.0), 6) if np.isfinite(schedule_load_ratio) else None,
        },
        {"check": "Energy reference file", "value": ENERGY_REFERENCE.get("source_path")},
        {"check": "Energy reference sheet", "value": ENERGY_REFERENCE.get("sheet_name")},
        {"check": "Energy reference loaded", "value": bool(ENERGY_REFERENCE.get("loaded", False))},
        {"check": "Energy reference rows parsed", "value": int(ENERGY_REFERENCE.get("rows_parsed", 0))},
        {"check": "Energy reference found for process", "value": bool(process_energy_ref_loaded)},
        {"check": "Energy reference kWh/machine-hour for process", "value": round(float(process_energy_ref_kwh), 6)},
        {"check": "Energy cost rate ($/kWh)", "value": round(float(energy_cost_rate), 6)},
        {"check": "Energy rate (kWh per machine-hour)", "value": round(float(energy_kwh_rate), 6)},
        {"check": "Total machine-hours (filtered)", "value": round(float(total_machine_hours), 6)},
        {"check": "Estimated total energy (kWh)", "value": round(float(total_energy_kwh), 6)},
        {"check": "Estimated total energy cost ($)", "value": round(float(total_energy_cost), 6)},
        {"check": "Energy formula", "value": "Cost = ($/kWh) * (kWh per machine-hour) * (total machine-hours)"},
        {"check": "Labor cost rate ($/labor-hour)", "value": round(float(labor_cost_rate), 6)},
        {"check": "Total labor-hours (filtered)", "value": round(float(total_labor_hours), 6)},
        {"check": "Estimated total labor cost ($)", "value": round(float(total_labor_cost), 6)},
        {"check": "Estimated total operating cost ($)", "value": round(float(total_operating_cost), 6)},
        {"check": "Labor formula", "value": "Cost = ($/labor-hour) * (total labor-hours); labor-hours ~= total machine-hours"},
        {"check": "Queue: time-varying arrivals enabled", "value": bool(queue_use_timevarying_arrivals)},
        {"check": "Queue: piece-dependent service enabled", "value": bool(queue_use_piece_service)},
        {"check": "Queue: downtime enabled", "value": bool(queue_use_downtime)},
        {"check": "Holdout ratio (time split)", "value": holdout_ratio_value},
        {"check": "Histogram bins (interactive)", "value": hist_bins},
        {"check": "Throughput rolling window days", "value": roll_days},
        {"check": "Pieces trendline enabled", "value": show_trendline},
        {"check": "Train rows (queue)", "value": int(len(train_df))},
        {"check": "Holdout rows (queue)", "value": int(len(holdout_df))},
        {"check": "Arrival generator (queue)", "value": arrival_sampler_label},
        {"check": "Service generator (queue)", "value": service_sampler_label},
    ]
    if interarrival_fit_rows:
        quality_rows.append(
            {
                "check": "Best interarrival family (AIC)",
                "value": f"{interarrival_fit_rows[0]['family']} (AIC={interarrival_fit_rows[0]['aic']:.2f})",
            }
        )
    if metric_fit_rows:
        quality_rows.append(
            {
                "check": f"Best {metric_label.lower()} family (AIC)",
                "value": f"{metric_fit_rows[0]['family']} (AIC={metric_fit_rows[0]['aic']:.2f})",
            }
        )
    if service_fit_rows_queue:
        quality_rows.append(
            {
                "check": "Best service family for queue (AIC)",
                "value": f"{service_fit_rows_queue[0]['family']} (AIC={service_fit_rows_queue[0]['aic']:.2f})",
            }
        )
    if interarrival_fit_rows_queue:
        quality_rows.append(
            {
                "check": "Best queue-train interarrival family (AIC)",
                "value": f"{interarrival_fit_rows_queue[0]['family']} (AIC={interarrival_fit_rows_queue[0]['aic']:.2f})",
            }
        )
    quality_rows.append(
        {
            "check": "Queue simulation engine",
            "value": queue_sim.get("sim_engine", "unknown"),
        }
    )
    quality_rows.append(
        {
            "check": "Queue simulation mode",
            "value": queue_model_label,
        }
    )
    quality_rows.append(
        {
            "check": "Allen-Cunneen Wq (h)",
            "value": round(float(allen["wq_h"]), 6) if np.isfinite(allen["wq_h"]) else None,
        }
    )
    quality_rows.append(
        {
            "check": "Gap SimPy Wq - Allen Wq (h)",
            "value": round(float(sim_vs_ac_wq_gap), 6) if np.isfinite(sim_vs_ac_wq_gap) else None,
        }
    )
    quality_rows.append(
        {
            "check": "Holdout error mean Wq (h)",
            "value": round(float(holdout_metrics["err_mean_wait_h"]), 6) if np.isfinite(holdout_metrics["err_mean_wait_h"]) else None,
        }
    )
    quality_rows.append(
        {
            "check": "Holdout error P90 Wq (h)",
            "value": round(float(holdout_metrics["err_p90_wait_h"]), 6) if np.isfinite(holdout_metrics["err_p90_wait_h"]) else None,
        }
    )
    quality_rows.append(
        {
            "check": "Holdout error P(wait>2h) (pp)",
            "value": round(float(holdout_metrics["err_p_wait_2h_pp"]), 6) if np.isfinite(holdout_metrics["err_p_wait_2h_pp"]) else None,
        }
    )

    outlier_audit = dff_outlier_pool.copy()
    outlier_audit["service_min_outlier_class"] = outlier_audit["service_min_outlier_class"].astype(str)
    outlier_audit["wait_min_outlier_class"] = outlier_audit["wait_min_outlier_class"].astype(str)
    outlier_audit["cycle_min_outlier_class"] = outlier_audit["cycle_min_outlier_class"].astype(str)
    outlier_audit["outlier_class_any"] = outlier_audit["outlier_class_any"].astype(str)
    outlier_audit = outlier_audit[
        outlier_audit["outlier_class_any"].isin(["mild", "extreme"])
    ].copy()
    if not outlier_audit.empty:
        outlier_audit = outlier_audit.sort_values(
            by=["outlier_rank_any", "service_min_outlier_score", "wait_min_outlier_score"],
            ascending=[False, False, False],
        )
    outlier_audit = outlier_audit[
        [
            "lot_id",
            "arrival_time",
            "machine_label",
            "operator_view",
            "service_hours",
            "wait_hours",
            "cycle_hours",
            "service_min_outlier_class",
            "wait_min_outlier_class",
            "cycle_min_outlier_class",
            "outlier_class_any",
        ]
    ].head(60)

    top_lots = (
        dff.groupby("lot_id", dropna=False)
        .agg(
            visits=("lot_id", "count"),
            pieces=("pieces", "mean"),
            mean_wait_hours=("wait_hours", "mean"),
            mean_service_hours=("service_hours", "mean"),
            mean_cycle_hours=("cycle_hours", "mean"),
            outlier_events=("is_outlier_any", "sum"),
            extreme_events=("outlier_class_any", lambda s: int((s == "extreme").sum())),
        )
        .sort_values(["outlier_events", "mean_wait_hours", "mean_cycle_hours"], ascending=False)
        .head(15)
        .reset_index()
    )

    top_lots["pieces"] = top_lots["pieces"].round(1)
    for c_name in ["mean_wait_hours", "mean_service_hours", "mean_cycle_hours"]:
        top_lots[c_name] = top_lots[c_name].round(2)

    dist_fit_rows = []
    dist_fit_rows.extend(fit_rows_for_table(interarrival_fit_rows, top_n=5))
    dist_fit_rows.extend(fit_rows_for_table(interarrival_fit_rows_queue, top_n=5))
    dist_fit_rows.extend(fit_rows_for_table(metric_fit_rows, top_n=5))
    dist_fit_rows.extend(fit_rows_for_table(service_fit_rows_queue, top_n=5))
    dist_fit_rows.extend(interarrival_fit_notes)
    dist_fit_rows.extend(interarrival_fit_notes_queue)
    dist_fit_rows.extend(metric_fit_notes)
    dist_fit_rows.extend(service_fit_notes_queue)
    if not dist_fit_rows:
        dist_fit_rows = [
            {
                "dataset": "all",
                "rank": None,
                "family": "N/A",
                "k_params": None,
                "log_likelihood": None,
                "aic": None,
                "delta_aic": None,
                "bic": None,
                "ks_d": None,
                "ks_p": None,
                "n": 0,
                "params": None,
                "aic_formula": None,
                "note": "No fit results",
            }
        ]

    quality_columns = [{"name": c, "id": c} for c in ["check", "value"]]
    outlier_columns = [{"name": c, "id": c} for c in outlier_audit.columns.tolist()]
    dist_fit_columns = [
        {"name": c, "id": c}
        for c in [
            "dataset",
            "rank",
            "family",
            "k_params",
            "log_likelihood",
            "aic",
            "delta_aic",
            "bic",
            "ks_d",
            "ks_p",
            "n",
            "params",
            "aic_formula",
            "note",
        ]
    ]
    top_columns = [{"name": c, "id": c} for c in top_lots.columns.tolist()]
    kpi_energy_cost = f"${safe_number(total_energy_cost, 2)}"
    kpi_labor_cost = f"${safe_number(total_labor_cost, 2)}"

    return (
        scope,
        f"{len(dff):,}",
        f"{safe_number(service_mean_min / time_divisor)} h",
        f"{safe_number(wait_p90_min / time_divisor)} h",
        f"{safe_number(arrival_rate)} lots/h",
        f"{safe_number(utilization, 3)}",
        f"{safe_number(outlier_rate_pct, 1)}%",
        kpi_energy_cost,
        kpi_labor_cost,
        fig_arrivals_hour,
        fig_cumulative,
        fig_interarrival,
        fig_duration,
        fig_machine,
        fig_operator,
        fig_pieces,
        fig_throughput,
        kpi_queue_model,
        kpi_queue_wq,
        kpi_queue_pwait,
        kpi_queue_lq,
        queue_recommendation,
        fig_queue_wait,
        fig_queue_survival,
        queue_sim_rows,
        queue_sim_columns,
        scenario_rows,
        queue_scenario_columns,
        quality_rows,
        quality_columns,
        outlier_audit.to_dict("records"),
        outlier_columns,
        dist_fit_rows,
        dist_fit_columns,
        process_line_rows,
        process_line_columns,
        top_lots.to_dict("records"),
        top_columns,
    )


@app.callback(
    Output("flow-lot-store", "data"),
    Output("flow-lot-table", "data"),
    Output("flow-lot-name", "value"),
    Output("flow-lot-pieces", "value"),
    Output("flow-lot-processes", "value"),
    Output("flow-lot-repeat", "value"),
    Input("flow-add-lot-btn", "n_clicks"),
    Input("flow-clear-lots-btn", "n_clicks"),
    State("flow-lot-name", "value"),
    State("flow-lot-pieces", "value"),
    State("flow-lot-processes", "value"),
    State("flow-lot-repeat", "value"),
    State("flow-lot-store", "data"),
    prevent_initial_call=False,
)
def manage_flow_lot_builder(
    add_clicks: int | None,
    clear_clicks: int | None,
    lot_name: str | None,
    lot_pieces: int | float | None,
    lot_processes: list[str] | None,
    lot_repeat: int | float | None,
    lot_store: list[dict[str, Any]] | None,
):
    _ = (add_clicks, clear_clicks)
    store = list(lot_store) if isinstance(lot_store, list) else []
    trigger = dash.ctx.triggered_id
    default_route = ["RASPADO", "BAUCE", "VACIO"] if all(x in PROCESS_OPTIONS for x in ["RASPADO", "BAUCE", "VACIO"]) else []

    if trigger == "flow-clear-lots-btn":
        return [], [], "Lote_1", 200, default_route, 1

    if trigger == "flow-add-lot-btn":
        pieces_val = pd.to_numeric(pd.Series([lot_pieces]), errors="coerce").iloc[0]
        route = [p for p in (lot_processes or []) if p in PROCESS_OPTIONS]
        repeat_val = pd.to_numeric(pd.Series([lot_repeat]), errors="coerce").iloc[0]
        repeat_n = int(max(1, min(500, int(repeat_val if pd.notna(repeat_val) else 1))))
        if pd.notna(pieces_val) and float(pieces_val) > 0 and route:
            base_name = str(lot_name).strip() if str(lot_name or "").strip() else f"Lote_{len(store) + 1}"
            for i in range(repeat_n):
                final_name = base_name if repeat_n == 1 else f"{base_name}#{i + 1}"
                store.append(
                    {
                        "lot_name": final_name,
                        "pieces": float(pieces_val),
                        "route": list(route),
                    }
                )

    table_rows = lot_plan_table_rows(store)
    next_name = f"Lote_{len(store) + 1}" if store else "Lote_1"
    pieces_out = int(lot_pieces) if pd.notna(pd.to_numeric(pd.Series([lot_pieces]), errors="coerce").iloc[0]) else 200
    route_out = lot_processes if isinstance(lot_processes, list) and lot_processes else default_route
    return store, table_rows, next_name, pieces_out, route_out, 1


@app.callback(
    Output("flow-sim-summary", "children"),
    Output("flow-cost-output", "children"),
    Output("flow-stage-table", "data"),
    Output("flow-stage-table", "columns"),
    Output("fig-flow-stage-wait", "figure"),
    Input("flow-run-sim-btn", "n_clicks"),
    State("flow-lot-store", "data"),
    State("date-range", "value"),
    State("strict-cleaning", "checked"),
    State("energy-cost-per-kwh", "value"),
    State("energy-kwh-per-machine-hour", "value"),
    State("labor-cost-per-hour", "value"),
)
def update_connected_flow_simulation(
    run_clicks: int | None,
    flow_lot_store: list[dict[str, Any]] | None,
    date_range: list[str] | None,
    strict_cleaning: bool,
    energy_cost_per_kwh: float | int | None,
    energy_kwh_per_machine_hour: float | int | None,
    labor_cost_per_hour: float | int | None,
):
    if not run_clicks:
        msg = "Add lots and click Simulate."
        return msg, msg, [], [], empty_figure("Waiting for simulation")

    if DATAFRAME.empty:
        msg = "No data loaded."
        return msg, msg, [], [], empty_figure("No data for connected flow simulation")

    lot_plan_raw = flow_lot_store if isinstance(flow_lot_store, list) else []
    if not lot_plan_raw:
        msg = "No lots configured. Add at least one lot and click Simulate."
        return msg, msg, [], [], empty_figure("No lots configured")

    lot_plan: list[dict[str, Any]] = []
    skipped_invalid = 0
    for i, lot in enumerate(lot_plan_raw, start=1):
        route = [p for p in lot.get("route", []) if p in PROCESS_OPTIONS]
        pieces_val = pd.to_numeric(pd.Series([lot.get("pieces")]), errors="coerce").iloc[0]
        lot_name = str(lot.get("lot_name", f"Lote_{i}")).strip() or f"Lote_{i}"
        if not route or pd.isna(pieces_val) or float(pieces_val) <= 0:
            skipped_invalid += 1
            continue
        lot_plan.append({"lot_name": lot_name, "pieces": float(pieces_val), "route": list(route)})

    if not lot_plan:
        msg = "Configured lots are invalid. Check pieces > 0 and at least one valid process."
        return msg, msg, [], [], empty_figure("Invalid lots")

    base = DATAFRAME.copy()
    if date_range and len(date_range) == 2 and date_range[0] and date_range[1]:
        start = pd.to_datetime(date_range[0], errors="coerce")
        end = pd.to_datetime(date_range[1], errors="coerce")
        if pd.notna(start) and pd.notna(end):
            base = base[(base["arrival_time"] >= start) & (base["arrival_time"] <= end + pd.Timedelta(days=1))].copy()

    if base.empty:
        msg = "No rows in selected date range for flow simulation."
        return msg, msg, [], [], empty_figure("No rows in scope")

    rng = np.random.default_rng(20260529)

    ordered_processes: list[str] = []
    for lot in lot_plan:
        for proc in lot["route"]:
            if proc not in ordered_processes:
                ordered_processes.append(proc)

    stage_catalog, missing_processes = build_stage_catalog_for_processes(
        base_df=base,
        process_list=ordered_processes,
        strict_cleaning=bool(strict_cleaning),
        queue_use_downtime=True,
        rng=rng,
    )

    if not stage_catalog:
        msg = "No valid stage data found after cleaning/outlier exclusion."
        return (
            msg,
            msg,
            [],
            [],
            empty_figure("Flow stages have no usable data"),
        )

    filtered_lot_plan: list[dict[str, Any]] = []
    skipped_lots = 0
    for lot in lot_plan:
        filtered_route = [p for p in lot["route"] if p in stage_catalog]
        if not filtered_route:
            skipped_lots += 1
            continue
        filtered_lot_plan.append(
            {
                "lot_name": lot["lot_name"],
                "pieces": float(lot["pieces"]),
                "route": filtered_route,
            }
        )

    if not filtered_lot_plan:
        msg = "All lots were removed because their routes have no usable stage data."
        return (
            msg,
            msg,
            [],
            [],
            empty_figure("No usable lots after stage filtering"),
        )

    arrival_obs = pd.to_datetime(base.get("arrival_time", pd.Series(dtype="datetime64[ns]")), errors="coerce").dropna().sort_values()
    ia_obs = arrival_obs.diff().dt.total_seconds() / 3600.0 if len(arrival_obs) > 1 else pd.Series(dtype=float)
    ia_obs = pd.to_numeric(ia_obs, errors="coerce").dropna()
    ia_obs = ia_obs[(ia_obs > 0) & np.isfinite(ia_obs)]
    if ia_obs.empty:
        ia_samples = np.full(len(filtered_lot_plan), 1.0, dtype=float)
    else:
        ia_samples = np.asarray(
            rng.choice(ia_obs.to_numpy(dtype=float), size=len(filtered_lot_plan), replace=True),
            dtype=float,
        )
        ia_samples = np.where(np.isfinite(ia_samples) & (ia_samples > 0), ia_samples, 1e-6)

    flow_sim = simulate_lot_plan_flow(stage_catalog=stage_catalog, lot_plan=filtered_lot_plan, interarrival_h=ia_samples)
    stage_events = pd.DataFrame(flow_sim.get("stage_rows", []))
    lot_events = pd.DataFrame(flow_sim.get("lot_rows", []))

    if stage_events.empty or lot_events.empty:
        msg = "Simulation completed but produced no stage events."
        return (
            msg,
            msg,
            [],
            [],
            empty_figure("No stage events in simulation"),
        )

    lambda_h = float(1.0 / np.nanmean(ia_samples)) if np.isfinite(np.nanmean(ia_samples)) and np.nanmean(ia_samples) > 0 else np.nan
    energy_cost_per_kwh_val = float(energy_cost_per_kwh) if energy_cost_per_kwh is not None else 0.12
    energy_kwh_per_machine_hour_val = float(energy_kwh_per_machine_hour) if energy_kwh_per_machine_hour is not None else DEFAULT_ENERGY_KWH_PER_MACHINE_HOUR
    labor_cost_per_hour_val = float(labor_cost_per_hour) if labor_cost_per_hour is not None else DEFAULT_LABOR_COST_PER_HOUR

    process_order = (
        stage_events.groupby("process")["route_step"].mean().sort_values().index.tolist()
        if "route_step" in stage_events.columns
        else sorted(stage_events["process"].dropna().unique().tolist())
    )

    stage_rows: list[dict[str, Any]] = []
    total_energy_cost = 0.0
    total_labor_cost = 0.0
    for order_idx, proc in enumerate(process_order, start=1):
        g = stage_events[stage_events["process"] == proc].copy()
        if g.empty:
            continue
        wait_arr = pd.to_numeric(g["wait_h"], errors="coerce").dropna().to_numpy(dtype=float)
        srv_arr = pd.to_numeric(g["service_h"], errors="coerce").dropna().to_numpy(dtype=float)
        stage_arr = pd.to_numeric(g["stage_time_h"], errors="coerce").dropna().to_numpy(dtype=float)
        dt_arr = pd.to_numeric(g["downtime_h"], errors="coerce").dropna().to_numpy(dtype=float)
        mean_srv = float(np.nanmean(srv_arr)) if srv_arr.size > 0 else np.nan
        c_servers = int(stage_catalog.get(proc, {}).get("servers", 1))
        rho_nominal = (lambda_h * mean_srv / max(1, c_servers)) if np.isfinite(lambda_h) and np.isfinite(mean_srv) else np.nan
        service_total_h = float(np.nansum(srv_arr)) if srv_arr.size > 0 else 0.0
        downtime_total_h = float(np.nansum(dt_arr)) if dt_arr.size > 0 else 0.0
        machine_hours = service_total_h + downtime_total_h
        energy_cost_proc = machine_hours * energy_kwh_per_machine_hour_val * energy_cost_per_kwh_val
        labor_cost_proc = service_total_h * labor_cost_per_hour_val
        total_cost_proc = energy_cost_proc + labor_cost_proc
        total_energy_cost += float(energy_cost_proc)
        total_labor_cost += float(labor_cost_proc)

        stage_rows.append(
            {
                "order": int(order_idx),
                "process": proc,
                "c_servers": c_servers,
                "visits": int(len(g)),
                "mean_wait_h": round(float(np.nanmean(wait_arr)), 4) if wait_arr.size > 0 else None,
                "mean_stage_time_h": round(float(np.nanmean(stage_arr)), 4) if stage_arr.size > 0 else None,
                "service_total_h": round(service_total_h, 4),
                "downtime_total_h": round(downtime_total_h, 4),
                "energy_cost": round(float(energy_cost_proc), 2),
                "labor_cost": round(float(labor_cost_proc), 2),
                "total_cost": round(float(total_cost_proc), 2),
                "rho_nominal": round(float(rho_nominal), 4) if np.isfinite(rho_nominal) else None,
            }
        )

    stage_cols = [
        {"name": c, "id": c}
        for c in [
            "order",
            "process",
            "c_servers",
            "visits",
            "mean_wait_h",
            "mean_stage_time_h",
            "service_total_h",
            "downtime_total_h",
            "energy_cost",
            "labor_cost",
            "total_cost",
            "rho_nominal",
        ]
    ]

    fig_flow = go.Figure()
    fig_flow.add_bar(
        x=[r["process"] for r in stage_rows],
        y=[r["energy_cost"] if r["energy_cost"] is not None else 0.0 for r in stage_rows],
        name="Energy cost",
        marker_color="#0891b2",
    )
    fig_flow.add_bar(
        x=[r["process"] for r in stage_rows],
        y=[r["labor_cost"] if r["labor_cost"] is not None else 0.0 for r in stage_rows],
        name="Labor cost",
        marker_color="#d97706",
    )
    fig_flow.update_layout(barmode="stack", yaxis={"title": "Cost"})
    fig_flow = style_figure(fig_flow, "Empirical simulation cost by process")

    system_sojourn_h = pd.to_numeric(lot_events.get("system_time_h", pd.Series(dtype=float)), errors="coerce").dropna().to_numpy(dtype=float)
    mean_system_h = float(np.nanmean(system_sojourn_h)) if system_sojourn_h.size > 0 else np.nan
    p90_system_h = float(np.nanpercentile(system_sojourn_h, 90)) if system_sojourn_h.size > 0 else np.nan
    total_pieces = float(pd.to_numeric(lot_events.get("pieces", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum())
    total_cost = float(total_energy_cost + total_labor_cost)
    cost_per_piece = float(total_cost / total_pieces) if total_pieces > 0 else np.nan

    summary = (
        f"Empirical simulation complete: lots={int(flow_sim['n'])}, pieces={safe_number(total_pieces, 0)}, "
        f"lead_time_mean={safe_number(mean_system_h, 4)} h, lead_time_p90={safe_number(p90_system_h, 4)} h. "
        f"Cost -> energy=${safe_number(total_energy_cost, 2)}, labor=${safe_number(total_labor_cost, 2)}, "
        f"total=${safe_number(total_cost, 2)}, cost_per_piece=${safe_number(cost_per_piece, 4)}. "
        f"sim_engine={flow_sim.get('sim_engine', 'unknown')}. "
        f"Skipped invalid lots={skipped_invalid}, skipped route-mismatch lots={skipped_lots}. "
        + (f"Missing stage data: {', '.join(sorted(set(missing_processes)))}." if missing_processes else "")
    )

    return summary, summary, stage_rows, stage_cols, fig_flow


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=8050)
