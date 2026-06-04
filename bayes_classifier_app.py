#!/usr/bin/env python3
from __future__ import annotations

import os
from typing import Any

import dash
import dash_mantine_components as dmc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, dcc, dash_table

import app as core

DEFAULT_CLASS_A = "MEDIDO"
DEFAULT_CLASS_B = "TAIC"
DEFAULT_CAP_PROCESS = "TAIC"
DEFAULT_CAP_H = 4.0
METRIC_COL = "service_hours"
METRIC_MIN_COL = "service_min"
EPS = 1e-12
PROCESS_OPTIONS = [str(p).strip().upper() for p in core.PROCESS_OPTIONS if str(p).strip()]


def option_data(values: list[str]) -> list[dict[str, str]]:
    return [{"label": str(v), "value": str(v)} for v in values]


def empty_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(height=360, margin={"l": 20, "r": 20, "t": 40, "b": 20})
    return fig


def _num(value: Any, default: float = np.nan) -> float:
    try:
        x = float(value)
        return x if np.isfinite(x) else default
    except Exception:
        return default


def normalize_pair(class_a: str | None, class_b: str | None) -> list[str]:
    a = str(class_a or DEFAULT_CLASS_A).strip().upper()
    b = str(class_b or DEFAULT_CLASS_B).strip().upper()
    if a not in PROCESS_OPTIONS:
        a = PROCESS_OPTIONS[0] if PROCESS_OPTIONS else DEFAULT_CLASS_A
    if b not in PROCESS_OPTIONS:
        b = DEFAULT_CLASS_B if DEFAULT_CLASS_B in PROCESS_OPTIONS else (PROCESS_OPTIONS[1] if len(PROCESS_OPTIONS) > 1 else a)
    if a == b:
        alternatives = [p for p in PROCESS_OPTIONS if p != a]
        b = alternatives[0] if alternatives else b
    return [a, b]


def clean_bayes_data(
    classes: list[str],
    date_range: list[str] | None,
    exclude_iqr: bool,
    cap_on: bool,
    cap_process: str | None,
    cap_h: float,
) -> tuple[pd.DataFrame, list[dict], dict[str, dict[str, int]]]:
    rows = []
    summary_rows = []
    counts_by_class: dict[str, dict[str, int]] = {}
    cap_key = str(cap_process or "NONE").strip().upper()

    for cls in classes:
        counts = {"raw_rows": 0, "strict_removed": 0, "iqr_found": 0, "iqr_removed": 0, "cap_removed": 0, "clean_n": 0}
        df = core.DATAFRAME[core.DATAFRAME["process"].astype(str).str.upper().str.strip() == cls].copy() if not core.DATAFRAME.empty else pd.DataFrame()
        counts["raw_rows"] = int(len(df))

        if date_range and len(date_range) == 2 and date_range[0] and date_range[1] and not df.empty:
            start = pd.to_datetime(date_range[0], errors="coerce")
            end = pd.to_datetime(date_range[1], errors="coerce")
            if pd.notna(start) and pd.notna(end):
                df = df[(df["arrival_time"] >= start) & (df["arrival_time"] <= end + pd.Timedelta(days=1))].copy()

        if df.empty:
            counts_by_class[cls] = counts
            summary_rows.append({"class": cls, **counts, "min_h": None, "max_h": None, "mean_h": None, "median_h": None})
            continue

        before_strict = int(len(df))
        df = core.add_quality_flags(df, process_value=cls)
        df = df[(~df["invalid_time_row"]) & (~df["missing_time_flag"])].copy()
        counts["strict_removed"] = int(before_strict - len(df))

        if not df.empty:
            df = core.apply_outlier_flags(df, method=core.FIXED_OUTLIER_METHOD, metrics=[METRIC_MIN_COL])
            class_col = f"{METRIC_MIN_COL}_outlier_class"
            if class_col in df.columns:
                is_iqr = df[class_col].astype(str).isin(["mild", "extreme"])
                counts["iqr_found"] = int(is_iqr.sum())
                if exclude_iqr:
                    before_iqr = int(len(df))
                    df = df[~is_iqr].copy()
                    counts["iqr_removed"] = int(before_iqr - len(df))

        df[METRIC_COL] = pd.to_numeric(df.get(METRIC_COL), errors="coerce")
        df = df[np.isfinite(df[METRIC_COL]) & (df[METRIC_COL] > 0)].copy()

        if cap_on and cap_key == cls:
            cap = max(0.01, _num(cap_h, DEFAULT_CAP_H))
            before_cap = int(len(df))
            df = df[df[METRIC_COL] <= cap].copy()
            counts["cap_removed"] = int(before_cap - len(df))

        vals = pd.to_numeric(df[METRIC_COL], errors="coerce").dropna().astype(float)
        counts["clean_n"] = int(len(vals))
        if len(vals):
            for _, r in df.iterrows():
                rows.append(
                    {
                        "class": cls,
                        "time_h": float(r[METRIC_COL]),
                        "time_min": float(r[METRIC_COL]) * 60.0,
                        "arrival_time": r.get("arrival_time"),
                        "lot_id": r.get("lot_id"),
                        "pieces": r.get("pieces"),
                    }
                )
            summary_rows.append(
                {
                    "class": cls,
                    **counts,
                    "min_h": round(float(vals.min()), 4),
                    "max_h": round(float(vals.max()), 4),
                    "mean_h": round(float(vals.mean()), 4),
                    "median_h": round(float(vals.median()), 4),
                }
            )
        else:
            summary_rows.append({"class": cls, **counts, "min_h": None, "max_h": None, "mean_h": None, "median_h": None})
        counts_by_class[cls] = counts

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["class", "time_h"]).reset_index(drop=True)
    return out, summary_rows, counts_by_class


def _normal_pdf(x: np.ndarray, values: np.ndarray) -> np.ndarray:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals) & (vals > 0)]
    if vals.size == 0:
        return np.full_like(x, EPS, dtype=float)
    mu = float(vals.mean())
    sigma = float(vals.std(ddof=1)) if vals.size > 1 else max(mu * 0.15, 1e-3)
    sigma = max(sigma, 1e-3)
    z = (x - mu) / sigma
    return np.maximum((1.0 / (sigma * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * z * z), EPS)


def _kde_pdf(x: np.ndarray, values: np.ndarray) -> np.ndarray:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals) & (vals > 0)]
    if vals.size < 3 or len(np.unique(np.round(vals, 8))) < 2 or not core.SCIPY_AVAILABLE or core.stats is None:
        return _normal_pdf(x, vals)
    try:
        kde = core.stats.gaussian_kde(vals)
        return np.maximum(np.asarray(kde(x), dtype=float), EPS)
    except Exception:
        return _normal_pdf(x, vals)


def class_densities(x: np.ndarray, data: pd.DataFrame, classes: list[str], model_kind: str) -> dict[str, np.ndarray]:
    densities: dict[str, np.ndarray] = {}
    for cls in classes:
        vals = data.loc[data["class"] == cls, "time_h"].to_numpy(dtype=float) if not data.empty else np.array([], dtype=float)
        densities[cls] = _normal_pdf(x, vals) if model_kind == "normal" else _kde_pdf(x, vals)
    return densities


def class_priors(data: pd.DataFrame, classes: list[str], use_empirical: bool, manual_prior_a: float) -> dict[str, float]:
    manual_a = min(0.99, max(0.01, _num(manual_prior_a, 0.5)))
    if not use_empirical or data.empty:
        return {classes[0]: manual_a, classes[1]: 1.0 - manual_a}
    counts = data["class"].value_counts().to_dict()
    total = float(sum(counts.get(c, 0) for c in classes))
    if total <= 0:
        return {classes[0]: manual_a, classes[1]: 1.0 - manual_a}
    return {c: float(counts.get(c, 0) / total) for c in classes}


def posterior_at(x_value: float, data: pd.DataFrame, classes: list[str], model_kind: str, priors: dict[str, float]) -> dict[str, float]:
    x = np.array([max(1e-9, _num(x_value, 1.0))], dtype=float)
    dens = class_densities(x, data, classes, model_kind)
    numerators = {cls: float(dens[cls][0] * priors.get(cls, 0.5)) for cls in classes}
    denom = max(EPS, sum(numerators.values()))
    return {cls: float(numerators[cls] / denom) for cls in classes}


def predict_class(x_value: float, data: pd.DataFrame, classes: list[str], model_kind: str, priors: dict[str, float]) -> tuple[str, dict[str, float]]:
    post = posterior_at(x_value, data, classes, model_kind, priors)
    pred = max(post, key=post.get)
    return pred, post


def loocv_rows(data: pd.DataFrame, classes: list[str], model_kind: str, use_empirical_prior: bool, manual_prior_a: float) -> tuple[list[dict], list[dict]]:
    if data.empty or len(data) < 4 or data["class"].nunique() < 2:
        return [], []
    preds = []
    reset = data.reset_index(drop=True)
    for idx, row in reset.iterrows():
        train = reset.drop(index=idx).copy()
        if any(train[train["class"] == c].empty for c in classes):
            continue
        priors = class_priors(train, classes, use_empirical_prior, manual_prior_a)
        pred, post = predict_class(float(row["time_h"]), train, classes, model_kind, priors)
        preds.append({"actual": row["class"], "predicted": pred, "time_h": float(row["time_h"]), f"p_{classes[0].lower()}": post[classes[0]], f"p_{classes[1].lower()}": post[classes[1]]})

    if not preds:
        return [], []
    pred_df = pd.DataFrame(preds)
    cm_rows = []
    for actual in classes:
        for predicted in classes:
            cm_rows.append({"actual": actual, "predicted": predicted, "count": int(((pred_df["actual"] == actual) & (pred_df["predicted"] == predicted)).sum())})
    acc = float((pred_df["actual"] == pred_df["predicted"]).mean())
    pos = classes[0]
    tp = int(((pred_df["actual"] == pos) & (pred_df["predicted"] == pos)).sum())
    fp = int(((pred_df["actual"] != pos) & (pred_df["predicted"] == pos)).sum())
    fn = int(((pred_df["actual"] == pos) & (pred_df["predicted"] != pos)).sum())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    metric_rows = [
        {"metric": "LOOCV accuracy", "value": round(acc, 4)},
        {"metric": f"{pos} precision", "value": round(float(precision), 4)},
        {"metric": f"{pos} recall", "value": round(float(recall), 4)},
        {"metric": "LOOCV observations", "value": int(len(pred_df))},
    ]
    return cm_rows, metric_rows


def class_colors(classes: list[str]) -> dict[str, str]:
    palette = ["#2563eb", "#dc2626"]
    return {cls: palette[i % len(palette)] for i, cls in enumerate(classes)}


def build_density_figure(data: pd.DataFrame, classes: list[str], model_kind: str, selected_time_h: float) -> go.Figure:
    if data.empty:
        return empty_figure("No clean data")
    xmax = max(float(data["time_h"].max()) * 1.15, selected_time_h * 1.15, 1.0)
    grid = np.linspace(0.001, xmax, 350)
    dens = class_densities(grid, data, classes, model_kind)
    fig = go.Figure()
    colors = class_colors(classes)
    soft = ["rgba(37,99,235,0.30)", "rgba(220,38,38,0.30)"]
    for i, cls in enumerate(classes):
        vals = data.loc[data["class"] == cls, "time_h"]
        fig.add_histogram(x=vals, histnorm="probability density", nbinsx=20, name=f"{cls} observed", marker_color=soft[i % len(soft)], opacity=0.65)
        fig.add_scatter(x=grid, y=dens[cls], mode="lines", name=f"{cls} likelihood", line={"color": colors[cls], "width": 2.5})
    fig.add_vline(x=selected_time_h, line_dash="dash", line_color="#111827", annotation_text=f"input={selected_time_h:.2f}h")
    fig.update_layout(barmode="overlay", height=420, margin={"l": 25, "r": 20, "t": 55, "b": 35})
    fig.update_xaxes(title="Service time (hours)")
    fig.update_yaxes(title="Density")
    return core.style_figure(fig, "Class-conditional time distributions")


def build_posterior_figure(data: pd.DataFrame, classes: list[str], model_kind: str, priors: dict[str, float], selected_time_h: float) -> go.Figure:
    if data.empty:
        return empty_figure("No clean data")
    xmax = max(float(data["time_h"].max()) * 1.15, selected_time_h * 1.15, 1.0)
    grid = np.linspace(0.001, xmax, 400)
    dens = class_densities(grid, data, classes, model_kind)
    nums = {cls: dens[cls] * priors.get(cls, 0.5) for cls in classes}
    denom = np.maximum(nums[classes[0]] + nums[classes[1]], EPS)
    post = {cls: nums[cls] / denom for cls in classes}
    colors = class_colors(classes)
    fig = go.Figure()
    for cls in classes:
        fig.add_scatter(x=grid, y=post[cls], mode="lines", name=f"P({cls} | time)", line={"color": colors[cls], "width": 3})
    fig.add_hline(y=0.5, line_dash="dot", line_color="#64748b", annotation_text="decision boundary")
    fig.add_vline(x=selected_time_h, line_dash="dash", line_color="#111827", annotation_text=f"input={selected_time_h:.2f}h")
    diff = post[classes[0]] - post[classes[1]]
    crossing_idx = np.where(np.sign(diff[:-1]) != np.sign(diff[1:]))[0]
    if crossing_idx.size:
        idx = int(crossing_idx[0])
        boundary = float((grid[idx] + grid[idx + 1]) / 2.0)
        fig.add_vline(x=boundary, line_dash="dot", line_color="#f59e0b", annotation_text=f"Bayes split ~{boundary:.2f}h")
    fig.update_yaxes(title="Posterior probability", range=[0, 1])
    fig.update_xaxes(title="Service time (hours)")
    fig.update_layout(height=420, margin={"l": 25, "r": 20, "t": 55, "b": 35})
    return core.style_figure(fig, "Bayesian posterior by time")


def observation_rows(data: pd.DataFrame) -> list[dict]:
    if data.empty:
        return []
    rows = []
    for _, r in data.sort_values(["class", "time_h"]).iterrows():
        rows.append(
            {
                "class": r.get("class"),
                "time_h": round(float(r.get("time_h")), 4),
                "time_min": round(float(r.get("time_min")), 2),
                "arrival_time": pd.to_datetime(r.get("arrival_time"), errors="coerce").strftime("%Y-%m-%d %H:%M") if pd.notna(pd.to_datetime(r.get("arrival_time"), errors="coerce")) else None,
                "lot_id": r.get("lot_id"),
                "pieces": round(float(r.get("pieces")), 2) if pd.notna(r.get("pieces")) else None,
            }
        )
    return rows


def date_bounds() -> tuple[str | None, str | None]:
    if core.DATAFRAME.empty:
        return None, None
    df = core.DATAFRAME[core.DATAFRAME["process"].astype(str).str.upper().str.strip().isin(PROCESS_OPTIONS)].copy()
    dts = pd.to_datetime(df.get("arrival_time"), errors="coerce").dropna()
    if dts.empty:
        return None, None
    return dts.min().date().isoformat(), dts.max().date().isoformat()


MIN_DATE, MAX_DATE = date_bounds()

app = dash.Dash(__name__)
app.title = "Flexible Bayesian Time Classifier"

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
                            dmc.Title("Bayesian Time Classifier", order=2),
                            dmc.Text("Choose two processes and classify a service time using Bayes: P(class | time) proportional to likelihood(time | class) times prior(class).", c="dimmed"),
                            dmc.Group(
                                grow=True,
                                children=[
                                    dmc.Select(id="class-a", label="Process A", data=option_data(PROCESS_OPTIONS), value=DEFAULT_CLASS_A if DEFAULT_CLASS_A in PROCESS_OPTIONS else (PROCESS_OPTIONS[0] if PROCESS_OPTIONS else None), searchable=True, allowDeselect=False),
                                    dmc.Select(id="class-b", label="Process B", data=option_data(PROCESS_OPTIONS), value=DEFAULT_CLASS_B if DEFAULT_CLASS_B in PROCESS_OPTIONS else (PROCESS_OPTIONS[1] if len(PROCESS_OPTIONS) > 1 else None), searchable=True, allowDeselect=False),
                                    dmc.DatePickerInput(id="date-range", label="Arrival date range", type="range", value=[MIN_DATE, MAX_DATE], clearable=False, valueFormat="YYYY-MM-DD"),
                                ],
                            ),
                            dmc.Group(
                                grow=True,
                                children=[
                                    dmc.Select(id="model-kind", label="Likelihood model", data=option_data(["kde", "normal"]), value="kde", allowDeselect=False),
                                    dmc.NumberInput(id="input-time-h", label="Time to classify (hours)", value=3.0, min=0.01, max=72, step=0.1, decimalScale=4),
                                    dmc.Select(id="cap-process", label="Operational cap process", data=option_data(["NONE"] + PROCESS_OPTIONS), value=DEFAULT_CAP_PROCESS if DEFAULT_CAP_PROCESS in PROCESS_OPTIONS else "NONE", searchable=True, allowDeselect=False),
                                ],
                            ),
                            dmc.Group(
                                grow=True,
                                children=[
                                    dmc.Switch(id="exclude-iqr", label="Exclude IQR outliers", checked=True),
                                    dmc.Switch(id="cap-on", label="Apply operational cap", checked=True),
                                    dmc.NumberInput(id="cap-h", label="Max hours for capped process", value=DEFAULT_CAP_H, min=0.1, max=72, step=0.25, decimalScale=3),
                                ],
                            ),
                            dmc.Group(
                                grow=True,
                                children=[
                                    dmc.Switch(id="empirical-prior", label="Use empirical prior", checked=True),
                                    dmc.NumberInput(id="manual-prior-a", label="Manual prior P(Process A)", value=0.5, min=0.01, max=0.99, step=0.01, decimalScale=3),
                                ],
                            ),
                            dmc.Alert(id="classifier-output", color="indigo", variant="light", children="Classifier output will appear here."),
                        ],
                    ),
                ),
                dmc.Group(
                    grow=True,
                    align="stretch",
                    children=[
                        dmc.Paper(withBorder=True, radius="lg", p="lg", children=dmc.Stack(gap="sm", children=[dmc.Text("Likelihoods", fw=700), dcc.Graph(id="density-fig", figure=empty_figure("Waiting for data"))])),
                        dmc.Paper(withBorder=True, radius="lg", p="lg", children=dmc.Stack(gap="sm", children=[dmc.Text("Posterior probability", fw=700), dcc.Graph(id="posterior-fig", figure=empty_figure("Waiting for data"))])),
                    ],
                ),
                dmc.Paper(
                    withBorder=True,
                    radius="lg",
                    p="lg",
                    children=dmc.Stack(
                        gap="sm",
                        children=[
                            dmc.Text("Clean data summary", fw=700),
                            dash_table.DataTable(
                                id="summary-table",
                                columns=[{"name": c, "id": c} for c in ["class", "raw_rows", "strict_removed", "iqr_found", "iqr_removed", "cap_removed", "clean_n", "min_h", "max_h", "mean_h", "median_h"]],
                                data=[], page_size=5, sort_action="native", filter_action="native", style_as_list_view=True, style_table={"overflowX": "auto"},
                                style_header={"fontWeight": "700", "backgroundColor": "#f8fafc"}, style_cell={"padding": "8px", "fontFamily": "IBM Plex Sans, sans-serif", "fontSize": "13px"},
                            ),
                            dmc.Text("Leave-one-out validation", fw=700),
                            dash_table.DataTable(
                                id="validation-table",
                                columns=[{"name": c, "id": c} for c in ["metric", "value"]],
                                data=[], page_size=5, sort_action="native", style_as_list_view=True, style_table={"overflowX": "auto"},
                                style_header={"fontWeight": "700", "backgroundColor": "#f8fafc"}, style_cell={"padding": "8px", "fontFamily": "IBM Plex Sans, sans-serif", "fontSize": "13px"},
                            ),
                            dash_table.DataTable(
                                id="confusion-table",
                                columns=[{"name": c, "id": c} for c in ["actual", "predicted", "count"]],
                                data=[], page_size=8, sort_action="native", style_as_list_view=True, style_table={"overflowX": "auto"},
                                style_header={"fontWeight": "700", "backgroundColor": "#f8fafc"}, style_cell={"padding": "8px", "fontFamily": "IBM Plex Sans, sans-serif", "fontSize": "13px"},
                            ),
                            dmc.Text("Clean observations used", fw=700),
                            dash_table.DataTable(
                                id="obs-table",
                                columns=[{"name": c, "id": c} for c in ["class", "time_h", "time_min", "arrival_time", "lot_id", "pieces"]],
                                data=[], page_size=12, sort_action="native", filter_action="native", style_as_list_view=True, style_table={"overflowX": "auto"},
                                style_header={"fontWeight": "700", "backgroundColor": "#f8fafc"}, style_cell={"padding": "8px", "fontFamily": "IBM Plex Sans, sans-serif", "fontSize": "13px"},
                            ),
                        ],
                    ),
                ),
            ],
        ),
    ),
)


@app.callback(
    Output("classifier-output", "children"),
    Output("density-fig", "figure"),
    Output("posterior-fig", "figure"),
    Output("summary-table", "data"),
    Output("validation-table", "data"),
    Output("confusion-table", "data"),
    Output("obs-table", "data"),
    Input("class-a", "value"),
    Input("class-b", "value"),
    Input("date-range", "value"),
    Input("model-kind", "value"),
    Input("input-time-h", "value"),
    Input("exclude-iqr", "checked"),
    Input("cap-on", "checked"),
    Input("cap-process", "value"),
    Input("cap-h", "value"),
    Input("empirical-prior", "checked"),
    Input("manual-prior-a", "value"),
)
def update_classifier(class_a, class_b, date_range, model_kind, input_time_h, exclude_iqr, cap_on, cap_process, cap_h, empirical_prior, manual_prior_a):
    classes = normalize_pair(class_a, class_b)
    model_kind = model_kind if model_kind in {"kde", "normal"} else "kde"
    selected_time = max(0.001, _num(input_time_h, 3.0))
    data, summary_rows, _counts = clean_bayes_data(classes, date_range, bool(exclude_iqr), bool(cap_on), cap_process, _num(cap_h, DEFAULT_CAP_H))

    if data.empty or data["class"].nunique() < 2:
        msg = f"Need clean observations for both {classes[0]} and {classes[1]}. Try disabling IQR exclusion or operational cap."
        return msg, empty_figure(msg), empty_figure(msg), summary_rows, [], [], []

    priors = class_priors(data, classes, bool(empirical_prior), _num(manual_prior_a, 0.5))
    pred, post = predict_class(selected_time, data, classes, model_kind, priors)
    density_fig = build_density_figure(data, classes, model_kind, selected_time)
    posterior_fig = build_posterior_figure(data, classes, model_kind, priors, selected_time)
    cm_rows, metric_rows = loocv_rows(data, classes, model_kind, bool(empirical_prior), _num(manual_prior_a, 0.5))

    class_counts = data["class"].value_counts().to_dict()
    model_txt = "empirical KDE" if model_kind == "kde" else "normal likelihood"
    cap_key = str(cap_process or "NONE").strip().upper()
    cap_txt = f"cap {cap_key} <= {_num(cap_h, DEFAULT_CAP_H):.2f}h ON" if cap_on and cap_key in classes else "cap OFF/not applied to selected pair"
    output = (
        f"Pair={classes[0]} vs {classes[1]}. Input time={selected_time:.3f} h ({selected_time*60:.1f} min). Prediction={pred}. "
        f"Posterior: P({classes[0]}|time)={post[classes[0]]:.3f}, P({classes[1]}|time)={post[classes[1]]:.3f}. "
        f"Model={model_txt}; priors: {classes[0]}={priors[classes[0]]:.3f}, {classes[1]}={priors[classes[1]]:.3f}; {cap_txt}. "
        f"Clean n: {classes[0]}={class_counts.get(classes[0], 0)}, {classes[1]}={class_counts.get(classes[1], 0)}. "
        f"This is a 1-feature Bayesian classifier, so use it as decision support, not as a final process label without context."
    )
    return output, density_fig, posterior_fig, summary_rows, metric_rows, cm_rows, observation_rows(data)


if __name__ == "__main__":
    port = int(os.getenv("BAYES_APP_PORT", "8054"))
    app.run(debug=False, host="127.0.0.1", port=port)
