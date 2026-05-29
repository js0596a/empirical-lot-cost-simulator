# Raspado EDA Dash App

Interactive Dash app with Plotly + Dash Mantine Components for process-level EDA, starting with **RASPADO**.

## Run

```bash
cd /Users/jeslgdo/Documents/Codex/2026-05-28/yo
python3 -m pip install -r requirements.txt
python3 app.py
```

Open: [http://127.0.0.1:8050](http://127.0.0.1:8050)

## Data source

By default the app reads:

`/Users/jeslgdo/Downloads/datos_raspado.xlsx` (sheet `BITACORA`)

Override with env vars:

```bash
export RASPADO_XLSX_PATH="/absolute/path/to/your_file.xlsx"
export RASPADO_SHEET="BITACORA"
python3 app.py
```

## What you get

- Filter panel (date range, machine, operator, client)
- Data-engineering controls:
  - outlier method (`IQR`, `Z-score`, `Quantile`)
  - outlier view (`Include`, `Exclude`, `Only outliers`)
  - strict preprocessing toggle (remove missing/invalid rows)
- Raspado machine catalog enforcement: official machines are `2, 3, 4, 5`
- KPI cards: lots, mean service, P90 wait, arrival rate, utilization, outlier rate
- Flow tab: arrivals by hour, cumulative arrivals, interarrival histogram
- Time Behavior tab: distribution selector, service by machine, wait by operator
- Capacity tab: pieces vs service, daily throughput/workload, top delayed lots
- Data Quality tab: preprocessing diagnostics + outlier audit table
