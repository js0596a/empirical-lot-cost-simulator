# Production Flow Decision Studio

A Dash + Plotly + SimPy operations research app for production-flow simulation, cost modeling, SPC/capability analysis, and Bayesian process classification.

The product workflow is simple:

1. Upload one or more Excel production logbooks.
2. Enter lots, pieces, and process routes.
3. Simulate plant flow and compare cost, time, capacity, bottlenecks, and process risk.

## Features

- Multi-file Excel upload for production time records / production logbooks.
- Empirical SimPy simulation for lot routing across multiple plant processes.
- Parallel machine flow with Gantt timeline output.
- Energy, labor, and drying gas cost estimates by process.
- Executive decision summary with total cost, cost per piece, lead time, and top cost driver.
- SPC control charts and capability analysis with `Cp`, `Cpk`, `Pp`, `Ppk`, `LSL`, and `USL`.
- Bayesian service-time classifier for comparing process labels.
- Product-style CSS formatting through `assets/studio.css`.

## Local Run

```bash
python3 -m pip install -r requirements.txt
SIM_APP_PORT=8050 python3 -u new_sim_app.py
```

Open [http://127.0.0.1:8050](http://127.0.0.1:8050).

If no default workbook exists locally, the app still starts and lets the user upload Excel files from the UI.

## Docker Run

Build and run with Docker:

```bash
docker build -t empirical-lot-cost-simulator .
docker run --rm -p 8050:8050 empirical-lot-cost-simulator
```

Open [http://127.0.0.1:8050](http://127.0.0.1:8050).

## Docker Compose

```bash
docker compose up --build
```

Optional local data mount:

```text
./data/production.xlsx -> /app/data/production.xlsx
./data/energy.xlsx     -> /app/data/energy.xlsx
```

The app also supports uploading Excel workbooks directly through the browser.

## Expected Excel Inputs

The parser looks for production logbook/time-record fields such as:

- `FECHA`
- `FECHA INICIAL`
- `FECHA FINAL`
- `PROCESO`
- `MAQUINA`
- `PIEZAS`
- `OPERADOR`

Spanish naming notes:

- `TIEMPOS` means `Times` or `Time Records`.
- `BITACORA` / `BITÁCORA` means `Logbook`, here best described as `Production Logbook`.

## Environment Variables

```bash
SIM_APP_HOST=127.0.0.1        # Use 0.0.0.0 in Docker
SIM_APP_PORT=8050
RASPADO_XLSX_PATH=/path/to/default_workbook.xlsx
RASPADO_SHEET=TIEMPOS
ENERGY_REF_XLSX_PATH=/path/to/energy_reference.xlsx
ENERGY_REF_SHEET=Hoja1
```

## Notes

Generated uploads and analysis outputs are intentionally ignored by Git. Keep sensitive plant data outside the repository.
