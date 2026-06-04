# Production Flow Decision Studio

A Dash + Plotly + SimPy application built for plant production planning, process EDA, lot-flow simulation, cost estimation, and process quality analysis.

This project was developed around a real operations problem: production data existed in Excel logbooks, but it was difficult to turn those records into useful decisions about lot routing, machine capacity, queueing behavior, process timing, and cost. The app converts those spreadsheets into an interactive decision tool where a user can upload data, define lots, choose process routes, simulate the plant flow, and review the expected timeline, cost, bottlenecks, and process risk.

## What I Built

I built an end-to-end operations research and data science workflow for a leather production environment:

- Cleaned and standardized Excel-based production logbook data.
- Engineered process timing metrics from arrival, start, and finish timestamps.
- Built per-process exploratory analysis to understand service-time behavior and outliers.
- Implemented an empirical SimPy simulation for user-defined lot routes.
- Added process-specific capacity rules, queueing logic, and machine utilization assumptions.
- Modeled energy, labor, and gas cost by process.
- Added Gantt charts to show how lots move through the plant over time.
- Added SPC charts and process capability analysis with `Cp`, `Cpk`, `Pp`, and `Ppk`.
- Added a Bayesian classifier to compare processes based on service-time behavior.
- Dockerized the app so it can be run consistently outside the development machine.

The goal is not just dashboarding. The goal is to make messy production records usable for planning decisions.

## Business Problem

The plant runs multiple production processes, each with different timing behavior, machine constraints, and cost structure. A planner may want to answer questions like:

- If I enter these lots today, how long will they take?
- Which process is driving most of the cost?
- Which lot route creates bottlenecks?
- How much labor, energy, and drying gas cost should I expect?
- Are certain processes behaving outside normal operating limits?
- Can we compare process timing patterns statistically instead of relying only on intuition?

The app lets a user enter lot configurations directly instead of manually calculating every step in Excel.

## Main App Workflow

1. Upload one or more Excel production logbooks.
2. The app parses and cleans the data.
3. The user enters lots, piece counts, and process routes.
4. The simulator runs the selected lot flow through the plant.
5. The app returns cost, timeline, queueing, quality, and classification outputs.

## Data Inputs

The app supports Excel workbooks that contain production time records. The parser looks for columns similar to:

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

The app can start without a default workbook. In that case, the user uploads Excel files from the browser.

## Data Engineering Pipeline

The data pipeline is built to handle imperfect production logs. The main steps are:

1. Read Excel sheets and identify usable production records.
2. Normalize process names so spelling variations map to consistent process labels.
3. Parse timestamp fields into arrival, start, and finish times.
4. Calculate service time from effective process duration.
5. Calculate arrival and waiting behavior where the fields are available.
6. Remove invalid records, missing timestamps, negative durations, and non-usable rows.
7. Apply outlier screening where requested, especially with IQR-based cleaning.
8. Rebuild process options, date ranges, machine counts, and simulation inputs from the cleaned data.

The app avoids treating all processes as identical. Each process keeps its own empirical timing distribution and capacity assumptions.

## Simulation Logic

The simulation is empirical and route-driven.

A user creates lots by entering:

- Lot name
- Number of pieces
- Number of repeated lots
- Process route, selected in any order

The simulator then runs those lots through a SimPy model. Each process acts like a station with a finite number of servers/machines. Lots can move through different process routes, and different lots can be active at the same time.

The simulation outputs:

- Total lead time
- Process service time
- Queue/wait time
- Between-process transfer gaps
- Machine/resource usage
- Cost by process
- Gantt chart by lot and process

Between process steps, the app adds a random transfer/setup gap using a uniform range of 20 to 30 minutes. This prevents every simulation from looking identical while keeping the transfer delay within the realistic operating range requested by the plant.

## Process-Specific Rules

Some process behavior is not estimated only from raw data. It also uses plant knowledge.

Examples:

- `RASPADO` is constrained to 4 machines based on business input.
- `RECURTIDO` is user-configurable because its duration depends on the selected operating plan rather than only piece count.
- Drying-related processes such as `LTD`, `TAIC`, and `AEREO` use operational cleaning/capping logic because some historical records contained unrealistic service times.
- Energy consumption is process-specific, loaded from the energy reference workbook when available.
- `RASPADO` has a manual one-time override of `44 kWh/hour` as requested during the project.

These rules are intentionally visible because they are business assumptions, not hidden model behavior.

## Cost Model

The cost model estimates cost by process and by full simulation run.

Cost components include:

- Energy cost
- Labor cost
- Drying gas cost

Energy cost is calculated from:

```text
machine hours * kWh per machine-hour * energy price per kWh
```

Labor cost is calculated from:

```text
labor hours * labor rate per hour
```

Drying gas cost is treated as a cost per cuero / piece for the drying processes where it applies.

The app summarizes:

- Total cost
- Cost per piece
- Cost by process
- Energy, labor, and gas cost split
- Highest cost driver

## Gantt Timeline

The Gantt chart is one of the main operational outputs. It shows each lot moving through each selected process over time.

The chart helps answer:

- When does each lot start and finish?
- Which process creates the longest block of time?
- Where are the gaps between process steps?
- Which lots overlap in the plant?
- How does a route change affect the schedule?

The Gantt is built from the simulated event table, not from a static chart template.

## SPC and Capability Analysis

The app includes a quality/statistical process control section for process timing metrics.

It supports:

- Individuals control chart
- Moving range chart
- Capability histogram
- Process comparison histogram
- User-entered `LSL` and `USL`
- `Cp`, `Cpk`, `Pp`, and `Ppk`

The capability metrics are calculated only when specification limits are provided. The app does not invent specification limits because those should come from engineering, production, or customer requirements.

The capability view is useful for discussing whether a process is stable and whether the observed process performance fits inside the required operating window.

## Bayesian Classifier

The app includes a simple Bayesian service-time classifier. It compares two process classes using service-time distributions and estimates the posterior probability of a class given a service time.

Example use case:

- Compare `MEDIDO` vs `TAIC`
- Enter a service time
- Estimate which process behavior that time more closely resembles

This is not used as a final truth label. It is a decision-support tool for comparing timing behavior when process distributions overlap.

## App Structure

```text
new_sim_app.py          Main product-style simulator app
app.py                  Core data cleaning, EDA, queueing, and helper logic
bayes_classifier_app.py Bayesian classifier module
process_eda.py          Standalone per-process EDA utility
assets/studio.css       App styling
Dockerfile              Container build definition
docker-compose.yml      Compose run configuration
requirements.txt        Python dependencies
```

Generated files, uploads, and plant data are intentionally not committed.

## Local Run

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Run the main app:

```bash
SIM_APP_PORT=8050 python3 -u new_sim_app.py
```

Open:

```text
http://127.0.0.1:8050
```

## Docker Run

Build the image:

```bash
docker build -t empirical-lot-cost-simulator .
```

Run the app:

```bash
docker run --rm -p 8050:8050 empirical-lot-cost-simulator
```

Open:

```text
http://127.0.0.1:8050
```

## Docker Compose

```bash
docker compose up --build
```

Optional local data mount:

```text
./data/production.xlsx -> /app/data/production.xlsx
./data/energy.xlsx     -> /app/data/energy.xlsx
```

The app also supports uploading Excel workbooks directly through the browser, so a default mounted workbook is not required.

## Environment Variables

```bash
SIM_APP_HOST=127.0.0.1        # Use 0.0.0.0 in Docker
SIM_APP_PORT=8050
RASPADO_XLSX_PATH=/path/to/default_workbook.xlsx
RASPADO_SHEET=TIEMPOS
ENERGY_REF_XLSX_PATH=/path/to/energy_reference.xlsx
ENERGY_REF_SHEET=Hoja1
```

## Applying This to Another Plant

The app can be adapted to another plant if the new plant has equivalent production timing records. The important requirement is not that the process names are identical, but that the data can be mapped into the same operational structure:

- Process name
- Arrival/start/end timestamps
- Machine or resource identifier
- Piece count or lot size
- Optional operator and cost references

For another plant, the main work would be:

1. Map the new plant's process names to normalized labels.
2. Confirm machine/server counts per process.
3. Confirm valid operating ranges for service times.
4. Replace energy and gas cost references.
5. Re-run per-process EDA before trusting simulation outputs.

## Limitations

This is an analytical planning tool, not a replacement for production supervision.

Important limitations:

- Historical data quality directly affects simulation quality.
- Unrealistic timestamps must be cleaned or capped with plant-approved rules.
- Specification limits for capability analysis should come from the business, not from the app.
- Docker currently runs the Dash development server; for external production deployment, use a production WSGI setup.
- The simulator estimates behavior based on available data and assumptions. It should be validated against real runs before being used for high-stakes scheduling decisions.

## Why This Matters

The value of the project is that it connects statistics, operations research, and business planning in one workflow. Instead of producing disconnected charts, the app turns raw production logs into a tool that helps answer practical plant questions about time, cost, capacity, and process behavior.
