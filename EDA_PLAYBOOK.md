# EDA Playbook (Per Process, Queueing-Ready)

This workflow matches your `Plan_2` structure: do EDA per process first, then choose queue models.

## 1) Expected data format
One row per lot-process visit (long format). Recommended columns:
- `lot_id`
- `product_type`
- `process` (A/B/C/...)
- `arrival_time`
- `start_time`
- `end_time`
- `machine_id`
- `operator_id` (optional)
- `batch_size` (optional)
- `shift` (optional)

## 2) Run

```bash
python3 /Users/jeslgdo/Documents/Codex/2026-05-28/yo/process_eda.py \
  --input /path/to/your_data.csv \
  --output /Users/jeslgdo/Documents/Codex/2026-05-28/yo/eda_output \
  --process-col process \
  --lot-col lot_id \
  --product-col product_type \
  --machine-col machine_id \
  --shift-col shift \
  --batch-col batch_size \
  --arrival-col arrival_time \
  --start-col start_time \
  --end-col end_time
```

## 3) Key outputs
In `eda_output/`:
- `queue_model_input_table.csv` (main summary for boss):
  - process
  - lots_processed
  - mean_service_time_min
  - cv_service
  - mean_wait_min
  - p90_wait_min
  - arrivals_per_hour
  - servers
  - utilization
  - queue_model_suggestion
- `process_comparison.csv`
- `wait_summary_by_process.csv`
- cross-process boxplots

In each process folder (`eda_output/A/`, etc.):
- `missing_summary.csv`
- `numeric_summary.csv`
- `categorical_top_values.csv`
- `queue_metric_stats.csv`
- `data_quality.json`
- `kpi_snapshot.json`
- arrival/service/wait charts in `charts/`

## 4) Interpretation rules
- `utilization >= 1`: overloaded/unstable candidate
- `utilization >= 0.85`: high queue risk
- `cv_service` and interarrival CV near `~1`: random/exponential-like behavior
- low arrival CV (`<0.6`): more scheduled arrivals
- large `mean_wait_min` or `p90_wait_min`: bottleneck indicator

## 5) Next step after EDA
Use `queue_model_input_table.csv` to choose per-process model assumptions (`M/M/1`, `M/M/c`, `D/G/c`, `G/G/c`, batch, or simulation).
