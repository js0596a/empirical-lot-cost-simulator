import fs from 'node:fs/promises';
import path from 'node:path';
import { Workbook, SpreadsheetFile } from '@oai/artifact-tool';

const payloadPath = '/Users/jeslgdo/Documents/Codex/2026-05-28/yo/raspado_findings_payload.json';
const outPath = '/Users/jeslgdo/Documents/Codex/2026-05-28/yo/raspado_important_findings.xlsx';

const payload = JSON.parse(await fs.readFile(payloadPath, 'utf8'));

const COLORS = {
  ink: '#0f172a',
  accent: '#0f766e',
  headerText: '#ffffff',
  muted: '#475569',
  light: '#e2e8f0',
};

function normalize(v) {
  if (v === undefined) return null;
  if (v === null) return null;
  if (typeof v === 'number' && Number.isFinite(v)) return v;
  if (typeof v === 'boolean' || typeof v === 'string') return v;
  if (v instanceof Date) return v;
  return JSON.stringify(v);
}

function writeTitle(ws, title, subtitle = '') {
  ws.getRange('A1:F1').merge();
  ws.getRange('A1').values = [[title]];
  ws.getRange('A1').format.font.bold = true;
  ws.getRange('A1').format.font.size = 18;
  ws.getRange('A1').format.font.color = COLORS.ink;

  if (subtitle) {
    ws.getRange('A2:F2').merge();
    ws.getRange('A2').values = [[subtitle]];
    ws.getRange('A2').format.font.color = COLORS.muted;
  }
}

function writeTable(ws, startRow, startCol, rows, headers, tableTitle = null) {
  let r = startRow;
  if (tableTitle) {
    ws.getRangeByIndexes(r, startCol, 1, Math.max(headers.length, 1)).merge();
    ws.getRangeByIndexes(r, startCol, 1, 1).values = [[tableTitle]];
    ws.getRangeByIndexes(r, startCol, 1, 1).format.font.bold = true;
    ws.getRangeByIndexes(r, startCol, 1, 1).format.font.color = COLORS.ink;
    r += 1;
  }

  ws.getRangeByIndexes(r, startCol, 1, headers.length).values = [headers];
  const headerRange = ws.getRangeByIndexes(r, startCol, 1, headers.length);
  headerRange.format.font.bold = true;
  headerRange.format.font.color = COLORS.headerText;
  headerRange.format.fill.color = COLORS.accent;
  headerRange.format.wrapText = true;

  const n = Array.isArray(rows) ? rows.length : 0;
  if (n > 0) {
    const matrix = rows.map((row) => headers.map((h) => normalize(row[h])));
    ws.getRangeByIndexes(r + 1, startCol, n, headers.length).values = matrix;
  }

  return r + 1 + n;
}

function setCommonSheetLayout(ws) {
  ws.getRange('A:Z').format.font.name = 'Calibri';
  ws.getRange('A:Z').format.font.size = 11;
  ws.getRange('A:A').format.columnWidthPx = 260;
  ws.getRange('B:B').format.columnWidthPx = 180;
  ws.getRange('C:C').format.columnWidthPx = 420;
  ws.getRange('D:D').format.columnWidthPx = 180;
  ws.getRange('E:E').format.columnWidthPx = 180;
  ws.getRange('F:F').format.columnWidthPx = 180;
  ws.getRange('G:G').format.columnWidthPx = 180;
  ws.getRange('H:H').format.columnWidthPx = 180;
}

const wb = Workbook.create();

// Executive Summary
const exec = wb.worksheets.add('Executive Summary');
setCommonSheetLayout(exec);
writeTitle(
  exec,
  'RASPADO - Important Findings',
  `Generated: ${payload?.meta?.generated_at || ''} | Rules: 4 machines (2,3,4,5), IQR outliers excluded, hours fixed`
);
const execHeaders = ['metric', 'value', 'notes'];
const lastExecRow = writeTable(exec, 3, 0, payload.executive_summary || [], execHeaders, 'Executive KPI Summary');

// Small key KPI block
const kpiRows = (payload.executive_summary || []).filter((x) => [
  'Rows used for analytics (final)',
  'Mean service hours',
  'Mean interarrival hours (by day)',
  'Arrival rate (lots/hour)',
].includes(x.metric));
writeTable(exec, 3, 4, kpiRows, execHeaders, 'Key KPIs');
exec.freezePanes.freezeRows(4);

// Reconciliation
const rec = wb.worksheets.add('Reconciliation');
setCommonSheetLayout(rec);
writeTitle(rec, 'Source Reconciliation', 'Compare app-derived metrics vs reference sheets HORAS_LOTE and INTERARRIBOS');
writeTable(rec, 3, 0, payload.reconciliation || [], ['measure', 'value'], 'Reconciliation Checks');
rec.freezePanes.freezeRows(4);

// Data Quality
const dq = wb.worksheets.add('Data Quality');
setCommonSheetLayout(dq);
writeTitle(dq, 'Data Quality Findings', 'Quality flags generated before modeling and queue metrics');
writeTable(dq, 3, 0, payload.data_quality || [], ['check', 'value', 'notes'], 'Quality Checks');
dq.freezePanes.freezeRows(4);

// Machine Summary
const ms = wb.worksheets.add('Machine Summary');
setCommonSheetLayout(ms);
writeTitle(ms, 'Machine Performance (RASPADO)', 'Official machines only: 2, 3, 4, 5');
writeTable(ms, 3, 0, payload.machine_summary || [], [
  'machine_label', 'lots', 'service_mean_h', 'service_median_h', 'service_p90_h', 'wait_mean_h', 'cycle_mean_h', 'total_service_h',
], 'Machine Stats');
ms.getRange('C:H').setNumberFormat('0.0000');
ms.freezePanes.freezeRows(4);

// Arrivals by Hour + chart
const ah = wb.worksheets.add('Arrivals by Hour');
setCommonSheetLayout(ah);
writeTitle(ah, 'Arrivals by Hour of Day', 'Counts after strict cleaning and outlier exclusion');
const arrivalsRows = payload.arrivals_by_hour || [];
writeTable(ah, 3, 0, arrivalsRows, ['arrival_hour', 'lots'], 'Hourly Arrival Counts');
ah.getRange('A:B').setNumberFormat('0');

const categories = arrivalsRows.map((r) => String(r.arrival_hour ?? ''));
const values = arrivalsRows.map((r) => Number(r.lots ?? 0));
if (categories.length > 0 && values.length > 0) {
  ah.charts.add('line', {
    title: 'Arrivals by Hour',
    categories,
    series: [{ name: 'Lots', values }],
    hasLegend: false,
    from: { row: 1, col: 3 },
    extent: { widthPx: 640, heightPx: 320 },
  });
}
ah.freezePanes.freezeRows(4);

// Daily Summary
const ds = wb.worksheets.add('Daily Summary');
setCommonSheetLayout(ds);
writeTitle(ds, 'Daily Flow Summary', 'Grouped by arrival day');
writeTable(ds, 3, 0, payload.daily_summary || [], [
  'arrival_day', 'lots', 'total_service_hours', 'avg_service_hours', 'avg_wait_hours', 'avg_cycle_hours', 'avg_interarrival_hours',
], 'Daily Metrics');
ds.getRange('C:G').setNumberFormat('0.0000');
ds.freezePanes.freezeRows(4);

// Operator Summary
const ops = wb.worksheets.add('Operator Summary');
setCommonSheetLayout(ops);
writeTitle(ops, 'Operator Summary (Top 10 by volume)', 'Post-cleaning and outlier exclusion');
writeTable(ops, 3, 0, payload.operator_summary || [], ['operator_std', 'lots', 'service_mean_h', 'wait_mean_h'], 'Operator Stats');
ops.getRange('C:D').setNumberFormat('0.0000');
ops.freezePanes.freezeRows(4);

// Outlier Summary
const out = wb.worksheets.add('Outlier Summary');
setCommonSheetLayout(out);
writeTitle(out, 'Outlier Summary (IQR)', 'Policy: always exclude from analytical outputs');
const nextRow = writeTable(out, 3, 0, payload.outlier_overview || [], [
  'rows_in_pool', 'normal_rows', 'mild_rows', 'extreme_rows', 'outlier_rate_pct', 'method', 'policy',
], 'Overall Outlier Overview');
writeTable(out, nextRow + 2, 0, payload.outlier_by_metric || [], ['metric', 'normal', 'mild', 'extreme', 'outlier_pct'], 'Outliers by Metric');
out.getRange('E:E').setNumberFormat('0.0000');
out.freezePanes.freezeRows(4);

// Outlier Examples
const oe = wb.worksheets.add('Outlier Examples');
setCommonSheetLayout(oe);
writeTitle(oe, 'Outlier Examples', 'Top rows flagged as mild/extreme in any metric');
writeTable(oe, 3, 0, payload.outlier_examples || [], [
  'lot_id', 'machine_label', 'operator_std', 'arrival_time', 'start_time', 'end_time',
  'service_hours', 'wait_hours', 'cycle_hours', 'outlier_class_any',
  'service_min_outlier_class', 'wait_min_outlier_class', 'cycle_min_outlier_class',
], 'Flagged Rows');
oe.getRange('G:I').setNumberFormat('0.0000');
oe.freezePanes.freezeRows(4);

// Method Notes
const notes = wb.worksheets.add('Method Notes');
setCommonSheetLayout(notes);
writeTitle(notes, 'Method and Scope Notes', 'How metrics were calculated');
const methodRows = [
  {
    item: 'Interarrival hours',
    definition: 'Computed as diff between consecutive LLEGADA timestamps grouped by day (arrival_day).',
  },
  {
    item: 'Service/Process hours',
    definition: 'Computed as FECHA FINAL - FECHA INICIAL in hours.',
  },
  {
    item: 'Outliers',
    definition: 'IQR method; rows flagged in service/wait/cycle are excluded from analytical outputs.',
  },
  {
    item: 'Machine scope',
    definition: 'RASPADO only includes machines 2,3,4,5 (max 4 machines).',
  },
  {
    item: 'Reference sheets',
    definition: 'HORAS_LOTE and INTERARRIBOS are included for reconciliation values.',
  },
];
writeTable(notes, 3, 0, methodRows, ['item', 'definition'], 'Definitions');
notes.freezePanes.freezeRows(4);

const outBlob = await SpreadsheetFile.exportXlsx(wb);
await fs.mkdir(path.dirname(outPath), { recursive: true });
await outBlob.save(outPath);
console.log(outPath);
