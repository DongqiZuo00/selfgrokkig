// Machine-readable control exports. All statistics are computed upstream from
// fixed endpoints; spreadsheet authoring must not recompute or round them.
import fs from 'node:fs/promises';
import path from 'node:path';
import { Workbook } from '@oai/artifact-tool';
import { pathToFileURL } from 'node:url';

export function tableSpecifications(bundle) {
  const data = bundle.analysis;
  if (data?.protocol !== 'verge_control_v1_endpoint_analysis' || data.segments !== 18 ||
      data.suite_complete !== false || data.official_test_opened !== false || data.total_compute_matched !== false) {
    throw new Error('Expected complete isolated endpoint analysis, not a suite-completion claim');
  }
  const specs = [
    ['condition_profiles', ['arm','round','endpoint','checkpoint','kind','metric','observed_rate','rollouts','loss_tokens_before_endpoint','source_artifact'],
      data.endpoint_rows.map(row => ({...row, source_artifact:
        `raw_results/verge_control_v1_${row.arm}_r${String(row.round).padStart(2,'0')}/${row.endpoint === 'initial' ? 'initial' : 'branches/0'}/${row.kind}.jsonl`}))],
    ['paired_contrasts', ['round','candidate','reference','kind','metric','difference','paired_95_lower','paired_95_upper','bootstrap_replicates'],
      data.within_control_contrasts.map(row => ({...row, paired_95_lower:row.paired_95_interval[0],
        paired_95_upper:row.paired_95_interval[1],bootstrap_replicates:data.bootstrap_replicates}))],
    ['generation_costs', ['arm','round','loss_tokens','nonzero_advantage_tokens','generated_training_tokens',
      'generated_initial_endpoint_tokens','generated_final_endpoint_tokens','generated_completion_tokens_all_phases',
      'non_loss_generation_tokens','raw_rollouts','binary_verifier_test_calls','challenger_generated_tokens'], data.generation_ledger],
    ['trajectory_events', ['arm','round','updates','optimizer_steps','training_full_successes',
      'first_full_success_observed','first_success_update','first_success_phase','generated_tokens_before_first_success_update',
      'scope_safe_vs_start_observed'], data.trajectory_events.map(row => ({...row,
        first_full_success_observed:row.first_training_full_success !== null,
        first_success_update:row.first_training_full_success?.update ?? null,
        first_success_phase:row.first_training_full_success?.phase ?? null,
        generated_tokens_before_first_success_update:row.first_training_full_success?.generated_tokens_before ?? null}))],
    ['slurm_allocations', ['job_id','arm','round','stage','state','elapsed_seconds','allocated_cpus',
      'allocated_gpus','allocated_cpu_memory_bytes','gpu_allocation_hours','cpu_allocation_core_hours',
      'batch_max_rss_bytes'], bundle.resources.allocations],
  ];
  const counts = [468,234,18,18];
  specs.forEach(([name,columns,records],i) => {
    if (!Array.isArray(records) || (i < 4 ? records.length !== counts[i] : records.length < 24)) {
      throw new Error(`Incomplete ${name} table`);
    }
    for (const row of records) for (const column of columns) {
      const value = row[column];
      if (value === undefined || (typeof value === 'number' && !Number.isFinite(value)) ||
          (value !== null && !['string','number','boolean'].includes(typeof value))) {
        throw new Error(`Missing or non-scalar ${name}.${column}`);
      }
      if (typeof value === 'string' && /^[=+@\-\t\r]/.test(value)) {
        throw new Error('Formula-like text is not an authorized scientific label');
      }
    }
  });
  return specs;
}

// The public Artifact Tool help has no CSV-export API in this runtime. Preserve
// the typed, inspected range values using RFC4180 serialization as the missing
// export capability. No formulas, number rounding, manual statistical recompute,
// or extra XLSX workbook is introduced by this fallback.
export function csvText(matrix) {
  return matrix.map(row => row.map(value => {
    if (value === null) return '';
    const text = String(value);
    return /[",\r\n]/.test(text) ? '"'+text.replaceAll('"','""')+'"' : text;
  }).join(',')).join('\r\n')+'\r\n';
}

export async function exportTables(bundle, outputDir, {render=true}={}) {
  const specs = tableSpecifications(bundle);
  return exportSpecifications(specs,outputDir,{render,synthetic:bundle.synthetic === true});
}

export function columnLetter(index) {
  if (!Number.isInteger(index) || index < 1) throw new Error('Positive column index required');
  let name='';
  while (index) { index--;name=String.fromCharCode(65+index%26)+name;index=Math.floor(index/26); }
  return name;
}

// Shared typed-range exporter. Caller-specific schemas keep each experiment
// table distinct; the legacy control CLI retains its original output contract.
export async function exportSpecifications(specs, outputDir, {render=true,synthetic=false,
    protocol='verge_control_v1_csv_export',completionField='control_block_complete',previewColumns=null}={}) {
  await fs.mkdir(outputDir,{recursive:true});
  const receipt = {protocol,authoring:'Artifact Tool typed ranges',
    serialization:'RFC4180 fallback: public CSV export unavailable',tables:[],synthetic,
    suite_complete:false,[completionField]:false};
  for (const [name,columns,records] of specs) {
    const workbook = Workbook.create();
    const sheet = workbook.worksheets.add(name);
    const matrix = [columns,...records.map(row => columns.map(column => row[column]))];
    const range = sheet.getRangeByIndexes(0,0,matrix.length,columns.length);
    range.values = matrix;
    range.format.font = {name:'Arial',size:10};
    for (let i=0;i<columns.length;i++) {
      const longest = Math.max(columns[i].length,...records.map(row =>
        typeof row[columns[i]] === 'string' ? row[columns[i]].length : 0));
      sheet.getRangeByIndexes(0,i,matrix.length,1).format.columnWidth = Math.max(16,Math.min(78,longest+3));
    }
    sheet.getRangeByIndexes(0,0,1,columns.length).format.font = {bold:true,name:'Arial',size:10};
    sheet.getRangeByIndexes(0,0,1,columns.length).format.wrapText = true;
    sheet.getRangeByIndexes(0,0,1,columns.length).format.rowHeight = 32;
    sheet.freezePanes.freezeRows(1);
    sheet.showGridLines = false;
    workbook.recalculate();
    const actual = range.values;
    if (JSON.stringify(actual) !== JSON.stringify(matrix)) throw new Error(`Typed values changed in ${name}`);
    const inspected = await workbook.inspect({kind:'table',range:`${name}!A1:D4`,include:'values,formulas',
      tableMaxRows:4,tableMaxCols:4,maxChars:800});
    const errors = await workbook.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!',
      options:{useRegex:true,maxResults:100},maxChars:1000,summary:'CSV typed-value error check'});
    if (range.formulas.some(row=>row.some(value=>value!==null && value!=='')) ||
        actual.some(row=>row.some(value=>typeof value==='string' && /^#(REF!|DIV\/0!|VALUE!|NAME\?|N\/A|NUM!|NULL!|SPILL!|CALC!)/.test(value)))) {
      throw new Error(`Unexpected formula or error cell in ${name}`);
    }
    if (render) {
      const last=previewColumns ? Math.min(previewColumns,columns.length) : columns.length;
      const preview = await workbook.render({sheetName:name,range:`A1:${columnLetter(last)}5`,scale:1,format:'png'});
      await fs.writeFile(path.join(outputDir,`${name}.preview.png`),new Uint8Array(await preview.arrayBuffer()));
    }
    await fs.writeFile(path.join(outputDir,`${name}.csv`),csvText(actual),'utf8');
    receipt.tables.push({name,rows:records.length,columns:columns.length,typed_values_unchanged:true,
      preview_rendered:render,inspection_available:Boolean(inspected.ndjson),formula_error_check_available:Boolean(errors.ndjson),
      preview_columns:previewColumns ? Math.min(previewColumns,columns.length) : columns.length});
  }
  await fs.writeFile(path.join(outputDir,'csv_export_receipt.json'),JSON.stringify(receipt,null,2)+'\n');
  return receipt;
}

if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) {
  const [input,output] = process.argv.slice(2);
  if (!input || !output) throw new Error('Usage: node verge_control_csv.mjs report_bundle.json output_dir');
  console.log(JSON.stringify(await exportTables(JSON.parse(await fs.readFile(input,'utf8')),output),null,2));
}
