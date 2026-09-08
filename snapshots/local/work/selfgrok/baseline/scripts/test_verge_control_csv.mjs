// Synthetic export tests only. Never run or report these as model experiments.
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import {tableSpecifications,csvText,exportTables} from './verge_control_csv.mjs';

const labels = ['binary','dense','dense_then_binary'];
const names = ['parse','parse_and_execute_all',
  ...['mean_normalized_lcp','exact_test_fraction'].flatMap(s => [1,2,3,4].map(i => `cumulative_${s}_threshold_${i}`)),'full_pass'];
const endpoint_rows=[],within_control_contrasts=[],generation_ledger=[],trajectory_events=[],allocations=[];
for (let r=0;r<6;r++) {
  for (const arm of labels) {
    for (const endpoint of ['initial','final']) for (const [kind,metrics] of [
      ['target',names],['scope',['full_pass_rate','mean_case_fraction']]]) for (const metric of metrics) {
      endpoint_rows.push({arm,round:r,endpoint,checkpoint:'synthetic_only',kind,metric,
        observed_rate:metric === 'parse' ? 1/512 : 0,rollouts:kind === 'target' ? 512 : 128,
        loss_tokens_before_endpoint:524288*(r+Number(endpoint === 'final'))});
    }
    generation_ledger.push({arm,round:r,loss_tokens:524288,nonzero_advantage_tokens:0,
      generated_training_tokens:524288,generated_initial_endpoint_tokens:100,
      generated_final_endpoint_tokens:100,generated_completion_tokens_all_phases:524488,
      non_loss_generation_tokens:200,raw_rollouts:512,binary_verifier_test_calls:1024,challenger_generated_tokens:0});
    trajectory_events.push({arm,round:r,updates:30,optimizer_steps:0,training_full_successes:0,
      first_training_full_success:null,scope_safe_vs_start_observed:true});
    allocations.push({job_id:`${100+r*2}_${labels.indexOf(arm)}`,arm,round:r,stage:'worker',state:'COMPLETED',
      elapsed_seconds:3600,allocated_cpus:8,allocated_gpus:1,allocated_cpu_memory_bytes:34359738368,
      gpu_allocation_hours:1,cpu_allocation_core_hours:8,batch_max_rss_bytes:null});
  }
  allocations.push({job_id:String(101+r*2),arm:'all',round:r,stage:'coordinator',state:'COMPLETED',
    elapsed_seconds:60,allocated_cpus:2,allocated_gpus:0,allocated_cpu_memory_bytes:4294967296,
    gpu_allocation_hours:0,cpu_allocation_core_hours:1/30,batch_max_rss_bytes:0});
  for (const [candidate,reference] of [['dense','binary'],['dense_then_binary','binary'],['dense_then_binary','dense']]) {
    for (const [kind,metrics] of [['target',names],['scope',['full_pass_rate','mean_case_fraction']]]) for (const metric of metrics) {
      within_control_contrasts.push({round:r,candidate,reference,kind,metric,difference:0,paired_95_interval:[0,0]});
    }
  }
}
const bundle = {synthetic:true,analysis:{protocol:'verge_control_v1_endpoint_analysis',segments:18,
  suite_complete:false,official_test_opened:false,total_compute_matched:false,bootstrap_replicates:2000,
  endpoint_rows,within_control_contrasts,generation_ledger,trajectory_events},resources:{allocations}};
assert.equal(tableSpecifications(bundle).length,5);
assert.equal(csvText([['text','count','rate','missing'],['a,b"c\nd',0,1/512,null]]),
  'text,count,rate,missing\r\n"a,b""c\nd",0,0.001953125,\r\n');
const incomplete=structuredClone(bundle); incomplete.analysis.endpoint_rows.pop();
assert.throws(() => tableSpecifications(incomplete),/Incomplete/);
const invalid=structuredClone(bundle); invalid.analysis.endpoint_rows[0].observed_rate=NaN;
assert.throws(() => tableSpecifications(invalid),/non-scalar/);
const formula=structuredClone(bundle); formula.analysis.endpoint_rows[0].checkpoint='=FORMULA()';
assert.throws(() => tableSpecifications(formula),/Formula-like/);
const output=process.argv[2];
if (!output) throw new Error('Pass a synthetic-test output directory');
const receipt=await exportTables(bundle,output);
assert.equal(receipt.tables.length,5);
assert.equal(receipt.synthetic,true);
assert.equal(receipt.suite_complete,false);
assert.equal(receipt.control_block_complete,false);
assert.ok(receipt.tables.every(t => t.typed_values_unchanged && t.preview_rendered));
const csv=await fs.readFile(`${output}/condition_profiles.csv`,'utf8');
assert.ok(csv.includes('0.001953125'));
assert.equal(csv.split('\r\n').length,470);
const events=await fs.readFile(`${output}/trajectory_events.csv`,'utf8');
assert.ok(events.includes(',false,,,,true'));
await fs.writeFile(`${output}/synthetic_source_bundle.json`,JSON.stringify(bundle));
console.log(JSON.stringify({synthetic_only:true,checks_passed:9,tables:receipt.tables,suite_complete:false},null,2));
