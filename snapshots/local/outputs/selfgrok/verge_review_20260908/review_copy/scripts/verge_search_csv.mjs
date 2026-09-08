// Fixed-checkpoint rates, branch-local events and total search costs stay separate.
// Serialize upstream results without re-estimating statistics or selecting a winner.
import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {exportSpecifications} from './verge_control_csv.mjs';

export function tableSpecifications(bundle) {
  const d=bundle.analysis;
  if (typeof bundle.synthetic!=='boolean' || d?.protocol!=='verge_search_v1_endpoint_analysis' ||
      d.rounds!==6 || d.branches!==24 || d.total_loss_tokens!==12582912 || d.search_block_complete!==false ||
      d.suite_complete!==false || d.official_test_opened!==false || d.total_compute_matched!==false ||
      d.cross_primary_paired_comparisons!==false) throw new Error('Expected isolated complete search analysis with explicit provenance');
  const specs=[
    ['condition_profiles',['round','endpoint','branch_index','kind','metric','observed_count','rollouts','observed_rate',
      'branch_training_loss_tokens','checkpoint','round_version','source_artifact'],
      d.endpoint_rows.map(r=>({...r,source_artifact:r.endpoint==='branch_endpoint'
        ? `raw_results/verge_search_v1_r${String(r.round).padStart(2,'0')}_b${r.branch_index}/branches/0/${r.kind}.jsonl`
        : `raw_results/verge_search_v1/rounds/${r.round_version}/${r.endpoint==='initial' ? `initial/${r.kind}` : 'completion'}.jsonl`}))],
    ['paired_contrasts',['round','candidate','reference','kind','metric','difference','paired_95_lower','paired_95_upper',
      'identical_policy_by_zero_update_provenance','bootstrap_replicates'],
      d.within_search_contrasts.map(r=>({...r,paired_95_lower:r.paired_95_interval[0],paired_95_upper:r.paired_95_interval[1],
        bootstrap_replicates:d.bootstrap_replicates}))],
    ['trajectory_events',['round','branch_index','event_observed','observation_loss_tokens','right_censored',
      'target_training_full_successes','target_endpoint_full_successes','optimizer_steps','updates',
      'planned_loss_tokens','executed_loss_tokens','budget_complete','first_event_role','first_event_update','first_event_group',
      'generated_tokens_through_first_group','training_rollouts_used_in_endpoint_rate','scope_successes_used_as_target_events',
      'clock_is_total_search_compute','version','checkpoint'],
      d.trajectory_events.map(r=>({...r,first_event_role:r.first_event?.role ?? null,first_event_update:r.first_event?.update ?? null,
        first_event_group:r.first_event?.group_index ?? null,generated_tokens_through_first_group:r.first_event?.generated_tokens_through_group ?? null}))],
    ['round_selections',['round','selected_name','selection_condition','selected_gain','completion_full_successes','completion_draws',
      'completion_used_for_selection','challenger_updated','round_loss_tokens_all_four_branches',
      'eligible_branches_json','selection_probabilities_json','selected_checkpoint'],
      d.round_selections.map(r=>({...r,eligible_branches_json:JSON.stringify(r.eligible_branches),
        selection_probabilities_json:r.selection_probabilities===null ? null : JSON.stringify(r.selection_probabilities)}))],
    ['generation_costs',['round','loss_tokens','generated_training_tokens','generated_initial_tokens','generated_worker_endpoint_tokens',
      'generated_selected_completion_tokens','generated_tokens_all_phases','non_loss_generation_tokens',
      'rollouts','binary_verifier_test_calls','challenger_generated_tokens'],d.generation_ledger],
    ['slurm_allocations',['job_id','round','search_branch_index','stage','state','elapsed_seconds','allocated_cpus','allocated_gpus',
      'allocated_cpu_memory_bytes','gpu_allocation_hours','cpu_allocation_core_hours','batch_max_rss_bytes','parent_job_id','version'],
      bundle.resources.allocations.map(r=>({...r,batch_max_rss_bytes:r.batch_max_rss_bytes ?? null}))],
  ];
  const counts=[456,234,24,6,6,null];
  for (const [i,[name,columns,rows]] of specs.entries()) {
    if (!Array.isArray(rows) || (counts[i]===null ? rows.length<36 : rows.length!==counts[i])) throw new Error(`Incomplete ${name}`);
    for (const r of rows) for (const c of columns) {
      const v=r[c];
      if (v===undefined || (typeof v==='number' && !Number.isFinite(v)) ||
          (v!==null && !['string','number','boolean'].includes(typeof v))) throw new Error(`Missing or non-scalar ${name}.${c}`);
      if (typeof v==='string' && /^[=+@\-\t\r]/.test(v)) throw new Error('Formula-like text is not a research identifier');
    }
  }
  return specs;
}

export async function exportTables(bundle,output,{render=true}={}) {
  return exportSpecifications(tableSpecifications(bundle),output,{render,synthetic:bundle.synthetic,
    protocol:'verge_search_v1_csv_export',completionField:'search_block_complete',previewColumns:8});
}

if (process.argv[1] && import.meta.url===pathToFileURL(path.resolve(process.argv[1])).href) {
  const [input,output]=process.argv.slice(2);
  if (!input || !output) throw new Error('Usage: node verge_search_csv.mjs report_bundle.json output_dir');
  console.log(JSON.stringify(await exportTables(JSON.parse(await fs.readFile(input,'utf8')),output),null,2));
}
