// Export separately verified fixed endpoints, events, ranks and resource records.
// Values stay full precision. No extra model sampling or statistical recompute.
import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {exportSpecifications} from './verge_control_csv.mjs';

export function tableSpecifications(bundle) {
  const d=bundle.analysis;
  if (d?.protocol!=='verge_followon_v1_all_candidate_analysis' || d.candidate_count!==72 || d.cohort_count!==24 ||
      d.suite_complete!==false || d.followon_block_complete!==false || d.official_test_opened!==false || d.total_compute_matched!==false) {
    throw new Error('Expected complete all-candidate analysis, not a book-completion claim');
  }
  const ranks=[],survival=[];
  for (const c of d.cohorts) {
    const context={source_arm:c.source_arm,source_round:c.source_round,source_cohort:c.source_cohort,
      source_solver_initial:c.source_solver_initial,pre_success_analysis:c.pre_success_analysis};
    for (const [strategy,s] of Object.entries(c.strategies)) {
      const comparable=c.strategies.condition.observed_order_concordance.comparable_pairs;
      for (const top of s.top_k) ranks.push({...context,...top,strategy,allowed_order_count:s.allowed_order_count,
        within_cohort_comparable_pairs:comparable,
        concordance:strategy==='random' ? s.expected_concordance_over_uniform_orders : s.observed_order_concordance.concordance,
        concordant_pair_credit:strategy==='random' ? comparable*.5 : s.observed_order_concordance.concordant_pair_credit});
    }
    for (const point of c.within_cohort_descriptive_survival) survival.push({...context,...point});
  }
  const specs=[
    ['condition_profiles',['array_index','source_arm','source_round','candidate_index','endpoint','metric','observed_rate','rollouts',
      'kind','observed_count','pre_success_analysis','followon_loss_tokens','source_endpoint_reused',
      'version','source_cohort','source_solver_initial','checkpoint','source_artifact'],
      d.endpoint_rows.map(r=>({...r,source_artifact:r.endpoint==='source'
        ? `raw_results/${r.source_cohort}/branches/${r.candidate_index}/${r.kind}.jsonl`
        : `raw_results/${r.version}/branches/0/${r.kind}.jsonl`}))],
    ['trajectory_events',['array_index','source_arm','source_round','candidate_index','event_observed','observation_loss_tokens',
      'right_censored','prior_source_event_recorded','pre_success_analysis','planned_loss_tokens','executed_loss_tokens','budget_complete',
      'target_training_full_successes','target_endpoint_full_successes','completion_full_successes','first_event_role',
      'first_event_update','first_event_group','generated_tokens_through_first_group','condition_rung_zero_based','condition_gain',
      'raw_endpoint_condition_gain','uncertainty_score','selected_in_primary','scope_safe_in_primary','eligible_in_primary',
      'scope_safe_vs_source_observed','optimizer_steps','version','source_cohort','source_solver_initial','source_checkpoint','final_checkpoint'],
      d.trajectory_events.map(r=>({...r,first_event_role:r.first_event?.role ?? null,first_event_update:r.first_event?.update ?? null,
        first_event_group:r.first_event?.group_index ?? null,generated_tokens_through_first_group:r.first_event?.generated_tokens_through_group ?? null}))],
    ['cohort_rankings',['source_arm','source_round','strategy','top_k','mean_observed_event_fraction','mean_any_observed_event_indicator',
      'concordance','within_cohort_comparable_pairs','concordant_pair_credit','allowed_order_count','pre_success_analysis',
      'source_cohort','source_solver_initial'],ranks],
    ['cohort_survival',['source_arm','source_round','loss_tokens','at_risk','observed_events','right_censored','survival_estimate',
      'pre_success_analysis','source_cohort','source_solver_initial'],survival],
    ['descriptive_aggregates',['stratum','source_arm','strategy','top_k','cohorts','candidates','observed_events','right_censored',
      'sum_cohort_top_k_event_fractions','mean_cohort_top_k_event_fraction','sum_cohort_top_k_any_event_indicators',
      'mean_cohort_top_k_any_event_indicator','within_cohort_comparable_pairs','concordant_pair_credit','pair_weighted_descriptive_concordance'],
      d.descriptive_aggregates],
    ['generation_costs',['array_index','loss_tokens','generated_training_tokens','generated_target_endpoint_tokens','generated_scope_endpoint_tokens',
      'generated_completion_tokens','generated_completion_tokens_all_phases','non_loss_generation_tokens','nonzero_advantage_tokens',
      'generated_initial_endpoint_tokens','raw_rollouts','binary_verifier_test_calls','source_endpoint_reused_without_generation',
      'uncommitted_failed_attempt_generation_included','version'],d.generation_ledger],
    ['slurm_allocations',['job_id','candidate_array_index','stage','state','elapsed_seconds','allocated_cpus','allocated_gpus',
      'allocated_cpu_memory_bytes','gpu_allocation_hours','cpu_allocation_core_hours','batch_max_rss_bytes','parent_job_id','version'],
      bundle.resources.allocations.map(r=>({...r,version:r.version ?? null}))],
  ];
  const sizes=[2664,72,216,null,135,72,null];
  for (const [i,[name,columns,records]] of specs.entries()) {
    if (!Array.isArray(records) || (sizes[i]!==null && records.length!==sizes[i]) ||
        (name==='cohort_survival' && (records.length<24 || records.length>72)) ||
        (name==='slurm_allocations' && records.length<73)) throw new Error(`Incomplete ${name}`);
    for (const r of records) for (const column of columns) {
      const v=r[column];
      if (v===undefined || (typeof v==='number' && !Number.isFinite(v)) ||
          (v!==null && !['string','number','boolean'].includes(typeof v))) throw new Error(`Missing or non-scalar ${name}.${column}`);
      if (typeof v==='string' && /^[=+@\-\t\r]/.test(v)) throw new Error('Formula-like text is not an authorized research label');
    }
  }
  return specs;
}

export async function exportTables(bundle,output,{render=true}={}) {
  return exportSpecifications(tableSpecifications(bundle),output,{render,synthetic:bundle.synthetic===true,
    protocol:'verge_followon_v1_csv_export',completionField:'followon_block_complete',previewColumns:8});
}

if (process.argv[1] && import.meta.url===pathToFileURL(path.resolve(process.argv[1])).href) {
  const [input,output]=process.argv.slice(2);
  if (!input || !output) throw new Error('Usage: node verge_followon_csv.mjs report_bundle.json output_dir');
  console.log(JSON.stringify(await exportTables(JSON.parse(await fs.readFile(input,'utf8')),output),null,2));
}
