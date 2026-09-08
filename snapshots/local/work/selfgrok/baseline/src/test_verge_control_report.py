"""Synthetic report/plot and accounting tests, never real experiment results."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import verge_control_report as report
from verge_control_controller import LABELS, write
from verge_round_core import condition_names


def fixture_data():
    endpoint_rows,events,costs = [],[],[]
    for ai,arm in enumerate(LABELS):
        for r in range(6):
            for endpoint in ('initial','final'):
                for kind,names in (('target',condition_names()),('scope',['full_pass_rate','mean_case_fraction'])):
                    for metric in names:
                        value = ((r+1)*(ai+1)/512) if metric == 'parse' else 0.
                        endpoint_rows.append({'arm':arm,'round':r,'endpoint':endpoint,'kind':kind,'metric':metric,
                            'observed_rate':value,'rollouts':512 if kind == 'target' else 128,
                            'loss_tokens_before_endpoint':524288*(r+(endpoint == 'final'))})
            events.append({'arm':arm,'round':r})
            costs.append({'arm':arm,'round':r,'loss_tokens':524288,'generated_completion_tokens_all_phases':800000})
    return {'protocol':'verge_control_v1_endpoint_analysis','segments':18,'endpoint_rows':endpoint_rows,
            'trajectory_events':events,'generation_ledger':costs,'official_test_opened':False,
            'suite_complete':False,'total_compute_matched':False,'dense_warmup_loss_tokens':786432,
            'statistical_note':'SYNTHETIC TEST DATA. Exploratory only.','cost_note':'SYNTHETIC TEST DATA. Allocated time is not FLOPs.'}


def fixture_resources():
    registered,allocations = {},[]
    for r in range(6):
        for i,arm in enumerate(LABELS):
            identity = f'{100+r*2}_{i}'
            registered[identity] = {'arm':arm,'round':r,'stage':'worker'}
            allocations.append(dict(job_id=identity,**registered[identity],state='COMPLETED',allocated_gpus=1,
                allocated_cpus=8,elapsed_seconds=3600,gpu_allocation_hours=1.,cpu_allocation_core_hours=8.))
        identity = str(101+r*2)
        registered[identity] = {'arm':'all','round':r,'stage':'coordinator'}
        allocations.append(dict(job_id=identity,**registered[identity],state='COMPLETED',allocated_gpus=0,
                allocated_cpus=2,elapsed_seconds=60,gpu_allocation_hours=0.,cpu_allocation_core_hours=2/60))
    snapshot = {'accounting_complete':True,'allocations':allocations,'missing_individual_accounting_rows':[],
                'gpu_allocation_hours':18.,'cpu_allocation_core_hours':144.2}
    return snapshot,registered


class ControlReportTests(unittest.TestCase):
    def test_all_endpoint_coordinates_are_required_and_zero_is_not_missing(self):
        data = fixture_data()
        self.assertEqual(len(report.validate_analysis(data)),468)
        data['endpoint_rows'].pop()
        with self.assertRaises(RuntimeError):
            report.validate_analysis(data)
        data = fixture_data()
        data['endpoint_rows'].append(data['endpoint_rows'][0])
        with self.assertRaises(RuntimeError):
            report.validate_analysis(data)

    def test_bad_denominator_tokens_and_nan_cannot_enter_the_plot(self):
        for field,value in (('rollouts',64),('loss_tokens_before_endpoint',42),('observed_rate',float('nan'))):
            data = fixture_data()
            data['endpoint_rows'][0][field] = value
            with self.assertRaises(RuntimeError):
                report.validate_analysis(data)

    def test_resource_join_reconciles_arm_and_shared_cost(self):
        snapshot,registered = fixture_resources()
        result = report.resource_totals(snapshot,registered)
        self.assertEqual(result['arms']['dense']['gpu_allocation_hours'],6.)
        self.assertAlmostEqual(result['shared_cpu_coordination']['cpu_allocation_core_hours'],.2)

    def test_live_empty_foreign_duplicate_or_unreconciled_accounting_rejected(self):
        for bad in ('empty','running','foreign','duplicate','total','ownership','gpu'):
            snapshot,registered = fixture_resources()
            if bad == 'empty':
                registered = {}
            elif bad == 'running':
                snapshot['allocations'][0]['state'] = 'RUNNING'
            elif bad == 'foreign':
                snapshot['allocations'][0]['job_id'] = 'OTHER'
            elif bad == 'duplicate':
                snapshot['allocations'].append(snapshot['allocations'][0])
            elif bad == 'total':
                snapshot['gpu_allocation_hours'] = 0.
            elif bad == 'ownership':
                snapshot['allocations'][0]['arm'] = 'other'
            else:
                snapshot['allocations'][3]['allocated_gpus'] = 1
            with self.assertRaises(RuntimeError):
                report.resource_totals(snapshot,registered)

    def test_recovered_failed_predecessor_still_costs_resources(self):
        snapshot,registered = fixture_resources()
        old = dict(snapshot['allocations'][0],job_id='99_0',state='FAILED')
        registered['99_0'] = {'arm':'binary','round':0,'stage':'worker'}
        snapshot['allocations'].append(old)
        snapshot['gpu_allocation_hours'] += 1
        snapshot['cpu_allocation_core_hours'] += 8
        result = report.resource_totals(snapshot,registered)
        self.assertEqual(result['arms']['binary']['gpu_allocation_hours'],7.)

    def test_report_does_not_promote_parse_to_full_success_or_claim_compute_match(self):
        snapshot,registered = fixture_resources()
        data = fixture_data()
        content = report.report(data,report.resource_totals(snapshot,registered))
        self.assertEqual(content.count('| 0/512 |'),3)
        self.assertIn('18/512',content)
        self.assertIn('786,432',content)
        self.assertIn('25.00%',content)
        self.assertIn('does not mark',content)
        self.assertIn('No cross-primary paired contrast',content)
        data['dense_warmup_loss_tokens'] = 524288
        self.assertIn('16.67%',report.report(data,report.resource_totals(snapshot,registered)))

    def test_full_collection_requires_last_coordinator_to_exit_successfully(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for r in range(6):
                write(root / 'jobs' / f'round_{r:02d}.json',{'workers':str(100+r*2),'coordinator':str(101+r*2)})
            snapshot,registered = fixture_resources()
            with patch.object(report.ctl,'ROOT',root), patch.dict(report.ctl.os.environ,{'VERGE_BOOK_SUITE':'verge_book_v2'}), patch.object(report.analysis,'collect',return_value=fixture_data()), patch.object(report.resources,'collect',return_value=snapshot), patch.object(report.resources,'registry',return_value=(registered,[])):
                report.collect()
                snapshot['allocations'][-1]['state'] = 'FAILED'
                with self.assertRaises(RuntimeError):
                    report.collect()

    def test_latex_uses_final_counts_and_same_resource_totals(self):
        snapshot,registered = fixture_resources()
        content = report.latex_table(fixture_data(),report.resource_totals(snapshot,registered))
        self.assertEqual(content.count('0/512'),3)
        self.assertIn('Dense to binary & 0/512 & 18/512 & 0/128 & 3,145,728 & 6.0000',content)
        self.assertEqual(content.count(r'\begin{tabular}'),1)
        self.assertEqual(content.count(r'\end{tabular}'),1)

    def test_synthetic_figure_is_rendered_and_explicitly_labeled(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'synthetic.png'
            report.figure(fixture_data(),path,synthetic=True)
            self.assertEqual(path.read_bytes()[:8],b'\x89PNG\r\n\x1a\n')
            self.assertGreater(path.stat().st_size,20000)


if __name__ == '__main__':
    unittest.main()
