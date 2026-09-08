"""Report and figure fixtures are synthetic, not pretrained-model experiments."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import verge_followon_report as report
from verge_followon_controller import version,write
from verge_followon_analysis import analyze
from test_verge_followon_analysis import fixture


def data_fixture():
    data=analyze(*fixture());costs=[]
    for i in range(72):
        costs.append({'array_index':i,'version':version(i),'loss_tokens':262144,
            'generated_training_tokens':262144,'generated_target_endpoint_tokens':1024,
            'generated_scope_endpoint_tokens':256,'generated_completion_tokens':128,
            'generated_initial_endpoint_tokens':0,'generated_completion_tokens_all_phases':263552,
            'non_loss_generation_tokens':1408})
    data.update(generation_ledger=costs,analysis_plan_frozen=True,synthetic=False)
    return data


def hardware_fixture():
    registered={};allocations=[]
    for i in range(72):
        identity=f'100_{i}';metadata={'stage':'worker','version':version(i),'candidate_array_index':i,'parent_job_id':'100'}
        registered[identity]=metadata
        allocations.append(dict(metadata,job_id=identity,state='COMPLETED',allocated_gpus=1,
            allocated_cpus=8,elapsed_seconds=3600,allocated_cpu_memory_bytes=32*1024**3,
            gpu_allocation_hours=1.,cpu_allocation_core_hours=8.))
    metadata={'stage':'coordinator','candidate_array_index':None,'parent_job_id':'101'}
    registered['101']=metadata
    allocations.append(dict(metadata,job_id='101',state='COMPLETED',allocated_gpus=0,
        allocated_cpus=2,elapsed_seconds=60,allocated_cpu_memory_bytes=4*1024**3,
        gpu_allocation_hours=0.,cpu_allocation_core_hours=2/60))
    return {'accounting_complete':True,'missing_individual_accounting_rows':[],
            'allocations':allocations,'gpu_allocation_hours':72.,'cpu_allocation_core_hours':576+2/60},registered


class ReportTests(unittest.TestCase):
    def test_complete_matrix_and_missing_rows(self):
        data=data_fixture();self.assertEqual(len(report.validate_analysis(data)),2664)
        data['endpoint_rows'].pop()
        with self.assertRaisesRegex(ValueError,'Incomplete endpoint'):report.validate_analysis(data)
        data=data_fixture();data['cohorts'].pop()
        with self.assertRaises(ValueError):report.validate_analysis(data)

    def test_endpoint_units_coordinates_and_count_cannot_change(self):
        for field,value in (('rollouts',64),('followon_loss_tokens',42),('checkpoint','foreign'),
                            ('observed_rate',float('nan')),('observed_count',1)):
            data=data_fixture();data['endpoint_rows'][0][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):report.validate_analysis(data)

    def test_missing_concordance_cannot_be_changed_into_zero(self):
        data=data_fixture();data['descriptive_aggregates'][0]['pair_weighted_descriptive_concordance']=0.
        with self.assertRaisesRegex(ValueError,'aggregate'):report.validate_analysis(data)
        data=data_fixture();data['cohorts'][0]['strategies']['condition']['observed_order_concordance']['concordance']=0.
        with self.assertRaisesRegex(ValueError,'ranking'):report.validate_analysis(data)

    def test_reused_source_cannot_add_generation_cost(self):
        data=data_fixture();data['generation_ledger'][0]['generated_initial_endpoint_tokens']=100
        with self.assertRaisesRegex(ValueError,'Generation'):report.validate_analysis(data)

    def test_resource_totals_keep_source_arm_and_coordination_separate(self):
        snapshot,registered=hardware_fixture();hardware=report.resource_totals(snapshot,registered)
        self.assertEqual(hardware['source_arms']['verge']['gpu_allocation_hours'],18.)
        self.assertAlmostEqual(hardware['shared_cpu_coordination']['cpu_allocation_core_hours'],2/60)

    def test_invalid_allocation_or_final_total_rejected(self):
        for key,value in (('state','RUNNING'),('allocated_gpus',2),('allocated_cpus',9),
                          ('allocated_cpu_memory_bytes',33*1024**3),('candidate_array_index',72)):
            snapshot,registered=hardware_fixture();snapshot['allocations'][0][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):report.resource_totals(snapshot,registered)
        snapshot,registered=hardware_fixture();snapshot['gpu_allocation_hours']=0.
        with self.assertRaises(ValueError):report.resource_totals(snapshot,registered)

    def test_historical_failure_cost_is_retained(self):
        snapshot,registered=hardware_fixture();old=dict(snapshot['allocations'][0],job_id='99_0',parent_job_id='99',state='FAILED')
        snapshot['allocations'].append(old);registered['99_0']=dict(registered['100_0'],parent_job_id='99')
        snapshot['gpu_allocation_hours']+=1;snapshot['cpu_allocation_core_hours']+=8
        self.assertEqual(report.resource_totals(snapshot,registered)['source_arms']['verge']['gpu_allocation_hours'],19.)

    def test_report_keeps_censoring_and_book_completion_distinct(self):
        snapshot,registered=hardware_fixture()
        text=report.report(data_fixture(),report.resource_totals(snapshot,registered),synthetic=True)
        self.assertIn('SYNTHETIC TEST DATA',text);self.assertIn('4,718,592',text)
        self.assertEqual(text.count('| 18 | 18 | 0 | 18 |'),4)
        self.assertIn('not complete target successes',text)
        self.assertIn('does not mark',text);self.assertIn('not six new training trials',text)

    def test_collect_requires_current_coordinator_success(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);write(root/'jobs/block.json',{'indices':list(range(72)),'workers':'100','coordinator':'101'})
            snapshot,registered=hardware_fixture()
            with patch.object(report.ctl,'ROOT',root),patch.dict(report.ctl.os.environ,
                    {'VERGE_BOOK_SUITE':'verge_book_v2','VERGE_FOLLOWON_SUITE':'verge_followon_v1'}), \
                    patch.object(report.analysis,'collect',return_value=data_fixture()), \
                    patch.object(report.resources,'collect',return_value=snapshot), \
                    patch.object(report.resources,'registry',return_value=(registered,[])):
                report.collect();snapshot['allocations'][-1]['state']='FAILED'
                with self.assertRaisesRegex(RuntimeError,'Current worker'):report.collect()

    def test_synthetic_figure_exists_with_zero_observations(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'synthetic.png';report.figure(data_fixture(),path,synthetic=True)
            self.assertEqual(path.read_bytes()[:8],b'\x89PNG\r\n\x1a\n')
            self.assertGreater(path.stat().st_size,20000)


if __name__=='__main__':unittest.main()
