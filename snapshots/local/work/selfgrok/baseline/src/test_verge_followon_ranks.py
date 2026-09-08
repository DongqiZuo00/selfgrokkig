"""Synthetic event fixtures, not results from pretrained-model continuations."""
import copy
import unittest
from verge_followon_ranks import (BUDGET,allowed_orders,concordance,order_summary,
                                 kaplan_meier,summarize_cohort,validate_events)


def events():
    return [{'candidate_index':i,'source_cohort':'cohort_fixture',
             'event_observed':i < 3,'observation_loss_tokens':i*100 if i < 3 else BUDGET,
             'planned_loss_tokens':BUDGET,'executed_loss_tokens':BUDGET,'budget_complete':True}
            for i in (1,2,3)]


def source():
    return {'cohort':'cohort_fixture','source_solver_initial':'source_fixture',
        'candidates':[{'candidate_index':i,'source_cohort':'cohort_fixture',
            'source_solver_initial':'source_fixture','condition_rung_zero_based':0,
            'condition_gain':4-i,'uncertainty_score':i} for i in (1,2,3)]}


class RankingTests(unittest.TestCase):
    def test_exact_ties_not_extra_random_seeds(self):
        self.assertEqual(allowed_orders({1:3,2:2,3:1}),[(1,2,3)])
        self.assertEqual(allowed_orders({1:3,2:3,3:1}),[(1,2,3),(2,1,3)])
        self.assertEqual(len(allowed_orders({1:0,2:0,3:0})),6)

    def test_all_censored_is_unidentified_not_a_correlation(self):
        data = events()
        for e in data:e.update(event_observed=False,observation_loss_tokens=BUDGET)
        c = concordance({1:3,2:2,3:1},data)
        self.assertIsNone(c['concordance'])
        self.assertEqual(c['comparable_pairs'],0)
        self.assertEqual(kaplan_meier(data)[-1]['right_censored'],3)
        self.assertEqual(kaplan_meier(data)[-1]['survival_estimate'],1)

    def test_predictor_direction_and_score_ties(self):
        self.assertEqual(concordance({1:3,2:2,3:1},events())['concordance'],1)
        self.assertEqual(concordance({1:1,2:2,3:3},events())['concordance'],0)
        self.assertEqual(concordance({1:0,2:0,3:0},events())['concordance'],.5)

    def test_event_at_budget_not_naively_ordered_ahead_of_censor(self):
        data = events()
        data[1]['observation_loss_tokens'] = BUDGET
        c = concordance({1:3,2:2,3:1},data)
        self.assertEqual(c['comparable_pairs'],2)
        self.assertEqual(c['equal_time_pairs_excluded'],1)
        curve = kaplan_meier(data)
        self.assertEqual(curve[-1]['observed_events'],1)
        self.assertEqual(curve[-1]['right_censored'],1)
        self.assertAlmostEqual(curve[-1]['survival_estimate'],1/3)

    def test_random_ranking_exact_top_one_and_all(self):
        result = summarize_cohort(source(),events())
        random = result['strategies']['random']
        self.assertEqual(random['allowed_order_count'],6)
        self.assertAlmostEqual(random['top_k'][0]['mean_observed_event_fraction'],2/3)
        self.assertEqual(random['top_k'][2]['mean_any_observed_event_indicator'],1)
        self.assertFalse(result['analysis_plan_frozen'])

    def test_missing_candidate_rejected(self):
        with self.assertRaises(ValueError):validate_events(events()[:2])
        with self.assertRaises(ValueError):allowed_orders({1:3,2:2})

    def test_incomplete_or_inconsistent_budget_rejected(self):
        for key,value in (('budget_complete',False),('executed_loss_tokens',BUDGET-1),
                          ('planned_loss_tokens',BUDGET*2),('event_observed',1),
                          ('observation_loss_tokens',-1)):
            data = events();data[0][key] = value
            with self.assertRaises(ValueError):validate_events(data)

    def test_censor_time_must_match_budget(self):
        data = events();data[2]['observation_loss_tokens'] = 10
        with self.assertRaises(ValueError):validate_events(data)

    def test_no_cross_cohort_pooling(self):
        for key,value in (('source_cohort','other'),('source_solver_initial','other'),
                          ('condition_rung_zero_based',2)):
            s = source();s['candidates'][2][key] = value
            with self.assertRaises(ValueError):summarize_cohort(s,events())
        data = events();data[2]['source_cohort'] = 'other'
        with self.assertRaises(ValueError):summarize_cohort(source(),data)

    def test_nonfinite_scores_rejected(self):
        for bad in (float('inf'),float('nan')):
            with self.assertRaises(ValueError):allowed_orders({1:1,2:2,3:bad})

    def test_inputs_unchanged(self):
        s,data = source(),events();before=copy.deepcopy((s,data))
        summarize_cohort(s,data)
        self.assertEqual((s,data),before)

    def test_tied_event_times_not_ordered(self):
        data = events();data[1]['observation_loss_tokens'] = 100
        self.assertEqual(concordance({1:3,2:2,3:1},data)['comparable_pairs'],2)


if __name__ == '__main__':
    unittest.main()
