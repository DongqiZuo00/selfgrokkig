"""Pure within-cohort ranking summaries with explicit right censoring.

Appendix D/O requires every candidate, a common source Solver and equal follow-on
budgets. This module neither trains nor selects candidates. Exact enumeration of
three-item orders handles ranking ties and the random-ranking reference without
additional random draws. It is not a frozen analysis plan or a pooled causal test.
"""
from itertools import combinations, permutations
import math

BUDGET = 262144


def validate_events(events):
    if len(events) != 3 or {e['candidate_index'] for e in events} != {1,2,3}:
        raise ValueError('All three candidate follow-ons must be retained')
    for event in events:
        t = event['observation_loss_tokens']
        if (type(t) is not int or not 0 <= t <= BUDGET
                or type(event['event_observed']) is not bool
                or event.get('planned_loss_tokens') != BUDGET
                or event.get('executed_loss_tokens') != BUDGET
                or event.get('budget_complete') is not True):
            raise ValueError('Incomplete or inconsistent equal-budget follow-on record')
        if not event['event_observed'] and t != BUDGET:
            raise ValueError('A complete event-free continuation is censored at its budget')
    return {e['candidate_index']:e for e in events}


def allowed_orders(scores):
    """All descending orders, uniform within exactly equal saved scores."""
    if set(scores) != {1,2,3} or any(not math.isfinite(v) for v in scores.values()):
        raise ValueError('Three finite saved scores are required')
    return [order for order in permutations((1,2,3))
            if all(scores[a] >= scores[b] for a,b in zip(order,order[1:]))]


def concordance(scores, events):
    """Conservative observed-order concordance, never ranking censored times.

The earlier observed time must be an event. Equal times are not ordered (including
event/censor ties at the final budget). Equal predictor scores receive half credit.
No comparable pair yields null, not zero or a claim that all candidates failed.
"""
    by_id = validate_events(events)
    allowed_orders(scores)
    comparable = 0
    correct = 0.0
    tied_event_times = 0
    for i,j in combinations((1,2,3),2):
        a,b = by_id[i],by_id[j]
        if a['observation_loss_tokens'] == b['observation_loss_tokens']:
            tied_event_times += 1
            continue
        early,late = sorted((a,b),key=lambda e:e['observation_loss_tokens'])
        if not early['event_observed']:
            continue
        comparable += 1
        delta = scores[early['candidate_index']]-scores[late['candidate_index']]
        correct += 1.0 if delta > 0 else .5 if delta == 0 else 0.0
    return {'comparable_pairs':comparable,'concordant_pair_credit':correct,
            'concordance':correct/comparable if comparable else None,
            'equal_time_pairs_excluded':tied_event_times}


def order_summary(orders, events):
    by_id = validate_events(events)
    if not orders or any(len(o) != 3 or set(o) != {1,2,3} for o in orders):
        raise ValueError('Only complete three-candidate orders are permitted')
    result = []
    for k in (1,2,3):
        counts = [sum(by_id[i]['event_observed'] for i in order[:k]) for order in orders]
        result.append({'top_k':k,
            'mean_observed_event_fraction':sum(counts)/(len(orders)*k),
            'mean_any_observed_event_indicator':sum(n > 0 for n in counts)/len(orders),
            'averaging':'Exact uniform averaging over allowed orders, not extra training replicates'})
    return {'allowed_order_count':len(orders),'top_k':result}


def kaplan_meier(events):
    """Descriptive within-cohort curve; no iid confidence interval across branches."""
    validate_events(events)
    at_risk = len(events)
    survival = 1.0
    result = []
    for t in sorted({e['observation_loss_tokens'] for e in events}):
        same = [e for e in events if e['observation_loss_tokens'] == t]
        observed = sum(e['event_observed'] for e in same)
        censored = len(same)-observed
        survival *= 1-observed/at_risk
        result.append({'loss_tokens':t,'at_risk':at_risk,'observed_events':observed,
                       'right_censored':censored,'survival_estimate':survival})
        at_risk -= len(same)
    return result


def summarize_cohort(source_cohort, events):
    candidates = source_cohort['candidates']
    if (len(candidates) != 3 or {c['candidate_index'] for c in candidates} != {1,2,3}
            or any(c['source_cohort'] != source_cohort['cohort']
                   or c['source_solver_initial'] != source_cohort['source_solver_initial']
                   for c in candidates)
            or len({c['condition_rung_zero_based'] for c in candidates}) != 1):
        raise ValueError('Do not pool candidates across cohorts, initial Solvers or credit rungs')
    if any(e.get('source_cohort') != source_cohort['cohort'] for e in events):
        raise ValueError('Follow-on event belongs to a different source cohort')
    validate_events(events)
    strategies = {}
    for label,key in (('condition','condition_gain'),('uncertainty','uncertainty_score')):
        scores = {c['candidate_index']:c[key] for c in candidates}
        strategies[label] = dict(order_summary(allowed_orders(scores),events),
                                 observed_order_concordance=concordance(scores,events))
    random_orders = list(permutations((1,2,3)))
    pair_count = strategies['condition']['observed_order_concordance']['comparable_pairs']
    strategies['random'] = dict(order_summary(random_orders,events),
        expected_concordance_over_uniform_orders=.5 if pair_count else None)
    return {'source_cohort':source_cohort['cohort'],
        'source_solver_initial':source_cohort['source_solver_initial'],
        'candidate_count':3,'strategies':strategies,
        'within_cohort_descriptive_survival':kaplan_meier(events),
        'observed_events_within_budget':sum(e['event_observed'] for e in events),
        'right_censored_at_budget':sum(not e['event_observed'] for e in events),
        'interpretation':'Observed near-term events only; no event is not proof of zero eventual success probability',
        'uncertainty_note':'No cross-seed or iid-branch confidence interval; cohort dependence must be retained in any later aggregate analysis',
        'analysis_plan_frozen':False,'follow_on_complete':False,'suite_complete':False}
