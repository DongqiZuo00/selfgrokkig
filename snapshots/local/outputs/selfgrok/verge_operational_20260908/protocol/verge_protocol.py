"""Small, model-independent VERGE protocol core. No training or dataset generation.

See README.md for original design rules versus 2026-09-08 operational additions.
All condition indices in the public API are ONE-BASED (1..6); saturated b is 7.
Synthetic tests exercise arithmetic and contracts, not research effectiveness.
"""

from dataclasses import dataclass, field
from fractions import Fraction
import math
import random
from typing import Mapping, Sequence


CONDITION_COUNT = 6
SATURATED = 7
DELTA = 0.1
STAGE_KINDS = frozenset({"registered", "tape_transform", "hinted_target"})


@dataclass(frozen=True)
class Verification:
    class_rates: Mapping[str, float]
    weakest_class_rate: float
    conditions: tuple[bool, ...]

    @property
    def binary_reward(self):
        return int(self.conditions[-1])


@dataclass(frozen=True)
class TestClasses:
    """An explicit, complete nonempty partition; no inferred or fallback classes.

    An outcome is True (exact pass), False, or None (missing/error/timeout/NONE).
    Failed and absent outputs retain their fixed place in the denominator.
    Missing manifest tests and unexpected tests are errors, not exclusions.
    """

    test_ids: tuple[str, ...]
    class_by_test: Mapping[str, str]
    expected_classes: tuple[str, ...] | None = None

    def __post_init__(self):
        if len(self.test_ids) < 4 or len(set(self.test_ids)) != len(self.test_ids):
            raise ValueError("need >=4 unique tests so the six conditions remain nested")
        if set(self.class_by_test) != set(self.test_ids):
            raise ValueError("class mapping must cover exactly all target tests")
        if any(not isinstance(v, str) or not v for v in self.class_by_test.values()):
            raise ValueError("every test needs a nonempty explicit class")
        if self.expected_classes is not None:
            expected = set(self.expected_classes)
            if not expected or len(expected) != len(self.expected_classes) or expected != set(self.class_by_test.values()):
                raise ValueError("every declared class must have members; no missing or unexpected classes")

    def evaluate(self, parsed: bool, outcomes: Mapping[str, bool | None]):
        if type(parsed) is not bool or set(outcomes) != set(self.test_ids):
            raise ValueError("provide parsed bool and one result slot per frozen test")
        if any(v is not None and type(v) is not bool for v in outcomes.values()):
            raise ValueError("test outcomes must be bool or explicit None failures")
        if not parsed and any(v is True for v in outcomes.values()):
            raise ValueError("an unparsable candidate cannot pass an executable test")
        members = {}
        for test_id in self.test_ids:
            members.setdefault(self.class_by_test[test_id], []).append(test_id)
        rates = {
            name: sum(outcomes[t] is True for t in tests) / len(tests)
            for name, tests in members.items()
        }
        f = min(rates.values())
        thresholds = (1 / len(self.test_ids), 0.25, 0.5, 0.75)
        conditions = (parsed,) + tuple(f >= threshold for threshold in thresholds) + (f == 1,)
        return Verification(rates, f, conditions)


@dataclass(frozen=True)
class StageSpec:
    kind: str
    spec: Mapping[str, object]

    def __post_init__(self):
        if self.kind not in STAGE_KINDS or not self.spec:
            raise ValueError("stage requires one of the three design interfaces and a specification")


@dataclass(frozen=True)
class BranchPlan:
    branch_id: str
    base_checkpoint: str
    stages: tuple[StageSpec, ...]
    stage_tokens: tuple[int, ...]
    tail_tokens: int
    milestones: tuple[int, ...]
    generated_token_budget: int
    fresh_optimizer: bool = True
    beta: float = 0.0
    weight_decay: float = 0.0
    solver_reward: str = "binary_full_pass"


@dataclass(frozen=True)
class RoundPlan:
    base_checkpoint: str
    curricula: tuple[BranchPlan, ...]
    direct: BranchPlan
    budget_unit: str = "generated_tokens"


def plan_round(base_checkpoint, curricula, B, eta):
    """Plan exact integer training budgets and direct checkpoint/probe boundaries.

    Operational rounding: floor(eta*B) tail tokens; divide remaining tokens
    equally, assigning single-token remainders to earlier stages. Direct gets B.
    Probes are excluded from training B and must be recorded as evaluation cost.
    """
    if not base_checkpoint or not curricula or type(B) is not int or B <= 0:
        raise ValueError("nonempty base/curricula and positive integer budget required")
    eta = Fraction(str(eta))
    if not 0 < eta < 1:
        raise ValueError("eta must leave positive stage and target-only-tail budgets")
    tail = int(eta * B)
    plans, direct_boundaries = [], {0, B}
    for name, supplied_stages in curricula.items():
        stages = tuple(supplied_stages)
        if name == "direct" or not name or not stages or not all(isinstance(s, StageSpec) for s in stages):
            raise ValueError("named curricula need typed stages; 'direct' is reserved")
        if tail < 1 or B - tail < len(stages):
            raise ValueError("budget cannot allocate positive tokens to each stage and tail")
        quotient, remainder = divmod(B - tail, len(stages))
        allocations = tuple(quotient + (i < remainder) for i in range(len(stages)))
        boundaries = [0]
        for tokens in allocations + (tail,):
            boundaries.append(boundaries[-1] + tokens)
        direct_boundaries.update(boundaries)
        plans.append(BranchPlan(name, base_checkpoint, stages, allocations, tail, tuple(boundaries), B))
    direct = BranchPlan("direct", base_checkpoint, (), (), B, tuple(sorted(direct_boundaries)), B)
    return RoundPlan(base_checkpoint, tuple(plans), direct)


@dataclass(frozen=True)
class Evaluation:
    """One training seed, fixed selection manifest, per-instance condition counts.

    counts[i][j] is the number of sample_ids satisfying condition j+1 on
    instance_ids[i]. These are integer empirical counts, not probabilities.
    A final evaluation has 32 sample IDs; a stage probe uses its first eight.
    sample IDs identify frozen generation seeds/slots, NOT independent instances.
    The producer is responsible for retaining raw per-rollout evidence.
    """

    checkpoint: str
    training_seed: int
    manifest_id: str
    instance_ids: tuple[str, ...]
    sample_ids: tuple[str, ...]
    counts: tuple[tuple[int, ...], ...]

    def __post_init__(self):
        if not self.checkpoint or not self.manifest_id:
            raise ValueError("checkpoint and frozen selection manifest identity required")
        if not self.instance_ids or len(set(self.instance_ids)) != len(self.instance_ids):
            raise ValueError("unique selection instances required")
        if not self.sample_ids or len(set(self.sample_ids)) != len(self.sample_ids):
            raise ValueError("unique sample slots required")
        if len(self.counts) != len(self.instance_ids):
            raise ValueError("counts must cover all selection instances")
        n = len(self.sample_ids)
        for row in self.counts:
            if len(row) != CONDITION_COUNT or any(type(x) is not int or x < 0 or x > n for x in row):
                raise ValueError("six integer counts bounded by rollout count required")
            if any(a < b for a, b in zip(row, row[1:])):
                raise ValueError("condition counts must be nested")

    def instance_rates(self, condition):
        if type(condition) is not int or not 1 <= condition <= CONDITION_COUNT:
            raise ValueError("condition index must be 1..6")
        return tuple(row[condition - 1] / len(self.sample_ids) for row in self.counts)

    def rate(self, condition):
        values = self.instance_rates(condition)
        return sum(values) / len(values)

    @property
    def success_count(self):
        return sum(row[-1] for row in self.counts)

    @property
    def parse_valid_count(self):
        return sum(row[0] for row in self.counts)


def frontier(evaluation, delta=DELTA):
    if not 0 < delta < 1:
        raise ValueError("delta must be between zero and one")
    return next((j for j in range(1, CONDITION_COUNT + 1) if evaluation.rate(j) < 1 - delta), SATURATED)


def _aligned(left, right, same_sample_ids=True):
    for attr in ("training_seed", "manifest_id", "instance_ids"):
        if getattr(left, attr) != getattr(right, attr):
            raise ValueError(f"paired evaluations must share {attr}")
    if same_sample_ids and left.sample_ids != right.sample_ids:
        raise ValueError("paired evaluations must share generation sample IDs")


@dataclass(frozen=True)
class PairedGain:
    estimate: float
    ci_low: float
    ci_high: float
    n_instances: int
    confidence: float

    @property
    def positive(self):
        return self.estimate > 0 and self.ci_low > 0


def _quantile(sorted_values, p):
    position = p * (len(sorted_values) - 1)
    lo, hi = math.floor(position), math.ceil(position)
    return sorted_values[lo] + (position - lo) * (sorted_values[hi] - sorted_values[lo])


def paired_gain(candidate, reference, condition, *, confidence=0.95, resamples=2000, random_seed=0):
    """Percentile paired bootstrap, resampling SELECTION INSTANCES within one seed.

    One selection instance is not enough to estimate between-instance uncertainty.
    This CI conditions on the trained checkpoints; it is not a cross-seed CI.
    """
    _aligned(candidate, reference)
    if len(candidate.instance_ids) < 2:
        raise ValueError("paired bootstrap needs at least two selection instances")
    if not 0 < confidence < 1 or type(resamples) is not int or resamples < 100:
        raise ValueError("valid confidence and >=100 resamples required")
    differences = tuple(a - b for a, b in zip(candidate.instance_rates(condition), reference.instance_rates(condition)))
    n, rng = len(differences), random.Random(random_seed)
    draws = sorted(sum(differences[rng.randrange(n)] for _ in range(n)) / n for _ in range(resamples))
    alpha = (1 - confidence) / 2
    return PairedGain(sum(differences) / n, _quantile(draws, alpha), _quantile(draws, 1 - alpha), n, confidence)


@dataclass(frozen=True)
class RoundScore:
    kappa: int
    reward_switched: bool
    gains: Mapping[str, PairedGain]
    j_plus: tuple[str, ...]


def score_round(base, direct, curricula, q, **bootstrap_options):
    """Freeze one kappa from all G+1 FINAL batches; compare every course to direct.

    The two-branch rule includes direct. q counts individual full-pass rollouts.
    Switch is evaluated per round, not permanently latched by an earlier round.
    """
    if type(q) is not int or q <= 0 or not curricula or "direct" in curricula:
        raise ValueError("positive q and named non-direct curricula required")
    _aligned(base, direct)
    branches = [direct] + list(curricula.values())
    for branch in branches:
        _aligned(base, branch)
        if len(branch.sample_ids) != 32:
            raise ValueError("round reward requires the 32-rollout final batch")
    switched = sum(branch.success_count >= q for branch in branches) >= 2
    kappa = CONDITION_COUNT if switched else min(frontier(base), CONDITION_COUNT)
    gains = {name: paired_gain(branch, direct, kappa, **bootstrap_options) for name, branch in curricula.items()}
    return RoundScore(kappa, switched, gains, tuple(name for name, gain in gains.items() if gain.positive))


@dataclass(frozen=True)
class Candidate:
    evaluation: Evaluation
    cycle_count: int

    def __post_init__(self):
        if len(self.evaluation.sample_ids) != 32:
            raise ValueError("archive descriptors require the 32-rollout final batch")
        if type(self.cycle_count) is not int or not 0 <= self.cycle_count <= self.evaluation.parse_valid_count:
            raise ValueError("cycle_count must count only parse-valid programs")

    @property
    def structure(self):
        """NEW convention: zero parse-valid -> s=0, with count still exposed."""
        n = self.evaluation.parse_valid_count
        return int(n > 0 and 2 * self.cycle_count >= n)

    @property
    def key(self):
        return frontier(self.evaluation), self.structure


@dataclass
class CheckpointArchive:
    """One incumbent per (b,s), including b=7 saturation; no lineage collapse.

    NEW conventions: saturated cells compare rho_6; exact lead ties preserve
    insertion order. A zero-parse batch remains explicitly count=0 with s=0.
    """

    cells: dict[tuple[int, int], Candidate] = field(default_factory=dict)

    def consider(self, candidate, **bootstrap_options):
        key = candidate.key
        incumbent = self.cells.get(key)
        if incumbent is None:
            if self.cells:
                _aligned(next(iter(self.cells.values())).evaluation, candidate.evaluation)
            self.cells[key] = candidate
            return "inserted"
        gain = paired_gain(candidate.evaluation, incumbent.evaluation, min(key[0], CONDITION_COUNT), **bootstrap_options)
        if gain.positive:
            self.cells[key] = candidate
            return "replaced"
        return "retained"

    @property
    def lead(self):
        if not self.cells:
            return None
        return max(self.cells.values(), key=lambda c: (c.key[0], c.evaluation.rate(min(c.key[0], CONDITION_COUNT))))

    @property
    def saturated(self):
        return self.lead is not None and self.lead.evaluation.rate(CONDITION_COUNT) >= 1 - DELTA


@dataclass(frozen=True)
class DeltaBreakdown:
    milestones: tuple[int, ...]
    kappa: int
    deltas: tuple[float, ...]
    sampling_correction: float
    gamma: float
    initial_gap: float
    final_probe_gap: float

    @property
    def telescoping_residual(self):
        return self.gamma - (sum(self.deltas) + self.initial_gap + self.sampling_correction)


def decompose_stages(curriculum_probe_evals, direct_probe_evals, final_curriculum, final_direct, milestones, kappa):
    """NEW 2026-09-08 decomposition; not a recovered historical definition.

    Let gap_l = rho_k(course_l; P8)-rho_k(direct_l; P8) at common
    generated-token boundaries, starting from the identical base at l=0.
    Delta_l = gap_l-gap_(l-1), including the target-only tail as last segment.
    correction = Gamma(P32)-gap_final(P8). Thus sum(Delta)+correction=Gamma.

    P8 is the first 8 frozen sample slots of final P32, on the same selection
    instances. Full final counts do not alone prove P8 is the actual prefix;
    raw rollout logs must support that provenance at integration time.
    """
    course, direct, marks = tuple(curriculum_probe_evals), tuple(direct_probe_evals), tuple(milestones)
    if len(marks) < 3 or marks[0] != 0 or any(type(x) is not int for x in marks):
        raise ValueError("need base, >=1 stage boundary, and target-only-tail boundary")
    if any(a >= b for a, b in zip(marks, marks[1:])) or len(course) != len(marks) or len(direct) != len(marks):
        raise ValueError("aligned strictly increasing generated-token boundaries required")
    if course[0].checkpoint != direct[0].checkpoint or course[0].counts != direct[0].counts:
        raise ValueError("both paths must begin at the same evaluated base checkpoint")
    _aligned(final_curriculum, final_direct)
    if len(final_curriculum.sample_ids) != 32:
        raise ValueError("final evaluation must use 32 samples per instance")
    if course[-1].checkpoint != final_curriculum.checkpoint or direct[-1].checkpoint != final_direct.checkpoint:
        raise ValueError("final probe and final batch must evaluate identical checkpoints")
    gaps = []
    for left, right in zip(course, direct):
        _aligned(left, right)
        _aligned(left, final_curriculum, same_sample_ids=False)
        if len(left.sample_ids) != 8 or left.sample_ids != final_curriculum.sample_ids[:8]:
            raise ValueError("stage probes need the common first eight final sample slots")
        gaps.append(left.rate(kappa) - right.rate(kappa))
    # Integer subset bounds detect impossible first-eight claims, without claiming
    # they replace the per-rollout provenance check in the training integration.
    for probe, final in ((course[-1], final_curriculum), (direct[-1], final_direct)):
        if any(not 0 <= full - part <= 24 for prow, frow in zip(probe.counts, final.counts) for part, full in zip(prow, frow)):
            raise ValueError("first-eight probe counts cannot be a subset of the final batch")
    gamma = final_curriculum.rate(kappa) - final_direct.rate(kappa)
    result = DeltaBreakdown(marks, kappa, tuple(b - a for a, b in zip(gaps, gaps[1:])), gamma - gaps[-1], gamma, gaps[0], gaps[-1])
    if not math.isclose(result.telescoping_residual, 0, abs_tol=1e-12):
        raise ArithmeticError("telescoping consistency failure")
    return result
