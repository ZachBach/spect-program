"""Subjects who progress over time, with a milestone to predict.

The cross-sectional phantom cannot support the survival question: it has no
time axis, so there is nothing for a time-to-milestone model to be trained
against. This adds one.

Four design decisions carry the module.

**The outcome depends on progression rate, not only on baseline state.** A
single scan shows how much binding is already lost. It cannot show how fast
it is being lost, because a rate is a difference and one measurement is not a
difference. Two scans can. So the latent severity that drives the hazard here
is a weighted sum of baseline loss and progression rate, which makes
"longitudinal beats cross-sectional" a hypothesis the fixture can actually
test rather than an assumption baked into it. Setting `severity_from_rate` to
zero removes the advantage, and a test asserts the difference disappears with
it.

**The outcome is stochastic, deliberately.** A milestone that was a
deterministic function of the scan would be perfectly predictable, the way
the classification task already saturates (see D3 in docs/decisions.md). Real
milestones depend on cognitive reserve, comorbidity, treatment and the
measurement of the milestone itself, none of which a scan sees. Event times
are drawn from an exponential with a proportional hazard, so there is an
achievable ceiling below 1.0 and `oracle_ceiling` measures it. A model should
be judged against that ceiling, not against perfection.

**Part of the outcome is visible only in the spatial pattern.** Gradient
steepness varies between subjects and can contribute to the hazard through
`severity_from_gradient`. It is exactly invisible to a region mean: the
gradient redistributes loss within the putamen while preserving its mean, so
the binding ratio is unchanged to four decimal places while the
posterior-to-anterior contrast moves from 0.99 to 0.89. Without a term like
this, the outcome is a function of quantities region means already measure,
and a model reading whole volumes has nothing left to find. See D5 in
docs/decisions.md.

**Binding decays exponentially toward total loss.** Loss is modelled as
1 - (1 - loss_0) * exp(-rate * t), so a subject loses a fraction of what
remains rather than a fixed amount. That keeps loss inside [0, 1] without
clipping and matches the shape of decline better than a straight line, which
would run past total loss and have to be truncated.

Numbers here are plausible magnitudes, not fitted ones. Nothing in this module
has been calibrated against a cohort and none of it may be cited as though it
had been.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np

from .phantom import Phantom, PhantomSpec, generate


@dataclass(frozen=True)
class ProgressionSpec:
    """Everything that determines a synthetic longitudinal cohort."""

    n_subjects: int = 120

    #: Imaging timepoints in years. The precedent for this program, Adams
    #: et al. 2021 (doi:10.1016/j.compbiomed.2021.104312), used year 0 and
    #: year 1 together.
    visit_years: tuple[float, ...] = (0.0, 1.0)

    baseline_loss_mean: float = 0.35
    baseline_loss_sd: float = 0.12

    #: Fractional loss of remaining binding per year.
    progression_rate_mean: float = 0.18
    progression_rate_sd: float = 0.08

    #: Per-subject steepness of the posterior-to-anterior gradient in putamen
    #: loss. Varies between subjects so it can carry information; the
    #: cross-sectional phantom holds it fixed.
    gradient_mean: float = 0.5
    gradient_sd: float = 0.25

    #: How much each contributes to the log hazard, in standard deviations.
    #:
    #: The three terms are deliberately visible to different kinds of model.
    #: Baseline loss is in any single scan. Rate needs two scans, because a
    #: rate is a difference. Gradient steepness needs the spatial pattern
    #: inside the structure, and is **exactly invisible to a region mean**:
    #: the gradient redistributes loss within the putamen while preserving its
    #: mean, so the binding ratio is unchanged to four decimal places while
    #: the posterior-to-anterior contrast moves from 0.99 to 0.89.
    #:
    #: That makes "a model reading the volume beats one reading region means"
    #: a claim this fixture can test rather than one it assumes. Set the
    #: weight to zero and the advantage must vanish, which is the control.
    severity_from_baseline: float = 0.5
    severity_from_rate: float = 0.7
    severity_from_gradient: float = 0.0

    #: Events per year at mean severity.
    baseline_hazard: float = 0.18

    #: Administrative censoring: the study ends here.
    follow_up_years: float = 8.0

    #: Yearly probability of leaving the study early.
    dropout_per_year: float = 0.05

    #: Asymmetry between sides at baseline, as in the cross-sectional phantom.
    asymmetry_mean: float = 0.20
    asymmetry_sd: float = 0.08

    #: When True, a subject who drops out keeps only the visits before they
    #: left. Realistic, and dangerous: the number of scans then carries
    #: information about the outcome, and a model can predict the milestone by
    #: counting images. Left False by default so the fixture does not hand a
    #: model a shortcut that would not exist in a properly analysed study.
    truncate_visits_at_dropout: bool = False

    seed: int = 0
    base_phantom: PhantomSpec = field(default_factory=PhantomSpec)

    def __post_init__(self) -> None:
        if self.n_subjects < 1:
            raise ValueError("n_subjects must be at least 1")
        if not self.visit_years:
            raise ValueError("at least one visit is required")
        if list(self.visit_years) != sorted(self.visit_years):
            raise ValueError("visit_years must be ascending")
        if any(v < 0 for v in self.visit_years):
            raise ValueError("visit_years must not be negative")
        if self.follow_up_years <= 0:
            raise ValueError("follow_up_years must be positive")
        if self.baseline_hazard <= 0:
            raise ValueError("baseline_hazard must be positive")
        if not 0.0 <= self.dropout_per_year < 1.0:
            raise ValueError("dropout_per_year must lie in [0, 1)")
        if not 0.0 <= self.gradient_mean <= 1.0:
            raise ValueError("gradient_mean must lie in [0, 1]")
        if self.gradient_sd < 0:
            raise ValueError("gradient_sd must not be negative")


@dataclass(frozen=True)
class Visit:
    """One imaging timepoint for one subject."""

    year: float
    putamen_loss: tuple[float, float]
    caudate_loss: tuple[float, float]

    @property
    def mean_putamen_loss(self) -> float:
        return sum(self.putamen_loss) / 2.0


@dataclass(frozen=True)
class Subject:
    """One synthetic participant, their visits, and their outcome.

    `latent_severity` and `milestone_time` are ground truth. They exist so a
    ceiling can be measured and must never be used as model inputs; a test
    guards the naming so they are hard to pass by accident.
    """

    subject_id: str
    site_id: str
    baseline_loss: tuple[float, float]
    progression_rate: float
    gradient: float
    latent_severity: float
    milestone_time: float
    observed_time: float
    event: bool
    visits: tuple[Visit, ...]

    @property
    def baseline_visit(self) -> Visit:
        return self.visits[0]

    @property
    def observed_progression(self) -> float:
        """Naive change in mean putamen loss per year.

        What a model gets by subtracting two scans. It is **confounded with
        baseline**, and not slightly: since loss decays as
        1 - (1 - L0) exp(-rt), the change over a year is (1 - L0)(1 - exp(-r)).
        A subject who has already lost a great deal changes less in absolute
        terms at the same underlying rate, so this reads low for exactly the
        people who are worst off.

        `observed_decay_rate` inverts the relationship properly. Both are
        provided because the gap between them is the point: a longitudinal
        model that differences two images without accounting for the baseline
        is measuring something other than progression.
        """
        if len(self.visits) < 2:
            return float("nan")
        first, last = self.visits[0], self.visits[-1]
        span = last.year - first.year
        if span <= 0:
            return float("nan")
        return (last.mean_putamen_loss - first.mean_putamen_loss) / span

    @property
    def observed_decay_rate(self) -> float:
        """Progression rate recovered from two visits, without the confound.

        Inverting the decay gives r = -ln((1 - L_last) / (1 - L_first)) / span,
        which returns the latent rate exactly when the losses are exact. On
        real data they are not, and the logarithm amplifies error as loss
        approaches total, which is a genuine limitation of estimating a rate
        from two noisy measurements rather than an artefact of the phantom.
        """
        if len(self.visits) < 2:
            return float("nan")
        first, last = self.visits[0], self.visits[-1]
        span = last.year - first.year
        if span <= 0:
            return float("nan")

        remaining_first = max(1e-9, 1.0 - first.mean_putamen_loss)
        remaining_last = max(1e-9, 1.0 - last.mean_putamen_loss)
        return -math.log(remaining_last / remaining_first) / span


def loss_at(baseline_loss: float, rate: float, years: float) -> float:
    """Exponential decay of remaining binding.

    Bounded above by 1.0 by construction, so no clipping is needed and the
    mean is not distorted near total loss.
    """
    return 1.0 - (1.0 - baseline_loss) * math.exp(-rate * years)


def generate_trajectories(spec: ProgressionSpec | None = None) -> list[Subject]:
    """Build a cohort's trajectories and outcomes, without rendering volumes.

    Separated from rendering because the tabular part is cheap and the
    volumes are not. The survival harness, the splits and the baseline can
    all be exercised on trajectories alone.
    """
    spec = spec or ProgressionSpec()
    rng = np.random.default_rng(spec.seed)

    baseline = np.clip(
        rng.normal(spec.baseline_loss_mean, spec.baseline_loss_sd, spec.n_subjects),
        0.0,
        0.95,
    )
    rates = np.clip(
        rng.normal(spec.progression_rate_mean, spec.progression_rate_sd, spec.n_subjects),
        0.0,
        None,
    )
    asymmetry = np.clip(
        rng.normal(spec.asymmetry_mean, spec.asymmetry_sd, spec.n_subjects), 0.0, None
    )
    gradients = np.clip(
        rng.normal(spec.gradient_mean, spec.gradient_sd, spec.n_subjects), 0.0, 1.0
    )

    # Standardize so the severity weights are in standard deviations and the
    # baseline hazard keeps its meaning when the distributions are changed.
    latent = spec.severity_from_baseline * _standardize(baseline)
    latent = latent + spec.severity_from_rate * _standardize(rates)
    latent = latent + spec.severity_from_gradient * _standardize(gradients)

    hazard = spec.baseline_hazard * np.exp(latent)
    milestone = rng.exponential(1.0 / hazard)

    dropout = (
        rng.exponential(1.0 / spec.dropout_per_year, spec.n_subjects)
        if spec.dropout_per_year > 0
        else np.full(spec.n_subjects, np.inf)
    )
    censor = np.minimum(dropout, spec.follow_up_years)

    observed = np.minimum(milestone, censor)
    event = milestone <= censor

    subjects: list[Subject] = []
    for i in range(spec.n_subjects):
        worse = float(min(1.0, baseline[i] + asymmetry[i] / 2.0))
        better = float(max(0.0, baseline[i] - asymmetry[i] / 2.0))
        if rng.random() < 0.5:
            worse, better = better, worse

        years = [
            y
            for y in spec.visit_years
            if not spec.truncate_visits_at_dropout or y <= observed[i]
        ]
        if not years:
            years = [spec.visit_years[0]]

        visits = tuple(
            Visit(
                year=float(y),
                putamen_loss=(
                    loss_at(worse, float(rates[i]), y),
                    loss_at(better, float(rates[i]), y),
                ),
                caudate_loss=(
                    loss_at(worse / 3.0, float(rates[i]), y),
                    loss_at(better / 3.0, float(rates[i]), y),
                ),
            )
            for y in years
        )

        subjects.append(
            Subject(
                subject_id=f"sub-{i:04d}",
                site_id=f"site-{i % 5}",
                baseline_loss=(worse, better),
                progression_rate=float(rates[i]),
                gradient=float(gradients[i]),
                latent_severity=float(latent[i]),
                milestone_time=float(milestone[i]),
                observed_time=float(observed[i]),
                event=bool(event[i]),
                visits=visits,
            )
        )

    return subjects


def _standardize(values: np.ndarray) -> np.ndarray:
    sd = values.std()
    if sd <= 0:
        return np.zeros_like(values)
    return (values - values.mean()) / sd


def render(subject: Subject, spec: ProgressionSpec | None = None) -> list[Phantom]:
    """Volumes for one subject's visits.

    Each visit gets its own noise seed derived from the subject and the year,
    so repeated calls are identical and no two visits share a noise field. A
    shared field would make the difference between visits cleaner than any
    real pair of scans, and a longitudinal model would look better than it is.
    """
    spec = spec or ProgressionSpec()
    volumes: list[Phantom] = []

    for index, visit in enumerate(subject.visits):
        seed = abs(hash((subject.subject_id, index))) % (2**31 - 1)
        volumes.append(
            generate(
                replace(
                    spec.base_phantom,
                    putamen_loss=visit.putamen_loss,
                    caudate_loss=visit.caudate_loss,
                    posterior_gradient=subject.gradient,
                    seed=seed,
                )
            )
        )

    return volumes


def oracle_ceiling(subjects: list[Subject]) -> float:
    """Concordance achievable by a model that knows the latent severity.

    The event times are drawn from an exponential, so even perfect knowledge
    of the hazard cannot order them perfectly. Reporting a model's concordance
    without this number invites reading 0.70 as a poor result when it may be
    close to everything the data contains.
    """
    from ..eval.survival import SurvivalData, concordance_index

    data = SurvivalData(
        time=np.array([s.observed_time for s in subjects]),
        event=np.array([s.event for s in subjects]),
    )
    return concordance_index(data, [s.latent_severity for s in subjects]).index


def to_scan_records(subjects: list[Subject]):
    """Adapt trajectories to the records the splitting module takes.

    One record per visit, all carrying the subject's identifier, so the split
    guards see repeat imaging for what it is.
    """
    from ..eval.splits import ScanRecord

    return [
        ScanRecord(
            scan_id=f"{s.subject_id}_y{visit.year:g}",
            subject_id=s.subject_id,
            site_id=s.site_id,
            visit=f"y{visit.year:g}",
            label="event" if s.event else "censored",
        )
        for s in subjects
        for visit in s.visits
    ]
