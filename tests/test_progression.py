"""Tests for the longitudinal phantom.

The pair that matters is `test_two_visits_beat_one_when_the_outcome_depends
_on_rate` and its companion with the rate weight removed. Together they show
the fixture can distinguish a real longitudinal advantage from an assumed
one. A fixture where two scans always win would prove nothing, because it
would win even for a model that learned nothing.
"""

from __future__ import annotations

import numpy as np
import pytest

from spect.eval.splits import Split, leave_one_site_out
from spect.eval.survival import SurvivalData, concordance_index
from spect.features import binding_ratios, regions_from_phantom
from spect.synthetic.phantom import PhantomSpec
from spect.synthetic.progression import (
    ProgressionSpec,
    generate_trajectories,
    loss_at,
    oracle_ceiling,
    render,
    to_scan_records,
)

FAST = PhantomSpec(shape=(45, 55, 45), voxel_mm=4.0)


def standardize(values):
    values = np.asarray(values, dtype=float)
    sd = values.std()
    return np.zeros_like(values) if sd <= 0 else (values - values.mean()) / sd


def survival_of(subjects) -> SurvivalData:
    return SurvivalData(
        time=np.array([s.observed_time for s in subjects]),
        event=np.array([s.event for s in subjects]),
    )


# --- the decay model --------------------------------------------------------


def test_loss_approaches_total_without_exceeding_it():
    """The decay is bounded above by 1.0 by construction, which is why no
    clipping is needed anywhere downstream.

    At extreme inputs it reaches exactly 1.0, because 1 - 0.1 * exp(-250)
    rounds to 1.0 in float64 rather than because the model allows more. Total
    loss is a valid state; exceeding it would not be, and is what this guards.
    """
    extreme = loss_at(0.9, rate=5.0, years=50.0)
    assert extreme <= 1.0

    # Away from the floating-point limit it stays strictly inside the range.
    assert 0.9 < loss_at(0.9, rate=0.2, years=5.0) < 1.0

    for baseline in (0.0, 0.3, 0.95):
        for rate in (0.0, 0.5, 20.0):
            for years in (0.0, 1.0, 100.0):
                assert 0.0 <= loss_at(baseline, rate, years) <= 1.0


def test_no_rate_means_no_progression():
    assert loss_at(0.4, rate=0.0, years=10.0) == pytest.approx(0.4)


def test_loss_is_monotone_across_visits():
    spec = ProgressionSpec(n_subjects=40, visit_years=(0.0, 1.0, 2.0, 4.0), seed=1)
    for subject in generate_trajectories(spec):
        losses = [v.mean_putamen_loss for v in subject.visits]
        assert all(b >= a - 1e-12 for a, b in zip(losses, losses[1:]))
        assert all(0.0 <= x <= 1.0 for x in losses)


def test_a_faster_rate_gives_more_loss_at_follow_up():
    assert loss_at(0.3, rate=0.30, years=1.0) > loss_at(0.3, rate=0.05, years=1.0)


# --- recovering the rate ----------------------------------------------------


def test_the_decay_rate_is_recovered_exactly_from_two_visits():
    """The losses are exact here, so the inversion should be exact. On real
    data it would not be, and the logarithm amplifies error near total loss."""
    subjects = generate_trajectories(ProgressionSpec(n_subjects=40, seed=2))
    for subject in subjects:
        assert subject.observed_decay_rate == pytest.approx(
            subject.progression_rate, rel=1e-6, abs=1e-9
        )


def test_the_naive_difference_is_confounded_by_baseline():
    """Two subjects progressing at the same underlying rate. The one who has
    already lost more changes less in absolute terms, so a model that simply
    subtracts two images reads them as progressing differently."""
    early = loss_at(0.20, rate=0.2, years=1.0) - 0.20
    late = loss_at(0.70, rate=0.2, years=1.0) - 0.70
    assert late < early

    subjects = generate_trajectories(ProgressionSpec(n_subjects=200, seed=3))
    naive = np.array([s.observed_progression for s in subjects])
    baseline = np.array([s.baseline_visit.mean_putamen_loss for s in subjects])
    # The confound is strong enough to be worth naming, not a rounding effect.
    assert np.corrcoef(naive, baseline)[0, 1] < -0.3


# --- the scientific claim the fixture exists to test ------------------------


def _concordance(subjects, risk) -> float:
    return concordance_index(survival_of(subjects), risk).index


def test_two_visits_beat_one_when_the_outcome_depends_on_rate():
    """Progression rate is invisible in a single scan, because a rate is a
    difference and one measurement is not a difference. When the hazard
    depends on it, a model with two timepoints has information a
    cross-sectional model cannot obtain at any sample size."""
    spec = ProgressionSpec(
        n_subjects=400,
        severity_from_baseline=0.5,
        severity_from_rate=0.7,
        seed=4,
    )
    subjects = generate_trajectories(spec)

    baseline_only = standardize([s.baseline_visit.mean_putamen_loss for s in subjects])
    longitudinal = baseline_only * spec.severity_from_baseline + standardize(
        [s.observed_decay_rate for s in subjects]
    ) * spec.severity_from_rate

    assert _concordance(subjects, longitudinal) > _concordance(subjects, baseline_only) + 0.03


def test_the_advantage_disappears_when_the_outcome_ignores_rate():
    """The control for the test above. If two visits won here too, the
    fixture would be rewarding the extra scan rather than the information in
    it, and any later comparison run on it would be meaningless."""
    spec = ProgressionSpec(
        n_subjects=400,
        severity_from_baseline=1.0,
        severity_from_rate=0.0,
        seed=5,
    )
    subjects = generate_trajectories(spec)

    baseline_only = standardize([s.baseline_visit.mean_putamen_loss for s in subjects])
    longitudinal = baseline_only + standardize(
        [s.observed_decay_rate for s in subjects]
    ) * 0.7

    assert _concordance(subjects, longitudinal) <= _concordance(subjects, baseline_only) + 0.01


# --- the ceiling ------------------------------------------------------------


def test_the_oracle_cannot_reach_one():
    """Event times are exponential, so even exact knowledge of the hazard
    cannot order them perfectly. A model reporting 0.70 against this ceiling
    is doing well, and reporting it without the ceiling invites the opposite
    reading."""
    subjects = generate_trajectories(ProgressionSpec(n_subjects=400, seed=6))
    ceiling = oracle_ceiling(subjects)
    assert 0.55 < ceiling < 0.95


def test_the_task_does_not_saturate_the_way_classification_did():
    """D3 recorded the cross-sectional task reaching AUC 1.000, leaving no
    headroom. The prognostic task must not, or it is equally useless for
    comparing methods."""
    subjects = generate_trajectories(ProgressionSpec(n_subjects=400, seed=7))
    assert oracle_ceiling(subjects) < 0.9


def test_a_noisier_outcome_lowers_the_ceiling():
    """Weakening the link between severity and hazard should reduce what any
    model can achieve, which is the knob for tuning headroom."""
    strong = generate_trajectories(
        ProgressionSpec(n_subjects=400, severity_from_baseline=1.2, severity_from_rate=1.2, seed=8)
    )
    weak = generate_trajectories(
        ProgressionSpec(n_subjects=400, severity_from_baseline=0.2, severity_from_rate=0.2, seed=8)
    )
    assert oracle_ceiling(strong) > oracle_ceiling(weak)


# --- censoring --------------------------------------------------------------


def test_no_one_is_observed_past_the_end_of_the_study():
    spec = ProgressionSpec(n_subjects=200, follow_up_years=5.0, seed=9)
    subjects = generate_trajectories(spec)
    assert max(s.observed_time for s in subjects) <= 5.0 + 1e-9


def test_some_subjects_reach_a_milestone_and_some_do_not():
    subjects = generate_trajectories(ProgressionSpec(n_subjects=200, seed=10))
    events = sum(s.event for s in subjects)
    assert 0 < events < len(subjects)


def test_an_event_is_recorded_only_when_it_precedes_censoring():
    for subject in generate_trajectories(ProgressionSpec(n_subjects=200, seed=11)):
        if subject.event:
            assert subject.observed_time == pytest.approx(subject.milestone_time)
        else:
            assert subject.observed_time <= subject.milestone_time + 1e-9


def test_dropout_can_be_switched_off():
    spec = ProgressionSpec(n_subjects=100, dropout_per_year=0.0, follow_up_years=4.0, seed=12)
    subjects = generate_trajectories(spec)
    for subject in subjects:
        if not subject.event:
            assert subject.observed_time == pytest.approx(4.0)


def test_visit_counts_carry_no_outcome_information_by_default():
    """Truncating visits at dropout is realistic and hands a model a
    shortcut: fewer scans would then imply an earlier exit. Off by default,
    and this asserts it."""
    subjects = generate_trajectories(ProgressionSpec(n_subjects=100, seed=13))
    assert len({len(s.visits) for s in subjects}) == 1


def test_truncation_is_available_and_does_leak():
    """Documenting the hazard rather than hiding it. With truncation on, the
    number of visits correlates with the outcome, which is why the default is
    off."""
    spec = ProgressionSpec(
        n_subjects=300, visit_years=(0.0, 1.0, 2.0, 3.0),
        truncate_visits_at_dropout=True, seed=14,
    )
    subjects = generate_trajectories(spec)
    counts = np.array([len(s.visits) for s in subjects])
    times = np.array([s.observed_time for s in subjects])
    assert len(set(counts.tolist())) > 1
    assert np.corrcoef(counts, times)[0, 1] > 0.5


# --- determinism and validation ---------------------------------------------


def test_trajectories_are_deterministic_in_the_seed():
    a = generate_trajectories(ProgressionSpec(n_subjects=50, seed=15))
    b = generate_trajectories(ProgressionSpec(n_subjects=50, seed=15))
    assert [s.observed_time for s in a] == [s.observed_time for s in b]
    assert [s.event for s in a] == [s.event for s in b]


def test_a_different_seed_gives_a_different_cohort():
    a = generate_trajectories(ProgressionSpec(n_subjects=50, seed=16))
    b = generate_trajectories(ProgressionSpec(n_subjects=50, seed=17))
    assert [s.observed_time for s in a] != [s.observed_time for s in b]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_subjects": 0},
        {"visit_years": ()},
        {"visit_years": (1.0, 0.0)},
        {"visit_years": (-1.0,)},
        {"follow_up_years": 0.0},
        {"baseline_hazard": 0.0},
        {"dropout_per_year": 1.0},
    ],
)
def test_invalid_specs_are_rejected(kwargs):
    with pytest.raises(ValueError):
        ProgressionSpec(**kwargs)


# --- the volumes ------------------------------------------------------------


def test_rendering_produces_one_volume_per_visit():
    spec = ProgressionSpec(n_subjects=1, visit_years=(0.0, 1.0, 2.0), base_phantom=FAST, seed=18)
    subject = generate_trajectories(spec)[0]
    assert len(render(subject, spec)) == 3


def test_binding_falls_between_the_baseline_and_follow_up_volumes():
    """The progression has to be visible in the images, not only in the
    trajectory table, or a model reading volumes has nothing to learn."""
    spec = ProgressionSpec(
        n_subjects=1,
        visit_years=(0.0, 4.0),
        progression_rate_mean=0.4,
        progression_rate_sd=0.0,
        base_phantom=FAST,
        seed=19,
    )
    subject = generate_trajectories(spec)[0]
    volumes = render(subject, spec)
    first, last = (
        binding_ratios(v.volume, regions_from_phantom(v)).lowest_putamen for v in volumes
    )
    assert last < first


def test_visits_do_not_share_a_noise_field():
    """A shared noise field would make the difference between two visits
    cleaner than any real pair of scans, and a longitudinal model would look
    better than it is."""
    spec = ProgressionSpec(
        n_subjects=1, visit_years=(0.0, 0.0), base_phantom=FAST, seed=20
    )
    subject = generate_trajectories(spec)[0]
    first, second = render(subject, spec)
    assert not np.array_equal(first.volume, second.volume)


def test_rendering_is_repeatable():
    spec = ProgressionSpec(n_subjects=1, base_phantom=FAST, seed=21)
    subject = generate_trajectories(spec)[0]
    assert np.array_equal(render(subject, spec)[0].volume, render(subject, spec)[0].volume)


# --- fitting into the rest of the pipeline ----------------------------------


def test_records_split_by_site_without_leaking():
    subjects = generate_trajectories(ProgressionSpec(n_subjects=100, seed=22))
    records = to_scan_records(subjects)
    assert len(records) == sum(len(s.visits) for s in subjects)

    for split in leave_one_site_out(records):
        assert isinstance(split, Split)
        assert not split.subjects("train") & split.subjects("test")


def test_every_visit_of_a_subject_lands_in_the_same_fold():
    subjects = generate_trajectories(ProgressionSpec(n_subjects=100, seed=23))
    records = to_scan_records(subjects)
    for split in leave_one_site_out(records):
        for subject_id in split.subjects("test"):
            expected = sum(1 for r in records if r.subject_id == subject_id)
            assert sum(1 for r in split.test if r.subject_id == subject_id) == expected
