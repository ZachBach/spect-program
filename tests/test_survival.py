"""Tests for the survival metrics.

Every expected value here is either hand-computed or forced by construction.
A metric checked only against another implementation inherits that
implementation's conventions, and the conventions are exactly where survival
metrics disagree.
"""

from __future__ import annotations

import numpy as np
import pytest

from spect.eval.survival import (
    SurvivalData,
    calibration,
    concordance_index,
    evaluate,
    evaluate_by_group,
    kaplan_meier,
    time_dependent_auc,
)


def survival(times, events) -> SurvivalData:
    return SurvivalData(time=np.array(times, dtype=float), event=np.array(events, dtype=bool))


# --- concordance: the three anchors -----------------------------------------


def test_perfect_ordering_scores_one():
    data = survival([1, 2, 3, 4], [1, 1, 1, 1])
    assert concordance_index(data, [4, 3, 2, 1]).index == 1.0


def test_reversed_ordering_scores_zero():
    data = survival([1, 2, 3, 4], [1, 1, 1, 1])
    assert concordance_index(data, [1, 2, 3, 4]).index == 0.0


def test_a_constant_predictor_scores_one_half():
    """Forced by the tied-prediction convention. If ties counted as zero, a
    useless model would score 0 and look worse than chance."""
    data = survival([1, 2, 3, 4], [1, 1, 1, 1])
    assert concordance_index(data, [7, 7, 7, 7]).index == 0.5


# --- concordance: hand-computed ---------------------------------------------


def test_matches_a_hand_computed_value():
    """Three subjects, all events, at times 1, 2, 3 with risks 1, 3, 2.

    Comparable pairs, being those where the earlier time had an event:
      (1,2): risk 1 vs 3, earlier subject has lower risk  -> discordant
      (1,3): risk 1 vs 2, lower                           -> discordant
      (2,3): risk 3 vs 2, higher                          -> concordant
    One of three.
    """
    data = survival([1, 2, 3], [1, 1, 1])
    result = concordance_index(data, [1, 3, 2])
    assert result.comparable == 3
    assert result.concordant == 1.0
    assert result.index == pytest.approx(1 / 3)


def test_censoring_removes_pairs_by_the_documented_rule():
    """A censored subject with the earliest time is comparable with nobody:
    they might have had an event after they were last seen.

    Subject A censored at 1, B event at 2, C event at 3. The only comparable
    pair is (B, C).
    """
    data = survival([1, 2, 3], [0, 1, 1])
    assert concordance_index(data, [3, 2, 1]).comparable == 1


def test_a_censored_subject_is_still_usable_as_the_later_half_of_a_pair():
    """Someone last seen event-free at year five did survive past year two,
    so they are comparable with anyone who had an event before then.
    Discarding them entirely would throw away real information."""
    data = survival([2, 5], [1, 0])
    assert concordance_index(data, [2, 1]).comparable == 1


def test_tied_times_contribute_nothing():
    """Nothing in the data says who failed first."""
    data = survival([3, 3], [1, 1])
    with pytest.raises(ValueError, match="no comparable pairs"):
        concordance_index(data, [2, 1])


def test_all_censored_is_refused_rather_than_reported_as_a_half():
    data = survival([1, 2, 3], [0, 0, 0])
    with pytest.raises(ValueError, match="no comparable pairs"):
        concordance_index(data, [3, 2, 1])


def test_comparable_count_is_reported_alongside_the_index():
    """0.80 from twelve pairs is not the claim that 0.80 from twelve thousand
    pairs is, and the index alone cannot tell them apart."""
    data = survival([1, 2, 3, 4], [1, 1, 1, 1])
    result = concordance_index(data, [4, 3, 2, 1])
    assert result.comparable == 6
    assert result.is_informative


def test_passing_survival_instead_of_risk_inverts_the_score():
    """The trap the docstring warns about. A survival probability runs
    opposite to a risk score, and the inverted value looks plausible."""
    data = survival([1, 2, 3, 4], [1, 1, 1, 1])
    risk = np.array([4.0, 3.0, 2.0, 1.0])
    correct = concordance_index(data, risk).index
    inverted = concordance_index(data, -risk).index
    assert correct == 1.0
    assert inverted == pytest.approx(1.0 - correct)


def test_length_mismatch_is_refused():
    data = survival([1, 2, 3], [1, 1, 1])
    with pytest.raises(ValueError, match="risk has"):
        concordance_index(data, [1, 2])


# --- Kaplan-Meier -----------------------------------------------------------


def test_without_censoring_it_is_the_empirical_survival():
    data = survival([1, 2, 3, 4], [1, 1, 1, 1])
    curve = kaplan_meier(data)
    assert curve.survival == pytest.approx([0.75, 0.5, 0.25, 0.0])


def test_matches_a_hand_computed_curve_with_censoring():
    """Times 1 to 5, with the subject at time 3 censored.

      t=1: 5 at risk, 1 event -> 1 - 1/5            = 0.8
      t=2: 4 at risk, 1 event -> 0.8 * (1 - 1/4)    = 0.6
      t=3: censored, no step
      t=4: 2 at risk, 1 event -> 0.6 * (1 - 1/2)    = 0.3
      t=5: 1 at risk, 1 event -> 0.3 * (1 - 1/1)    = 0.0

    The censored subject counts in the risk set at t=1 and t=2 and leaves
    before t=4. Treating censoring as survival would hold the curve too high;
    dropping the subject outright would pull it too low.
    """
    data = survival([1, 2, 3, 4, 5], [1, 1, 0, 1, 1])
    curve = kaplan_meier(data)
    assert curve.times == pytest.approx([1, 2, 4, 5])
    assert curve.survival == pytest.approx([0.8, 0.6, 0.3, 0.0])
    assert curve.n_at_risk == pytest.approx([5, 4, 2, 1])


def test_survival_is_one_before_the_first_event():
    curve = kaplan_meier(survival([5, 6], [1, 1]))
    assert curve.at(0.0) == 1.0
    assert curve.at(4.9) == 1.0


def test_with_no_events_the_curve_stays_at_one():
    curve = kaplan_meier(survival([1, 2, 3], [0, 0, 0]))
    assert curve.at(10.0) == 1.0


def test_the_curve_never_increases():
    rng = np.random.default_rng(0)
    data = survival(rng.uniform(1, 10, 60), rng.random(60) < 0.6)
    curve = kaplan_meier(data)
    assert np.all(np.diff(curve.survival) <= 1e-12)


# --- time-dependent AUC -----------------------------------------------------


def test_auc_anchors():
    data = survival([1, 2, 3, 4], [1, 1, 0, 0])
    assert time_dependent_auc(data, [4, 3, 2, 1], horizon=2.5).auc == 1.0
    assert time_dependent_auc(data, [1, 2, 3, 4], horizon=2.5).auc == 0.0
    assert time_dependent_auc(data, [9, 9, 9, 9], horizon=2.5).auc == 0.5


def test_cases_and_controls_are_counted_by_the_documented_rule():
    data = survival([1, 2, 3, 4], [1, 1, 0, 0])
    result = time_dependent_auc(data, [4, 3, 2, 1], horizon=2.5)
    assert (result.cases, result.controls) == (2, 2)
    assert result.excluded_censored == 0


def test_subjects_censored_before_the_horizon_are_excluded_and_counted():
    """The exclusion is the method's main limitation, so its size is
    reported rather than left implicit. Censoring that tracks risk biases
    this upward, and inverse-probability weighting is the correction that is
    deliberately not implemented yet."""
    data = survival([1, 2, 3, 4], [1, 0, 1, 0])
    result = time_dependent_auc(data, [4, 3, 2, 1], horizon=2.5)
    assert result.cases == 1
    assert result.controls == 2
    assert result.excluded_censored == 1


def test_a_horizon_with_no_cases_is_refused():
    data = survival([5, 6, 7], [1, 1, 1])
    with pytest.raises(ValueError, match="case"):
        time_dependent_auc(data, [3, 2, 1], horizon=1.0)


def test_a_horizon_past_all_follow_up_is_refused():
    data = survival([1, 2, 3], [1, 1, 1])
    with pytest.raises(ValueError, match="control"):
        time_dependent_auc(data, [3, 2, 1], horizon=99.0)


# --- calibration ------------------------------------------------------------


def _two_band_cohort():
    """Forty subjects in two bands whose true survival at horizon 10 is known.

    Band one: 20 subjects, 10 with an event at t=5, so survival is 0.50.
    Band two: 20 subjects, 2 with an event at t=5, so survival is 0.90.
    Everyone else is censored at t=20, past the horizon.
    """
    times, events = [], []
    for n_events in (10, 2):
        times += [5.0] * n_events + [20.0] * (20 - n_events)
        events += [True] * n_events + [False] * (20 - n_events)
    return survival(times, events)


def test_well_calibrated_predictions_show_almost_no_error():
    data = _two_band_cohort()
    predicted = np.array([0.5] * 20 + [0.9] * 20)
    result = calibration(data, predicted, horizon=10.0, bins=2)
    assert result.max_error < 1e-9
    assert [b.observed for b in result.bins] == pytest.approx([0.5, 0.9])


def test_overconfident_predictions_are_detected():
    data = _two_band_cohort()
    claimed = np.array([0.9] * 40)
    result = calibration(data, claimed, horizon=10.0, bins=2)
    assert result.max_error == pytest.approx(0.4, abs=1e-9)


def test_observed_survival_uses_kaplan_meier_not_a_raw_proportion():
    """Each band holds four subjects: one event at t=5, one censored at t=6,
    two still event-free at t=20. Horizon is 10.

      Kaplan-Meier:  t=5, 4 at risk, 1 event -> 0.75, and the subject
                     censored at t=6 simply leaves the risk set.
      Counting only those known to pass the horizon: 2 of 4 -> 0.50.

    The censored subject is the whole difference. Counting them as a failure
    understates survival, and the naive proportion is the one that does it.
    """
    times = [5, 6, 20, 20] * 2
    events = [True, False, False, False] * 2
    data = survival(times, events)

    result = calibration(data, np.full(8, 0.75), horizon=10.0, bins=2)
    observed = [b.observed for b in result.bins]

    assert observed == pytest.approx([0.75, 0.75])
    naive = 2 / 4
    assert all(o > naive for o in observed)


def test_a_risk_score_passed_as_a_probability_is_refused():
    data = _two_band_cohort()
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        calibration(data, np.arange(40, dtype=float), horizon=10.0, bins=2)


def test_calibration_error_is_weighted_by_band_size():
    data = _two_band_cohort()
    result = calibration(data, np.array([0.5] * 20 + [0.9] * 20), horizon=10.0, bins=2)
    assert result.mean_absolute_error == pytest.approx(0.0, abs=1e-9)


# --- validation of the inputs themselves ------------------------------------


def test_mismatched_time_and_event_lengths_are_refused():
    with pytest.raises(ValueError, match="entries"):
        SurvivalData(time=np.array([1.0, 2.0]), event=np.array([True]))


def test_negative_times_are_refused():
    with pytest.raises(ValueError, match="negative"):
        survival([-1, 2], [1, 1])


def test_an_empty_cohort_is_refused():
    with pytest.raises(ValueError, match="no subjects"):
        SurvivalData(time=np.array([]), event=np.array([]))


def test_censoring_rate_is_reported():
    data = survival([1, 2, 3, 4], [1, 1, 0, 0])
    assert data.n_events == 2
    assert data.censoring_rate == pytest.approx(0.5)


# --- the bundle and per-site reporting --------------------------------------


def test_evaluate_bundles_the_three_measures():
    data = survival([1, 2, 3, 4, 5, 6], [1, 1, 1, 0, 0, 0])
    result = evaluate(
        data,
        risk=[6, 5, 4, 3, 2, 1],
        horizon=3.5,
        predicted_survival=[0.1, 0.2, 0.3, 0.7, 0.8, 0.9],
        bins=2,
        label="baseline",
    )
    assert result.concordance.index == 1.0
    assert result.auc is not None and result.auc.auc == 1.0
    assert result.calibration_result is not None
    assert "baseline" in result.describe()
    assert "censored" in result.describe()


def test_per_site_reporting_evaluates_each_site_separately():
    """A pooled metric can look healthy while one centre sits at chance,
    which is the failure a multi-site study exists to detect."""
    times = list(range(1, 13))
    events = [1] * 12
    data = survival(times, events)
    # Site A ordered correctly; site B ordered backwards.
    risk = [6, 5, 4, 3, 2, 1] + [1, 2, 3, 4, 5, 6]
    groups = ["A"] * 6 + ["B"] * 6

    results = evaluate_by_group(data, risk, groups, minimum_events=5)
    assert results["A"].concordance.index == 1.0
    assert results["B"].concordance.index == 0.0


def test_a_site_with_too_few_events_is_named_not_scored():
    """A concordance index from three events is noise wearing a decimal
    point, so it is reported as a sentence instead of a number."""
    data = survival([1, 2, 3, 4, 5, 6, 7, 8], [1, 1, 1, 1, 1, 1, 0, 0])
    groups = ["big"] * 6 + ["tiny"] * 2
    results = evaluate_by_group(data, list(range(8, 0, -1)), groups, minimum_events=5)
    assert isinstance(results["tiny"], str)
    assert "not evaluated" in results["tiny"]
    assert results["big"].concordance.index == 1.0
