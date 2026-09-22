"""Tests for the Cox baseline.

The partial log-likelihood is checked against values worked out by hand
before anything else is trusted. A fit that converges proves the optimizer
works, not that it optimized the right function.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from spect.eval.survival import SurvivalData, concordance_index
from spect.models.cox import CoxModel, fit_cox, partial_log_likelihood


def survival(times, events) -> SurvivalData:
    return SurvivalData(time=np.array(times, float), event=np.array(events, bool))


def simulate(n=3000, beta=(0.8, -0.5), censor_at=3.0, seed=0):
    """Draw from a Cox model with a constant baseline hazard.

    An exponential with rate exp(x'b) is exactly proportional hazards with a
    flat baseline, so the coefficients the fit should recover are known
    rather than assumed.
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, size=(n, len(beta)))
    rate = np.exp(x @ np.array(beta))
    event_time = rng.exponential(1.0 / rate)
    observed = np.minimum(event_time, censor_at)
    return x, survival(observed, event_time <= censor_at)


# --- the objective, by hand -------------------------------------------------


def test_partial_log_likelihood_at_zero():
    """Three subjects, all events at times 1, 2, 3, covariate 0, 1, 2.

    At beta = 0 every weight is 1, so each risk set contributes the log of
    its size:

        -log(3) - log(2) - log(1) = -log(3) - log(2)
    """
    data = survival([1, 2, 3], [1, 1, 1])
    x = np.array([[0.0], [1.0], [2.0]])
    expected = -math.log(3) - math.log(2)
    assert partial_log_likelihood(np.array([0.0]), x, data) == pytest.approx(expected)


def test_partial_log_likelihood_at_one():
    """Same data at beta = 1. Writing the sum out and cancelling gives

        -log(1 + e + e^2) - log(1 + e)

    which is about -3.7209. The cancellation is worth doing by hand once:
    the third subject's own term vanishes because they are the only member of
    their own risk set.
    """
    data = survival([1, 2, 3], [1, 1, 1])
    x = np.array([[0.0], [1.0], [2.0]])
    expected = -math.log(1 + math.e + math.e**2) - math.log(1 + math.e)
    assert partial_log_likelihood(np.array([1.0]), x, data) == pytest.approx(expected)
    assert expected == pytest.approx(-3.720868, abs=1e-6)


def test_breslow_gives_tied_events_the_same_risk_set():
    """Times 1, 1, 2 with all three events. Breslow puts both subjects tied at
    time 1 against the full risk set of three, so at beta = 0:

        -log(3) - log(3) - log(1) = -2 log(3)

    Efron would instead discount the second tied event and give a slightly
    larger value. That correction is not implemented, and this test pins which
    convention is in use.
    """
    data = survival([1, 1, 2], [1, 1, 1])
    x = np.array([[0.0], [1.0], [2.0]])
    assert partial_log_likelihood(np.array([0.0]), x, data) == pytest.approx(
        -2 * math.log(3)
    )


def test_censored_subjects_contribute_only_to_risk_sets():
    """A censored subject has no term of their own but still sits in the
    denominator for everyone who failed before they left."""
    with_censored = survival([1, 2, 3], [1, 0, 0])
    x = np.array([[0.0], [1.0], [2.0]])
    assert partial_log_likelihood(np.array([0.0]), x, with_censored) == pytest.approx(
        -math.log(3)
    )


def test_shifting_the_linear_predictor_does_not_change_the_likelihood():
    """The centring used for numerical stability cancels exactly, because the
    same factor appears in the numerator and in every risk-set term."""
    data = survival([1, 2, 3, 4], [1, 1, 0, 1])
    x = np.array([[0.0], [1.0], [2.0], [3.0]])
    base = partial_log_likelihood(np.array([0.7]), x, data)
    shifted = partial_log_likelihood(np.array([0.7]), x + 100.0, data)
    assert shifted == pytest.approx(base)


# --- recovery ---------------------------------------------------------------


def test_coefficients_are_recovered_from_data_generated_by_the_model():
    beta = (0.8, -0.5)
    x, data = simulate(n=4000, beta=beta, seed=1)
    model = fit_cox(x, data, feature_names=("a", "b"))

    # The model fits on standardized features, so the target is scaled too.
    expected = np.array(beta) * x.std(axis=0)
    assert model.coefficients == pytest.approx(expected, abs=0.08)
    assert model.converged


def test_a_covariate_with_no_effect_gets_a_coefficient_near_zero():
    x, data = simulate(n=3000, beta=(0.8,), seed=2)
    # A generator seeded independently of simulate's. Reusing seed 2 here
    # reproduces x exactly, which is a different test entirely; see below.
    noise = np.random.default_rng(90210).normal(size=(len(data), 1))
    model = fit_cox(np.hstack([x, noise]), data, feature_names=("signal", "noise"))
    assert abs(model.coefficients[1]) < 0.08
    assert model.coefficients[0] > 0.5


def test_two_identical_features_split_one_coefficient_between_them():
    """Found by accident, from a seed collision that made a supposed noise
    column a copy of the signal.

    With collinear inputs the partial likelihood has no unique maximum: any
    pair summing to the true effect fits equally well. The ridge term picks
    the minimum-norm solution, which splits the effect evenly. Worth pinning,
    because a halved coefficient on a real covariate usually means it is
    duplicated by another rather than that it matters half as much.
    """
    x, data = simulate(n=3000, beta=(0.8,), seed=2)
    alone = fit_cox(x, data).coefficients[0]
    duplicated = fit_cox(np.hstack([x, x]), data).coefficients

    assert duplicated[0] == pytest.approx(duplicated[1], rel=1e-6)
    assert duplicated.sum() == pytest.approx(alone, rel=1e-3)


def test_hazard_ratios_are_the_exponentiated_coefficients():
    x, data = simulate(n=1000, beta=(0.6,), seed=3)
    model = fit_cox(x, data, feature_names=("a",))
    assert model.hazard_ratios["a"] == pytest.approx(math.exp(model.coefficients[0]))


def test_fitting_improves_on_the_null_model():
    x, data = simulate(n=1500, beta=(0.9,), seed=4)
    model = fit_cox(x, data)
    assert model.log_likelihood > model.null_log_likelihood
    assert model.likelihood_ratio > 0


def test_a_useless_covariate_barely_improves_the_likelihood():
    rng = np.random.default_rng(5)
    n = 800
    data = survival(rng.exponential(2.0, n), rng.random(n) < 0.7)
    model = fit_cox(rng.normal(size=(n, 1)), data)
    assert model.likelihood_ratio < 6.0


# --- risk and survival ------------------------------------------------------


def test_risk_runs_in_the_direction_concordance_expects():
    """Higher risk must mean earlier expected event. Returning the opposite
    sign would give a concordance near 1 - C, which looks plausible."""
    x, data = simulate(n=2000, beta=(1.0,), seed=6)
    model = fit_cox(x, data)
    assert concordance_index(data, model.risk(x)).index > 0.6


def test_predicted_survival_is_a_probability_that_falls_over_time():
    x, data = simulate(n=1500, beta=(0.7,), seed=7)
    model = fit_cox(x, data)
    early = model.predict_survival(x, horizon=0.5)
    late = model.predict_survival(x, horizon=2.5)
    assert np.all((early >= 0.0) & (early <= 1.0))
    assert np.all(late <= early + 1e-12)


def test_survival_before_the_first_event_is_one():
    x, data = simulate(n=500, beta=(0.5,), seed=8)
    model = fit_cox(x, data)
    assert model.predict_survival(x, horizon=0.0) == pytest.approx(1.0)


def test_higher_risk_subjects_get_lower_predicted_survival():
    x, data = simulate(n=1500, beta=(1.0,), seed=9)
    model = fit_cox(x, data)
    survival_probability = model.predict_survival(x, horizon=1.0)
    risk = model.risk(x)
    assert np.corrcoef(risk, survival_probability)[0, 1] < -0.9


def test_predicted_survival_feeds_the_calibration_check():
    """The pieces have to fit together: the model emits a probability and
    `calibration` consumes one."""
    from spect.eval.survival import calibration

    x, data = simulate(n=2000, beta=(0.8,), seed=10)
    model = fit_cox(x, data)
    result = calibration(data, model.predict_survival(x, 1.0), horizon=1.0, bins=4)
    assert result.max_error < 0.15


# --- failure modes ----------------------------------------------------------


def test_no_events_is_refused():
    data = survival([1, 2, 3], [0, 0, 0])
    with pytest.raises(ValueError, match="no events"):
        fit_cox(np.array([[1.0], [2.0], [3.0]]), data)


def test_row_count_mismatch_is_refused():
    x, data = simulate(n=100, seed=11)
    with pytest.raises(ValueError, match="rows"):
        fit_cox(x[:50], data)


def test_a_constant_feature_does_not_crash_the_fit():
    """A zero-variance column makes the information matrix singular. The
    small ridge term is there to turn that into a coefficient of zero rather
    than a linear algebra error far from its cause."""
    x, data = simulate(n=600, beta=(0.7,), seed=12)
    padded = np.hstack([x, np.ones((len(data), 1))])
    model = fit_cox(padded, data, feature_names=("signal", "constant"))
    assert abs(model.coefficients[1]) < 1e-6


def test_predicting_with_the_wrong_feature_count_is_refused():
    x, data = simulate(n=200, beta=(0.5,), seed=13)
    model = fit_cox(x, data)
    with pytest.raises(ValueError, match="fitted on 1 feature"):
        model.risk(np.zeros((10, 3)))


def test_mismatched_feature_names_are_refused():
    x, data = simulate(n=200, beta=(0.5, 0.2), seed=14)
    with pytest.raises(ValueError, match="name"):
        fit_cox(x, data, feature_names=("only-one",))


def test_non_finite_features_are_refused():
    x, data = simulate(n=100, seed=15)
    x[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        fit_cox(x, data)


def test_a_negative_penalty_is_refused():
    x, data = simulate(n=100, seed=16)
    with pytest.raises(ValueError, match="l2"):
        fit_cox(x, data, l2=-1.0)


def test_a_larger_penalty_shrinks_the_coefficients():
    x, data = simulate(n=1500, beta=(1.0,), seed=17)
    weak = fit_cox(x, data, l2=1e-6)
    strong = fit_cox(x, data, l2=500.0)
    assert abs(strong.coefficients[0]) < abs(weak.coefficients[0])


def test_the_fit_is_deterministic():
    x, data = simulate(n=800, beta=(0.6,), seed=18)
    assert fit_cox(x, data).coefficients == pytest.approx(fit_cox(x, data).coefficients)


def test_the_model_reports_what_it_was_fitted_on():
    x, data = simulate(n=700, beta=(0.5, 0.3), seed=19)
    model = fit_cox(x, data, feature_names=("a", "b"))
    assert isinstance(model, CoxModel)
    assert model.n == len(data)
    assert model.n_events == data.n_events
    assert model.feature_names == ("a", "b")
    assert 0 < model.iterations <= 100
