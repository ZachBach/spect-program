"""Cox proportional hazards, fitted by Newton-Raphson on the partial likelihood.

This is the baseline the hybrid model has to beat. The falsifier this program
committed to is comparative, so the baseline is not a formality: a weak one
makes the hybrid look good for free, and that is the most common way a
comparison gets reported without meaning anything.

**Why it is written out rather than imported.** lifelines and scikit-survival
are the standard implementations and either would be a reasonable dependency.
Neither is installed, and more to the point, a fitted coefficient from an
imported library is trusted rather than checked. Everything here is tested
against values computed by hand in the test docstrings, including the partial
log-likelihood itself, which is exposed as a public function so it can be
audited rather than inferred from a fit result. If a standard library is
adopted later, these tests become the cross-check for it.

**Ties are handled by Breslow.** For tied event times the risk set is everyone
still at risk, and each event contributes separately. Efron's correction is
more accurate when ties are common and is not implemented. With continuous
follow-up times exact ties are rare, but if visit-rounded times ever produce
many, this will be slightly biased toward zero and that is the direction to
remember.

**Proportional hazards is an assumption, not a fact.** The model says a
covariate multiplies the hazard by a constant factor at every time. If a
binding ratio predicts an early milestone strongly and a late one weakly,
that assumption fails and a single coefficient averages over the difference.
Nothing here tests it. That check belongs with the real data, and it belongs
before the result is believed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..eval.survival import SurvivalData


def _sorted_risk_sets(data: SurvivalData) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Index order by ascending time, plus the first index of each tie group.

    Sorting ascending makes the risk set for row i the contiguous block from i
    to the end, so a reverse cumulative sum gives every risk-set total in one
    pass. Tied times share the risk set of the earliest row in their group,
    which is what Breslow specifies.
    """
    order = np.argsort(data.time, kind="stable")
    times = data.time[order]
    first_of_tie = np.searchsorted(times, times, side="left")
    return order, times, first_of_tie


def partial_log_likelihood(
    beta: np.ndarray, features: np.ndarray, data: SurvivalData
) -> float:
    """Breslow partial log-likelihood.

    Public so it can be checked directly. A fit that converges is not evidence
    that the objective it converged on is the right one.

    The linear predictor is centred before exponentiating. That is a
    numerical measure only: shifting every value by a constant cancels exactly
    in the ratio, since the same factor appears in the numerator and in every
    term of the risk-set sum.
    """
    beta = np.asarray(beta, dtype=np.float64).ravel()
    eta = features @ beta
    order, _, first_of_tie = _sorted_risk_sets(data)

    eta_sorted = eta[order] - eta.mean()
    event_sorted = data.event[order]

    weights = np.exp(eta_sorted)
    reverse_cumsum = np.cumsum(weights[::-1])[::-1]
    risk_totals = reverse_cumsum[first_of_tie]

    return float((eta_sorted[event_sorted] - np.log(risk_totals[event_sorted])).sum())


def _gradient_and_hessian(
    beta: np.ndarray, features: np.ndarray, data: SurvivalData
) -> tuple[np.ndarray, np.ndarray]:
    """Score vector and observed information for the Breslow likelihood."""
    eta = features @ beta
    order, _, first_of_tie = _sorted_risk_sets(data)

    x = features[order]
    eta_sorted = eta[order] - eta.mean()
    event_sorted = data.event[order]

    weights = np.exp(eta_sorted)

    # Reverse cumulative sums give every risk set at once.
    sum_w = np.cumsum(weights[::-1])[::-1][first_of_tie]
    sum_wx = np.cumsum((weights[:, None] * x)[::-1], axis=0)[::-1][first_of_tie]

    outer = weights[:, None, None] * (x[:, :, None] * x[:, None, :])
    sum_wxx = np.cumsum(outer[::-1], axis=0)[::-1][first_of_tie]

    mean_x = sum_wx / sum_w[:, None]

    gradient = (x[event_sorted] - mean_x[event_sorted]).sum(axis=0)

    covariance = (
        sum_wxx[event_sorted] / sum_w[event_sorted][:, None, None]
        - mean_x[event_sorted][:, :, None] * mean_x[event_sorted][:, None, :]
    )
    hessian = -covariance.sum(axis=0)

    return gradient, hessian


@dataclass(frozen=True)
class CoxModel:
    """A fitted proportional hazards model.

    Features are standardized internally, so `coefficients` are per standard
    deviation of the training data and are not comparable with coefficients
    fitted on raw units. `hazard_ratios` reports exp of them, which is the
    scale most readers expect.
    """

    coefficients: np.ndarray
    feature_names: tuple[str, ...]
    means: np.ndarray
    scales: np.ndarray
    log_likelihood: float
    null_log_likelihood: float
    n: int
    n_events: int
    converged: bool
    iterations: int
    l2: float
    baseline_times: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0))
    baseline_cumulative_hazard: np.ndarray = field(
        repr=False, default_factory=lambda: np.empty(0)
    )

    @property
    def hazard_ratios(self) -> dict[str, float]:
        return {
            name: float(np.exp(value))
            for name, value in zip(self.feature_names, self.coefficients)
        }

    @property
    def likelihood_ratio(self) -> float:
        """Twice the gain in log-likelihood over the null model.

        Asymptotically chi-squared with one degree of freedom per covariate.
        Reported as the raw statistic rather than a p-value, because a p-value
        computed here would need a chi-squared quantile and scipy is not a
        dependency; converting it is the reader's decision anyway.
        """
        return 2.0 * (self.log_likelihood - self.null_log_likelihood)

    def _standardize(self, features: np.ndarray) -> np.ndarray:
        features = np.asarray(features, dtype=np.float64)
        if features.ndim != 2:
            raise ValueError("features must be two-dimensional")
        if features.shape[1] != len(self.feature_names):
            raise ValueError(
                f"model was fitted on {len(self.feature_names)} feature(s), "
                f"got {features.shape[1]}"
            )
        return (features - self.means) / self.scales

    def risk(self, features: np.ndarray) -> np.ndarray:
        """Log hazard ratio. Higher means earlier expected event.

        This is the quantity `concordance_index` expects. Passing a predicted
        survival probability there instead would invert the result, so the
        model returns the risk scale rather than leaving the choice open.
        """
        return self._standardize(features) @ self.coefficients

    def predict_survival(self, features: np.ndarray, horizon: float) -> np.ndarray:
        """Probability of remaining event-free to `horizon`, per subject.

        Uses the Breslow estimate of the baseline cumulative hazard, so it is
        a step function of the training event times and holds flat past the
        last one. Predicting far beyond observed follow-up is extrapolation
        and the flatness will understate risk there.
        """
        if horizon < 0:
            raise ValueError("horizon must not be negative")
        if self.baseline_times.size == 0:
            return np.ones(np.asarray(features).shape[0])

        index = np.searchsorted(self.baseline_times, horizon, side="right") - 1
        cumulative = 0.0 if index < 0 else float(self.baseline_cumulative_hazard[index])
        return np.exp(-cumulative * np.exp(self.risk(features)))


def fit_cox(
    features: np.ndarray,
    data: SurvivalData,
    feature_names: tuple[str, ...] | list[str] | None = None,
    l2: float = 1e-6,
    max_iterations: int = 100,
    tolerance: float = 1e-9,
) -> CoxModel:
    """Fit by Newton-Raphson with step halving.

    `l2` is a small ridge penalty, on the standardized scale. It is not there
    for regularization in any meaningful sense at the default value; it keeps
    the information matrix invertible when a feature is constant or two are
    collinear, which otherwise fails with a linear algebra error far from its
    cause.
    """
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2:
        raise ValueError("features must be two-dimensional")
    if features.shape[0] != len(data):
        raise ValueError(
            f"features has {features.shape[0]} rows, survival data has {len(data)}"
        )
    if not np.all(np.isfinite(features)):
        raise ValueError("features contain non-finite values")
    if data.n_events == 0:
        raise ValueError(
            "no events: a partial likelihood is built entirely from event times, "
            "so there is nothing to fit"
        )
    if l2 < 0:
        raise ValueError("l2 must not be negative")

    n_features = features.shape[1]
    names = tuple(feature_names) if feature_names else tuple(
        f"x{i}" for i in range(n_features)
    )
    if len(names) != n_features:
        raise ValueError(f"{len(names)} name(s) for {n_features} feature(s)")

    means = features.mean(axis=0)
    scales = features.std(axis=0)
    scales[scales <= 0] = 1.0  # a constant column carries no information
    standardized = (features - means) / scales

    beta = np.zeros(n_features)
    null_ll = partial_log_likelihood(beta, standardized, data)
    current_ll = null_ll
    converged = False
    iterations = 0

    for iterations in range(1, max_iterations + 1):
        gradient, hessian = _gradient_and_hessian(beta, standardized, data)
        penalized_gradient = gradient - l2 * beta
        penalized_hessian = hessian - l2 * np.eye(n_features)

        try:
            step = np.linalg.solve(penalized_hessian, penalized_gradient)
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                "the information matrix is singular, usually because a feature is "
                "constant or two are collinear. Raise l2 or drop the feature."
            ) from exc

        # Newton direction is -H^-1 g; H is negative definite at a maximum.
        direction = -step

        scale = 1.0
        improved = False
        for _ in range(30):
            candidate = beta + scale * direction
            candidate_ll = partial_log_likelihood(candidate, standardized, data)
            penalty = 0.5 * l2 * float(candidate @ candidate)
            current_penalty = 0.5 * l2 * float(beta @ beta)
            if candidate_ll - penalty >= current_ll - current_penalty:
                beta = candidate
                improved = True
                break
            scale *= 0.5

        if not improved:
            break

        previous_ll, current_ll = current_ll, candidate_ll
        if abs(current_ll - previous_ll) < tolerance:
            converged = True
            break

    baseline_times, baseline_hazard = _breslow_baseline(beta, standardized, data)

    return CoxModel(
        coefficients=beta,
        feature_names=names,
        means=means,
        scales=scales,
        log_likelihood=current_ll,
        null_log_likelihood=null_ll,
        n=len(data),
        n_events=data.n_events,
        converged=converged,
        iterations=iterations,
        l2=l2,
        baseline_times=baseline_times,
        baseline_cumulative_hazard=baseline_hazard,
    )


def _breslow_baseline(
    beta: np.ndarray, standardized: np.ndarray, data: SurvivalData
) -> tuple[np.ndarray, np.ndarray]:
    """Breslow estimate of the cumulative baseline hazard.

    Uncentred here, unlike the likelihood. The centring that cancels in a
    ratio does not cancel in an absolute hazard, and applying it would scale
    every predicted survival probability by a constant factor.
    """
    eta = standardized @ beta
    order, times, first_of_tie = _sorted_risk_sets(data)

    weights = np.exp(eta[order])
    event_sorted = data.event[order]
    risk_totals = np.cumsum(weights[::-1])[::-1][first_of_tie]

    event_times = np.unique(times[event_sorted])
    if event_times.size == 0:
        return np.empty(0), np.empty(0)

    increments = np.empty(event_times.size)
    for index, t in enumerate(event_times):
        at_t = (times == t) & event_sorted
        increments[index] = at_t.sum() / risk_totals[np.argmax(at_t)]

    return event_times, np.cumsum(increments)
