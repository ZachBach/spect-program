"""Survival metrics for the time-to-milestone question.

The outcome for question one is the time to a first clinically meaningful
milestone, which is right-censored: most participants have not reached one by
the end of follow-up, and "has not yet" is not the same as "will not". Every
metric here exists to handle that asymmetry, and every one of them is easy to
get subtly wrong in a way that inflates the result.

**Censoring makes pairs incomparable.** If one participant was last seen at
three years without a milestone and another reached one at five years, nothing
is known about their order: the first might have reached a milestone at four.
Counting that pair as if the second failed first is the most common way a
concordance index comes out too high. Only pairs where the earlier time is an
observed event are usable.

**Accuracy is the wrong frame entirely.** There is no threshold to be right or
wrong about. A model outputs a risk ordering and a survival curve, and the
questions are whether the ordering is correct (concordance), whether it
separates cases from controls at a horizon anyone cares about (time-dependent
AUC), and whether its stated probabilities mean what they say (calibration).
A model can rank perfectly and be badly calibrated, and the falsifier this
program committed to requires calibrated probabilities specifically.

**Conventions are stated, not assumed.** Concordance implementations differ on
tied times and tied predictions, so two correct-looking numbers from two
packages need not agree. What this module does is written down below and
tested against hand-computed cases.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class SurvivalData:
    """Observed follow-up for a set of subjects.

    time:  time to the event, or to last contact if censored
    event: True if the milestone was observed, False if censored
    """

    time: np.ndarray
    event: np.ndarray

    def __post_init__(self) -> None:
        time = np.asarray(self.time, dtype=np.float64)
        event = np.asarray(self.event, dtype=bool)
        if time.ndim != 1:
            raise ValueError("time must be one-dimensional")
        if time.shape != event.shape:
            raise ValueError(
                f"time has {time.shape[0]} entries and event has {event.shape[0]}"
            )
        if time.size == 0:
            raise ValueError("no subjects")
        if np.any(~np.isfinite(time)):
            raise ValueError("time contains non-finite values")
        if np.any(time < 0):
            raise ValueError("time must not be negative")
        object.__setattr__(self, "time", time)
        object.__setattr__(self, "event", event)

    def __len__(self) -> int:
        return int(self.time.size)

    @property
    def n_events(self) -> int:
        return int(self.event.sum())

    @property
    def censoring_rate(self) -> float:
        return 1.0 - self.n_events / len(self)


# --- concordance ------------------------------------------------------------


@dataclass(frozen=True)
class Concordance:
    """Harrell's concordance index, with the counts that produced it.

    `comparable` is reported because the index alone hides how much evidence
    it rests on. Heavy censoring can leave a handful of usable pairs, and a
    C-index of 0.80 from twelve pairs is not the same claim as 0.80 from
    twelve thousand.
    """

    index: float
    concordant: float
    comparable: int

    @property
    def is_informative(self) -> bool:
        return self.comparable > 0


def concordance_index(data: SurvivalData, risk: Sequence[float]) -> Concordance:
    """Fraction of comparable pairs the risk score orders correctly.

    Higher risk must mean earlier expected event. If a model outputs a
    survival probability or an expected time instead, negate it first;
    passing it directly gives a value near 1 - C, which looks plausible and
    is wrong.

    Conventions, which differ between implementations:

    - A pair is comparable when the earlier of the two times is an observed
      event. Two censored subjects are never comparable, and a censored
      subject with the earlier time is not comparable with anyone.
    - Tied times contribute nothing, including when both are events. Nothing
      in the data says who failed first.
    - Tied risk scores on a comparable pair count as half concordant, which
      is what makes a constant predictor score 0.5 rather than 0.
    """
    risk = np.asarray(risk, dtype=np.float64)
    if risk.shape != data.time.shape:
        raise ValueError(
            f"risk has {risk.shape[0]} entries, survival data has {len(data)}"
        )
    if np.any(~np.isfinite(risk)):
        raise ValueError("risk contains non-finite values")

    # Pair (i, j) is comparable when t_i < t_j and i had an event.
    earlier = data.time[:, None] < data.time[None, :]
    comparable = earlier & data.event[:, None]
    n_comparable = int(comparable.sum())

    if n_comparable == 0:
        raise ValueError(
            "no comparable pairs: every pair is either tied in time or has a "
            "censored subject with the earlier time. A concordance index "
            "cannot be computed, and reporting one would be reporting nothing."
        )

    higher = risk[:, None] > risk[None, :]
    tied = risk[:, None] == risk[None, :]
    concordant = float((comparable & higher).sum() + 0.5 * (comparable & tied).sum())

    return Concordance(
        index=concordant / n_comparable,
        concordant=concordant,
        comparable=n_comparable,
    )


# --- Kaplan-Meier -----------------------------------------------------------


@dataclass(frozen=True)
class KaplanMeier:
    """A right-continuous step estimate of the survival function."""

    times: np.ndarray
    survival: np.ndarray
    n_at_risk: np.ndarray

    def at(self, horizon: float) -> float:
        """Survival at `horizon`, holding the last step before it.

        Before the first event the estimate is 1.0 by definition. Past the
        last observation it holds flat, which is an extrapolation and should
        be treated as one.
        """
        if horizon < 0:
            raise ValueError("horizon must not be negative")
        index = np.searchsorted(self.times, horizon, side="right") - 1
        return 1.0 if index < 0 else float(self.survival[index])


def kaplan_meier(data: SurvivalData) -> KaplanMeier:
    """Non-parametric survival, accounting for censoring.

    Censored subjects contribute to the risk set until they leave it, which
    is the whole point: ignoring them and treating censoring as survival
    would bias the curve upward, and dropping them entirely would bias it
    downward.
    """
    order = np.argsort(data.time, kind="stable")
    time = data.time[order]
    event = data.event[order]

    event_times = np.unique(time[event])
    if event_times.size == 0:
        return KaplanMeier(
            times=np.array([0.0]),
            survival=np.array([1.0]),
            n_at_risk=np.array([float(len(data))]),
        )

    survival = np.empty(event_times.size, dtype=np.float64)
    at_risk_counts = np.empty(event_times.size, dtype=np.float64)
    running = 1.0

    for index, t in enumerate(event_times):
        at_risk = int((time >= t).sum())
        deaths = int(((time == t) & event).sum())
        at_risk_counts[index] = at_risk
        if at_risk > 0:
            running *= 1.0 - deaths / at_risk
        survival[index] = running

    return KaplanMeier(times=event_times, survival=survival, n_at_risk=at_risk_counts)


# --- time-dependent AUC -----------------------------------------------------


@dataclass(frozen=True)
class TimeDependentAUC:
    horizon: float
    auc: float
    cases: int
    controls: int
    excluded_censored: int

    @property
    def is_informative(self) -> bool:
        return self.cases > 0 and self.controls > 0


def time_dependent_auc(
    data: SurvivalData, risk: Sequence[float], horizon: float
) -> TimeDependentAUC:
    """Cumulative-dynamic AUC at a horizon.

    Cases are subjects with an observed event at or before `horizon`.
    Controls are subjects known to be event-free past it. Subjects censored
    before the horizon are neither, and are excluded.

    That exclusion is a real limitation, not a detail. It is unbiased only if
    censoring is independent of risk. If sicker participants drop out earlier,
    which is the usual direction, this reads high. Inverse-probability-of-
    censoring weighting is the standard correction and is not implemented
    here; `excluded_censored` is returned so the size of the exposure is
    visible rather than implicit.
    """
    risk = np.asarray(risk, dtype=np.float64)
    if risk.shape != data.time.shape:
        raise ValueError("risk and survival data have different lengths")
    if horizon <= 0:
        raise ValueError("horizon must be positive")

    is_case = data.event & (data.time <= horizon)
    is_control = data.time > horizon
    excluded = int((~is_case & ~is_control).sum())

    n_cases = int(is_case.sum())
    n_controls = int(is_control.sum())
    if n_cases == 0 or n_controls == 0:
        raise ValueError(
            f"horizon {horizon} yields {n_cases} case(s) and {n_controls} "
            "control(s); an AUC needs both. Choose a horizon inside the "
            "follow-up where events have actually accrued."
        )

    case_risk = risk[is_case][:, None]
    control_risk = risk[is_control][None, :]
    wins = (case_risk > control_risk).sum()
    ties = (case_risk == control_risk).sum()

    return TimeDependentAUC(
        horizon=horizon,
        auc=float((wins + 0.5 * ties) / (n_cases * n_controls)),
        cases=n_cases,
        controls=n_controls,
        excluded_censored=excluded,
    )


# --- calibration ------------------------------------------------------------


@dataclass(frozen=True)
class CalibrationBin:
    lower: float
    upper: float
    n: int
    predicted: float
    observed: float

    @property
    def error(self) -> float:
        return self.predicted - self.observed


@dataclass(frozen=True)
class Calibration:
    """Predicted against observed survival, by band of predicted risk.

    Observed survival per band comes from Kaplan-Meier rather than a raw
    proportion, because within a band some subjects are censored before the
    horizon and a raw proportion would count them as survivors.
    """

    horizon: float
    bins: tuple[CalibrationBin, ...]

    @property
    def max_error(self) -> float:
        """Largest gap between stated and observed survival.

        Reported rather than a single slope because a slope near 1 can hide a
        band that is badly wrong, and it is usually the high-risk band that
        matters and that has fewest subjects.
        """
        return max((abs(b.error) for b in self.bins), default=float("nan"))

    @property
    def mean_absolute_error(self) -> float:
        weighted = sum(abs(b.error) * b.n for b in self.bins)
        total = sum(b.n for b in self.bins)
        return weighted / total if total else float("nan")


def calibration(
    data: SurvivalData,
    predicted_survival: Sequence[float],
    horizon: float,
    bins: int = 4,
) -> Calibration:
    """Compare predicted survival probability at a horizon with observed.

    `predicted_survival` is a probability of being event-free at `horizon`,
    not a risk score. The two run in opposite directions and swapping them
    produces a confidently miscalibrated result.
    """
    predicted = np.asarray(predicted_survival, dtype=np.float64)
    if predicted.shape != data.time.shape:
        raise ValueError("predicted_survival and survival data have different lengths")
    if np.any((predicted < 0.0) | (predicted > 1.0)):
        raise ValueError(
            "predicted_survival must lie in [0, 1]; a risk score passed here "
            "would be silently reinterpreted as a probability"
        )
    if bins < 2:
        raise ValueError("bins must be at least 2")
    if horizon <= 0:
        raise ValueError("horizon must be positive")

    order = np.argsort(predicted, kind="stable")
    groups = np.array_split(order, bins)

    rows: list[CalibrationBin] = []
    for group in groups:
        if group.size == 0:
            continue
        subset = SurvivalData(time=data.time[group], event=data.event[group])
        observed = kaplan_meier(subset).at(horizon)
        rows.append(
            CalibrationBin(
                lower=float(predicted[group].min()),
                upper=float(predicted[group].max()),
                n=int(group.size),
                predicted=float(predicted[group].mean()),
                observed=float(observed),
            )
        )

    return Calibration(horizon=horizon, bins=tuple(rows))


# --- bundle -----------------------------------------------------------------


@dataclass(frozen=True)
class SurvivalEvaluation:
    """Everything reported for one model on one test set."""

    label: str
    n: int
    n_events: int
    censoring_rate: float
    concordance: Concordance
    auc: TimeDependentAUC | None = None
    calibration_result: Calibration | None = None

    def describe(self) -> str:
        parts = [
            f"{self.label}: n={self.n}, events={self.n_events} "
            f"({self.censoring_rate:.0%} censored)",
            f"C={self.concordance.index:.3f} over {self.concordance.comparable} pairs",
        ]
        if self.auc is not None:
            parts.append(
                f"AUC@{self.auc.horizon:g}={self.auc.auc:.3f} "
                f"({self.auc.cases}/{self.auc.controls}, "
                f"{self.auc.excluded_censored} excluded)"
            )
        if self.calibration_result is not None:
            parts.append(f"max calibration error={self.calibration_result.max_error:.3f}")
        return " | ".join(parts)


def evaluate(
    data: SurvivalData,
    risk: Sequence[float],
    label: str = "model",
    horizon: float | None = None,
    predicted_survival: Sequence[float] | None = None,
    bins: int = 4,
) -> SurvivalEvaluation:
    """Concordance always; AUC and calibration when a horizon is supplied."""
    auc = None
    if horizon is not None:
        auc = time_dependent_auc(data, risk, horizon)

    calibration_result = None
    if horizon is not None and predicted_survival is not None:
        calibration_result = calibration(data, predicted_survival, horizon, bins=bins)

    return SurvivalEvaluation(
        label=label,
        n=len(data),
        n_events=data.n_events,
        censoring_rate=data.censoring_rate,
        concordance=concordance_index(data, risk),
        auc=auc,
        calibration_result=calibration_result,
    )


def evaluate_by_group(
    data: SurvivalData,
    risk: Sequence[float],
    groups: Sequence[str],
    horizon: float | None = None,
    minimum_events: int = 5,
) -> dict[str, SurvivalEvaluation | str]:
    """Per-site results, which the analysis plan requires alongside the pooled
    number.

    A pooled metric can look healthy while one centre is at chance, and that
    is exactly the failure a multi-site study is run to detect. Groups too
    small to evaluate return a string saying so rather than a number, because
    a C-index from three events is noise wearing a decimal point.
    """
    risk = np.asarray(risk, dtype=np.float64)
    groups = np.asarray(groups)
    if groups.shape != data.time.shape:
        raise ValueError("groups and survival data have different lengths")

    results: dict[str, SurvivalEvaluation | str] = {}
    for name in sorted(set(groups.tolist())):
        mask = groups == name
        subset = SurvivalData(time=data.time[mask], event=data.event[mask])
        if subset.n_events < minimum_events:
            results[name] = (
                f"not evaluated: {subset.n_events} event(s), "
                f"minimum is {minimum_events}"
            )
            continue
        try:
            results[name] = evaluate(
                subset, risk[mask], label=str(name), horizon=horizon
            )
        except ValueError as exc:
            results[name] = f"not evaluated: {exc}"
    return results
