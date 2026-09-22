"""Cohorts of phantoms, with between-subject variation.

A normal database needs a spread, and the spread has to come from the right
place. Generating twenty phantoms that differ only in their noise seed gives a
standard deviation reflecting counting statistics alone, which is far smaller
than the real variation between healthy people. Z-scores against such a
database come out enormous, every subject looks abnormal, and the failure is
invisible because the arithmetic is correct throughout.

So variation here is layered the way it is in practice:

**Shared, per subject.** Injected dose, camera sensitivity, body habitus and
scan timing scale the whole striatum together. This is the largest component
and it is the one that mostly divides out of a putamen-to-caudate ratio, which
is exactly why that ratio is a more stable feature than either structure alone.

**Independent, per structure.** Genuine anatomical and physiological
differences between one person's caudate and another's.

**Asymmetry.** Healthy striata are not perfectly symmetric. Without this,
any asymmetry at all would read as pathological, and a classifier trained on
such a cohort would learn a threshold that does not exist.

The coefficients below are plausible magnitudes, not measured ones. They have
not been fitted to any cohort and must not be cited as though they were.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .phantom import Phantom, PhantomSpec, generate

#: Spread of the subject-level scale that moves the whole striatum together.
DEFAULT_SHARED_CV = 0.12

#: Spread of the independent per-structure variation.
DEFAULT_STRUCTURE_CV = 0.06

#: Spread of healthy left/right asymmetry, as a fraction of the mean.
DEFAULT_ASYMMETRY_CV = 0.05


def healthy_cohort(
    n: int = 24,
    shared_cv: float = DEFAULT_SHARED_CV,
    structure_cv: float = DEFAULT_STRUCTURE_CV,
    asymmetry_cv: float = DEFAULT_ASYMMETRY_CV,
    seed: int = 0,
    base: PhantomSpec | None = None,
) -> list[Phantom]:
    """`n` phantoms with no loss, varying the way healthy people vary.

    Deterministic in `seed`. The base spec supplies everything not varied
    here, so a cohort can be generated at a different resolution or count
    level by passing one in.
    """
    if n < 1:
        raise ValueError("a cohort needs at least one subject")
    for name, value in (
        ("shared_cv", shared_cv),
        ("structure_cv", structure_cv),
        ("asymmetry_cv", asymmetry_cv),
    ):
        if value < 0:
            raise ValueError(f"{name} must not be negative")

    base = base or PhantomSpec()
    rng = np.random.default_rng(seed)
    cohort: list[Phantom] = []

    for index in range(n):
        shared = 1.0 + rng.normal(0.0, shared_cv)
        putamen_scale = shared * (1.0 + rng.normal(0.0, structure_cv))
        caudate_scale = shared * (1.0 + rng.normal(0.0, structure_cv))

        # Asymmetry is expressed as a loss on whichever side comes out lower,
        # because the spec carries loss rather than a per-side ratio. Healthy
        # asymmetry is small, so this stays near zero.
        tilt = rng.normal(0.0, asymmetry_cv)
        left_loss = max(0.0, tilt)
        right_loss = max(0.0, -tilt)

        spec = replace(
            base,
            healthy_putamen_ratio=max(0.05, base.healthy_putamen_ratio * putamen_scale),
            healthy_caudate_ratio=max(0.05, base.healthy_caudate_ratio * caudate_scale),
            putamen_loss=(min(1.0, left_loss), min(1.0, right_loss)),
            caudate_loss=(min(1.0, left_loss / 2.0), min(1.0, right_loss / 2.0)),
            seed=int(rng.integers(0, 2**31 - 1)),
        )
        cohort.append(generate(spec))

    return cohort


def parkinsonian_cohort(
    n: int = 24,
    severity: float = 0.5,
    severity_sd: float = 0.15,
    asymmetry: float = 0.25,
    asymmetry_sd: float = 0.10,
    shared_cv: float = DEFAULT_SHARED_CV,
    seed: int = 0,
    base: PhantomSpec | None = None,
) -> list[Phantom]:
    """`n` phantoms with the asymmetric, putamen-first pattern.

    The worse side is chosen at random per subject rather than always being
    the same one. A cohort where every patient is worse on the left would let
    a model separate groups on side alone, which is an artefact of the
    generator and not a finding.
    """
    if n < 1:
        raise ValueError("a cohort needs at least one subject")

    base = base or PhantomSpec()
    rng = np.random.default_rng(seed)
    cohort: list[Phantom] = []

    for _ in range(n):
        shared = 1.0 + rng.normal(0.0, shared_cv)
        subject_severity = float(np.clip(rng.normal(severity, severity_sd), 0.0, 1.0))
        subject_asymmetry = float(np.clip(rng.normal(asymmetry, asymmetry_sd), 0.0, 1.0))

        worse = float(np.clip(subject_severity + subject_asymmetry / 2.0, 0.0, 1.0))
        better = float(np.clip(subject_severity - subject_asymmetry / 2.0, 0.0, 1.0))
        if rng.random() < 0.5:
            worse, better = better, worse

        spec = replace(
            base,
            healthy_putamen_ratio=max(0.05, base.healthy_putamen_ratio * shared),
            healthy_caudate_ratio=max(0.05, base.healthy_caudate_ratio * shared),
            putamen_loss=(worse, better),
            caudate_loss=(worse / 3.0, better / 3.0),
            seed=int(rng.integers(0, 2**31 - 1)),
        )
        cohort.append(generate(spec))

    return cohort
