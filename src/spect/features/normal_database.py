"""Z-scores against a database of healthy binding ratios.

This is the baseline the hybrid model has to beat. Arnaldi et al. 2024
(Ann Neurol, doi:10.1002/ana.26902) reached 85% sensitivity and 86%
specificity separating converters to Parkinson's disease from converters to
dementia with Lewy bodies using 14 regional z-scores across 13 centres. A
volumetric model that cannot beat a vector of z-scores is not earning its
complexity, which is the falsifier this program committed to before any data
was requested.

A z-score is (subject - healthy mean) / healthy standard deviation, so it
inherits everything about the cohort the database was built from. A database
assembled on one camera, one reconstruction and one set of region definitions
does not transfer to another, and the resulting z-scores will be confidently
wrong rather than obviously wrong. Real quantification packages ship a normal
database precisely because it cannot be improvised.

The database built here comes from phantoms. It is labelled as such, and
`z_scores` refuses to score real data against it. That refusal is the point
of the module as much as the arithmetic is.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np

from ..provenance import Provenance, ProvenanceError
from .binding_ratio import RegionRatios

#: Below this, a "database" is a handful of observations and its standard
#: deviation is mostly sampling noise. Real normal databases run to dozens.
MINIMUM_SUBJECTS = 8


@dataclass(frozen=True)
class NormalDatabase:
    """Healthy means and standard deviations for each binding-ratio feature.

    `source` records what the database was built from in enough detail that
    someone reading a z-score six months later can tell whether it applies to
    the scan in front of them. It is required, not optional, for the same
    reason a reference interval without a citation is not usable.
    """

    means: Mapping[str, float]
    sds: Mapping[str, float]
    n: int
    provenance: Provenance
    source: str

    def __post_init__(self) -> None:
        if not self.source.strip():
            raise ValueError(
                "a normal database without a stated source cannot be interpreted "
                "later; record the cohort, camera, reconstruction and region set"
            )
        if set(self.means) != set(self.sds):
            raise ValueError("means and sds must cover the same features")

    @property
    def features(self) -> tuple[str, ...]:
        return tuple(sorted(self.means))

    def z_scores(
        self,
        ratios: RegionRatios,
        subject_provenance: Provenance,
    ) -> dict[str, float]:
        """Z-score every feature this database covers.

        Negative means below the healthy mean, which for binding ratios is
        the direction of disease. `subject_provenance` is required rather
        than defaulted, so that scoring real data against a phantom-derived
        database has to be written out explicitly and then fails.
        """
        if (
            self.provenance is Provenance.SYNTHETIC
            and subject_provenance is Provenance.REAL
        ):
            raise ProvenanceError(
                "refusing to z-score a real scan against a synthetic normal "
                "database. The phantom cohort has no relationship to human "
                "between-subject variation, so the z-scores would be "
                "arithmetically valid and meaningless. Build a database from "
                "real healthy controls acquired on the same camera and "
                "reconstruction."
            )

        subject = ratios.as_dict()
        missing = [f for f in self.features if f not in subject]
        if missing:
            raise KeyError(f"subject is missing features: {', '.join(missing)}")

        scores: dict[str, float] = {}
        for feature in self.features:
            sd = self.sds[feature]
            value = subject[feature]
            if sd <= 0 or not math.isfinite(sd):
                scores[feature] = float("nan")
                continue
            scores[feature] = (value - self.means[feature]) / sd
        return scores

    def most_abnormal(
        self, ratios: RegionRatios, subject_provenance: Provenance
    ) -> tuple[str, float]:
        """The feature furthest below the healthy mean, and its z.

        Lowest rather than largest in magnitude: a binding ratio above the
        healthy mean is not evidence of a dopaminergic deficit, and treating
        it as abnormal would make high-binding healthy people look diseased.
        """
        scores = {
            k: v for k, v in self.z_scores(ratios, subject_provenance).items()
            if math.isfinite(v)
        }
        if not scores:
            raise ValueError("no finite z-scores available")
        feature = min(scores, key=lambda k: scores[k])
        return feature, scores[feature]


def build_normal_database(
    healthy: Iterable[RegionRatios],
    provenance: Provenance,
    source: str,
    minimum: int = MINIMUM_SUBJECTS,
) -> NormalDatabase:
    """Summarize healthy binding ratios into means and standard deviations.

    Uses the sample standard deviation (ddof=1). The population form would
    understate the spread of a small cohort, which biases every later z-score
    away from zero and makes healthy subjects look abnormal.
    """
    rows = [r.as_dict() for r in healthy]
    if len(rows) < minimum:
        raise ValueError(
            f"{len(rows)} subjects is below the minimum of {minimum}. A standard "
            "deviation from fewer is mostly sampling noise, and every z-score "
            "computed against it inherits that."
        )

    features = sorted(rows[0])
    for row in rows:
        if sorted(row) != features:
            raise ValueError("all subjects must carry the same features")

    means: dict[str, float] = {}
    sds: dict[str, float] = {}
    for feature in features:
        values = np.array([row[feature] for row in rows], dtype=np.float64)
        finite = values[np.isfinite(values)]
        if finite.size < 2:
            means[feature] = float("nan")
            sds[feature] = float("nan")
            continue
        means[feature] = float(finite.mean())
        sds[feature] = float(finite.std(ddof=1))

    return NormalDatabase(
        means=means,
        sds=sds,
        n=len(rows),
        provenance=provenance,
        source=source,
    )
