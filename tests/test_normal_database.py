"""Tests for the z-score baseline.

Two things are being checked. That the arithmetic separates patients from
controls, and that the database refuses to be used where it does not apply.
The second matters more: a z-score against the wrong normal database is
wrong in a way that looks entirely normal.
"""

from __future__ import annotations

import numpy as np
import pytest

from spect.features import (
    MINIMUM_SUBJECTS,
    NormalDatabase,
    binding_ratios,
    build_normal_database,
    regions_from_phantom,
)
from spect.provenance import Provenance, ProvenanceError
from spect.synthetic import PhantomSpec, healthy_cohort, parkinsonian_cohort

#: Smaller than the default so the suite stays quick. Physical extent is
#: unchanged: 45 voxels at 4 mm covers the same field as 91 at 2 mm.
FAST = PhantomSpec(shape=(45, 55, 45), voxel_mm=4.0)

SOURCE = "Phantom cohort from spect.synthetic.healthy_cohort; testing only."


def ratios_of(phantoms):
    return [binding_ratios(p.volume, regions_from_phantom(p)) for p in phantoms]


@pytest.fixture(scope="module")
def controls():
    return ratios_of(healthy_cohort(n=20, seed=1, base=FAST))


@pytest.fixture(scope="module")
def patients():
    return ratios_of(parkinsonian_cohort(n=20, severity=0.55, seed=2, base=FAST))


@pytest.fixture(scope="module")
def database(controls):
    return build_normal_database(controls, Provenance.SYNTHETIC, SOURCE)


# --- construction -----------------------------------------------------------


def test_database_summarizes_every_feature(database, controls):
    assert database.n == len(controls)
    assert set(database.features) == set(controls[0].as_dict())
    assert all(np.isfinite(database.means[f]) for f in database.features)


def test_too_few_subjects_is_refused(controls):
    with pytest.raises(ValueError, match="below the minimum"):
        build_normal_database(controls[: MINIMUM_SUBJECTS - 1], Provenance.SYNTHETIC, SOURCE)


def test_a_database_without_a_source_is_refused():
    with pytest.raises(ValueError, match="source"):
        NormalDatabase(
            means={"putamen_left": 1.0},
            sds={"putamen_left": 0.1},
            n=20,
            provenance=Provenance.SYNTHETIC,
            source="   ",
        )


def test_means_and_sds_must_cover_the_same_features():
    with pytest.raises(ValueError, match="same features"):
        NormalDatabase(
            means={"putamen_left": 1.0, "caudate_left": 1.0},
            sds={"putamen_left": 0.1},
            n=20,
            provenance=Provenance.SYNTHETIC,
            source=SOURCE,
        )


# --- the cohort has to vary for the right reason ----------------------------


def test_the_cohort_has_biological_spread_not_just_counting_noise(database):
    """If the only variation were Poisson noise the standard deviation would
    be tiny, every z-score would be enormous, and the arithmetic would still
    be correct. That is the failure this asserts against."""
    cv = database.sds["putamen_left"] / abs(database.means["putamen_left"])
    assert 0.03 < cv < 0.40


def test_the_putamen_caudate_ratio_is_more_stable_than_either_structure(database):
    """Dose, camera sensitivity and habitus scale both structures together,
    so dividing one by the other removes most of that shared variation. This
    is why the ratio is a useful feature and not merely a derived one."""
    putamen_cv = database.sds["putamen_left"] / abs(database.means["putamen_left"])
    ratio_cv = database.sds["putamen_caudate_left"] / abs(
        database.means["putamen_caudate_left"]
    )
    assert ratio_cv < putamen_cv


# --- separation -------------------------------------------------------------


def test_a_control_scores_near_zero(database, controls):
    scores = database.z_scores(controls[0], Provenance.SYNTHETIC)
    assert abs(scores["putamen_left"]) < 3.0


def test_controls_are_centred_on_zero(database, controls):
    """Scoring the cohort against its own summary should recover a mean of
    zero and a standard deviation of one, within sampling error."""
    values = np.array(
        [database.z_scores(c, Provenance.SYNTHETIC)["putamen_left"] for c in controls]
    )
    assert abs(values.mean()) < 0.2
    assert values.std(ddof=1) == pytest.approx(1.0, abs=0.15)


def test_patients_score_well_below_controls(database, patients):
    values = [database.z_scores(p, Provenance.SYNTHETIC)["lowest_putamen"] for p in patients]
    assert np.median(values) < -2.0


def test_the_groups_separate(database, controls, patients):
    def worst(ratios):
        return database.z_scores(ratios, Provenance.SYNTHETIC)["lowest_putamen"]

    control_scores = np.array([worst(c) for c in controls])
    patient_scores = np.array([worst(p) for p in patients])
    assert patient_scores.max() < control_scores.min()


def test_the_most_abnormal_feature_is_a_putamen_measure(database, patients):
    """The putamen goes first. If the most abnormal feature were the caudate,
    the phantom's disease model or the calculator would be inverted."""
    feature, score = database.most_abnormal(patients[0], Provenance.SYNTHETIC)
    assert "putamen" in feature
    assert score < 0


def test_most_abnormal_looks_downward_not_outward(database, controls):
    """A binding ratio above the healthy mean is not a dopaminergic deficit.
    Ranking by magnitude would call a high-binding healthy subject abnormal."""
    feature, score = database.most_abnormal(controls[0], Provenance.SYNTHETIC)
    scores = database.z_scores(controls[0], Provenance.SYNTHETIC)
    assert score == min(v for v in scores.values() if np.isfinite(v))


# --- the refusal ------------------------------------------------------------


def test_a_synthetic_database_refuses_to_score_a_real_scan(database, patients):
    """The load-bearing test in this file. Phantom cohorts have no
    relationship to human between-subject variation, so these z-scores would
    be arithmetically valid and medically meaningless."""
    with pytest.raises(ProvenanceError, match="synthetic normal"):
        database.z_scores(patients[0], Provenance.REAL)


def test_the_refusal_also_covers_most_abnormal(database, patients):
    with pytest.raises(ProvenanceError):
        database.most_abnormal(patients[0], Provenance.REAL)


def test_a_real_database_may_score_a_real_scan(controls, patients):
    """The guard is directional. A database built from real controls is the
    thing you actually want, and nothing here should block it."""
    real = build_normal_database(
        controls, Provenance.REAL, "Stand-in for a real control cohort; test only."
    )
    scores = real.z_scores(patients[0], Provenance.REAL)
    assert np.isfinite(scores["putamen_left"])


def test_subject_provenance_has_no_default():
    """Scoring real data against a phantom database should require writing
    the claim out, so that it can then be refused. A default would let it
    happen silently."""
    import inspect

    signature = inspect.signature(NormalDatabase.z_scores)
    assert signature.parameters["subject_provenance"].default is inspect.Parameter.empty


# --- degenerate input -------------------------------------------------------


def test_a_zero_standard_deviation_yields_nan_rather_than_infinity():
    database = NormalDatabase(
        means={"putamen_left": 1.0},
        sds={"putamen_left": 0.0},
        n=20,
        provenance=Provenance.SYNTHETIC,
        source=SOURCE,
    )

    class Stub:
        def as_dict(self):
            return {"putamen_left": 2.0}

    assert np.isnan(database.z_scores(Stub(), Provenance.SYNTHETIC)["putamen_left"])


def test_a_subject_missing_a_feature_is_reported(database):
    class Stub:
        def as_dict(self):
            return {"putamen_left": 1.0}

    with pytest.raises(KeyError, match="missing features"):
        database.z_scores(Stub(), Provenance.SYNTHETIC)
