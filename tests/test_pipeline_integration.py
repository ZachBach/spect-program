"""The whole pipeline, end to end, on phantoms.

Unit tests prove each piece works. This proves they compose, which is a
different claim and the one that breaks first when an interface changes. It
runs the path a real analysis would: trajectories, rendered volumes, binding
ratios read off those volumes, leave-one-site-out, a Cox fit on training
sites only, and evaluation on the held-out site.

Kept small on purpose. It is a wiring check, not a measurement, and no number
it produces means anything about real scans.
"""

from __future__ import annotations

import numpy as np
import pytest

from spect.eval.splits import leave_one_site_out
from spect.eval.survival import SurvivalData, concordance_index
from spect.features import binding_ratios, regions_from_phantom
from spect.models import fit_cox
from spect.synthetic.phantom import PhantomSpec
from spect.synthetic.progression import (
    ProgressionSpec,
    generate_trajectories,
    oracle_ceiling,
    render,
    to_scan_records,
)

SMALL = PhantomSpec(shape=(31, 37, 31), voxel_mm=6.0)
FEATURES = ("baseline_putamen", "baseline_putamen_caudate", "delta_putamen_per_year")


@pytest.fixture(scope="module")
def cohort():
    spec = ProgressionSpec(n_subjects=40, base_phantom=SMALL, seed=101)
    subjects = generate_trajectories(spec)

    rows = []
    for subject in subjects:
        ratios = [
            binding_ratios(v.volume, regions_from_phantom(v))
            for v in render(subject, spec)
        ]
        first, last = ratios[0], ratios[-1]
        span = subject.visits[-1].year - subject.visits[0].year
        rows.append(
            [
                first.lowest_putamen,
                first.putamen_caudate_ratio[0],
                (last.lowest_putamen - first.lowest_putamen) / span,
            ]
        )

    return spec, subjects, np.array(rows), {s.subject_id: i for i, s in enumerate(subjects)}


def survival_for(subjects, indices) -> SurvivalData:
    return SurvivalData(
        time=np.array([subjects[i].observed_time for i in indices]),
        event=np.array([subjects[i].event for i in indices]),
    )


def test_features_come_out_of_the_volumes_usable():
    """Guards the seam most likely to break silently: a NaN here would
    propagate into a fit and out the other side as a coefficient."""
    spec = ProgressionSpec(n_subjects=3, base_phantom=SMALL, seed=102)
    for subject in generate_trajectories(spec):
        for volume in render(subject, spec):
            ratios = binding_ratios(volume.volume, regions_from_phantom(volume))
            assert all(np.isfinite(v) for v in ratios.as_dict().values())


def test_the_pipeline_runs_from_phantoms_to_a_held_out_score(cohort):
    spec, subjects, features, index_of = cohort
    records = to_scan_records(subjects)

    scored = 0
    for split in leave_one_site_out(records):
        train = [index_of[s] for s in sorted(split.subjects("train"))]
        test = [index_of[s] for s in sorted(split.subjects("test"))]

        training = survival_for(subjects, train)
        if training.n_events == 0:
            continue

        model = fit_cox(features[train], training, feature_names=FEATURES)
        held_out = survival_for(subjects, test)
        try:
            result = concordance_index(held_out, model.risk(features[test]))
        except ValueError:
            continue

        assert 0.0 <= result.index <= 1.0
        scored += 1

    assert scored >= 3, "most folds should be scorable at this cohort size"


def test_the_model_is_never_fitted_on_a_held_out_subject(cohort):
    """The composition failure that matters. Each piece can be correct while
    the wiring between them leaks."""
    _, subjects, _, _ = cohort
    records = to_scan_records(subjects)
    for split in leave_one_site_out(records):
        assert not split.subjects("train") & split.subjects("test")
        assert not split.sites("train") & split.sites("test")


def test_risk_runs_the_right_way_through_the_whole_chain(cohort):
    """A sign error anywhere between the volume and the fit would show up as
    a concordance below chance. Checked on the full cohort, where the sample
    is large enough for the direction to be unambiguous."""
    _, subjects, features, _ = cohort
    data = survival_for(subjects, range(len(subjects)))
    model = fit_cox(features, data, feature_names=FEATURES)
    assert concordance_index(data, model.risk(features)).index > 0.55


def test_more_binding_lowers_the_hazard(cohort):
    """These features are binding ratios, so higher means healthier. A hazard
    ratio above 1 for baseline putamen binding would mean the chain had
    inverted somewhere, most likely by confusing loss with binding."""
    _, subjects, features, _ = cohort
    data = survival_for(subjects, range(len(subjects)))
    model = fit_cox(features, data, feature_names=FEATURES)
    assert model.hazard_ratios["baseline_putamen"] < 1.0


def test_the_ceiling_is_available_next_to_the_score(cohort):
    """A concordance reported without the achievable maximum invites being
    read as mediocre when it is close to everything the data contains."""
    _, subjects, features, _ = cohort
    data = survival_for(subjects, range(len(subjects)))
    model = fit_cox(features, data, feature_names=FEATURES)

    achieved = concordance_index(data, model.risk(features)).index
    ceiling = oracle_ceiling(subjects)
    assert ceiling > 0.5
    assert achieved <= ceiling + 0.15
