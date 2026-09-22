"""Tests for the spatial descriptors, and the D5 fix they enable.

The pair at the bottom is the point of the module. One shows that making the
outcome depend on spatial pattern gives a pattern-reading model an advantage
a region mean cannot have. The other removes the dependence and requires the
advantage to vanish. Without the second, the fixture would reward extra
features rather than the information in them.
"""

from __future__ import annotations

import numpy as np
import pytest

from spect.eval.survival import SurvivalData, concordance_index
from spect.features import (
    binding_ratios,
    posterior_anterior_contrast,
    regions_from_phantom,
    spatial_features,
)
from spect.models import fit_cox
from spect.synthetic.phantom import PhantomSpec, generate
from spect.synthetic.progression import ProgressionSpec, generate_trajectories, render

SMALL = PhantomSpec(shape=(31, 37, 31), voxel_mm=6.0)


def phantom_with(gradient: float, loss: float = 0.5):
    return generate(
        PhantomSpec(putamen_loss=(loss, loss), posterior_gradient=gradient, noise=False)
    )


# --- the property the D5 fix rests on ---------------------------------------


def test_the_region_mean_is_blind_to_the_gradient():
    """The gradient redistributes loss within the putamen while preserving
    its mean, so a binding ratio cannot see it at all. Identical to four
    decimal places is not an approximation here; it is the construction."""
    ratios = [
        binding_ratios(p.volume, regions_from_phantom(p)).putamen[0]
        for p in (phantom_with(g) for g in (0.0, 0.3, 0.6, 0.9))
    ]
    assert ratios == pytest.approx([ratios[0]] * 4, abs=1e-4)


def test_the_spatial_contrast_does_see_it_and_moves_monotonically():
    indices = [
        spatial_features(p.volume, regions_from_phantom(p))["gradient_index_mean"]
        for p in (phantom_with(g) for g in (0.0, 0.3, 0.6, 0.9))
    ]
    assert all(b > a for a, b in zip(indices, indices[1:]))
    assert indices[0] == pytest.approx(0.0, abs=0.01)


def test_a_uniform_structure_has_no_contrast():
    phantom = phantom_with(gradient=0.0, loss=0.0)
    contrast = posterior_anterior_contrast(
        phantom.volume, phantom.masks["putamen_left"]
    )
    assert contrast.gradient_index == pytest.approx(0.0, abs=0.02)
    assert contrast.ratio == pytest.approx(1.0, abs=0.05)


def test_the_posterior_half_is_the_worse_one():
    """The direction that matters clinically. If this inverted, every later
    gradient feature would carry the wrong sign."""
    phantom = phantom_with(gradient=0.9)
    contrast = posterior_anterior_contrast(
        phantom.volume, phantom.masks["putamen_left"]
    )
    assert contrast.posterior < contrast.anterior
    assert contrast.ratio < 1.0


def test_the_split_uses_the_structures_own_centroid():
    """So it needs no registration and does not depend on where the structure
    sits in the field of view."""
    phantom = phantom_with(gradient=0.6)
    contrast = posterior_anterior_contrast(
        phantom.volume, phantom.masks["putamen_left"]
    )
    assert abs(contrast.n_posterior - contrast.n_anterior) <= max(
        4, contrast.n_posterior // 4
    )


# --- failure modes ----------------------------------------------------------


def test_an_empty_mask_is_refused():
    phantom = phantom_with(0.5)
    with pytest.raises(ValueError, match="empty"):
        posterior_anterior_contrast(phantom.volume, np.zeros_like(phantom.masks["putamen_left"]))


def test_a_mask_in_the_wrong_space_is_refused():
    phantom = phantom_with(0.5)
    with pytest.raises(ValueError, match="different space"):
        posterior_anterior_contrast(phantom.volume, np.ones((5, 5, 5), dtype=bool))


def test_an_axis_out_of_range_is_refused():
    phantom = phantom_with(0.5)
    with pytest.raises(ValueError, match="out of range"):
        posterior_anterior_contrast(phantom.volume, phantom.masks["putamen_left"], axis=7)


def test_all_four_spatial_features_are_finite():
    phantom = phantom_with(0.6)
    values = spatial_features(phantom.volume, regions_from_phantom(phantom))
    assert set(values) == {
        "gradient_index_left",
        "gradient_index_right",
        "gradient_index_mean",
        "gradient_index_asymmetry",
    }
    assert all(np.isfinite(v) for v in values.values())


# --- the D5 validation pair -------------------------------------------------


def _fit_and_score(spec: ProgressionSpec, use_spatial: bool) -> float:
    subjects = generate_trajectories(spec)
    rows = []
    for subject in subjects:
        volumes = render(subject, spec)
        first, last = volumes[0], volumes[-1]
        regions = regions_from_phantom(first)
        base = binding_ratios(first.volume, regions)
        follow = binding_ratios(last.volume, regions_from_phantom(last))
        span = subject.visits[-1].year - subject.visits[0].year

        row = [
            base.lowest_putamen,
            (follow.lowest_putamen - base.lowest_putamen) / span,
        ]
        if use_spatial:
            row.append(spatial_features(first.volume, regions)["gradient_index_mean"])
        rows.append(row)

    data = SurvivalData(
        time=np.array([s.observed_time for s in subjects]),
        event=np.array([s.event for s in subjects]),
    )
    features = np.array(rows)
    model = fit_cox(features, data)
    return concordance_index(data, model.risk(features)).index


def test_spatial_features_help_when_the_outcome_depends_on_pattern():
    """The D5 fix, validated. Gradient steepness is invisible to a region
    mean by construction, so an outcome depending on it gives a
    pattern-reading model information the baseline cannot obtain at any
    sample size."""
    spec = ProgressionSpec(
        n_subjects=200,
        severity_from_baseline=0.3,
        severity_from_rate=0.3,
        severity_from_gradient=1.4,
        base_phantom=SMALL,
        seed=201,
    )
    assert _fit_and_score(spec, True) > _fit_and_score(spec, False) + 0.03


def test_spatial_features_do_not_help_when_the_outcome_ignores_pattern():
    """The control. If the extra feature won here too, the fixture would be
    rewarding the feature count rather than the information, and any later
    comparison against a network would be meaningless."""
    spec = ProgressionSpec(
        n_subjects=200,
        severity_from_baseline=0.6,
        severity_from_rate=0.6,
        severity_from_gradient=0.0,
        base_phantom=SMALL,
        seed=202,
    )
    assert _fit_and_score(spec, True) <= _fit_and_score(spec, False) + 0.02
