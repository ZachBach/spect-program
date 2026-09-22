"""Tests for the binding ratio calculator.

These prove arithmetic, not anatomy. Every test hands the calculator the same
masks the phantom was built from, so a region that landed in the wrong place
would still score perfectly here. Registration needs separate evidence and
does not have any yet.
"""

from __future__ import annotations

import numpy as np
import pytest

from spect.features import (
    binding_ratios,
    regions_from_phantom,
    specific_binding_ratio,
)
from spect.synthetic import PhantomSpec, generate, healthy, parkinsonian


def ratios_for(phantom, **kwargs):
    return binding_ratios(phantom.volume, regions_from_phantom(phantom), **kwargs)


# --- recovery ---------------------------------------------------------------


def test_recovers_the_nominal_ratio_without_blur_or_noise():
    phantom = generate(PhantomSpec(fwhm_mm=0.0, noise=False))
    result = ratios_for(phantom)
    assert result.putamen[0] == pytest.approx(phantom.truth.putamen_ratio[0], rel=0.02)
    assert result.caudate[0] == pytest.approx(phantom.truth.caudate_ratio[0], rel=0.02)


def test_partial_volume_biases_every_structure_downward():
    phantom = generate(PhantomSpec(noise=False))
    result = ratios_for(phantom)
    assert result.putamen[0] < phantom.truth.putamen_ratio[0]
    assert result.caudate[0] < phantom.truth.caudate_ratio[0]


def test_the_larger_structure_keeps_more_of_its_signal():
    """Partial volume loss is driven by a structure's volume relative to the
    point spread, not by its shape. The putamen ellipsoid is about 1.5 times
    the caudate's volume here, matching the real anatomy, so it should retain
    the larger fraction.

    The volume check is part of the test rather than a comment. Tying the
    assertion to its cause means that changing the geometry fails this with a
    readable reason instead of leaving a bare inequality to be puzzled over.
    """
    phantom = generate(PhantomSpec(noise=False))
    result = ratios_for(phantom)

    caudate_voxels = int(phantom.masks["caudate_left"].sum())
    putamen_voxels = int(phantom.masks["putamen_left"].sum())
    assert putamen_voxels > caudate_voxels, "geometry changed; revisit the claim below"

    putamen_kept = result.putamen[0] / phantom.truth.putamen_ratio[0]
    caudate_kept = result.caudate[0] / phantom.truth.caudate_ratio[0]
    assert putamen_kept > caudate_kept


# --- derived measures -------------------------------------------------------


def test_a_symmetric_phantom_has_near_zero_asymmetry():
    result = ratios_for(generate(PhantomSpec(noise=False)))
    assert result.putamen_asymmetry < 1.0


def test_asymmetric_loss_is_reported_with_the_correct_side():
    phantom = generate(PhantomSpec(putamen_loss=(0.7, 0.2), noise=False))
    result = ratios_for(phantom)
    assert result.putamen_asymmetry > 10.0
    assert result.worse_putamen_side == "left"
    assert result.lowest_putamen == result.putamen[0]


def test_putamen_caudate_ratio_falls_with_disease():
    """The putamen-before-caudate gradient, expressed as a single internally
    normalized number."""
    control = ratios_for(generate(PhantomSpec(noise=False)))
    patient = ratios_for(
        generate(PhantomSpec(putamen_loss=(0.6, 0.6), caudate_loss=(0.2, 0.2), noise=False))
    )
    assert patient.putamen_caudate_ratio[0] < control.putamen_caudate_ratio[0]


def test_striatum_is_the_union_not_the_average_of_two_ratios():
    """Averaging the caudate and putamen ratios would weight the structures
    equally regardless of size and silently report a different quantity."""
    phantom = generate(PhantomSpec(putamen_loss=(0.8, 0.8), noise=False))
    result = ratios_for(phantom)
    naive_average = (result.caudate[0] + result.putamen[0]) / 2.0
    assert result.striatum[0] != pytest.approx(naive_average, rel=1e-6)
    assert min(result.caudate[0], result.putamen[0]) <= result.striatum[0]
    assert result.striatum[0] <= max(result.caudate[0], result.putamen[0])


def test_feature_vector_covers_every_reported_measure():
    features = ratios_for(healthy()).as_dict()
    assert set(features) == {
        "caudate_left",
        "caudate_right",
        "putamen_left",
        "putamen_right",
        "striatum_left",
        "striatum_right",
        "putamen_caudate_left",
        "putamen_caudate_right",
        "putamen_asymmetry",
        "lowest_putamen",
    }
    assert all(np.isfinite(v) for v in features.values())


# --- estimators -------------------------------------------------------------


def test_median_and_mean_agree_on_a_noiseless_uniform_region():
    phantom = generate(PhantomSpec(posterior_gradient=0.0, fwhm_mm=0.0, noise=False))
    by_mean = ratios_for(phantom, estimator="mean")
    by_median = ratios_for(phantom, estimator="median")
    assert by_median.putamen[0] == pytest.approx(by_mean.putamen[0], rel=1e-9)


def test_the_median_resists_a_hot_voxel_that_moves_the_mean():
    """A single bright artefact should not move a region's level much. This is
    the reason the estimator is selectable rather than fixed."""
    phantom = generate(PhantomSpec(fwhm_mm=0.0, noise=False))
    regions = regions_from_phantom(phantom)

    spiked = phantom.volume.copy()
    index = tuple(coord[0] for coord in np.nonzero(regions["putamen_left"]))
    spiked[index] = phantom.volume.max() * 500.0

    clean_mean = binding_ratios(phantom.volume, regions, estimator="mean").putamen[0]
    spiked_mean = binding_ratios(spiked, regions, estimator="mean").putamen[0]
    clean_median = binding_ratios(phantom.volume, regions, estimator="median").putamen[0]
    spiked_median = binding_ratios(spiked, regions, estimator="median").putamen[0]

    assert abs(spiked_median - clean_median) < abs(spiked_mean - clean_mean)


# --- failure modes ----------------------------------------------------------


def test_missing_region_names_are_reported_together():
    phantom = healthy()
    regions = regions_from_phantom(phantom)
    del regions["putamen_left"]
    del regions["reference"]
    with pytest.raises(KeyError) as excinfo:
        binding_ratios(phantom.volume, regions)
    message = str(excinfo.value)
    assert "putamen_left" in message and "reference" in message


def test_an_empty_region_is_refused():
    phantom = healthy()
    regions = regions_from_phantom(phantom)
    regions["putamen_left"] = np.zeros_like(regions["putamen_left"])
    with pytest.raises(ValueError, match="empty"):
        binding_ratios(phantom.volume, regions)


def test_a_region_in_the_wrong_space_is_refused():
    """The failure this is guarding is a mask drawn at a different matrix size
    than the image, which would otherwise raise an obscure indexing error far
    from its cause."""
    phantom = healthy()
    regions = regions_from_phantom(phantom)
    regions["putamen_left"] = np.ones((10, 10, 10), dtype=bool)
    with pytest.raises(ValueError, match="different space"):
        binding_ratios(phantom.volume, regions)


def test_an_empty_reference_region_is_refused_rather_than_dividing_by_zero():
    phantom = healthy()
    regions = regions_from_phantom(phantom)
    volume = np.zeros_like(phantom.volume)
    with pytest.raises(ValueError, match="denominator"):
        binding_ratios(volume, regions)


def test_single_region_helper_matches_the_batch_calculation():
    phantom = parkinsonian(severity=0.5)
    regions = regions_from_phantom(phantom)
    single = specific_binding_ratio(
        phantom.volume, regions["putamen_left"], regions["reference"]
    )
    batch = binding_ratios(phantom.volume, regions).putamen[0]
    assert single == pytest.approx(batch, rel=1e-12)
