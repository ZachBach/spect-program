"""Tests for the synthetic phantom generator.

The point of a phantom is that the answer is known before the code runs, so
these tests assert recovery rather than plausibility. The important ones are
the gradient and the partial volume tests: they check that the fixture is hard
in the ways a real scan is hard, rather than being an easier problem wearing
the same shape.
"""

from __future__ import annotations

import numpy as np
import pytest

from spect.synthetic import PhantomSpec, generate, healthy, parkinsonian
from spect.synthetic.phantom import PUTAMEN_LEFT


def measured_ratio(volume: np.ndarray, structure: np.ndarray, reference: np.ndarray) -> float:
    """Specific binding ratio: (structure - reference) / reference.

    Deliberately written here rather than imported. The binding-ratio
    calculator is the next piece of work, and a test that used it would be
    checking two unproven things against each other.
    """
    ref = float(volume[reference].mean())
    return (float(volume[structure].mean()) - ref) / ref


# --- determinism ------------------------------------------------------------


def test_same_spec_gives_an_identical_volume():
    assert np.array_equal(generate(PhantomSpec(seed=7)).volume, generate(PhantomSpec(seed=7)).volume)


def test_different_seed_gives_a_different_volume():
    assert not np.array_equal(generate(PhantomSpec(seed=1)).volume, generate(PhantomSpec(seed=2)).volume)


def test_noise_can_be_turned_off_for_a_deterministic_baseline():
    a = generate(PhantomSpec(noise=False, seed=1)).volume
    b = generate(PhantomSpec(noise=False, seed=999)).volume
    assert np.array_equal(a, b), "with noise off the seed must not matter"


def test_shape_follows_the_spec():
    phantom = generate(PhantomSpec(shape=(41, 49, 41)))
    assert phantom.volume.shape == (41, 49, 41)


# --- structure --------------------------------------------------------------


def test_striatum_is_brighter_than_the_reference_region():
    phantom = healthy()
    striatum = phantom.masks["putamen_left"] | phantom.masks["caudate_left"]
    assert phantom.volume[striatum].mean() > phantom.volume[phantom.masks["occipital"]].mean()


def test_outside_the_head_is_empty_before_the_point_spread():
    phantom = generate(PhantomSpec(fwhm_mm=0.0, noise=False))
    assert phantom.volume[~phantom.masks["brain"]].max() == pytest.approx(0.0, abs=1e-9)


def test_the_point_spread_does_carry_activity_past_the_head_boundary():
    """Not a defect. A scanner resolving at 10 mm puts counts outside the
    head, and any preprocessing step that assumes a hard edge there would be
    tuned against a phantom that is cleaner than the real thing."""
    phantom = generate(PhantomSpec(noise=False))
    outside = phantom.volume[~phantom.masks["brain"]]
    assert outside.max() > 0.0

    # It should still fall away: the far field is near empty.
    n = phantom.spec.shape[0]
    x_mm = (np.arange(n) - (n - 1) / 2.0) * phantom.spec.voxel_mm
    far = np.abs(x_mm) > 85.0
    if far.any():
        assert phantom.volume[far, :, :].max() < phantom.spec.reference_counts * 0.01


def test_structures_do_not_overlap():
    phantom = healthy()
    caudate = phantom.masks["caudate_left"]
    putamen = phantom.masks["putamen_left"]
    assert not (caudate & putamen).any()


def test_occipital_reference_excludes_the_striatum():
    phantom = healthy()
    for name in ("putamen_left", "putamen_right", "caudate_left", "caudate_right"):
        assert not (phantom.masks["occipital"] & phantom.masks[name]).any()


# --- the degradations that make the fixture honest --------------------------


def test_partial_volume_pulls_the_measured_ratio_below_the_truth():
    """A scanner resolving at 10 mm cannot hold all the activity inside a
    structure 12 mm across. If this ever passes at equality, the blur has been
    lost and every downstream accuracy number becomes optimistic."""
    phantom = generate(PhantomSpec(noise=False))
    measured = measured_ratio(
        phantom.volume, phantom.masks["putamen_left"], phantom.masks["occipital"]
    )
    nominal = phantom.truth.putamen_ratio[0]
    assert measured < nominal
    assert measured > nominal * 0.3, "blur should bias the ratio, not destroy it"


def test_no_blur_recovers_the_nominal_ratio():
    """The complement of the test above: with the point spread removed the
    estimator should land on the truth, which localizes any bias to the blur
    rather than to the geometry."""
    phantom = generate(PhantomSpec(fwhm_mm=0.0, noise=False))
    measured = measured_ratio(
        phantom.volume, phantom.masks["putamen_left"], phantom.masks["occipital"]
    )
    assert measured == pytest.approx(phantom.truth.putamen_ratio[0], rel=0.02)


def test_noise_scales_as_poisson():
    """Variance tracking the mean is what separates counting noise from an
    additive Gaussian, and it is why low-count regions are the unreliable
    ones."""
    low = generate(PhantomSpec(reference_counts=40.0, fwhm_mm=0.0, seed=3))
    high = generate(PhantomSpec(reference_counts=400.0, fwhm_mm=0.0, seed=3))

    def relative_sd(phantom):
        values = phantom.volume[phantom.masks["occipital"]]
        return values.std() / values.mean()

    # Ten times the counts should roughly halve the relative spread.
    assert relative_sd(low) > relative_sd(high) * 2.0


# --- the disease pattern ----------------------------------------------------


def test_loss_lowers_the_measured_ratio():
    intact = measured_ratio(
        (p := generate(PhantomSpec(noise=False))).volume,
        p.masks["putamen_left"],
        p.masks["occipital"],
    )
    depleted = measured_ratio(
        (q := generate(PhantomSpec(putamen_loss=(0.7, 0.7), noise=False))).volume,
        q.masks["putamen_left"],
        q.masks["occipital"],
    )
    assert depleted < intact


def test_complete_loss_brings_the_putamen_to_background():
    phantom = generate(PhantomSpec(putamen_loss=(1.0, 1.0), fwhm_mm=0.0, noise=False))
    ratio = measured_ratio(
        phantom.volume, phantom.masks["putamen_left"], phantom.masks["occipital"]
    )
    assert ratio == pytest.approx(0.0, abs=1e-6)


def test_asymmetric_loss_shows_up_as_a_side_difference():
    phantom = parkinsonian(severity=0.6, asymmetry=0.4)
    left = measured_ratio(phantom.volume, phantom.masks["putamen_left"], phantom.masks["occipital"])
    right = measured_ratio(phantom.volume, phantom.masks["putamen_right"], phantom.masks["occipital"])
    assert left < right, "the worse side should carry the lower ratio"
    assert phantom.truth.putamen_asymmetry > 0.1


def test_caudate_is_relatively_spared_relative_to_putamen():
    """The putamen-before-caudate ordering is the characteristic pattern. A
    phantom that lost both equally would be a different and easier problem."""
    phantom = parkinsonian(severity=0.6)
    putamen = measured_ratio(
        phantom.volume, phantom.masks["putamen_left"], phantom.masks["occipital"]
    )
    caudate = measured_ratio(
        phantom.volume, phantom.masks["caudate_left"], phantom.masks["occipital"]
    )
    assert caudate > putamen


def test_posterior_putamen_is_worse_than_anterior():
    """The posterior-to-anterior gradient, which is the spatial signal a
    volumetric model could use and a single mean binding ratio throws away."""
    phantom = generate(
        PhantomSpec(putamen_loss=(0.6, 0.6), posterior_gradient=0.8, fwhm_mm=0.0, noise=False)
    )
    mask = phantom.masks["putamen_left"]
    # Split the structure at its own centre along the anterior axis.
    n = phantom.spec.shape[1]
    y_mm = (np.arange(n) - (n - 1) / 2.0) * phantom.spec.voxel_mm
    anterior_of_centre = y_mm > PUTAMEN_LEFT.centre[1]
    anterior = mask & anterior_of_centre[None, :, None]
    posterior = mask & ~anterior_of_centre[None, :, None]

    assert phantom.volume[posterior].mean() < phantom.volume[anterior].mean()


def test_a_flat_gradient_leaves_the_putamen_uniform():
    phantom = generate(
        PhantomSpec(putamen_loss=(0.6, 0.6), posterior_gradient=0.0, fwhm_mm=0.0, noise=False)
    )
    values = phantom.volume[phantom.masks["putamen_left"]]
    assert values.std() == pytest.approx(0.0, abs=1e-9)


# --- validation -------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"putamen_loss": (1.5, 0.0)},
        {"putamen_loss": (-0.1, 0.0)},
        {"caudate_loss": (0.0, 2.0)},
        {"posterior_gradient": 1.5},
        {"reference_counts": 0.0},
        {"fwhm_mm": -1.0},
    ],
)
def test_invalid_specs_are_rejected(kwargs):
    with pytest.raises(ValueError):
        PhantomSpec(**kwargs)
