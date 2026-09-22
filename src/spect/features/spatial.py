"""Spatial descriptors a region mean throws away.

A binding ratio reduces a structure to one number. That is the right summary
for how much tracer is bound, and it is blind by construction to where inside
the structure the binding is. In Parkinson's disease the posterior putamen
goes before the anterior, so the shape of the loss carries information that
the amount of it does not.

This module exists to make that concrete before any network is trained. The
posterior-to-anterior contrast below is a two-number summary computed with
arithmetic, and on the phantom it recovers gradient steepness that the region
mean cannot see at all: across gradients of 0.0 to 0.9 the putamen binding
ratio is identical to four decimal places while this contrast moves from 0.99
to 0.89.

That matters for interpreting a later result. If a 3D network beats the
region-mean baseline on this fixture, the honest question is whether it beat
*this* too, because anything a posterior/anterior split can find is not
evidence that a network was needed. And if a network fails to beat the region
mean, this establishes the information was there to be found, so the failure
belongs to the model rather than the data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Which array axis runs anterior. This is the phantom's own convention.
#: Real data must establish orientation from the file, never by assumption;
#: a flipped axis here would invert every contrast silently.
ANTERIOR_AXIS = 1


@dataclass(frozen=True)
class SpatialContrast:
    """Posterior and anterior halves of one structure."""

    posterior: float
    anterior: float
    n_posterior: int
    n_anterior: int

    @property
    def ratio(self) -> float:
        """Posterior over anterior. Below 1 means the posterior is worse.

        A ratio rather than a difference, so it is not confounded by how much
        binding the structure has overall. Two subjects with very different
        absolute uptake and the same spatial pattern should land here
        together, which is the point of measuring pattern separately from
        amount.
        """
        return self.posterior / self.anterior if self.anterior else float("nan")

    @property
    def gradient_index(self) -> float:
        """Signed contrast, scaled so 0 means uniform.

        (anterior - posterior) / (anterior + posterior). Bounded, symmetric,
        and finite when either half is near zero, which the raw ratio is not.
        """
        total = self.anterior + self.posterior
        return (self.anterior - self.posterior) / total if total else float("nan")


def posterior_anterior_contrast(
    volume: np.ndarray,
    mask: np.ndarray,
    axis: int = ANTERIOR_AXIS,
) -> SpatialContrast:
    """Split a structure at its own centroid and compare the halves.

    The split uses the structure's centroid rather than a fixed coordinate, so
    it does not depend on where the structure sits in the field of view and
    does not need the image to be registered to anything. That keeps this
    usable on the phantom and on a real scan whose alignment has not yet been
    established.
    """
    if mask.shape != volume.shape:
        raise ValueError(
            f"mask has shape {mask.shape}, volume has {volume.shape}; a mask "
            "drawn in a different space gives a confidently wrong contrast"
        )
    if not mask.any():
        raise ValueError("mask is empty")
    if not 0 <= axis < volume.ndim:
        raise ValueError(f"axis {axis} is out of range for a {volume.ndim}D volume")

    coordinates = np.nonzero(mask)[axis]
    centroid = coordinates.mean()

    along = np.arange(volume.shape[axis], dtype=np.float64)
    shape = [1] * volume.ndim
    shape[axis] = volume.shape[axis]
    is_anterior = (along > centroid).reshape(shape)

    anterior = mask & is_anterior
    posterior = mask & ~is_anterior

    if not anterior.any() or not posterior.any():
        raise ValueError(
            "the structure does not straddle its own centroid along this axis, "
            "which usually means the axis is wrong"
        )

    return SpatialContrast(
        posterior=float(volume[posterior].mean()),
        anterior=float(volume[anterior].mean()),
        n_posterior=int(posterior.sum()),
        n_anterior=int(anterior.sum()),
    )


def spatial_features(
    volume: np.ndarray,
    regions: dict[str, np.ndarray],
    axis: int = ANTERIOR_AXIS,
) -> dict[str, float]:
    """Gradient indices for both putamina, plus their mean and asymmetry.

    Returned as a flat dict so it concatenates with `RegionRatios.as_dict()`
    into one feature vector for the baseline.
    """
    left = posterior_anterior_contrast(volume, regions["putamen_left"], axis)
    right = posterior_anterior_contrast(volume, regions["putamen_right"], axis)

    return {
        "gradient_index_left": left.gradient_index,
        "gradient_index_right": right.gradient_index,
        "gradient_index_mean": (left.gradient_index + right.gradient_index) / 2.0,
        "gradient_index_asymmetry": abs(left.gradient_index - right.gradient_index),
    }
