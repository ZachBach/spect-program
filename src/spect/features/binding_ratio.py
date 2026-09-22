"""Specific binding ratios from a DaT-SPECT volume and a set of regions.

The ratio itself is arithmetic:

    SBR = (target mean - reference mean) / reference mean

Everything difficult is upstream of that line. Which voxels are the putamen,
whether the reference region is free of specific binding, and whether the
volume is in the space the regions were drawn in are all harder problems than
the division, and all of them can be wrong while the division stays right.

That is why regions are passed in rather than derived here. A test that hands
this function the masks a phantom was built from proves the arithmetic and
nothing else. Proving that regions land in the right place on a real scan is
registration work, it is a separate problem, and it needs its own evidence.
Keeping the two apart in the API is what stops one from being mistaken for
the other later.

Terminology: "specific binding ratio" is used here for the target-to-reference
ratio above. Published DaT literature also calls this the striatal binding
ratio or the specific uptake ratio, and the absolute values are not comparable
across software packages because the region definitions and reference regions
differ. Comparisons belong within one pipeline, not across them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping

import numpy as np

Side = Literal["left", "right"]
Estimator = Literal["mean", "median"]


@dataclass(frozen=True)
class RegionRatios:
    """Binding ratios for one scan, per structure and side.

    Tuples are ordered (left, right) throughout, matching the phantom's own
    convention. On real data the left/right assignment comes from the image
    orientation and must be established by the loader, never assumed.
    """

    caudate: tuple[float, float]
    putamen: tuple[float, float]
    striatum: tuple[float, float]
    reference_level: float
    estimator: Estimator

    @property
    def putamen_caudate_ratio(self) -> tuple[float, float]:
        """Putamen over caudate, per side.

        Falls with disease because the putamen loses binding first. Useful
        because it is internally normalized: it divides out anything that
        scales both structures together, including much of the variation in
        injected dose and in the reference region.
        """
        return tuple(
            p / c if c else float("nan")
            for p, c in zip(self.putamen, self.caudate)
        )  # type: ignore[return-value]

    @property
    def putamen_asymmetry(self) -> float:
        """Percent difference between sides, relative to their mean.

        Reported unsigned, because which side is worse is a separate fact
        from how unequal the sides are, and mixing them into one number makes
        both harder to use. `worse_putamen_side` carries the direction.
        """
        left, right = self.putamen
        mean = (left + right) / 2.0
        if not mean:
            return float("nan")
        return abs(left - right) / mean * 100.0

    @property
    def worse_putamen_side(self) -> Side:
        return "left" if self.putamen[0] <= self.putamen[1] else "right"

    @property
    def lowest_putamen(self) -> float:
        """The more affected side.

        Often more informative than the mean of the two, because early
        disease is asymmetric and averaging dilutes the side that carries the
        signal.
        """
        return min(self.putamen)

    def as_dict(self) -> dict[str, float]:
        """Flat feature vector, for a baseline model or a z-score lookup."""
        return {
            "caudate_left": self.caudate[0],
            "caudate_right": self.caudate[1],
            "putamen_left": self.putamen[0],
            "putamen_right": self.putamen[1],
            "striatum_left": self.striatum[0],
            "striatum_right": self.striatum[1],
            "putamen_caudate_left": self.putamen_caudate_ratio[0],
            "putamen_caudate_right": self.putamen_caudate_ratio[1],
            "putamen_asymmetry": self.putamen_asymmetry,
            "lowest_putamen": self.lowest_putamen,
        }


def _region_level(
    volume: np.ndarray, mask: np.ndarray, estimator: Estimator, name: str
) -> float:
    if mask.shape != volume.shape:
        raise ValueError(
            f"region '{name}' has shape {mask.shape}, volume has {volume.shape}. "
            "A region drawn in a different space than the image is the most "
            "common way to get a confidently wrong number here."
        )
    if not mask.any():
        raise ValueError(f"region '{name}' is empty")

    values = volume[mask]
    return float(np.median(values) if estimator == "median" else values.mean())


def specific_binding_ratio(
    volume: np.ndarray,
    target: np.ndarray,
    reference: np.ndarray,
    estimator: Estimator = "mean",
    target_name: str = "target",
) -> float:
    """One ratio, for one region against one reference region.

    `estimator` selects how a region is reduced to a level. The mean is
    conventional. The median is more robust to a few hot voxels, which is
    worth having when a reconstruction artefact or a stray high-count voxel
    would otherwise drag a small region's mean. They are not interchangeable
    and a value computed one way should not be compared with one computed the
    other.
    """
    reference_level = _region_level(volume, reference, estimator, "reference")
    if reference_level <= 0:
        raise ValueError(
            f"reference level is {reference_level}, which cannot be a denominator. "
            "An empty or near-empty reference region usually means the region "
            "and the image are misaligned."
        )
    target_level = _region_level(volume, target, estimator, target_name)
    return (target_level - reference_level) / reference_level


def binding_ratios(
    volume: np.ndarray,
    regions: Mapping[str, np.ndarray],
    estimator: Estimator = "mean",
) -> RegionRatios:
    """All striatal ratios for one scan.

    `regions` must supply `caudate_left`, `caudate_right`, `putamen_left`,
    `putamen_right` and `reference`. A phantom's `masks` satisfies this except
    for the reference key, which is named `occipital` there; pass it through
    `regions_from_phantom` rather than renaming by hand at each call site.

    The striatal ratio is computed over the union of caudate and putamen
    rather than by averaging their two ratios. Averaging would weight the
    structures equally regardless of size, which silently changes the quantity
    being reported.
    """
    required = ("caudate_left", "caudate_right", "putamen_left", "putamen_right", "reference")
    missing = [name for name in required if name not in regions]
    if missing:
        raise KeyError(f"missing regions: {', '.join(missing)}")

    reference = regions["reference"]
    reference_level = _region_level(volume, reference, estimator, "reference")
    if reference_level <= 0:
        raise ValueError(f"reference level is {reference_level}, which cannot be a denominator")

    def ratio(mask: np.ndarray, name: str) -> float:
        return (_region_level(volume, mask, estimator, name) - reference_level) / reference_level

    caudate = (
        ratio(regions["caudate_left"], "caudate_left"),
        ratio(regions["caudate_right"], "caudate_right"),
    )
    putamen = (
        ratio(regions["putamen_left"], "putamen_left"),
        ratio(regions["putamen_right"], "putamen_right"),
    )
    striatum = (
        ratio(regions["caudate_left"] | regions["putamen_left"], "striatum_left"),
        ratio(regions["caudate_right"] | regions["putamen_right"], "striatum_right"),
    )

    return RegionRatios(
        caudate=caudate,
        putamen=putamen,
        striatum=striatum,
        reference_level=reference_level,
        estimator=estimator,
    )


def regions_from_phantom(phantom) -> dict[str, np.ndarray]:
    """Adapt a phantom's masks to the region names this module expects.

    Deliberately a thin, explicit adapter rather than special-casing phantoms
    inside `binding_ratios`. When real data arrives its regions will come from
    a registered atlas through a different adapter, and the calculator should
    not be able to tell the difference.
    """
    masks = phantom.masks
    return {
        "caudate_left": masks["caudate_left"],
        "caudate_right": masks["caudate_right"],
        "putamen_left": masks["putamen_left"],
        "putamen_right": masks["putamen_right"],
        "reference": masks["occipital"],
    }
