"""Synthetic DaT-SPECT phantoms for testing the pipeline end to end.

These are not simulations of anything. They are geometric stand-ins with
roughly the right structure, built so that code can be exercised, split logic
can be tested, and metrics can be checked against a known answer before any
participant data exists. Nothing measured on a phantom says anything about
real scans, and no number produced from one belongs in a result.

What the phantom does model, because leaving it out would make the fixture
misleading:

**Partial volume effect.** A SPECT scanner resolves at roughly 8 to 12 mm
FWHM, which is comparable to the width of the putamen itself. Activity
therefore blurs out of small structures and the measured binding ratio comes
in below the true one. A phantom without this would let a binding-ratio
calculator look perfectly accurate when the real thing is biased low. The
truth record keeps the nominal ratio, so a test can assert the direction and
rough size of that bias rather than pretending it is absent.

**Counting noise.** SPECT is a photon-counting measurement, so the noise is
Poisson and scales with the square root of counts. Gaussian noise would make
low-count regions look better behaved than they are.

**A posterior-to-anterior gradient.** In Parkinson's disease the putamen
loses binding before the caudate, and the posterior putamen before the
anterior. That gradient is the pattern a model would have to learn, so a
phantom that lost binding uniformly would be an easier problem than the real
one, in the specific way that matters.

The geometry is approximate and deliberately so. Ellipsoid centres and radii
are rounded numbers chosen to sit in about the right place at about the right
size. They have not been checked against an atlas and must not be used as
one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

#: Conversion from full width at half maximum to Gaussian sigma.
_FWHM_TO_SIGMA = 1.0 / (2.0 * math.sqrt(2.0 * math.log(2.0)))


@dataclass(frozen=True)
class Ellipsoid:
    """One structure, in millimetres relative to the volume centre.

    `centre` is (x, y, z) with x lateral (positive is the subject's left in
    this phantom's own convention), y anterior, z superior. The convention is
    internal to the phantom and is not a claim about any real orientation
    standard; loaders for real data must establish orientation from the file.
    """

    centre: tuple[float, float, float]
    radii: tuple[float, float, float]


# Approximate striatal geometry. Rounded, not atlas-derived. The caudate head
# sits anterior, superior and medial; the putamen is lateral and runs
# posteriorly, which is what gives the pair its comma shape in an axial slice.
CAUDATE_LEFT = Ellipsoid(centre=(13.0, 14.0, 12.0), radii=(7.0, 9.0, 9.0))
CAUDATE_RIGHT = Ellipsoid(centre=(-13.0, 14.0, 12.0), radii=(7.0, 9.0, 9.0))
PUTAMEN_LEFT = Ellipsoid(centre=(26.0, 2.0, 2.0), radii=(6.0, 14.0, 10.0))
PUTAMEN_RIGHT = Ellipsoid(centre=(-26.0, 2.0, 2.0), radii=(6.0, 14.0, 10.0))

#: The reference region. Occipital cortex is used because it is largely free
#: of dopamine transporters, which is what makes it a usable denominator.
OCCIPITAL = Ellipsoid(centre=(0.0, -55.0, 0.0), radii=(28.0, 18.0, 20.0))

#: Rough extent of the head, so that background is zero outside it.
BRAIN = Ellipsoid(centre=(0.0, 0.0, 5.0), radii=(70.0, 85.0, 60.0))


@dataclass(frozen=True)
class PhantomSpec:
    """Everything that determines a phantom. Same spec, same volume.

    putamen_loss and caudate_loss are (left, right) fractions in [0, 1].
    0.0 leaves the structure at its healthy ratio; 1.0 removes all specific
    binding, leaving it at background.
    """

    shape: tuple[int, int, int] = (91, 109, 91)
    voxel_mm: float = 2.0

    putamen_loss: tuple[float, float] = (0.0, 0.0)
    caudate_loss: tuple[float, float] = (0.0, 0.0)

    #: How much of the putamen loss is redistributed posterior-to-anterior
    #: rather than applied flat. At 0.0 the loss is uniform. Above that the
    #: posterior edge takes more than the nominal loss and the anterior edge
    #: less, with the mean across the structure preserved.
    #:
    #: The gradient scales with the binding still present, so it vanishes as
    #: the loss approaches total. Without that, a structure with complete
    #: loss would keep binding at its anterior end, which is both wrong and
    #: the kind of wrong that quietly makes a classifier's job easier.
    posterior_gradient: float = 0.6

    healthy_putamen_ratio: float = 3.0
    healthy_caudate_ratio: float = 3.5

    #: Mean counts per voxel in the reference region. Raising this lowers
    #: the relative Poisson noise.
    reference_counts: float = 120.0

    fwhm_mm: float = 10.0
    noise: bool = True
    seed: int = 0

    def __post_init__(self) -> None:
        for name in ("putamen_loss", "caudate_loss"):
            for side in getattr(self, name):
                if not 0.0 <= side <= 1.0:
                    raise ValueError(f"{name} entries must lie in [0, 1], got {side}")
        if not 0.0 <= self.posterior_gradient <= 1.0:
            raise ValueError("posterior_gradient must lie in [0, 1]")
        if self.reference_counts <= 0:
            raise ValueError("reference_counts must be positive")
        if self.fwhm_mm < 0:
            raise ValueError("fwhm_mm must not be negative")


@dataclass(frozen=True)
class PhantomTruth:
    """What the phantom was built from.

    These are the ratios written into the activity map before blurring and
    noise. A binding ratio measured off the output will read lower, because
    the point-spread function pulls activity out of structures this small.
    That gap is the partial volume effect and it is expected; see the module
    docstring.

    The size of the gap is substantial. At the default 10 mm FWHM a nominal
    putamen ratio of 3.0 measures at about 1.39, down 54%. So these numbers
    are not comparable with published DaTscan binding ratios, which are
    measured quantities from a calibrated normal database. To make a phantom
    whose *measured* ratio lands in a realistic range, raise the nominal
    rather than reducing the blur.
    """

    putamen_ratio: tuple[float, float]
    caudate_ratio: tuple[float, float]
    putamen_loss: tuple[float, float]
    caudate_loss: tuple[float, float]

    @property
    def putamen_asymmetry(self) -> float:
        """Absolute difference between sides, as a fraction of their mean.

        Asymmetry is one of the features that separates Parkinson's disease
        from the symmetric loss seen in some other causes of reduced binding.
        """
        left, right = self.putamen_ratio
        mean = (left + right) / 2.0
        return abs(left - right) / mean if mean else 0.0


@dataclass(frozen=True)
class Phantom:
    """A generated volume and everything needed to check work against it."""

    volume: np.ndarray
    masks: dict[str, np.ndarray]
    truth: PhantomTruth
    spec: PhantomSpec
    #: Noise-free, unblurred activity map. Useful for isolating whether an
    #: error came from the estimator or from the degradations.
    clean: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0))


def _grid_mm(spec: PhantomSpec) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Voxel-centre coordinates in millimetres, origin at the volume centre."""
    axes = []
    for n in spec.shape:
        idx = np.arange(n, dtype=np.float64)
        axes.append((idx - (n - 1) / 2.0) * spec.voxel_mm)
    return np.meshgrid(*axes, indexing="ij")


def _ellipsoid_mask(
    ellipsoid: Ellipsoid, grid: tuple[np.ndarray, np.ndarray, np.ndarray]
) -> np.ndarray:
    x, y, z = grid
    cx, cy, cz = ellipsoid.centre
    rx, ry, rz = ellipsoid.radii
    return ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 + ((z - cz) / rz) ** 2 <= 1.0


def _gaussian_kernel(sigma_voxels: float) -> np.ndarray:
    """A normalized 1D Gaussian, truncated at four sigma."""
    if sigma_voxels <= 0:
        return np.array([1.0])
    radius = max(1, int(math.ceil(4.0 * sigma_voxels)))
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-(offsets**2) / (2.0 * sigma_voxels**2))
    return kernel / kernel.sum()


def _blur(volume: np.ndarray, sigma_voxels: float) -> np.ndarray:
    """Separable Gaussian blur, standing in for the scanner's point spread.

    Separable because a 3D Gaussian is the product of three 1D Gaussians, so
    three passes cost far less than one 3D convolution and give the same
    result. Edges are handled by edge replication, which is closer to a real
    reconstruction than zero padding would be.
    """
    if sigma_voxels <= 0:
        return volume.copy()

    kernel = _gaussian_kernel(sigma_voxels)
    pad = len(kernel) // 2
    out = volume.astype(np.float64, copy=True)

    for axis in range(volume.ndim):
        moved = np.moveaxis(out, axis, -1)
        padded = np.pad(moved, [(0, 0)] * (moved.ndim - 1) + [(pad, pad)], mode="edge")
        # Sliding windows turn the convolution into one contraction, which
        # avoids a Python-level loop over several tens of thousands of lines.
        windows = np.lib.stride_tricks.sliding_window_view(padded, len(kernel), axis=-1)
        moved = windows @ kernel[::-1]
        out = np.moveaxis(moved, -1, axis)

    return out


def generate(spec: PhantomSpec | None = None) -> Phantom:
    """Build a phantom. Deterministic: the same spec gives the same volume."""
    spec = spec or PhantomSpec()
    grid = _grid_mm(spec)
    x, y, _ = grid

    brain = _ellipsoid_mask(BRAIN, grid)
    masks = {
        "caudate_left": _ellipsoid_mask(CAUDATE_LEFT, grid),
        "caudate_right": _ellipsoid_mask(CAUDATE_RIGHT, grid),
        "putamen_left": _ellipsoid_mask(PUTAMEN_LEFT, grid),
        "putamen_right": _ellipsoid_mask(PUTAMEN_RIGHT, grid),
        "occipital": _ellipsoid_mask(OCCIPITAL, grid) & brain,
        "brain": brain,
    }

    # Background is flat inside the head and zero outside it. Real scans have
    # structured non-specific binding; this does not, and any method that
    # depends on that structure will look better here than it should.
    activity = np.where(brain, spec.reference_counts, 0.0)

    putamen_ratios: list[float] = []
    caudate_ratios: list[float] = []

    for side_index, side in enumerate(("left", "right")):
        # --- putamen, with the posterior-to-anterior gradient ---
        loss = spec.putamen_loss[side_index]
        mask = masks[f"putamen_{side}"]
        nominal = spec.healthy_putamen_ratio * (1.0 - loss)
        putamen_ratios.append(nominal)

        if mask.any():
            centre_y = PUTAMEN_LEFT.centre[1]
            radius_y = PUTAMEN_LEFT.radii[1]
            # -1 at the posterior edge, +1 at the anterior edge.
            position = np.clip((y - centre_y) / radius_y, -1.0, 1.0)
            # Posterior voxels take more than the nominal loss, anterior less.
            # The gradient is damped by the binding that remains, so at total
            # loss it disappears and the structure goes uniformly to
            # background. That damping also keeps local_loss inside [0, 1]
            # for every combination of loss and gradient, so no clipping is
            # needed and the mean across the structure stays equal to `loss`.
            gradient = spec.posterior_gradient * (1.0 - loss)
            local_loss = loss * (1.0 + gradient * -position)
            local_ratio = spec.healthy_putamen_ratio * (1.0 - local_loss)
            activity = np.where(
                mask, spec.reference_counts * (1.0 + local_ratio), activity
            )

        # --- caudate, uniform ---
        loss = spec.caudate_loss[side_index]
        ratio = spec.healthy_caudate_ratio * (1.0 - loss)
        caudate_ratios.append(ratio)
        activity = np.where(
            masks[f"caudate_{side}"], spec.reference_counts * (1.0 + ratio), activity
        )

    clean = activity.copy()

    sigma_voxels = spec.fwhm_mm * _FWHM_TO_SIGMA / spec.voxel_mm
    volume = _blur(activity, sigma_voxels)

    if spec.noise:
        rng = np.random.default_rng(spec.seed)
        volume = rng.poisson(np.clip(volume, 0.0, None)).astype(np.float64)

    truth = PhantomTruth(
        putamen_ratio=(putamen_ratios[0], putamen_ratios[1]),
        caudate_ratio=(caudate_ratios[0], caudate_ratios[1]),
        putamen_loss=spec.putamen_loss,
        caudate_loss=spec.caudate_loss,
    )

    return Phantom(volume=volume, masks=masks, truth=truth, spec=spec, clean=clean)


def healthy(seed: int = 0) -> Phantom:
    """A phantom with no loss on either side."""
    return generate(PhantomSpec(seed=seed))


def parkinsonian(
    severity: float = 0.6, asymmetry: float = 0.25, seed: int = 0
) -> Phantom:
    """A phantom with the asymmetric, putamen-first pattern of Parkinson's.

    severity is the mean putamen loss; asymmetry is the difference between
    sides. The caudate takes a third of the putamen loss, which is the usual
    direction of the gradient but a made-up number.
    """
    worse = min(1.0, severity + asymmetry / 2.0)
    better = max(0.0, severity - asymmetry / 2.0)
    return generate(
        PhantomSpec(
            putamen_loss=(worse, better),
            caudate_loss=(worse / 3.0, better / 3.0),
            seed=seed,
        )
    )
