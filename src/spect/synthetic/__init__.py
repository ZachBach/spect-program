"""Synthetic phantoms for testing the pipeline without participant data."""

from .cohort import healthy_cohort, parkinsonian_cohort
from .phantom import (
    Phantom,
    PhantomSpec,
    PhantomTruth,
    generate,
    healthy,
    parkinsonian,
)

__all__ = [
    "Phantom",
    "healthy_cohort",
    "parkinsonian_cohort",
    "PhantomSpec",
    "PhantomTruth",
    "generate",
    "healthy",
    "parkinsonian",
]
