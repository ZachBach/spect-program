"""Synthetic phantoms for testing the pipeline without participant data."""

from .cohort import healthy_cohort, parkinsonian_cohort
from .progression import (
    ProgressionSpec,
    Subject,
    Visit,
    generate_trajectories,
    oracle_ceiling,
    render,
    to_scan_records,
)
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
    "ProgressionSpec",
    "Subject",
    "Visit",
    "generate_trajectories",
    "oracle_ceiling",
    "render",
    "to_scan_records",
    "healthy_cohort",
    "parkinsonian_cohort",
    "PhantomSpec",
    "PhantomTruth",
    "generate",
    "healthy",
    "parkinsonian",
]
