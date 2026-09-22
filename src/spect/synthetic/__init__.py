"""Synthetic phantoms for testing the pipeline without participant data."""

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
    "PhantomSpec",
    "PhantomTruth",
    "generate",
    "healthy",
    "parkinsonian",
]
