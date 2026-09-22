"""Where a number came from, carried alongside the number.

Synthetic and real data are not interchangeable and the difference is not
recoverable from the values themselves. A binding ratio computed from a
phantom looks exactly like one computed from a person. Once the two are in
the same array, nothing downstream can separate them, and a result quietly
becomes uninterpretable.

So provenance travels with the data and is checked at the points where mixing
would matter. This mirrors the benchmark gate in the LabOptimal repository,
where a synthetic corpus can never produce a passing verdict regardless of
its score. The principle is the same: the check is structural, so it cannot be
forgotten under deadline.
"""

from __future__ import annotations

from enum import Enum


class Provenance(str, Enum):
    """Whether data came from a generator or from a person."""

    #: Produced by spect.synthetic. Exercises code; measures nothing.
    SYNTHETIC = "synthetic"

    #: Acquired from a participant under a data use agreement.
    REAL = "real"


class ProvenanceError(RuntimeError):
    """Raised when synthetic and real data would be mixed improperly.

    Deliberately not a warning. A warning would be printed into a log nobody
    reads at the moment the result stops meaning anything.
    """
