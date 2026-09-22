"""Train and test splits that cannot leak a subject across the boundary.

This module exists because of one measured failure. Yagis et al. 2021
(Sci Rep, doi:10.1038/s41598-021-01681-w) split PPMI at the slice level rather
than the subject level and found accuracy inflated by 48 percentage points.
With the labels randomly shuffled, the same setup still scored about 96%
against a 50% chance baseline. The model had learned to recognise the subject,
not the disease, and every number it produced looked publishable.

That failure is silent. There is no error, no warning, and no diagnostic that
distinguishes it from success. It is only visible if the split is checked, and
by then the result has usually been written up.

So the check is structural here. `Split.__post_init__` validates on
construction, which means a leaking split cannot be made, passed around, or
serialized. There is no flag to skip it, for the same reason the data guard
has no override: the moment someone wants to skip it is the moment it is
doing its job.

Three leaks this guards, in increasing order of subtlety:

**The same scan in both halves.** Rare and usually a copy-paste error.

**Two scans of one subject split apart.** Common and easy to write. PPMI
participants are imaged repeatedly, and the closest deep-learning precedent
for this program, Adams et al. 2021 (Comput Biol Med,
doi:10.1016/j.compbiomed.2021.104312), used year 0 and year 1 scans together.
Splitting on scans rather than subjects puts a person's baseline in training
and their follow-up in test.

**A subject who appears at two sites.** The one that survives review. Leave
one site out looks airtight until a participant transferred centres, at which
point they are in both halves and nothing in the site grouping notices.
`leave_one_site_out` refuses to build folds from such a cohort rather than
quietly producing them.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Sequence


class LeakageError(RuntimeError):
    """Raised when a split would place one subject on both sides.

    An exception rather than a warning. A warning is printed at the moment a
    result stops meaning anything, into a log nobody reads until later.
    """


@dataclass(frozen=True)
class ScanRecord:
    """One acquired scan.

    `subject_id` is the grouping unit and the only one. A scan identifier is
    not a person, and neither is a slice, a volume, or an augmented copy. If
    a future format introduces some finer unit, it belongs in `scan_id`;
    `subject_id` must keep meaning one human being.

    `site_id` is the acquisition centre, which in practice also carries the
    camera and reconstruction. Held-out sites are the falsifier this program
    committed to, so it is required rather than optional.
    """

    scan_id: str
    subject_id: str
    site_id: str
    visit: str = "baseline"
    label: str | None = None

    def __post_init__(self) -> None:
        for name in ("scan_id", "subject_id", "site_id"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} must not be blank")


@dataclass(frozen=True)
class Split:
    """One train, validation and test partition.

    Validated on construction. A `Split` that exists is a `Split` that does
    not leak.
    """

    name: str
    train: tuple[ScanRecord, ...]
    validation: tuple[ScanRecord, ...] = ()
    test: tuple[ScanRecord, ...] = ()

    def __post_init__(self) -> None:
        parts = {
            "train": self.train,
            "validation": self.validation,
            "test": self.test,
        }

        seen_scans: dict[str, str] = {}
        for part, records in parts.items():
            for record in records:
                prior = seen_scans.get(record.scan_id)
                if prior is not None:
                    raise LeakageError(
                        f"split '{self.name}': scan {record.scan_id!r} appears in "
                        f"both '{prior}' and '{part}'"
                    )
                seen_scans[record.scan_id] = part

        subject_parts: dict[str, set[str]] = defaultdict(set)
        for part, records in parts.items():
            for record in records:
                subject_parts[record.subject_id].add(part)

        straddling = sorted(s for s, p in subject_parts.items() if len(p) > 1)
        if straddling:
            shown = ", ".join(straddling[:5])
            more = f" and {len(straddling) - 5} more" if len(straddling) > 5 else ""
            raise LeakageError(
                f"split '{self.name}': {len(straddling)} subject(s) appear in more "
                f"than one partition ({shown}{more}). A model can recognise a "
                "person across their own scans, so this inflates every metric "
                "downstream without producing any visible error. Group by "
                "subject_id before splitting."
            )

    # --- composition ------------------------------------------------------

    @property
    def all_records(self) -> tuple[ScanRecord, ...]:
        return self.train + self.validation + self.test

    def subjects(self, part: str = "test") -> set[str]:
        records = {"train": self.train, "validation": self.validation, "test": self.test}[part]
        return {r.subject_id for r in records}

    def sites(self, part: str = "test") -> set[str]:
        records = {"train": self.train, "validation": self.validation, "test": self.test}[part]
        return {r.site_id for r in records}

    def label_counts(self, part: str = "test") -> Counter:
        records = {"train": self.train, "validation": self.validation, "test": self.test}[part]
        return Counter(r.label for r in records if r.label is not None)

    def describe(self) -> str:
        """A one-line summary, for a log that someone will actually read."""
        pieces = []
        for part in ("train", "validation", "test"):
            records = getattr(self, part)
            if not records:
                continue
            pieces.append(
                f"{part}={len(records)} scans/{len({r.subject_id for r in records})} subjects"
            )
        return f"{self.name}: " + ", ".join(pieces)


@dataclass
class CohortAudit:
    """What is wrong, or merely worth knowing, about a set of records."""

    subjects_at_multiple_sites: dict[str, set[str]] = field(default_factory=dict)
    duplicate_scan_ids: list[str] = field(default_factory=list)
    site_subject_counts: dict[str, int] = field(default_factory=dict)
    label_counts: Counter = field(default_factory=Counter)

    @property
    def is_splittable_by_site(self) -> bool:
        return not self.subjects_at_multiple_sites and not self.duplicate_scan_ids


def audit(records: Sequence[ScanRecord]) -> CohortAudit:
    """Inspect a cohort before splitting it.

    Separated from the split functions so a cohort can be checked while it is
    still being assembled, which is when these problems are cheap to fix.
    """
    if not records:
        raise ValueError("no records to audit")

    sites_per_subject: dict[str, set[str]] = defaultdict(set)
    scan_counts: Counter = Counter()
    for record in records:
        sites_per_subject[record.subject_id].add(record.site_id)
        scan_counts[record.scan_id] += 1

    subjects_per_site: dict[str, set[str]] = defaultdict(set)
    for record in records:
        subjects_per_site[record.site_id].add(record.subject_id)

    return CohortAudit(
        subjects_at_multiple_sites={
            subject: sites for subject, sites in sites_per_subject.items() if len(sites) > 1
        },
        duplicate_scan_ids=sorted(sid for sid, n in scan_counts.items() if n > 1),
        site_subject_counts={site: len(subs) for site, subs in subjects_per_site.items()},
        label_counts=Counter(r.label for r in records if r.label is not None),
    )


def _require_clean(records: Sequence[ScanRecord]) -> CohortAudit:
    report = audit(records)

    if report.duplicate_scan_ids:
        shown = ", ".join(report.duplicate_scan_ids[:5])
        raise ValueError(
            f"{len(report.duplicate_scan_ids)} duplicate scan id(s) ({shown}). "
            "A scan counted twice is weighted twice."
        )

    if report.subjects_at_multiple_sites:
        lines = [
            f"  {subject}: {', '.join(sorted(sites))}"
            for subject, sites in sorted(report.subjects_at_multiple_sites.items())[:5]
        ]
        raise LeakageError(
            f"{len(report.subjects_at_multiple_sites)} subject(s) appear at more than "
            "one site, so no site can be held out cleanly:\n"
            + "\n".join(lines)
            + "\nDecide which site each of these belongs to, or exclude them. "
            "Leaving them in puts the same person on both sides of a "
            "leave-one-site-out boundary, which is invisible in the results."
        )

    return report


def leave_one_site_out(
    records: Sequence[ScanRecord],
    minimum_test_subjects: int = 1,
) -> list[Split]:
    """One fold per site, each holding that site out for testing.

    The primary evaluation for this program. The falsifier requires the
    hybrid to beat the binding-ratio baseline on held-out sites, because a
    model that has seen a camera during training can learn that camera's
    idiosyncrasies and carry them into its test score. Dzialas et al. 2024
    (Mov Disord, doi:10.1002/mds.30002) found only 8% of 244 Parkinson's
    neuroimaging AI studies used an external test set at all.
    """
    report = _require_clean(records)

    if len(report.site_subject_counts) < 2:
        raise ValueError(
            "leave-one-site-out needs at least two sites; "
            f"got {sorted(report.site_subject_counts)}"
        )

    by_site: dict[str, list[ScanRecord]] = defaultdict(list)
    for record in records:
        by_site[record.site_id].append(record)

    splits: list[Split] = []
    for site in sorted(by_site):
        test = tuple(by_site[site])
        train = tuple(r for r in records if r.site_id != site)

        held_out_subjects = len({r.subject_id for r in test})
        if held_out_subjects < minimum_test_subjects:
            raise ValueError(
                f"site {site!r} has {held_out_subjects} subject(s), below the "
                f"minimum of {minimum_test_subjects}. A fold this small gives a "
                "confidence interval too wide to act on; either merge the site or "
                "exclude it, as a documented decision."
            )

        splits.append(Split(name=f"leave-out-{site}", train=train, test=test))

    return splits


def grouped_k_fold(
    records: Sequence[ScanRecord],
    k: int = 5,
    seed: int = 0,
    validation_fraction: float = 0.0,
) -> list[Split]:
    """K folds, grouped so every scan of a subject travels together.

    For use when site holdout does not apply, not as a substitute for it. A
    random split across sites measures how well a model does on cameras it has
    already seen, which is a different and easier question.

    Folds are balanced by subject count rather than scan count, and subjects
    are assigned largest-first so that a subject with many visits does not
    land unevenly. Deterministic in `seed`.
    """
    _require_clean(records)

    if k < 2:
        raise ValueError("k must be at least 2")
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must lie in [0, 1)")

    by_subject: dict[str, list[ScanRecord]] = defaultdict(list)
    for record in records:
        by_subject[record.subject_id].append(record)

    subjects = sorted(by_subject)
    if len(subjects) < k:
        raise ValueError(
            f"{len(subjects)} subjects cannot fill {k} folds. Splitting on scans "
            "instead would fill them and leak."
        )

    rng = random.Random(seed)
    rng.shuffle(subjects)
    # Largest first, ties broken by the shuffled order, so fold sizes stay
    # close even when visit counts differ a lot between subjects.
    subjects.sort(key=lambda s: len(by_subject[s]), reverse=True)

    folds: list[list[str]] = [[] for _ in range(k)]
    scan_totals = [0] * k
    for subject in subjects:
        target = min(range(k), key=lambda i: (scan_totals[i], len(folds[i])))
        folds[target].append(subject)
        scan_totals[target] += len(by_subject[subject])

    splits: list[Split] = []
    for index, held_out in enumerate(folds):
        test_subjects = set(held_out)
        remaining = [s for s in subjects if s not in test_subjects]

        validation_subjects: set[str] = set()
        if validation_fraction > 0.0:
            n_validation = max(1, round(len(remaining) * validation_fraction))
            picker = random.Random(seed + index + 1)
            validation_subjects = set(picker.sample(remaining, n_validation))

        train = tuple(
            r
            for s in remaining
            if s not in validation_subjects
            for r in by_subject[s]
        )
        validation = tuple(r for s in sorted(validation_subjects) for r in by_subject[s])
        test = tuple(r for s in sorted(test_subjects) for r in by_subject[s])

        splits.append(
            Split(name=f"fold-{index + 1}", train=train, validation=validation, test=test)
        )

    return splits


def coverage(splits: Iterable[Split]) -> dict[str, int]:
    """How many times each scan is used for testing across a set of folds.

    A correct cross-validation tests every scan exactly once. Anything else
    means a fold was dropped or a record double-counted, and the aggregate
    metric is then a weighted average nobody intended.
    """
    counts: Counter = Counter()
    for split in splits:
        for record in split.test:
            counts[record.scan_id] += 1
    return dict(counts)
