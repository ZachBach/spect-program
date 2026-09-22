"""Tests for split construction.

The first test in this file is the reason the module exists. Everything else
is support for it.
"""

from __future__ import annotations

import pytest

from spect.eval.splits import (
    CohortAudit,
    LeakageError,
    ScanRecord,
    Split,
    audit,
    coverage,
    grouped_k_fold,
    leave_one_site_out,
)


def cohort(
    n_subjects: int = 12,
    visits: int = 2,
    sites: tuple[str, ...] = ("siteA", "siteB", "siteC"),
) -> list[ScanRecord]:
    """A cohort where every subject is imaged more than once.

    Repeat imaging is the normal case, not an edge case: PPMI images
    participants across visits, and the closest precedent for this program
    used year 0 and year 1 scans together.
    """
    records = []
    for i in range(n_subjects):
        subject = f"sub-{i:03d}"
        site = sites[i % len(sites)]
        for v in range(visits):
            records.append(
                ScanRecord(
                    scan_id=f"{subject}_v{v}",
                    subject_id=subject,
                    site_id=site,
                    visit=f"v{v}",
                    label="patient" if i % 2 else "control",
                )
            )
    return records


# --- the failure this module exists to prevent ------------------------------


def test_splitting_on_scans_instead_of_subjects_is_refused():
    """The Yagis 2021 failure, reproduced and caught.

    Splitting a repeat-imaged cohort at the scan level puts a person's
    baseline in training and their follow-up in test. The model then scores
    well by recognising the subject. Yagis measured 48 points of inflation
    from exactly this, and about 96% accuracy on randomly shuffled labels.

    There is no error message in that failure. This is the error message.
    """
    records = cohort(n_subjects=10, visits=2)
    naive_train = tuple(records[::2])
    naive_test = tuple(records[1::2])

    with pytest.raises(LeakageError, match="more than one partition"):
        Split(name="scan-level", train=naive_train, test=naive_test)


def test_the_refusal_names_the_subjects_and_says_why():
    records = cohort(n_subjects=10, visits=2)
    with pytest.raises(LeakageError) as excinfo:
        Split(name="scan-level", train=tuple(records[::2]), test=tuple(records[1::2]))
    message = str(excinfo.value)
    assert "sub-000" in message
    assert "group by subject_id" in message.lower()


def test_there_is_no_flag_to_skip_validation():
    """A split that exists is a split that does not leak. If an escape hatch
    is ever added, it will be used on the day it matters most."""
    import inspect

    source = inspect.getsource(Split)
    for escape in ("force", "skip_validation", "allow_leak", "unchecked"):
        assert escape not in source


# --- Split construction -----------------------------------------------------


def test_a_clean_split_constructs():
    records = cohort(n_subjects=6, visits=2)
    held_out = {"sub-000", "sub-001"}
    split = Split(
        name="clean",
        train=tuple(r for r in records if r.subject_id not in held_out),
        test=tuple(r for r in records if r.subject_id in held_out),
    )
    assert split.subjects("test") == held_out
    assert not split.subjects("train") & split.subjects("test")


def test_the_same_scan_in_both_halves_is_refused():
    records = cohort(n_subjects=4, visits=1)
    with pytest.raises(LeakageError, match="appears in"):
        Split(name="dup", train=tuple(records), test=(records[0],))


def test_validation_is_checked_too():
    """Three-way splits are where a subject slips through, because the
    train/test check passes and nobody looks at validation."""
    records = cohort(n_subjects=6, visits=2)
    with pytest.raises(LeakageError):
        Split(
            name="three-way",
            train=(records[0],),
            validation=(records[1],),  # same subject as records[0]
            test=(records[4],),
        )


def test_blank_identifiers_are_refused():
    with pytest.raises(ValueError, match="subject_id"):
        ScanRecord(scan_id="s1", subject_id="  ", site_id="siteA")


def test_describe_is_readable():
    records = cohort(n_subjects=6, visits=2)
    held_out = {"sub-000"}
    split = Split(
        name="clean",
        train=tuple(r for r in records if r.subject_id not in held_out),
        test=tuple(r for r in records if r.subject_id in held_out),
    )
    text = split.describe()
    assert "train=10 scans/5 subjects" in text
    assert "test=2 scans/1 subjects" in text


# --- leave one site out -----------------------------------------------------


def test_one_fold_per_site():
    splits = leave_one_site_out(cohort())
    assert [s.name for s in splits] == ["leave-out-siteA", "leave-out-siteB", "leave-out-siteC"]


def test_the_held_out_site_never_appears_in_training():
    """The falsifier requires held-out sites, because a model that has seen a
    camera can carry its idiosyncrasies into the test score."""
    for split in leave_one_site_out(cohort()):
        site = split.name.removeprefix("leave-out-")
        assert split.sites("test") == {site}
        assert site not in split.sites("train")


def test_every_scan_is_tested_exactly_once_across_the_folds():
    records = cohort()
    used = coverage(leave_one_site_out(records))
    assert set(used) == {r.scan_id for r in records}
    assert set(used.values()) == {1}


def test_a_subject_imaged_at_two_sites_blocks_site_holdout():
    """The leak that survives review. Site grouping looks airtight until a
    participant transferred centres, and nothing in the grouping notices."""
    records = cohort(n_subjects=6, visits=2)
    moved = ScanRecord(
        scan_id="sub-000_v2", subject_id="sub-000", site_id="siteB", visit="v2"
    )
    with pytest.raises(LeakageError, match="more than one site"):
        leave_one_site_out(records + [moved])


def test_the_transfer_message_names_the_subject_and_both_sites():
    records = cohort(n_subjects=6, visits=2)
    moved = ScanRecord(scan_id="sub-000_v2", subject_id="sub-000", site_id="siteB")
    with pytest.raises(LeakageError) as excinfo:
        leave_one_site_out(records + [moved])
    message = str(excinfo.value)
    assert "sub-000" in message
    assert "siteA" in message and "siteB" in message


def test_a_single_site_cannot_be_held_out():
    with pytest.raises(ValueError, match="at least two sites"):
        leave_one_site_out(cohort(sites=("only",)))


def test_a_site_too_small_to_test_on_is_refused():
    records = cohort(n_subjects=9, visits=1, sites=("big", "big", "tiny"))
    with pytest.raises(ValueError, match="below the minimum"):
        leave_one_site_out(records, minimum_test_subjects=5)


def test_duplicate_scan_ids_are_refused():
    records = cohort(n_subjects=4, visits=1)
    with pytest.raises(ValueError, match="duplicate scan id"):
        leave_one_site_out(records + [records[0]])


# --- grouped k-fold ---------------------------------------------------------


def test_no_subject_appears_in_two_folds():
    splits = grouped_k_fold(cohort(n_subjects=20), k=5)
    for split in splits:
        assert not split.subjects("train") & split.subjects("test")


def test_a_subjects_visits_are_never_separated():
    records = cohort(n_subjects=20, visits=3)
    for split in grouped_k_fold(records, k=4):
        for subject in split.subjects("test"):
            in_test = {r.scan_id for r in split.test if r.subject_id == subject}
            expected = {r.scan_id for r in records if r.subject_id == subject}
            assert in_test == expected


def test_k_fold_tests_every_scan_exactly_once():
    records = cohort(n_subjects=20, visits=2)
    used = coverage(grouped_k_fold(records, k=5))
    assert set(used) == {r.scan_id for r in records}
    assert set(used.values()) == {1}


def test_folds_are_deterministic_in_the_seed():
    records = cohort(n_subjects=20)
    a = [sorted(s.subjects("test")) for s in grouped_k_fold(records, k=5, seed=3)]
    b = [sorted(s.subjects("test")) for s in grouped_k_fold(records, k=5, seed=3)]
    assert a == b


def test_a_different_seed_gives_different_folds():
    records = cohort(n_subjects=20)
    a = [sorted(s.subjects("test")) for s in grouped_k_fold(records, k=5, seed=1)]
    b = [sorted(s.subjects("test")) for s in grouped_k_fold(records, k=5, seed=2)]
    assert a != b


def test_folds_are_roughly_balanced_when_visit_counts_differ():
    """A subject with many visits should not make one fold twice the size of
    another, which would quietly weight that fold's sites and labels."""
    records = []
    for i in range(20):
        visits = 5 if i < 4 else 1
        for v in range(visits):
            records.append(
                ScanRecord(
                    scan_id=f"sub-{i:03d}_v{v}",
                    subject_id=f"sub-{i:03d}",
                    site_id="siteA",
                    visit=f"v{v}",
                )
            )
    sizes = [len(s.test) for s in grouped_k_fold(records, k=4)]
    assert max(sizes) - min(sizes) <= 2


def test_validation_is_carved_out_by_subject():
    records = cohort(n_subjects=20, visits=2)
    for split in grouped_k_fold(records, k=5, validation_fraction=0.25):
        assert split.validation
        assert not split.subjects("train") & split.subjects("validation")
        assert not split.subjects("validation") & split.subjects("test")


def test_too_few_subjects_for_k_is_refused():
    with pytest.raises(ValueError, match="cannot fill"):
        grouped_k_fold(cohort(n_subjects=3), k=5)


def test_k_below_two_is_refused():
    with pytest.raises(ValueError, match="at least 2"):
        grouped_k_fold(cohort(), k=1)


# --- audit ------------------------------------------------------------------


def test_audit_reports_composition_without_raising():
    report = audit(cohort(n_subjects=12, visits=2))
    assert isinstance(report, CohortAudit)
    assert report.is_splittable_by_site
    assert sum(report.site_subject_counts.values()) == 12
    assert report.label_counts["patient"] == 12


def test_audit_surfaces_a_transferred_subject_without_raising():
    """Auditing should be usable while a cohort is still being assembled,
    which is when these are cheap to fix."""
    records = cohort(n_subjects=6, visits=1)
    records.append(ScanRecord(scan_id="sub-000_x", subject_id="sub-000", site_id="siteZ"))
    report = audit(records)
    assert "sub-000" in report.subjects_at_multiple_sites
    assert not report.is_splittable_by_site


def test_auditing_nothing_is_an_error():
    with pytest.raises(ValueError, match="no records"):
        audit([])
