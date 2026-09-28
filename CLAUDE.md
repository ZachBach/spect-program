# CLAUDE.md

Guidance for Claude Code working in this repository.

## What this is

Hybrid CNNs for brain SPECT, predicting time to a first clinically meaningful
milestone in early Parkinson's disease. Research use only. Owner: Zachary
Auerbach, doing business as Aurelius Dynamic.

No data has been received. Everything here runs on synthetic phantoms.

## Rule 1, which overrides convenience

**Never load participant-level PPMI data into an AI tool, including this one.**

PPMI's Data Use Agreement v5.0 (April 2026) prohibits AI tools that cannot
guarantee containment of the data. That covers this session. Work from code,
file schemas, synthetic phantoms and aggregate outputs only.

This is easier to break than it looks. Reading a participant file into
context, printing one with a shell command, or pasting an error containing a
row of it all move the data somewhere it may not be. The rule is about where
bytes end up, not about intent. If a task seems to need participant-level
data in context, stop and say so rather than finding a way.

Treat NAPS2, Dementias Platform UK and every other clinical dataset the same
way until their own terms say otherwise.

Two consequences that follow:

- **Trained weights stay private.** A model trained on participant-level PPMI
  data may only be shared through routes MJFF approves. Code ships MIT.
- **No clinical claims.** FDA's clinical decision support guidance, revised
  6 January 2026, treats software that analyses a medical image to produce
  findings as a medical device. Nothing here diagnoses.

## Commands

```sh
git config core.hooksPath .githooks    # once per clone, before anything else
python -m pytest                       # 213 tests, about 25 seconds
python -m pytest tests/test_survival.py
python -m pytest -k "leak or provenance"
```

`pyproject.toml` sets `pythonpath = ["src"]`, so no install is needed to run
tests. Core dependencies are numpy and pytest. scipy is deliberately not one;
the point-spread blur is implemented directly in `phantom.py`.

torch is optional and only `models/hybrid.py` imports it, so the rest of the
codebase runs without it and `tests/test_hybrid.py` skips. CPU build:
`pip install torch --index-url https://download.pytorch.org/whl/cpu`

## Layout

```
src/spect/
  provenance.py       Provenance enum and ProvenanceError; shared
  synthetic/
    phantom.py        DaT-SPECT phantoms, with PSF, Poisson noise, gradient
    cohort.py         Cohorts with between-subject variation
    progression.py    Longitudinal trajectories, milestones, oracle ceiling
  features/
    binding_ratio.py  SBR per structure; asymmetry; putamen/caudate ratio
    spatial.py        Posterior/anterior contrast; what a region mean discards
    normal_database.py  Z-score baseline, with the provenance guard
  eval/
    splits.py         Leak-proof split construction
    survival.py       Concordance, Kaplan-Meier, time-dependent AUC, calibration
  models/
    cox.py            Cox PH baseline, Newton-Raphson, Breslow ties
    hybrid.py         3D CNN plus tabular fusion, Cox loss (needs torch)
  io/ preprocess/            empty; see README
```

## Guards, and why none of them have an override

Four checks are structural rather than advisory. Each guards a failure that
produces no error and no warning, only a wrong number that looks right. Do not
add a bypass flag to any of them; the moment someone wants one is the moment
it is working.

**`.githooks/pre-commit`** blocks imaging, participant tables and weights from
git. Checks DICOM magic bytes at offset 128, so a renamed volume is still
caught. History is permanent and there is no clean undo.

**`Split.__post_init__`** refuses any partition placing one subject on both
sides. Yagis 2021 (doi:10.1038/s41598-021-01681-w) measured 48 points of
accuracy inflation from slice-level splits on PPMI, and about 96% accuracy on
randomly shuffled labels. The subtle case is a participant who transferred
centres: leave-one-site-out looks airtight and is not.

**`NormalDatabase.z_scores`** refuses to score a real scan against a
synthetic database, and `subject_provenance` has no default so the claim must
be written out before it can be refused.

**`concordance_index`** raises when no pairs are comparable rather than
returning 0.5. A censored subject with the earlier time is comparable with
nobody.

## Conventions this codebase holds to

**State the convention, do not assume it.** Concordance implementations
disagree on tied times and tied predictions. What `survival.py` does is
written in its docstrings and tested against hand-computed values, not
against another library.

**Test against known truth, not against plausibility.** Phantoms carry the
nominal ratios they were built from. Survival metrics are checked against
values computed by hand in the test docstring. A test that only asserts a
number looks reasonable will pass when the code is wrong.

**Model the degradations.** The phantom includes partial volume blur, Poisson
noise and the posterior-to-anterior gradient because omitting any of them
makes the fixture easier than reality in the direction that flatters a model.

**Regions are inputs, never derived inside a calculator.** Testing arithmetic
and testing registration must stay separable. Every current test proves the
arithmetic and nothing about anatomy.

## Known limits, recorded so they are not rediscovered

- **The phantom saturates.** The z-score baseline reaches AUC 1.000 at 0.40
  putamen loss, so phantoms cannot compare baseline against hybrid. See D3 in
  `docs/decisions.md`.
- **The prognostic task does have headroom**, unlike classification: the
  oracle ceiling sits at 0.708 against a baseline-only 0.618 at default
  settings. Always report a concordance next to `oracle_ceiling`, because
  0.70 is at the ceiling rather than mediocre. See D4.
- **Time-dependent AUC excludes subjects censored before the horizon.** That
  is unbiased only if censoring is independent of risk, which is usually
  optimistic. Inverse-probability-of-censoring weighting is the correction and
  is not implemented. `excluded_censored` reports the exposure.
- **`severity_from_gradient` defaults to zero.** With it at zero the outcome
  is a function of quantities region means measure directly, so a network is
  learning to reconstruct a binding ratio and its score is a smoke test. Set
  it above zero before any hybrid result is read off this fixture, and check
  the network also beats `features/spatial.py`. See D5.
- **Seed the model as well as the training loop.** Weights are initialized at
  construction, so `train(seed=...)` alone leaves runs starting from different
  weights while appearing seeded.
- **The GitKraken plugin and three account-synced plugins are disabled here**
  because their hooks can see prompts, tool output or the transcript. Neither
  override has been verified empirically, and the synced set can change
  without any change to this repository. See D2 and D6 in `docs/decisions.md`.

## Writing style for this project

From the owner's handoff, and it applies to code comments, docs and replies:

- No em dashes, and avoid en dashes too.
- Plain, human voice.
- Every factual claim gets a source. Anything unverified is labelled
  unverified. Citations in this repo carry a DOI or a PubMed id.
- Push back when there is substance to push back on.

Console output stays ASCII. Source files are UTF-8, but printed output runs
through a cp1252 terminal on Windows.

## Decision record

`docs/decisions.md`, newest first. Add an entry for anything that would
otherwise be re-argued later, including decisions that went against the
handoff and why.
