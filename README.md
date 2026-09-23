# spect-program

Hybrid CNNs for brain SPECT. Program 10 on the Aurelius Dynamic research page,
under the track NUCLEAR NEUROIMAGING.

**Research use only.** Nothing here produces a diagnosis, and nothing here is
clinical decision support. FDA's clinical decision support guidance, revised
6 January 2026 and reissued 29 January 2026, treats software that analyses a
medical image to produce findings as a medical device. This program stays on
the research side of that line and makes no clinical claims.

Owner: Zachary Auerbach, doing business as Aurelius Dynamic.

**No participant data has been requested, received or processed.** Everything
runs on synthetic phantoms. Every number this repository can currently produce
is a property of a fixture, not a finding about anyone's brain.

## Install

```sh
git config core.hooksPath .githooks    # once per clone, before anything else
python -m pytest                       # 213 tests, about 25 seconds
```

Without the hook configuration the data guard does not run. See "The guard".

`pyproject.toml` sets `pythonpath = ["src"]`, so no install is needed to run
the tests. Core dependencies are numpy and pytest.

torch is optional and only `models/hybrid.py` imports it, so everything else
runs without it and those tests skip:

```sh
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Data rules

These are not conventions. They come from data use agreements with teeth.

**1. No participant-level data in any AI tool.** Not this repository's
assistant, not Claude Code, not Copilot, not any cloud assistant. PPMI's Data
Use Agreement v5.0 (April 2026) prohibits AI tools that do not guarantee
containment. Work from code, file schemas, synthetic phantoms and aggregate
outputs. Treat NAPS2, Dementias Platform UK and every other clinical dataset
the same way until their own terms say otherwise.

This rule is easier to break than it looks. An assistant reading a file, or a
shell command printing one, moves the contents into a context you may not
control. The rule is about where bytes end up, not about intent.

**2. Trained weights stay private.** A model trained on participant-level PPMI
data may only be shared through routes MJFF approves. Code ships MIT. Weights
do not go to GitHub or Hugging Face.

**3. Participant data never enters git.** History is permanent. A file
committed and then deleted is still in the object store, in every clone and in
every fork. Tables and volumes live outside the repository and loaders take a
path from config.

## The guards

Five checks are structural rather than advisory. Each one guards a failure
that produces no error and no warning, only a wrong number that looks right.
None of them has an override, and a test asserts that none is added.

**The data guard.** `.gitignore` stops the accident. `.githooks/pre-commit`
stops everything else: it reads what is actually staged, so it covers
`git add -f` and tooling that stages files without being asked. It blocks
imaging formats, anything inside a data directory, trained weights and
credentials, checks DICOM magic bytes at offset 128 so a volume renamed to
`.txt` is still caught, and refuses anything over 2 MB.

Verified by test before it was trusted: a force-added `.nii` is blocked, a
DICOM renamed to `meeting_notes.txt` is blocked on its magic bytes, a
participant CSV is blocked, and an allow-listed synthetic fixture passes.

**Splits that cannot leak.** `Split.__post_init__` refuses any partition
placing one subject on both sides, so a leaking split cannot be constructed,
passed around or serialized. Yagis et al. 2021 measured 48 points of accuracy
inflation from slice-level splits on PPMI, with randomly shuffled labels still
scoring about 96%. The subtle case is a participant who transferred centres:
leave-one-site-out looks airtight and is not.

**Provenance that will not mix.** `NormalDatabase.z_scores` refuses to score a
real scan against a database built from phantoms, and `subject_provenance` has
no default, so the claim has to be written out before it can be refused.

**Concordance that refuses to guess.** `concordance_index` raises when no
pairs are comparable rather than returning 0.5. A censored subject with the
earlier time is comparable with nobody.

**A measured ceiling.** `oracle_ceiling` reports what a model knowing the true
latent severity would score. Event times are stochastic, so perfect ordering is
impossible, and a concordance published without its ceiling invites being read
as mediocre when it is at the limit of the data.

## Layout

```
src/spect/
  provenance.py       Provenance enum and ProvenanceError; shared
  synthetic/
    phantom.py        DaT-SPECT phantoms: PSF, Poisson noise, spatial gradient
    cohort.py         Cohorts with between-subject variation
    progression.py    Longitudinal trajectories, milestones, oracle ceiling
  features/
    binding_ratio.py  SBR per structure; asymmetry; putamen/caudate ratio
    normal_database.py  Z-score baseline, with the provenance guard
    spatial.py        Posterior/anterior contrast; what a region mean discards
  eval/
    splits.py         Leak-proof split construction, leave-one-site-out
    survival.py       Concordance, Kaplan-Meier, time-dependent AUC, calibration
  models/
    cox.py            Cox PH baseline, Newton-Raphson, Breslow ties
    hybrid.py         3D CNN plus tabular fusion, Cox loss (needs torch)
  io/ preprocess/     empty; loaders and spatial normalization go here
configs/              experiment configs; data paths live here, never in code
tests/                213 tests
docs/decisions.md     decision record, newest first
```

## Research question

In early disease, stages 2B and 3 of the NSD-ISS biological staging system,
does a hybrid model predict time to the first clinically meaningful milestone
better than baseline binding ratios, age and motor score?

**Falsifier.** The hybrid must beat the binding-ratio model on held-out sites,
at a matched share of cases flagged inconclusive, and with calibrated
probabilities. If it cannot, the added complexity is not justified, and that
null gets published at full weight.

The reason to state the falsifier before the data arrives: Dzialas et al. 2024
(Mov Disord, doi:10.1002/mds.30002) found that of 244 PD neuroimaging AI
studies, 20% passed five minimal quality criteria and only 8% used an external
test set. Yagis et al. 2021 (Sci Rep, doi:10.1038/s41598-021-01681-w) showed
slice-level splits on PPMI inflate accuracy by 48%, with random labels scoring
about 96% against a 50% chance baseline. Both failure modes are invisible
unless the analysis plan is fixed first.

## On reading anything this repository produces

The phantom is a geometric stand-in with invented parameters. It models
partial volume blur, Poisson counting noise and the posterior-to-anterior
gradient, because omitting any of them would make the fixture easier than
reality in the direction that flatters a model. It is still not a scan, and
nothing measured on it says anything about real data.

Two limits are recorded in `docs/decisions.md` and matter before any result is
read off this fixture:

- **Separating patients from controls saturates.** The z-score baseline
  reaches AUC 1.000 at moderate loss, leaving no headroom. That mirrors the
  published saturation of the same task and is why prognosis, not
  classification, is the target. See D3.
- **`severity_from_gradient` defaults to zero.** With it at zero the outcome
  is a function of quantities region means measure directly, so a network is
  learning to reconstruct a binding ratio and its score is a smoke test. Set
  it above zero before reading any hybrid result, and check the network also
  beats `features/spatial.py`: anything a posterior/anterior split can find is
  not evidence a network was needed. See D5.

Every synthetic comparison carries a control. A fixture where two scans always
beat one would reward the second scan rather than the information in it, so
the generator's severity weights can be set to zero and a test requires the
advantage to vanish when they are.

## Status

Analysis pipeline built and validated on synthetic fixtures. No data has been
requested, received or processed. The PPMI application is not yet submitted.

Order of work departed from the plan on one point: the data guard was built
first rather than last. A repository that will hold imaging paths should not
exist for a single commit without the thing that stops imaging from being
committed. That decision is D1 in the decision record, along with the reason
it was reinforced.

## Licence

Code: MIT. Trained weights: not distributed, see data rule 2.
