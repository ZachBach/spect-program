# spect-program

Hybrid CNNs for brain SPECT. Program 10 on the Aurelius Dynamic research page,
under the track NUCLEAR NEUROIMAGING.

**Research use only.** Nothing here produces a diagnosis, and nothing here is
clinical decision support. FDA's clinical decision support guidance, revised
6 January 2026 and reissued 29 January 2026, treats software that analyses a
medical image to produce findings as a medical device. This program stays on
the research side of that line and makes no clinical claims.

Owner: Zachary Auerbach, doing business as Aurelius Dynamic.

## Install

```sh
git config core.hooksPath .githooks
```

Once per clone, before anything else. Without it the pre-commit guard does not
run. See "The guard" below for what it stops.

## Data rules

These are not conventions. They come from data use agreements with teeth.

**1. No participant-level data in any AI tool.** Not this repository's assistant,
not Claude Code, not Copilot, not any cloud assistant. PPMI's Data Use
Agreement v5.0 (April 2026) prohibits AI tools that do not guarantee
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

## The guard

Two layers, because one is not enough.

`.gitignore` stops the accident: imaging formats, archives, participant
tables, trained weights, credentials.

`.githooks/pre-commit` stops everything else. It reads what is actually staged,
so it covers `git add -f` and tooling that stages files without being asked.
It blocks the same extensions, blocks anything inside a data directory, checks
DICOM magic bytes at offset 128 so a volume renamed to `.txt` is still caught,
and refuses anything over 2 MB.

There is no override flag. If the hook is wrong, change the hook in a commit
someone reviews.

Verified by test before it was committed: a force-added `.nii` is blocked, a
DICOM renamed to `meeting_notes.txt` is blocked on its magic bytes, a
participant CSV is blocked, and an allow-listed synthetic fixture passes.

## Layout

```
configs/          experiment configs; data paths live here, never in code
src/io/           DICOM and NIfTI loaders (no real paths committed)
src/preprocess/   spatial normalization, occipital reference, striatal crop
src/features/     binding ratio and z-score baselines
src/models/       baseline survival model; hybrid CNN plus tabular fusion
src/eval/         C-index, time-dependent AUC, calibration, leave-site-out
src/synthetic/    phantom generator for end-to-end tests
tests/
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

## Status

Scaffolding only. No data has been requested, received or processed. The PPMI
application is not yet submitted.

Order of work, which differs from the handoff on one point: the guard was
built first rather than last. The handoff listed it as task 6. A repository
that will hold imaging paths should not exist for a single commit without the
thing that stops imaging from being committed.

## Licence

Code: MIT. Trained weights: not distributed, see data rule 2.
