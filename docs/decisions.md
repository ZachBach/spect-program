# Decision record

One entry per decision that would otherwise be re-argued later. Newest first.

---

## D4. The prognostic task has headroom, unlike the classification task

**Date:** 2026-09-22
**Status:** Measured. Resolves the blocker recorded in D3.

D3 recorded that the cross-sectional phantom saturates: the z-score baseline
reaches AUC 1.000 at 0.40 putamen loss, leaving nothing for a hybrid model to
demonstrate. The longitudinal phantom was built to see whether the prognostic
question behaves differently. It does.

Latent severity drives an exponential hazard and is a weighted sum of
baseline loss and progression rate. Rate is the part a single scan cannot
see, because a rate is a difference and one measurement is not a difference.
Six hundred subjects, 66% reaching a milestone:

| rate weight | oracle ceiling | baseline only | two visits | gain |
|---|---|---|---|---|
| 0.0 | 0.644 | 0.644 | 0.644 | +0.000 |
| 0.3 | 0.659 | 0.638 | 0.659 | +0.021 |
| 0.7 | 0.708 | 0.618 | 0.708 | +0.090 |
| 1.2 | 0.769 | 0.590 | 0.769 | +0.179 |

Three things follow.

**The ceiling is well below 1.0 at every setting.** Event times are drawn
from an exponential, so even exact knowledge of the hazard cannot order them
perfectly. There is no saturation to hide behind, and the room between the
baseline and the ceiling is where a model has to earn its complexity. At the
default rate weight of 0.7 that room is 0.618 to 0.708, about nine points.

**The control holds.** At a rate weight of zero the two-visit model matches
the baseline exactly. The fixture rewards the information in a second scan,
not the existence of one. Without that check a longitudinal model would beat
a cross-sectional one on this data no matter what it had learned, and any
comparison run on it would have been worthless.

**The ceiling must be reported next to any result.** A model scoring 0.70
here is at the ceiling, not mediocre. Reporting concordance without the
achievable maximum invites exactly the wrong reading, which is why
`oracle_ceiling` ships alongside the generator rather than being left for
someone to reconstruct.

**What this does not establish.** The advantage is measured against a model
given the true severity weights and an exactly recovered rate. Real
measurements are noisy, and recovering a rate by inverting the decay
amplifies that noise as loss approaches total. A real longitudinal model will
land somewhere between 0.618 and 0.708, not at the top of it. The phantom
says the information is present and recoverable in principle. It says nothing
about whether a network will find it in a real scan.

---

## D3. The phantom saturates, so it cannot be used to compare methods

**Date:** 2026-09-22
**Status:** Recorded as a limitation. No change needed yet.

The z-score baseline was measured against phantom cohorts of 30 controls and
30 patients, scoring the worst putamen z-score:

| putamen loss | control z | patient z | AUC |
|---|---|---|---|
| 0.15 | 0.00 +/- 0.98 | -1.99 +/- 1.64 | 0.852 |
| 0.25 | 0.00 +/- 0.98 | -2.80 +/- 1.66 | 0.926 |
| 0.40 | 0.00 +/- 0.98 | -4.11 +/- 1.61 | 0.992 |
| 0.55 | 0.00 +/- 0.98 | -5.41 +/- 1.50 | 1.000 |

At moderate loss the baseline separates the groups perfectly. There is no
headroom left for a hybrid model to demonstrate anything, so **a comparison
run on phantoms at these severities cannot produce a meaningful result** in
either direction. Only the narrow band around 0.15 loss has any room, and even
there the phantom is a geometric toy rather than a scan.

This is the same saturation the handoff already recorded in the real
literature: an SVM on binding ratios passed 96% for Parkinson's versus
controls in 2014, and a plain ResNet50 reached 100% in a 2026 reproduction,
which is why that classification task was excluded as a target. The phantom
reproduces the saturation, which is mild evidence that the phantom is
behaving, and strong evidence that separating patients from controls is the
wrong question regardless of the data.

**Consequence for the analysis plan.** The phantom's job is to prove the
pipeline runs, that splits do not leak, and that metrics behave. It is not a
testbed for whether the hybrid beats the baseline. That question needs the
real prognostic outcome, time to milestone, which the phantom has no way to
generate because nothing in it varies over time.

**Implied next piece of work.** If the phantom is to support the survival
arm at all, it needs a longitudinal mode: a subject whose loss progresses
across simulated visits at a rate that the model is then asked to recover.
Without that, the survival harness has nothing to be tested against.

---

## D2. Disable the GitKraken Claude Code plugin in this repository

**Date:** 2026-09-22
**Status:** Applied, with a verification step still outstanding.

### What was found

The GitKraken Claude Code plugin registers hooks on **every lifecycle event**,
including prompt submission and tool use, before and after. Hooks receive the
event payload on stdin, so in principle prompts and tool results pass through
the vendor's local binary. The build in use was an alpha, with automatic
updates enabled.

GitKraken's public position, per their documentation and a GitLens issue
thread, is that only event metadata is forwarded and that prompt content,
agent output and source code are not captured. A local inspection found
nothing contradicting that, though absence of evidence in a diagnostic log is
weak: a payload can be transmitted without being logged.

The forensic detail of that inspection is deliberately not recorded here.
Local paths, cache sizes and installed versions describe a particular machine
rather than this project, and a public repository is the wrong place to
publish a software fingerprint of the person running it.

### Why it is disabled here anyway

Not because GitKraken is believed to be doing anything improper. The argument
does not depend on that.

PPMI's Data Use Agreement v5.0 requires **guaranteeing** containment of
participant-level data. An auto-updating alpha build that subscribes to every
tool-use event cannot clear that bar, because the guarantee has to hold for
whatever version installs itself next, not only the one audited today. A
vendor statement about current behaviour is not a containment guarantee, and
the obligation under the agreement sits with the researcher, not the vendor.

The cost of being wrong is asymmetric. Disabling it here loses a convenience
feature in one repository. Leaving it enabled risks a disclosure that cannot
be undone and that would be discovered, if at all, after the fact.

### What was done

`.claude/settings.json` in this repository sets the plugin to `false`. Project
settings take precedence over user settings, so the plugin stays available for
every other repository, including labOptimal and the website, where no data
use agreement applies.

### Outstanding, and it matters

1. **The override has not been empirically verified.** It follows from the
   documented precedence order, but it was not observed working, because that
   requires a session restart. Confirm the plugin is inactive here before any
   data arrives. `/plugin` should show it disabled.
2. **GitKraken Desktop may reinstall the hooks independently.** A GitKraken
   Desktop release is reported to have installed Claude Code hooks without
   prompting, with the opt-out added in a later version. Disabling the Claude
   Code plugin may not stop a separate installation path. Check GitKraken
   Desktop's own settings for the agent-session opt-out.
3. **Ask GitKraken directly**, in writing, what a hook payload contains and
   whether any of it leaves the machine. A written answer is worth more than
   this inference, and it is the kind of thing an IRB or a data controller
   will ask for.

### Related

An unexplained `git add -A` staged 51 files in the labOptimal repository
during an active session on 2026-09-21. The cause was never established;
candidates include this integration, the VS Code extension, or a concurrent
session. It is the reason the data guard in this repository was built before
anything else rather than last. See D1.

---

## D1. Build the data guard before anything it guards

**Date:** 2026-09-21
**Status:** Applied.

The handoff listed the pre-commit hook as task 6 of 6. It was built as task 1.

A repository that will hold DICOM paths, NIfTI volumes and PPMI tables should
not exist for even one commit without the thing that stops those from being
committed. Git history is permanent; a file committed and then deleted remains
in the object store, in every clone and in every fork.

This was reinforced by the staging incident in D1's "Related" note above:
something in this environment stages files without being asked, so the hook is
not a backstop for human error alone.

The guard was tested before it was trusted. A force-added `.nii` is blocked, a
DICOM renamed to `meeting_notes.txt` is blocked on its magic bytes at offset
128, a participant CSV is blocked, and an allow-listed synthetic fixture
passes.
