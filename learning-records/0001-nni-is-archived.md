# 0001 — NNI is archived (not actively maintained)

**Date:** 2026-08-10
**Status:** stable fact (verified via GitHub API)

## Observation
Microsoft NNI on GitHub returns `archived: true`. Stars ~14.4k. The repo is
frozen — no new features, no bug fixes, no security patches. The ReadTheDocs
docs are likewise frozen at the last published version.

## Why it matters
- Any lesson or plan must treat NNI as a *reference/legacy* toolkit, not a live
  one. Saying "just `pip install nni` and go" hides the maintenance risk.
- For Tyler's local-R530 hobby, archived status means: it runs today, but if it
  breaks on a newer Python/TF/Keras (e.g. Python 3.12, Keras 3), there will be
  no upstream fix. Pinning an environment is the practical mitigation.
- Comparison to active alternatives (Optuna, Ray Tune, AutoKeras, Keras Tuner)
  is a first-class part of "learning NNI" — knowing *when not to use it* is as
  valuable as knowing how.

## Implications for future sessions
- When building lessons, lead with the four pillars + the archived caveat.
- If a lesson needs a runnable example, prefer a frozen, pinned environment so
  it stays reproducible despite the archival.
- Don't imply NNI is the default modern choice.

## Open question
Does Tyler want to actually *run* NNI examples, or purely study it? Affects how
much we invest in a pinned Colab/local env. (Asked indirectly; pending.)
