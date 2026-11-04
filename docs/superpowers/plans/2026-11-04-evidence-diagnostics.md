# Evidence diagnostics implementation plan

> **For agentic workers:** Use superpowers:executing-plans inline. User requested minimal subagents; implement and review these scoped diagnostics inline.

**Goal:** Audit true observation gains and measure fixed-state ActionValue learnability using existing training data.
**Architecture:** Diagnostic helpers reconstruct saved states and validate provenance; an independent CLI replays frozen runtimes, fits original ActionValue, and renders JSON/HTML/plots. Production model and training code remain unchanged.
**Tech Stack:** Python, PyTorch, NumPy, Pillow, matplotlib, unittest.
**Spec:** `docs/superpowers/specs/2026-11-04-evidence-diagnostics-design.md` (approved conversational scope).

## Constraints and review focus
- Position 64 only for model input/optimization; reject labels outside their recorded training roles.
- Check upstream fingerprints, source hashes, real replay inputs/targets; no guessed server-to-local images.
- Deduplicate set-equivalent histories; report raw and unique counts, label conflicts.
- Tiny fits use identical nested selected states across branches, fixed optimizer settings, final checkpoint; no best-result selection.
- Ties are not errors; report ranking at both existing tolerance and meaningful gain gap 0.01. Never present training scores as generalization.
- New output folder only. Reference/history/new-image HTML uses verified source paths, copied preview assets, escaped text.

## Task 1: audit and selection helpers
- [x] Add meaningful tests for state reconstruction, illegal histories, transition categories, pair ranking tolerance, state deduplication/conflicts and paired selection.
- [x] Observe RED (diagnostic module absent), implement `dinoexp/evidence_diagnostics.py`, verify GREEN (6 tests).
- [x] Implement real counterfactual audit, inspect all remaining actions, verify frozen labels and inputs within CPU/CUDA tolerance (1198 transitions, max label error 3.159e-6).

## Task 2: fitting and report CLI
- [x] Test fixed-state fit isolation/gradients and CLI overwrite refusal.
- [x] Implement `run_evidence_diagnostics.py`, fit helper and portable HTML/report/curves.
- [x] Execute 2000 updates, lr=0.001, seed=7 for 1/4/16 states on both branches; follow up with lr=0.0001, identical labels/initialization/curriculum. No production setting changed.
- [x] Review plots, interpretation, tests and lint; document results and commands in `docs/evidence-diagnostics.md`.

## Ledger
- Base: 833530c; existing isolated worktree `D:/Work/DINO-active-evidence`, branch `codex/active-evidence-pilot`, clean before work.
- Ruling: prior conversational design plus “继续做吧” authorizes execution; no redundant approval gate. Inline review honors user's subagent preference.
- Pre-flight: task 2 consumes reconstructed states, paired record indices and audit rows from task 1; interfaces implemented together in diagnostic-only helpers. No production interface changes.
- Data inspection: 96 saved states, 87 unique, 46 with target range >=0.05 in both branches, all from pair1. Fitting deliberately targets this informative subset; audit covers all 87 states and reports this selection restriction.
- Task 1 complete: all unique histories reconstructed; source tensors and labels numerically reproduced, input hashes unchanged after each run.
- Final review fixed: exact input-collision diagnostics initially used full tokens while ActionValue consumes token means. Added permutation/collision regression, observed FAIL, switched to effective means, observed PASS. Both real branches have no exact conflicts under the corrected check. No optimization code changed.
- Task 2 complete: twelve fixed-state fits (2 branches x 3 sizes x 2 learning rates), saved final weights and every update. Pairwise verification confirms same initialization, selected states, labels and sample sequence.
- Verification: full unittest suite 191/191 passed in 72.354s; Ruff clean. HTML resources validated (1198 cards,168 images), curves visually checked. Browser preview timed out twice; interaction verification unavailable, recorded honestly.
- Final review: inline self-review per user preference; no production model/training changes, no new test-set use. Unresolved research issue is multi-state optimization stability, not a claimed software fix. Ledger retained here with reproducible results.
