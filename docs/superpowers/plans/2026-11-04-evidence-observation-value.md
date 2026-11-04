# Evidence Observation Value Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans in the current worktree. User prefers no unnecessary subagents.

**Goal:** Compare stable full-state updates and privileged real-observation value inputs with frozen upstream models.

**Architecture:** An isolated diagnostic CLI rebuilds and verifies records, fits four controlled conditions, and evaluates the final checkpoints on the existing development role. Reuse masked state losses and phase weights; keep production actions free of future observations.

**Tech Stack:** Existing Python, PyTorch, NumPy, matplotlib, unittest; no new dependencies.

**Spec:** `docs/superpowers/specs/2026-11-04-evidence-observation-value-design.md`

## Global Constraints

- Frozen upstream, unchanged true-margin labels, all unique training states.
- Exact phase weights with global normalization across gradient chunks.
- Paired initialization for controls; fixed final checkpoint; separate development role.
- No production future-information path; diagnostic checkpoints rejected by production loader.
- Existing outputs are never overwritten; provenance saved and checked.

## Review Focus

- Variable action counts and padding cannot change loss/gradients.
- Real future features must align to the saved action order; failure images must not be read.
- Targets and actual post-action scores must not enter network inputs.
- Duplicate histories and development role must not inflate or contaminate training.
- Equal-update and equal-state-work comparisons must remain distinguishable.

### Task 1: Observation interface and isolated data construction

Files: `dinoexp/evidence_model.py`, `dinoexp/evidence_observation_diagnostics.py`, `dinoexp/evidence_training.py`, `tests/test_evidence_observation_diagnostics.py`.

- [x] Write and run failing tests for paired initial output, invalid-feature masking, action alignment and frozen inputs; reject diagnostic checkpoint loading.
- [x] Extract ActionValue action representation without changing original outputs/checkpoint names. Add ObservationValue residual branch and verified observation record builder.
- [x] Run targeted tests, including existing model/update tests (15 passed); 10 expanded diagnostic tests passed.

### Task 2: Full-state controlled fitting

Files: diagnostic module and tests above; reuse `evidence_update_diagnostics.py` helpers.

- [x] Test chunked weighted gradients against a full batch with variable action counts; verify zero/real capacity control and work counters.
- [x] Implement four conditions, fixed-step learning curves, final-only development evaluation, work-matched checkpoints and per-state/group metrics.
- [x] Test immutable records, reproducibility and no-overwrite behavior.

### Task 3: CLI, real experiment and report

Files: `run_evidence_observation_diagnostics.py`, `dinoexp/evidence_observation_report.py`, tests, `docs/evidence-observation-diagnostics.md`.

- [x] Test CLI rejects reused output before data loading and supports explicit development opt-in.
- [x] Verify source run/cache/model fingerprints, disjoint roles, paired records; save source hashes and failed status on exception.
- [x] Run the prescribed first-seed experiment; inspect metrics and plots. Run full tests and Ruff/diff checks, self-review, document limitations.
- [ ] Commit/push the authorized experiment branch.

## Progress

- Baseline: 195 tests passed. Existing isolated worktree `D:/Work/DINO-active-evidence`, branch `codex/active-evidence-pilot`.
- Authorization: user said “继续做吧” after the concrete two-part proposal; execute inline without another approval round.
- Ruling: observation construction lives in `evidence_observation_data.py` to keep fitting and provenance code separate; no behavioral scope change.
- Ruling: use this plan's Progress section as durable ledger and existing `out/` for run artifacts; keep the user's existing worktree and self-review preference.
- Source inspection: actual upstream dim is 64, so observations are 5×64; all interfaces derive dimensions from the checkpoint.
- Real run: `out/evidence_observation_diagnostics_20261104_01`, seed 7, 2000 updates, two branches × four conditions, development opt-in. No parameter selection based on intermediate results.
- Tasks 1–3 implementation and experiment complete: 205 tests passed; Ruff clean; eight real fits complete; source hashes/upstream unchanged; curves visually checked.
- Final review: self-review following user's preference; checked provenance, target isolation, global phase normalization, candidate/action padding, fixed development evaluation, and matched-work accounting. No outstanding correctness findings.
- Result: observation real vs zero development regret 0.231801→0.155058 (alpha=0), 0.053639→0.097036 (alpha=.1). Training fit improved in both but development benefit is inconsistent. Do not proceed to prediction training on a false success claim.
