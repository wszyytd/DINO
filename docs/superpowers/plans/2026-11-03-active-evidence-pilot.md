# Active Evidence Pilot Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task-by-task. The user explicitly authorized direct implementation and experiments.

**Goal:** Implement and run an offline multiview-memory / auxiliary-prediction / direct-action-value experiment on the existing frozen DINOv2 cache.

**Architecture:** Keep real observations in immutable, deduplicated per-candidate histories. Train a set memory and reference-conditioned matcher, optionally with held-out-view prediction; freeze it and train a separate action-value network on real before/after margins. Evaluate all policies against the same replay environment.

**Tech Stack:** Python, PyTorch, NumPy, existing feature cache, unittest. No new dependencies, no CARLA process required for the first experiment.

**Spec:** D:/Data/Notes/Obsidian/计算机/科研/世界模型/流形辅助的主动证据收集模块-详细设计-20261103.md

## Global Constraints

- DINOv2 stays frozen; predicted observations never enter the memory.
- Ground-truth identity and future observations remain outside policy inputs.
- Exact duplicate / invalid observations do not change the state; attempted actions consume budget.
- Preserve the main checkout's existing files and experiment outputs.
- The local complete training captures contain 26 episodes at three locations. Use explicit pilot location separation: representation=42, value=64, evaluation=105. Identities overlap; do not claim held-out-identity generalization or calibrated stopping.
- Official val/test raw captures are incomplete locally; their cached features alone do not supply reliable replay geometry. Do not treat the stale local collection.json as server status.
- Existing full-cache feature hashes are verified before use. Use raw world camera and candidate poses, not the shared camera-orbit label, to compute candidate-relative pose.
- CPU pilot may use spatial pooling and smaller configured hidden dimensions; record these numerical choices in every artifact.

## Review Focus

1. Missing/rejected action results must not disappear through future-quality filtering.
2. Changing the strongest distractor requires recomputing both margins independently.
3. Synthetic model tests do not establish real-image scientific results.
4. Different candidate counts, empty/padded histories and duplicate observation IDs need explicit behavior.
5. Cached value labels/checkpoints must reject mismatched upstream model/config hashes.

## Task 1: Replay data and state

Files: dinoexp/evidence_data.py, dinoexp/evidence_state.py; tests/test_evidence_replay.py.
Interfaces: load_episodes(root, cache), split_by_position(episodes, mapping), initial_state(episode), update_state(state, candidate, observation), legal_actions(episode,state), true_margin(scores,target).

- [x] Write tests for exact-repeat idempotence, immutable counterfactual updates, rejected-result accounting, split overlap, relative pose, distractor switching.
- [x] Observe expected missing-module failure, implement and run tests to green.
- [x] Audit actual local records and preserve incomplete-capture diagnostics.

## Task 2: Neural representation and value heads

Files: dinoexp/evidence_model.py; tests/test_evidence_model.py.
Interfaces: EvidenceModel.encode_observations / aggregate / match / predict; ActionValue.forward; teacher_targets; prediction_loss.

- [x] Test permutation invariance, padding isolation, empty history, candidate permutation equivariance, gradients for matching and prediction, fixed teacher loss masking.
- [x] Observe failure, implement with existing SpatialAdapter / CrossBlock, then verify.

## Task 3: Staged training and provenance

Files: dinoexp/evidence_training.py, dinoexp/evidence_labels.py; tests/test_evidence_experiment.py.
Interfaces: train_memory, generate_labels, train_value, checkpoint loading with feature/model fingerprint.

- [x] Test that labels use real transitions, freeze leaves upstream weights unchanged, and model/cache mismatches fail.
- [x] Implement deterministic sampling, separate fit/value/evaluation roles and signed MSE value learning.
- [x] Run a small end-to-end synthetic training/reload exercise.

## Task 4: Replay policies and runnable experiment

Files: dinoexp/evidence_policy.py, run_evidence_pilot.py, docs/active-evidence-pilot.md.

- [x] Test fixed budgets, failure counting, oracle isolation, stopping statuses and deterministic random baselines.
- [x] Add random/fixed/coverage/learned/oracle policies, CLS-mean comparator, artifact reports and manifest provenance.
- [x] Run the real CPU pilot with alpha=0 and alpha=0.1, fixed seed/budget, no test-set selection or parameter tuning.
- [x] Run full tests and lint; review all changes; document empirical results and limits.

## Execution ledger

- Worktree: D:/Work/DINO-active-evidence, branch codex/active-evidence-pilot, base f0adab5.
- Ruling: native managed-worktree tool rejects the source repository outside this chat's workspace roots. Git worktree fallback created a sibling checkout; all original files stay intact.
- Ruling: current data supports a location-separated pipeline pilot, not the design's final identity-disjoint/cross-background experiment. Keep this limitation explicit in data audit and reports.
- Baseline before changes: 156 unittest tests passed. New module tests first failed with missing imports; implementation brought the suite to green. Auxiliary-ablation curriculum test exposed missing audit IDs and now verifies identical matching samples across auxiliary weights.
- New evidence tests: 16 passed; Ruff passed on all added Python files. Additional real-file tests exercise rejected results, missing captures, reference leakage, cross-role leakage, padding-aware pooling, and altered cache bytes.
- Real experiment: out/active_evidence_20261103_01, status complete. At budget 8, CLS-mean fixed order = 100%, learned policy alpha=0 and 0.1 = 50%. Value MSE exceeds the zero predictor in both branches. Eight evaluation episodes share one position: no claim of statistical/generalization success.
- Ruling: default fixed-budget rankings are a pipeline diagnostic. Reliable threshold-crossing efficiency, identity-disjoint splits, cross-background queries and flight integration require later data/calibration; no stop threshold was selected on this evaluation set.
- Reproduction: docs/active-evidence-pilot.md contains the command and limitations; run.json stores source hashes, parameters, feature config, roles and outcome. Original main checkout is preserved.
- Final review: fresh read-only reviewer inspected f0adab5..c6d3da4, reused modules and experiment artifacts; independently reran the evidence tests. No Critical/Important issues. One reporting issue: repeated random rollouts inflated the `episodes` metadata. Reproduced with a failing regression, then split unique `episodes` and `rollouts`; regression passed. Corrected existing summaries from saved traces, preserving all curves/checkpoints and recording the reporting correction in run.json.
- Final verification after the fix: 173 unittest tests passed (29.646 s), including 17 evidence tests; Ruff and git diff --check passed. Original experimental source hashes matched the committed source before the reporting-only fix. Training and raw rollout artifacts were not regenerated.
- Final ruling on declined judgments: scientific superiority/manifold support is an empirical outcome, not an implementation acceptance criterion; this run is negative. Identity-disjoint and cross-background generalization require new data and remain outside this pilot. Calibrated stopping requires independent calibration and remains disabled. GPU behavior is unverified and not claimed. Physical flight/routes/simultaneous observations are outside equal-cost offline replay. Confidence across independent locations is not reported because evaluation has only one location.
- Integration: keep codex/active-evidence-pilot in D:/Work/DINO-active-evidence for continued experiments; no merge or push requested. No temporary per-plan review workspace was created.
