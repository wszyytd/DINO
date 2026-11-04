# Evidence experiment v2 implementation plan

User authorized implementation and server handoff. Execute inline; user requests minimal subagents. Existing worktree: D:/Work/DINO-active-evidence, base 300fcf8. The briefly started expansion agent was stopped before changes.

Goal: retain crop experiments, repair gain-score invariance, reuse frozen upstream checkpoints, provide reproducible grouped learning curves and explicit expansion of saved capture plans.

Architecture: versioned relative softmax scores; downstream-only runs load existing representation checkpoints and regenerate all value labels. Explicit episode manifests keep representation/value/development/reserved test groups apart. Nested representation subsets preserve A/B pairs; training steps may scale per episode. Capture expansion restores omitted original cases while retaining current revised cases.

Constraints: no remote CARLA/training; no overwrite of old experiments; no reuse of old value labels under new score version; no test-based automatic selection; default candidate crop task retained. Expanded original plan does not invent new assets/background conditions.

- [x] Score invariance, finite inputs, fingerprint isolation: regression tests then implementation.
- [x] Frozen-upstream reuse, train-only value diagnostics, training diagnostics and real replay verification.
- [x] Explicit role manifests, leakage checks, paired nested subsets and per-episode training budgets.
- [x] Read-only expansion preview and safe explicit resume expansion; preserve revisions and split isolation.
- [x] Document exact server commands; full tests/lint; inline review.

Review focus: wrong upstream/data roles, legacy label contamination, split A/B leakage, reserved test exposure, restored plans undoing corrected geometry, report semantics for runs without evaluation.

Verification: 183 unittest tests passed in 55.195 s; Ruff and diff checks passed. Score/manifest/expansion tests first reproduced missing behavior, then passed. The CLI integration test verifies no evaluation features/policies in train-only mode and byte-identical upstream checkpoints. Expansion tests execute the mocked 6-to-60-scene capture flow, retain original photos, preserve camera revisions, and verify preview writes nothing.

Real run: out/evidence_v2_value_fit_20261104 reused both server 2000-step memory checkpoints. Each value branch trained 300 steps on 96 states; source checkpoint SHA256 unchanged; no evaluation results produced. Training MSE versus zero predictor: alpha0 0.219305 vs 0.243488, alpha0.1 0.162813 vs 0.180670. These are training diagnostics, not evidence of improved policy/generalization. Later changes added cache-lineage preflight and reporting/role validation; actual-data dry-run passed after those changes.

Review: self-review inline, honoring the user's request to minimize subagents; no new reviewer dispatched. Checked scoring bounds/invariance, checkpoint/version separation, paired roles, reserved data access, original-plan merging and exact server commands. Additional missing-identity and cache-manifest regressions cover issues found during this review.

Scope rulings: expansion restores saved original cases, not new assets or cross-background references; actual CARLA capture and expanded learning curves remain server work. Full-collection parsing audits reserved test image metadata but does not fit or evaluate on reserved test features. Grouped learning curves vary representation data while keeping value/development groups fixed. No calibrated stopping claim.

Integration target: commit and push codex/active-evidence-pilot; leave main and old experiment artifacts intact. Server handoff: docs/evidence-experiment-v2.md.
