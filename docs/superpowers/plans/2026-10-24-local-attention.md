# Learned local reference matching implementation plan

User authorized local implementation and pushing to the existing remote, then manual server training.
This first experiment compares reference/candidate crops, not the teacher's full dense search encoder.

Architecture: frozen DINOv2 intermediate patch features, separate trainable adapters,
reference local pooling and learned queries, reference-conditioned cross attention and pair logit.
Cache the frozen backbone once, train the small head separately. Preserve CLS cosine as a baseline.

- [x] Dataset and integrity: portable pair manifests from complete CARLA runs, SHA256 binding,
  explicit capture group IDs, reject cross-split group/image overlap. Existing pilot is train only.
- [x] Encoder and cache: mask padding, retain multi-layer spatial features, learn reference matching,
  validate shapes and prove gradient flow with CPU tests.
- [x] Training and reporting: reproducible CLI, train-only smoke mode, real validation optional,
  checkpoint metadata, per-view paired ranking and image report, Chinese server instructions.
- [x] Verify original and new tests, real-image backbone extraction and short CPU training;
  review final diff, commit only new feature files, push main without overwriting upstream changes.

Constraints: no automatic exclusion or calibrated-probability claim; no part annotations required;
no inferred semantic labels for learned queries; same-capture validation is prohibited;
no claim of generalization from these two vehicles. No original untracked user scripts are modified.

Verification ledger: original baseline 59 tests passed. New version 76 tests passed with Torch,
including a synthetic held-out validation/checkpoint/reload/report integration test.
Real DINOv2 extraction on 21 images, 20 real pairs, 12 CPU epochs and evaluation succeeded.
The default one-command pipeline also completed two CPU epochs on the same real images.
New/changed Python files pass Ruff. Repository-wide Ruff has a pre-existing B905 in run_probe.py:53;
that unrelated original file is not modified. Source checkout is separate because D:/Work/DINO
contains user-owned untracked scripts; remote main was confirmed at 2831c0c before push.
Independent review found two issues, both reproduced by failing tests and fixed: custom patch
layers must not change the final-layer CLS baseline; repeated named views across different
weather/position conditions must not collapse into one ranking group. Final real-image one-command
pipeline passed again after those fixes. No additional Important review findings remain.
