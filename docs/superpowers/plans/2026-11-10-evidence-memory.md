# Evidence memory implementation plan

Goal: implement and run stage-one oracle-associated vehicle evidence fusion on the local teacher replica, without modifying its trained weights.

Spec: the approved conversation design: frozen teacher, full aerial images, GT masks/identities for training evidence routing only, per-vehicle 8x256 recurrent slots, reference matching; no automatic association, global search memory, or action training yet.

Execution: inline, no subagents per user preference. Working on codex/evidence-memory-v1 in the existing checkout so local caches and user's untracked diagnostics remain available. No merging, pushing, or overwriting existing runs.

Tasks:
- [x] Test and expose reference-independent teacher view features, preserving original forward behavior.
- [x] Test candidate permutation, variable count, missing observation preservation, temporal gradients, baseline behavior.
- [x] Implement memory/current/mean models sharing the same writer and matching architecture.
- [x] Build hash-bound full-view evidence cache using existing DINO caches where available, otherwise official local ViT-B/14; GT-mask weighted spatial pooling, no camera pose input.
- [x] Build deterministic image-disjoint diagnostic splits, exclude teacher fitting images from probe; same-layout scope explicit. Train all modes on equal routes and paired references; candidate order shuffled. No future features.
- [x] Run local experiment and report per-step and current-visible matching, history ablation, repeat drift, seed variability, scope limitations.
- [x] Run relevant existing tests, review code, write reproducible commands and actual results.

Rulings: use only the frozen full-image branch for the first CPU experiment; label as full-view ablation, not the teacher's full 16-view pipeline. Shared memory/actions deferred as agreed. A single layout permits diagnostics only, not independent-layout validation. No hyperparameter tuning on probe. Keep training-ID routing outside model inputs.

Outcome: implementation and nine training runs completed; saved checkpoints reproduce reported metrics. Learned memory failed this diagnostic (both-visible probe mean 4.4%, current 22.1%, mean 18.0%). See docs/evidence-memory-stage1-results.md. Full suite: 179 passed, 17 subtests passed; Ruff passed. No positive efficacy claim or action-training continuation.
