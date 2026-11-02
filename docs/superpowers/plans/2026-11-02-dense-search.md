# Dense Search Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans inline, with a fresh final reviewer. The user's instruction is to implement the just-reviewed teacher design, and authorizes execution without a further design/permission round.

**Goal:** Restore dense reference-conditioned localization and mask supervision, with a usable server capture-to-training workflow.

**Architecture:** Frozen multiscale DINOv2 feeds dual adapters, reference tokens and one global query. Sixteen spatial views receive geometry/context/reference conditioning, then patch/Null probabilities are fused into a spatial distribution plus global Null.

**Tech Stack:** Existing Python3.11+, NumPy, Pillow, PyTorch and local DINOv2; CARLA0.10 only for server capture.

**Spec:** docs/superpowers/specs/2026-11-02-dense-search-design.md

## Global Constraints

- Preserve old crop matcher and all original user untracked files/data/weights.
- Teacher preset518/256/8heads, frozen layers3/6/9/12, one reference query/two reference blocks, full+15views. Explicitly label all smoke presets.
- Never manufacture visible masks from boxes; no fake camera state, no actor-ID assumption, no calibrated identity claim.
- Split by physical group and all image hashes; real unknown regions excluded, real Null/state frozen.
- No automatic CARLA execution or full training on the remote server; user runs documented commands after code push.

## Review Focus

- Letterbox and boundary geometry must not shift localization or count padding.
- Overlapping/duplicate views cannot gain probability merely through coverage count.
- Empty, absent and partially confirmed masks must not silently become wrong supervision.
- Sensor failures and segmentation ID mismatches must fail closed and clean up owned resources.
- Missing backbone/masks, changed cache or split leakage must stop before creating a training run.

### Task 1: Spatial views, manifests and labels

Files: dinoexp/dense_views.py, dense_data.py, dense_loss.py; tests/test_dense_views.py, test_dense_data.py, test_dense_loss.py.

- [ ] Write and run failing tests: hand-derived bilinear mapping, padding exclusion, binary mask/shape validation, split leakage through shared references, absent/unknown contradictions, equal mass for differently sized positive instances, zero unknown-region gradients.
- [ ] Implement `make_views(image, image_size=518, tile_size=512, rows=3, cols=5, grid_size=32)` returning prepared images, valid masks, geometry, bilinear indices/weights and resolution priors. Implement strict manifest reading and spatial targets. Implement simulation focal CE and confirmed-region conditional loss.
- [ ] Run these tests and the original full suite; commit.

### Task 2: Dense model and feature/training pipeline

Files: dinoexp/dense_model.py, dense_cache.py, dense_training.py; cache_dense_features.py, train_dense_search.py, evaluate_dense_search.py, run_dense_search.py; tests/test_dense_model.py, test_dense_pipeline.py.

- [ ] Failing behavioral tests: probabilities sum to1, invalid padding contributes0, duplicate-view coverage invariance, changed references affect output, gradients reach all trainable branches, dense supervision can fit tiny spatial/Null fixtures, strict checkpoint and cache contracts, real freezing.
- [ ] Implement model and frozen per-view cache; real and simulation stages; held-out selection/evaluation and original-image reports. Defaults follow spec; source paths resolved before any outputs. Do not download weights implicitly.
- [ ] Run targeted/full tests and a real-backbone local smoke with explicit small preset. Commit.

### Task 3: Verified mask capture and resumable collection

Files: dinoexp/dense_capture.py, capture_dense_scene.py, collect_dense_search.py; tests/test_dense_capture.py, test_dense_collection.py.

- [ ] Failing tests: G-low/B-high decode, opaque ID association using disappearance/reappearance, reject ambiguous or unchanged probes, synchronized frame matching, sensor/spawn/cleanup errors, safe resume/archive, confirmed Null after owned target removal.
- [ ] Implement standalone capture and collector using saved complete collection configs; write pixel masks and native full images, no bbox mask substitution. Export strict dense manifest with positives and Null examples.
- [ ] Full suite and offline collection dry-run against existing actual configs; commit.

### Task 4: Integration, instructions and review

Files: docs/dense-search.md, README.md, configs/dense_search_teacher.json; final tests/bug fixes as necessary.

- [ ] Write exact server commands for dry-run, mask pilot, teacher cache/training/evaluation and real adaptation; explain old data/weights incompatibility and backbone requirements.
- [ ] Complete a tiny end-to-end fixture and render/report inspection, full regression suite and fresh code review.
- [ ] Fix important findings with RED→GREEN tests, commit, integrate and push to origin/main under existing session authorization. Report exact verified scope and remaining live capture/training step.

## Execution Ledger

- Design and execution authorized by user instruction following the detailed teacher-method comparison.
- Native worktree creation could not target D:\Work\DINO because it is outside the chat's registered workspace roots; manual sibling worktree D:\Work\DINO-teacher-dense, branch codex/teacher-dense-search created instead. Original checkout and untracked files untouched.
- Builder check: duplicate source rectangles receive inverse multiplicity in fusion and summary/Null aggregation; full-rectangle duplicates do not receive extra global-context addition. Confirmed-region samples without positives are rejected, not silently counted. Dense capture rejects environment-hiding configs before mutation because the prior state cannot be restored reliably from those configs.
- Task 1 complete: tests first failed on missing dense modules, then all10 targeted and full127 tests passed. Baseline117 tests passed before changes. RGB/masks are still untouched.
