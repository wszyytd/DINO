# Vehicle parts inference implementation plan

**Goal:** Apply public AutoInspect weights to CARLA reference/query crops on Linux and export auditable native-resolution masks and parts.

**Approved scope:** The user requested implementation and push. Output part segmentation first, with exact-class ROI JSON for the existing DINO matcher. No identity probability or exclusion rule.

**Architecture:** `dinoexp/parts.py` handles scene discovery, native masks, records and ROI selection independently of the ML framework. `segment_parts.py` loads Ultralytics and local/HF weights lazily and writes CSV, JSON and portable HTML. Optional dependencies are separate.

## Tasks

- [x] Write tests, observe failure: suite discovery, incomplete capture, native masks, no detections, ambiguous/opposite-side ROI, dry-run without ML dependencies.
- [x] Implement the core and CLI; preserve rejected rows and failure manifests; reject incompatible models and output overwrite.
- [x] Add optional requirements and Linux commands for single scene/suite, offline checkpoint and DINO ROI export.
- [x] Run tests, real CARLA-data dry-run and synthetic adapter smoke; inspect report; document whether public weights were actually tested.
- [ ] Review, commit intended files, fast-forward main and push authorized origin/main; verify remote commit.

## Constraints

- Existing candidate crops use CARLA projected boxes, not a new whole-vehicle detector.
- Model confidence is not visibility, testability or identity probability.
- Saved masks use original crop coordinates; never infer quality from upsampled pixels.
- Multiple same-class detections are ambiguous; opposite sides are not interchangeable.
- Model labels, checkpoint SHA256, metadata hashes, parameters and source provenance are recorded.
- Existing untracked analysis scripts are not included.

## Verification record

Baseline: 29 unittest tests passed. New tests first failed because dinoexp.parts did not exist; implementation then passed 8 new tests. Real model_suite02 input dry-run found two scenes and 22 crops. The injected-result smoke verifies native coordinates, portable assets, ROI JSON and incomplete-run handling; no claim of real public-checkpoint inference or model accuracy is made. Local and Linux paths are documented. Review focused on native mask alignment, exact-side matching, tiny masks, output overwrite, and false confidence interpretation.
