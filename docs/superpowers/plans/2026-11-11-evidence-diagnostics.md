# Evidence diagnostics implementation plan

Goal: implement the approved minimal P0–P2 diagnosis against the server 200-epoch run, without retraining or changing existing model/training behavior.

Spec: user-reviewed 排查与解决方案.md plus the review corrections in this conversation: latest server results, legacy/physical truth separation, same-weight controls, single-frame history comparison and gate/update statistics.

Execution: inline on codex/evidence-memory-v1; no subagents. Existing untracked diagnostics and teacher documents remain untouched. Push to the already authorized remote branch.

Ruling: write tests first, but do not execute them locally; the user forbids local model execution, including forward-pass tests. Local validation is AST parsing, lint and diff inspection only. Server tests and runtime replay remain explicitly pending.

Ruling: no changes to EvidenceMatcher, pack, score_batch or training. Use external hooks for update summaries and standalone replay matching the original batching. Use fixed existing probe routes and fit seed 902. Full normal replay must match stored metrics; partial smoke is labelled partial and cannot certify replay.

Ruling: all new outputs go to a fresh directory outside source run/cache/data. Preserve cache identity and raw label/image hashes. Source images copied only for a bounded sample HTML report.

Tasks:
- [x] Write server tests: error categorization, legacy truth, slot-only swaps, observer transparency, replay equivalence, duplicate no-op, output protection.
- [x] Implement raw metadata validation, frame quality/position and provenance.
- [x] Implement batched causal replay, scores, two margins, update statistics, reference switch, reset/pause/swap controls.
- [x] Implement bounded repeat checks with empty/one-usable/mixed history, distinct subsequent observation and duplicate no-op.
- [x] Implement JSONL/CSV summaries and bounded HTML examples; labels, counts and limitations explicit.
- [x] Add CLI and server run guide, static checks and manual review.

Interface: diagnose_evidence_memory.py delegates to dinoexp.evidence_diagnostics.run_diagnostics(args); dinoexp.evidence_diagnostic_reports.py owns pure-Python scoring summaries and reports. Existing models/checkpoints unchanged.

Implementation ruling: source validation/physical metadata extracted into evidence_diagnostic_data.py, separate from model replay and report formatting.

Verification ledger: nine server tests written before/alongside implementation, not run locally. Python 3.10 AST parsing and Ruff passed. Manual review checked checkpoint/source binding, same-batch legacy replay, immutable intervention forks, first-write gate semantics, partial-run claims, and empty-report handling. No local Torch import, forward pass, pytest/unittest execution or training performed during implementation. Runtime test and full checkpoint replay acceptance are pending user execution on the server.

Review focus: reference/candidate mapping, null vs wrong-car, physical visibility vs cached validity, truth independent of manipulated state, different devices/replay mismatch, immutable intervention forks, no unseen-vehicle swaps, no future feature input, no unreported sample limits.
