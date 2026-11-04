# Fixed-state update comparison

> Execute inline with writing-plans/executing-plans and test-driven-development. User authorized this specific next experiment with “开始做”; keep subagent use minimal.

**Goal:** Test whether state-wise stochastic updates contribute to the observed 16-state fitting oscillation.
**Spec:** The approved conversational design: fixed 16 states, original network and sample weights, compare single-state and joint updates before changing the production trainer.
**Architecture:** Separate diagnostic module and CLI. Pack variable numbers of actions with a loss mask, preserving a mean loss per state. Existing model and trainer stay unchanged.
**Tech Stack:** PyTorch, unittest, matplotlib, existing cached label records.

## Controls decided before running
- Reuse the 16 indices from `evidence_diagnostics_20261104_lr0001/run.json` (position64, pair1 only). No additional data selection or held-out evaluation.
- Both representation branches; lr=0.0001, AdamW wd=0.01, gradient clip=5, 2000 updates, seeds 7/17/27.
- Single mode reproduces original alternating hard-half/all-state sampler. Joint mode uses the exact expected gradient of that same phase: hard states have equal weight on odd steps, all states equal weight on even steps. Two-step average weights remain identical.
- Joint forwards all 16 states, masks zero-weight states in odd-step loss. Record total forwarded states, active loss states and actions. 125 joint updates = 2000 forwarded states; compare this to 2000 single updates, and also show equal-update comparison (joint costs 16x state forwards). These are proxy compute counts, not FLOPs or equal optimizer histories.
- Full-set evaluation does not train or consume random draws. Save per-state losses each update, final checkpoint and last-500-step stability. Gradient cosine snapshots are descriptive; actual cross-state loss increases are also measured. Opposed gradients alone do not prove an architectural defect.

## Tasks / ledger
- [x] Add RED tests (module absent), then GREEN for masked packed-loss and gradient equivalence, phase weights, conflicting/zero gradients and run accounting.
- [x] Implement diagnostic helpers and independent CLI; seed7 single mode reproduces previous final weights exactly on both branches (max error 0).
- [x] Run all 12 fits, save loss/gradient/budget comparisons and standalone curves.
- [x] Review interpretation, run appropriate regression checks and document results; push after final verification.

Base: 4a238ec, existing clean `codex/active-evidence-pilot` worktree. No production changes; prior authorization covers implementation and the experiment branch push. Scope is training dynamics, not generalization. Plan self-review found no missing inter-task interface: packed inputs and masks are consumed by the new runner only.

Review ruling: add an equal-forward-budget comparison at the largest EVEN joint step <=2000/16 (124 versus 1984 single steps), using already-recorded per-step values. The original 125/2000 comparison ends in different hard/all phases. Preserve both rather than select by result; this adjustment changes analysis only.

Validation so far: full suite 195/195 passed in 100.034s while experiment was running; Ruff clean. All real runs use unchanged production ActionValue.

Completed output: `out/evidence_update_comparison_20261104_01`. Paired initialization and average weights match in every run; seed7 matches previous final weights exactly. All 6 joint fits have lower final MSE and tail standard deviation at 2000 updates. Equal-state-forward comparisons do NOT show uniform superiority; this qualification is in the result report, not omitted as a failed comparison.

Final review: inline self-review per user preference. Packed loss and gradients tested against individual-state sums; source label/memory hashes unchanged; no held-out data read. Standalone plots inspected. No unresolved software findings; scientific limits remain training-only, small correlated state set and different optimizer-step counts under matched forward budgets.
