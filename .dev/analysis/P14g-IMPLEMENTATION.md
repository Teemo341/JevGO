# P14g memory selection and stop semantics

This change covers AR-5/7 and EVAL, on integrated P14f baseline `48b367f`.
It does not train, change environment success, or introduce a stopping threshold.

## Configuration and selection

- `memory_records=M` retains its positive integer default 32.
- `memory_pool_policy=legacy` is the default and preserves the old bounded archive
  suffix capacity `max(M, candidate_count)`.
- `memory_pool_policy=history_tail` requires an explicit positive integer
  `memory_pool_records=K >= M`. This pool is independent of candidate_count.
  Supplying K under legacy is rejected rather than silently ignored.
- `memory_selection=judge` is the default. `tail` selects the last M entries from
  the **same pool**, with no evidence judge call. Other enabled judge roles keep
  their normal behavior. A tail-only configuration with other judge roles off
  does not require a judge backend.

Both pools use the current branch's full immutable evidence archive, filtered to
exclude judge opinions. They do not use retained memory plus new records as a
recall policy. K and M count records, not tokens; full actions, transient working
memory and repeated judge context still occupy the model window and incur costs.
The existing evidence scoring instruction/content format is preserved.

G0 always skips evidence scoring when every pool record fits in M. Selection IDs
and chronological order match the old algorithm, including normal M4/M32 with
candidate_count4, but fees and available budgets can change. This is not evidence
that one can subtract old useless calls to predict a new real trajectory.

With K>M, judge selection uses stable highest-score ranking with old-first ties,
then restores chronological order among retained entries. Same-pool tail is the
independent comparator. Old counterevidence may be recalled and a low-score new
record omitted. Raw evidence is never deleted or modified, sibling archives are
never merged, and judge opinions never become selected facts.

`memory_selection` events include the input state ID/phase, configured K/M,
actual pool size, ordered pool IDs with kind/origin/source/branch, pool rule,
prior memory IDs, comparator tail IDs, scores or skip reason, selected IDs,
selection status and archive size. `pool_fits_retention` explains no-choice skips;
`same_pool_tail_comparator`, `evidence_role_disabled`, or `memory_disabled` explain
other skips. Failed scoring records `status=failed`, `selection_applied=false`,
`selected_ids=null`, preserved prior memory, pool and error before rethrowing.
Backend failure/partial fees stay in the authoritative ledger.

## Stop and result metadata

Existing `judge_stop=false` and baseline method rules yield mode `off`; enabled
B1/B3 stop yields `advisory`. Advice only creates a judge-opinion record. It does
not set objective success, select an action, stop search, or end an episode.
Scoring expenses can reduce the remaining budget, and backend failure can still
end a run under existing error semantics. A committed action, its observation
and completed memory selection survive a later stop failure.

`RunResult.metadata.policy_mechanisms` is generated from actual policy state:

- `scope=all_search_and_commit_policy_calls`.
- `memory`: enabled, pool_policy/capacity, retention_limit, selection,
  evidence_judge_enabled, scoring_applied, retention_changed_vs_tail,
  selections_completed, no_choice_skips, score_calls_attempted/completed,
  scored_selections, different_from_same_pool_tail.
- `stop`: mode, controls_success=false, controls_action_or_search=false,
  cost_can_reduce_remaining_budget, backend_failure_can_terminate,
  advice_recorded, calls_attempted, advice_records.

These counters describe logical attempts and completed policy operations; they
never replace ledger actual requests/tokens or prove algorithmic effectiveness.
Counts include simulated branches and actual commits, as the scope states.
Initialization failure before a policy exists records metadata as null.

## Validation boundary

CPU fixtures cover zero new judge requests/tokens at M4/M32, exact archive/order
preservation, K independent of candidate_count, actual next generation prompts
under exchanged scores, counterevidence, new low scores, stable ties, same-pool
tail, branch isolation, retained partial failure costs, and stop budget/error
semantics. One pre-existing reward-backup test now uses K4/M1 so it still exercises
an actual post-transition evidence-choice budget interruption; its former M32
pool correctly has no such interruption after G0.

All required research/evaluation/runner/core tests and affected MCTS tests passed
(224). CPU fixture billing is not real model inference. Real G0 before/after,
G1 judge/tail and stop costs require the parent's bounded experiments and exact
checkpoint/request artifacts. This task starts neither GPU inference nor P15.
