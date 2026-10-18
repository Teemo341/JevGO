# P14f bounded candidate coverage

This change covers AR-2, with AR-1/6/7 telemetry and EVAL checks. Parameters remain
frozen. Default `options.candidate_coverage` is `off`. `bounded` enables one extra
same-prompt batch only when the initial batch has at least one strictly valid
action but fewer than two distinct actions. A public admissible set containing
only one distinct action lowers the target to one. No public action list leaves
the target at two; it does not authorize inventing or injecting an action.

All non-B0 ResearchPolicy calls require `candidate_count <= 4`; larger values
raise explicitly before any backend call. The general RunConfig and backend APIs
retain their existing range. B0 continues to request one sample and ignores
coverage. Every decision has at most two logical generation calls and eight
samples: initial plus repair if all initial samples fail, or initial plus coverage
if there are valid but insufficiently distinct actions. Coverage never triggers
another repair. Reaching the limit with one action is a recorded negative result.

The `generation_request`, `proposal_samples`, and `candidate_sources` events
preserve `source=initial|repair|coverage` and the attempt number. Final candidates
record returned and valid sample counts, strict valid fraction, distinct action
count, single-candidate flag, attempted/completed calls, coverage rounds/status,
source references, and the actual final priors. Aggregate single-candidate ratios
must be derived from these events; requested n is never called action diversity.
Frequency counts use strictly valid samples only. Coverage merges both batches
under the identical original prompt; repair frequencies are explicitly labeled
conditional on the repair prompt. The existing deterministic seed derivation is
unchanged: config seed + twice action-history length + attempt. Same-depth seeds
and backend per-sample ranges can overlap; this change claims no independence.

Backend budget admission remains authoritative and uses the existing phase and
commit reserve. If optional coverage raises BudgetExceeded with no new ledger
entry, identical Usage, and no pre-existing overage, existing candidates are kept
and `coverage_budget_limited` records the refusal without invented charges.
Any new entry (even zero usage), measured overage or other backend error retains
its original exception semantics. Partial work and failures remain charged.
Candidate JEV scoring retains its existing admission/error rules after coverage.

The MCTS optional `candidates_for_scope` hook labels root/tree/rollout generation
at the actual call site; domains implementing only the existing `candidates`
protocol still work. Sequential decisions are root proposals; standalone calls
are unspecified unless explicitly labeled. Reusing cached candidates does not
create a new sample event. Search events additionally report actual root candidate
edges and root edges with N>0. Scope affects only telemetry, not selection or
backup. Existing MCTS private `_expand` signature remains compatible with the
historical CPU replay subclass.

CPU tests exercise valid/invalid sample merges, repair precedence, call limits,
uncharged versus charged budget errors, commit reserve, default/B0/singleton cases,
actual ResearchPolicy priors entering PUCT, two visited edges and exact reward
backup, and unchanged search results with telemetry enabled. They do not prove
real JEV identifies collisions or that online sampling supplies useful actions.
Fixed-state JEV probes use the existing development remediation harness's `judge`
condition with tester-provided candidates; these never enter the online generator.
Real-model coverage, extra costs, ranking stability and root visits remain for
the parent's finite experiments. No P15 benchmark or training is started here.
