# JevGO 公共契约 v1

权威实现为 `jevgo/core.py`。所有模型参数保持冻结；`RunConfig.learning_mode` 仅接受 `none`，其余训练模式显式校验失败。正式代码不读取开发目录。

## 数据和可见性

Pydantic 公共模型均冻结且拒绝多余字段；JSON payload 在构造、复制时递归复制并冻结，数组成为 tuple、对象成为只读 mapping。`model_dump(mode="json")` / `model_dump_json()` 输出独立标准 JSON；`model_validate_json()` 可恢复。非有限数、非字符串对象键、任意 Python 对象均拒绝。`model_copy(update=...)` 也重新校验。

- `EpisodeSpec(environment, task, episode_id=<uuid>, seed=0, split="development", success_criteria=..., options={})`：任务/seed/公开配置。
- `RunConfig`：`method=B0|B1|B2|B3`；`candidate_count=4, simulations=8, max_depth=4, rollout_depth=2, max_steps=100, c_puct=1.5, gamma=1`；`budget: BudgetLimits`、`commit_reserve: Usage`；`judge_candidates/judge_evidence/judge_stop/use_memory` 与 JSON `options`。这些为验收参数，不是效果结论。
- `Action(name, arguments={})`：一个环境原子动作；`key()` 产生稳定去重键。
- `Observation(text="", data={}, admissible_actions=(), image_paths=())`：仅 agent 可见信息，不能含计分标签、隐藏地图、恢复句柄。
- `StepResult(observation, action_success=None, reward=0, terminated=False, truncated=False, success=False, objective_score=None, info={})`：环境客观反馈；action_success 区分动作执行成功/无效/未提供，不等同任务 success；`done` 为三个终止/成功标记的或。
- `EvidenceRecord(episode_id, branch_id, kind, origin, source, content, raw={}, state_id=None, action=None, references=(), conditions=())`：生成 `evidence_id`；`kind=observation|hypothesis|inference|judge_opinion|prior`。只有环境来源可标 observation，JEV 意见只能是模型来源。
- `ResearchState(episode, observation, ...)`：`state_id/parent_id/version/branch_id`、原始 `evidence`、引用实测证据的 `fact_ids`、`hypotheses/inferences/working_memory/constraints/unresolved`、`actions/experiment_ids`、`last_reward/objective_score/terminated/truncated/success`。`evolve(**changes)` 自动形成下一版本，禁止删改已有证据前缀；可压缩工作记忆。分支继承父证据，新增证据注明新分支；互斥分支证据不能自动合并成事实。
- `Candidate(action, candidate_id=<uuid>, hypothesis="", rationale="", prior=1, sample_count=1, judge_score=None)`：`prior` 有限且在 [0,1]；域返回候选时负责整体归一化。低先验不等于非法。
- `ExperimentPlan(candidate_id, actions, ...)`：非空动作序列、前置条件、预期观测、失败判据及 `estimated_usage`。
- `EnvironmentSnapshot(owner_id, payload, snapshot_id=<uuid>)` 是 opaque Python handle；由 adapter 拥有并确保恢复隔离。不得序列化到事件、研究状态或模型 prompt。快照不包含账本。

## 环境协议：唯一签名

```python
class EnvironmentAdapter(Protocol):
    capabilities: EnvironmentCapabilities
    def reset(self, spec: EpisodeSpec, *, ledger: BudgetLedger,
              phase: UsagePhase = UsagePhase.COMMIT) -> StepResult: ...
    def step(self, action: Action, *, ledger: BudgetLedger,
             phase: UsagePhase = UsagePhase.COMMIT) -> StepResult: ...
    def snapshot(self, *, ledger: BudgetLedger,
                 phase: UsagePhase = UsagePhase.SEARCH) -> EnvironmentSnapshot: ...
    def restore(self, snapshot: EnvironmentSnapshot, *, ledger: BudgetLedger,
                phase: UsagePhase = UsagePhase.SEARCH) -> StepResult: ...
    def observe(self) -> Observation: ...
    def close(self) -> None: ...
```

Adapter 独占环境计费；上层不得再次收同一费用。`observe()` 只返回缓存观测；任何真实刷新成本归 `reset/step/restore`。`EnvironmentCapabilities(snapshot_restore=False, deterministic_replay=False, partial_observability=True, objective_success=True, details={})` 的 `can_branch` 由前两项决定；`require_branching()` 对不可恢复环境抛 `UnsupportedCapability`。能力必须由真实验证支持，不凭环境名称声明。恢复需验证句柄归属，恢复 seed/RNG/历史/终止标志；恢复失败不得声称已恢复。

重放一次动作增加 `env_steps=1, replay_steps=1`，并使用 phase=`replay`。恢复内部 reset 同样增加 `env_resets`；每次恢复调用另加 `env_restores`。这些是独立计数，不把 replay_steps 再加进 env_steps。快照按 `env_snapshots` 记录。真实观察刷新按 `env_observations` 记录，即使环境允许免费观察；缓存 observe 不计费。所有搜索、失败、重放和未提交动作都保留真实成本。

## 模型协议：唯一签名

```python
class GenerationBackend(Protocol):
    identity: ModelIdentity
    def generate(self, request: GenerationRequest, *, ledger: BudgetLedger,
                 phase: UsagePhase = UsagePhase.SEARCH) -> GenerationResponse: ...
    def close(self) -> None: ...

class JudgeBackend(Protocol):
    identity: ModelIdentity
    def score(self, request: JudgeRequest, *, ledger: BudgetLedger,
              phase: UsagePhase = UsagePhase.SEARCH) -> JudgeResponse: ...
    def close(self) -> None: ...
```

`GenerationRequest(prompt, system="", n=1, max_new_tokens=256, temperature=0.7, seed=0, image_paths=(), options={})` → `GenerationResponse(texts, identity, usage, metadata={})`。texts 顺序固定且长度为 n，由 backend 校验。

`JudgeRequest(role, context, items, image_paths=(), options={})` → `JudgeResponse(scores, identity, usage, rationale=(), metadata={})`。role 为 `candidate/evidence/state_value/stop`；每 item 一个有限 score，顺序一致、越高越偏好。后端在 metadata 说明原始评分尺度；scores 不是跨后端校准概率，更不是客观成功。候选偏好、证据有用性、未来价值与停止辅助必须区分角色。

`ModelIdentity(backend, model, revision=None, adapter=None, mode="frozen", metadata={})` 保留实际模型、revision、adapter/head 与调用模式。Backend 独占计费；响应 usage 是本次调用完整费用的报告，上层只读，不再入账。`generation_calls/judge_calls` 计公开调用，`model_requests` 计其内部实际请求；单次调用可能产生多个内部请求，不能因包装批处理而漏计。失败调用仍记录已发生请求和已知 token，未知费用须明确 metadata，不能伪造为完整零成本。

## 账本与预留

`Usage` 和 `BudgetLimits` 的字段相同：`env_steps, env_resets, env_snapshots, env_restores, env_observations, replay_steps, generation_calls, judge_calls, model_requests, input_tokens, output_tokens, rollouts, wall_seconds`。Usage 默认零，整数计数严格非负；Limits 默认 None（无限），零表示禁止该资源。wall_seconds 是实际操作耗时，顺序执行时累计，不重复把外层总时长加入内部操作耗时。

```python
ledger = BudgetLedger(limits=None, *, commit_reserve=None, event_store=None)
ledger.reserve_commit(usage)  # 替换保护额度，不消费、不退款
ledger.ensure_available(usage, *, phase=COMMIT, budget_phase=None)
ledger.can_afford(usage, *, phase=COMMIT, budget_phase=None) -> bool
ledger.charge(usage, *, source, phase=COMMIT, budget_phase=None,
              failed=False, metadata=None) -> LedgerEntry
ledger.record(usage, *, source, phase=COMMIT, budget_phase=None,
              failed=False, metadata=None) -> LedgerEntry
ledger.remaining(*, phase=COMMIT, budget_phase=None) -> dict
ledger.used -> Usage
ledger.entries -> tuple[LedgerEntry, ...]
ledger.commit_reserve -> Usage
ledger.over_budget -> tuple[str, ...]
ledger.summary() -> dict  # JSON-safe
```

- 同一 run/全部分支共享一个 ledger；环境恢复只恢复世界，不回退账本。
- `ensure_available` 在任意资源已经超额或当前阶段剩余 wall_seconds 为零时拒绝所有新操作，拒绝发生在收费/触网/环境变更之前。已发生费用仍可由 `record` 追加。
- `charge` 用于已知固定成本，原子检查后在执行前计费；拒绝时不产生费用、也不执行真实操作。已经计费后失败不退款，可以追加零费用 failure entry。
- `record` 用于已发生的实际费用，即使超出额度仍完整记录；调用方先用 `ensure_available` 检查 token 上界等可知预算，再执行并记录实际 token/耗时。无法提前知道的耗时不能保证零超额，超额在 `over_budget` 显示，后续操作应停止。不要重复记录已由 charge 计入的调用次数。
- `phase=commit/search/replay` 是实际操作标签。search/replay 默认可用额度为硬上限减累计成本再减 commit reserve；commit 可用完整剩余额度。只有 replay 可显式设置 `budget_phase="commit"|"search"`，继承实际父操作的预算阶段。搜索恢复必须传 search，不能借 replay 绕过提交预留；实际提交恢复可以传 commit。
- Reserve 应覆盖实际提交及必要恢复成本，并在每次重新规划前调整。仅预留一个 env step 不足以保护有额外 reset/replay 上限的恢复。
- `LedgerEntry` 保存 `sequence/source/phase/budget_phase/usage/failed/metadata`。费用方法可由多个线程安全调用；算法本身为同步单智能体。

## 搜索域与 MCTS

```python
SearchState(research: ResearchState, snapshot: EnvironmentSnapshot | None = None)
SearchTransition(state: SearchState, reward: float)
ValueEstimate(value: float, source: Literal["environment", "heuristic"], rationale="")

class SearchDomain(Protocol):
    ledger: BudgetLedger
    capabilities: EnvironmentCapabilities
    def candidates(self, state: SearchState, *,
                   phase: UsagePhase = UsagePhase.SEARCH) -> tuple[Candidate, ...]: ...
    def transition(self, state: SearchState, candidate: Candidate, *,
                   phase: UsagePhase = UsagePhase.SEARCH) -> SearchTransition: ...
    def evaluate(self, state: SearchState, *,
                 phase: UsagePhase = UsagePhase.SEARCH) -> ValueEstimate: ...
```

`SearchState.terminal` 来自 research.done。模型只能看 `.research`，不能看 opaque snapshot。

`transition` **独占恢复父快照→一个真实原子动作→后继快照**，只有已知环境准确位于该父状态时可省恢复；MCTS 不再自行恢复。search 在不可恢复环境必须在任何修改前拒绝。commit 支持不可恢复环境的顺序执行。后继不得覆盖父研究状态和证据；域、adapter 和模型后端共用域的 ledger。

`SearchTransition.reward` 是这条边即时奖励。`evaluate` 只给额外未来回报估计，不包含进入此状态的奖励、不执行环境动作；终局返回 0，避免重复算终局奖励。环境来源与启发式来源明确分开，JEV 未校准评分不得伪装为环境长期价值。

MCTS 通过连续调用 `transition` 执行真实 rollout，按每条边即时奖励和 gamma 回传，单智能体不翻转符号；每次真正启动 rollout，由 MCTS 单独记录 `Usage(rollouts=1)`，不重收环境/model费用。这样 MCTS 不需要了解 benchmark 或 LLM。候选 prior 的来源及归一化由研究域负责。搜索结束后，由 runner 使用实际根 SearchState 和选中候选执行 commit；搜索叶状态不能直接宣称为真实轨迹。

## 事件与结束

`EventRecord(event_type, episode_id=None, branch_id=None, payload={})` 自动带 uuid、UTC timestamp、schema_version=1。`JSONLEventStore(path).append(event)` 只追加一行 UTF-8 JSON，`read()` 返回事件 tuple；坏行明确报告行号，不静默跳过。一个 run 使用单个 store writer，ledger 可自动写 `usage` 事件。原始环境证据、候选/评分、状态转换和失败由对应上层写事件；不把 opaque 快照写入 JSONL。

错误类型：`JevGOError`；`BudgetExceeded(resources, phase)`、`EnvironmentFailure`、`ModelFailure`、`InvalidCandidate`、`UnsupportedCapability`，均有 `category: ErrorCategory`。

`RunResult(episode, config, final_state, termination_reason, usage, answer="", error=None, events_path=None, metadata={})` 可序列化。结束原因：`success/budget_exhausted/environment_terminated/environment_truncated/environment_failure/model_failure/invalid_candidates/no_candidates/max_steps/cancelled`。`success` 必须与 final_state 的客观 success 一致；预算耗尽与 JEV 停止意见不构成成功。预算耗尽允许保留部分答案、证据和未决问题。
