# P14f 候选覆盖与固定状态JEV对照

## 需求与定位

主环节AR-2，关联AR-1/6/7、EVAL。候选数4目前表示样本数，去重后只有一个动作符合旧契约；有用动作覆盖是待验证的策略扩展。依赖P14d严格入口和P14e稳定公开接口；离线工具可提前准备。

## 实施大纲

1. 增加可观测性：样本数、严格合法率、不同合法动作数、单候选比例、根实际访问边数、候选来源和追加轮数，区分root/tree/rollout。不得将n=4写成4个动作。
2. 比较原一次采样与有界补采。第一开发候选可设目标至少2种动作（公开合法集合仅1种时不要求2种）、最多额外一轮4样本；这是待验参数而非最优设置。固定调用顺序：初始4样本全坏时只走P14d的一次repair并结束该轮提议，不再补采；初始已有合法候选但种类不足时只追加一次原提示采样，不为补采再加repair。每决策最多两次generation调用、8个样本；B0保持原单样本/一次repair规则。追加请求使用预先记录的确定性seed，全部计费且预留真实提交预算。
   频次只统计实际严格合法样本：补采条件合并两批相同提示的合法计数；全坏后repair条件仅有repair来源的合法计数，标明条件化提示，不称为原始提示分布。补采准入预算不足时保留已有合法候选并记录coverage_budget_limited；补采正常返回但全坏时同样保留已有候选。真实后端调用错误仍按明确的原错误语义处理、保留费用，不以“补采”名义静默吞异常。达到上限仍不足时记录退化，不循环至满意。
3. 不向在线候选注入已知成功路线或强塞left/right，不用JEV低分永久删边。B0仍为顺序单样本；若B1/B2/B3增加采样，标记为方法/成本变化，比较时保持覆盖策略及预算一致。
4. 固定公开状态JEV探针单独运行：初始、两次forward后、重复forward后；显式提供forward/left/right中性候选、相同假设风格，做固定排列及次序交换。分别配对旧/E1/E1+E2表述，保留raw评分、偏好和选项对应。探针候选由测试者给定，不算生成器输出，不注入在线策略。
5. 上层同深度seed相关性仅记录；改变seed推导算法单列条件，不和补采一起归因。若采用条件化“生成不同动作”提示，必须单列策略并重新说明先验语义，不能继续把频次宣称原采样分布。

## 验收标准

- 确定性CPU替身验证重复样本去重、补采上限、频次回传、预算不足及失败费用，候选有真实来源，接口仍支持单候选并如实标记。
- 固定两条不同动作路径的CPU测试证明JEV偏好可改变先验、PUCT确有多边可选且回传正确；这不证明真实JEV能分辨碰墙。
- 真实模型报告覆盖率、根实际探索数、增量费用和JEV排序稳定性。全高分、排序错误或仍单候选都是可接受的负结果；无阈值校准时不把分数当成功概率。
- 不要求实际轨迹Q必须非零：零环境奖励和零bootstrap可能合法。若根仍单候选，不能称该真实试验验证了有效树搜索/候选重排贡献。

## 修改范围与验证入口

- 任务branch：`codex/ar-p14f-candidate-coverage`。
- 任务worktree：`/mnt/nas-tianwen/home/ssy/JevGO/.dev/worktrees/P14f`。
- owner范围：jevgo/research.py的候选策略与事件；必要配置owner协调；固定状态JEV开发工具和相关测试。涉及共享契约/CLI时由主控唯一owner协调；正式代码不依赖开发目录。

在该worktree扩展以下已有测试文件覆盖本卡验收场景，然后执行：

```bash
cd /mnt/nas-tianwen/home/ssy/JevGO/.dev/worktrees/P14f
JEVGO_ASSETS_ROOT=/mnt/nas-tianwen/home/ssy/JevGO /mnt/nas-tianwen/home/ssy/JevGO/.venv/bin/python -B -m pytest -q -p no:cacheprovider \
  .dev/tests/autoresearch/test_research.py .dev/tests/autoresearch/test_mcts.py .dev/tests/autoresearch/test_runner.py
```

以上为本卡回归入口；实际受测文件集合、准确命令及退出码见文末验证记录。CPU 通过与真实模型结果分别报告，不累加重复测试数。

## 基线、工作环境与交接

状态：complete_with_negative_results。2026-10-19 已完成本卡范围的实现、独立审查及有限验证；以下保留原验收约定，实际记录见文末。证据审查基线 `305fed8c4a1698604eb5699a8afac898ffe4b3d5`；启动时复核最新集成HEAD并记录实际基线，继承已有AutoDL等变更，不回退历史推理版本。

项目与资源根 `/mnt/nas-tianwen/home/ssy/JevGO`，解释器 `/mnt/nas-tianwen/home/ssy/JevGO/.venv/bin/python`；worktree使用显式 `JEVGO_ASSETS_ROOT`，不用会切回主项目的env.sh。读取AGENTS及autoresearch/长期记忆/release-layout；真实GPU前再读取gpu-access/autodl技能。原checkpoint/revision/精度、`learning_mode=none`保持。只读CPU准备不触发GPU、push或远端流程。

本卡分支/worktree 已创建并保留，实际基线与交付提交见文末；子Agent优先gpt-6-astra / ultra，按实际工具能力记录。可复用开发源码和小fixture进Git，运行产物独立保存于本卡worktree的 `.dev/artifacts/` 并在回收清单记录绝对路径/哈希。原P14证据只读；其他任务的notification目录不清理、不提交。必要GPU命令需在首次启动前补入本卡执行记录，包含实际硬件、模型路径、服务/客户端窗口、预算和唯一输出目录，不能将未实现的探针写成已经可运行。

交付：实现commit、受测commit、准确命令/退出码、CPU与真实模型验证范围、逐条件结果/费用/失败、已知局限和清理记录；主控审查后合入 `codex/autoresearch`。本地提交不自动push；确需AutoDL时按当前技能同步已审查代码与指定commit。所有任务遵循[修正总计划](P14-REMEDIATION.md)，通过、失败、未覆盖和资源阻塞分开报告。

## 本轮实际交付（2026-10-19）

有界补采及 root/tree/rollout 遥测完成。两模型、两状态均 4→8 样本而唯一动作仍为 1；固定候选 JEV 顺序敏感性单独记录。

- 实际分支起点：`c90304dac6617e63f89e8e409ef82347a39f488d`；任务提交 → 集成提交：`c8fe0fb4413c5b66806d502033beaf3a45af76ad` → `48b367f5e4a8b2a1ba8dbcb3656451ad48fea3a2`。
- 本卡阶段 CPU：150 passed，退出码 0。最终统一代码 `9c88efce2a37a47571d30720e7d61dbabdc9582d` 的完整 HOST 集合为 527 passed，不与阶段/审查测试相加。
- 唯一运行根：`/mnt/nas-tianwen/home/ssy/JevGO/.dev/artifacts/P14-remediation-20261019-r1`；[原任务产物](../../../.dev/artifacts/P14-remediation-20261019-r1/harvest/P14f/)、[完整产物库存](../../../.dev/artifacts/P14-remediation-20261019-r1/artifact-inventory.json)记录源绝对路径、归档目标与 SHA，工作树保留。P14h 补存的尝试标记和 P14i 非审查产物位于 `harvest/inventory-supplement/`；尝试标记不单独证明模型已执行。独立审查由主控在活动审查结束后最终归档至 `harvest/P14i-review/`。机器总入口 `report.json`，逐版本映射 `implementation-commits.json`，独立 CPU 入口 `cpu-final-summary.json`。
- 真实条件清单、受测提交和命令见各批 `execution-record.json`；原始模型输入/输出、token IDs、事件、ledger/result 和失败均保留。全部 learning_mode=none，原权重/BF16；无训练、下载或 AutoDL。
- 综合结论、剩余范围、容量留痕局限与清理记录见 [本轮结果](P14-REMEDIATION-RESULTS.md)。原 P14 历史失败未覆盖；P15 与正式矩阵未启动。
