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

- 计划branch：`codex/ar-p14f-candidate-coverage`。
- 计划worktree：`/mnt/nas-tianwen/home/ssy/JevGO/.dev/worktrees/P14f`。
- owner范围：jevgo/research.py的候选策略与事件；必要配置owner协调；固定状态JEV开发工具和相关测试。涉及共享契约/CLI时由主控唯一owner协调；正式代码不依赖开发目录。

在该worktree扩展以下已有测试文件覆盖本卡验收场景，然后执行：

```bash
cd /mnt/nas-tianwen/home/ssy/JevGO/.dev/worktrees/P14f
JEVGO_ASSETS_ROOT=/mnt/nas-tianwen/home/ssy/JevGO /mnt/nas-tianwen/home/ssy/JevGO/.venv/bin/python -B -m pytest -q -p no:cacheprovider \
  .dev/tests/autoresearch/test_research.py .dev/tests/autoresearch/test_mcts.py .dev/tests/autoresearch/test_runner.py
```

这些是实施后的回归入口，当前没有执行；新开发工具/探针的准确CLI、固定条件清单与预算在实现后、GPU执行前记录。测试通过不替代本卡所列真实模型对照。

## 基线、工作环境与交接

状态：planned_not_started。本卡于2026-10-19编写，仅形成实施约定，未实施或运行新实验。证据审查基线 `305fed8c4a1698604eb5699a8afac898ffe4b3d5`；启动时复核最新集成HEAD并记录实际基线，继承已有AutoDL等变更，不回退历史推理版本。

项目与资源根 `/mnt/nas-tianwen/home/ssy/JevGO`，解释器 `/mnt/nas-tianwen/home/ssy/JevGO/.venv/bin/python`；worktree使用显式 `JEVGO_ASSETS_ROOT`，不用会切回主项目的env.sh。读取AGENTS及autoresearch/长期记忆/release-layout；真实GPU前再读取gpu-access/autodl技能。原checkpoint/revision/精度、`learning_mode=none`保持。只读CPU准备不触发GPU、push或远端流程。

本卡分支/worktree为计划名称，尚未创建；子Agent优先gpt-6-astra / ultra，按实际工具能力记录。可复用开发源码和小fixture进Git，运行产物独立保存于本卡worktree的 `.dev/artifacts/` 并在回收清单记录绝对路径/哈希。原P14证据只读；其他任务的notification目录不清理、不提交。必要GPU命令需在首次启动前补入本卡执行记录，包含实际硬件、模型路径、服务/客户端窗口、预算和唯一输出目录，不能将未实现的探针写成已经可运行。

交付：实现commit、受测commit、准确命令/退出码、CPU与真实模型验证范围、逐条件结果/费用/失败、已知局限和清理记录；主控审查后合入 `codex/autoresearch`。本地提交不自动push；确需AutoDL时按当前技能同步已审查代码与指定commit。所有任务遵循[修正总计划](P14-REMEDIATION.md)，通过、失败、未覆盖和资源阻塞分开报告。
