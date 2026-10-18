# P14e 公开观测表达与可见差分

## 需求与定位

主环节AR-4/5，关联AR-2、EVAL。输入只能来自公开Observation及公开动作，输出为无额外隐藏信息的表述和差分。逻辑上可独立准备，正式接线在P14d后；public_state契约由唯一owner修改。

已确认转换无错、模型原文误读；替代表述能否改善行为仍待验证。现有MiniGrid action_success由私有_world_bytes比较得到，不能直接公开给模型。

## 实施大纲

1. 独立条件E1补充语义：零起始局部坐标、image[x][y][channel]、对象/颜色/状态编码、agent局部(3,6)、前方(3,5)、全局compass与旋转局部视图的关系。不改变环境状态、动作或benchmark随机机制，不提示求解路线。
2. 独立条件E2在E1基础上生成前后公开观测差分。使用白名单可见字段而非raw StepResult、私有指纹、agent_pos或全局grid；没有可见变化时只能说“公开观测未变化”，不能一般化为“没有移动”。
3. 声明差分是一步即时反馈还是工作记忆；建议作为统一即时反馈，no_memory仅关闭持久记忆，文档/消融据此明确。所有B0–B3在同条件获得相同公开信息，原始观测与别名可追溯。
4. 固定公开状态对照：旧表达、E1、E1+E2；生成和JEV分别保存完整提示/hash和行为。更改采样或候选覆盖不混入本项因果判断。

## 验收标准

- CPU验证四朝向、局部轴/编码、遮挡unknown、墙前、相似走廊/重复外观。相同公开输入必须产生相同表述，改变隐藏世界但不改变公开输入不得改变差分。
- 兄弟分支不串前一状态，恢复后差分与实际路径一致；恢复句柄、私有成功判断和全局坐标不进入提示。
- E1与E2分别记录token开销。固定状态生成诊断统计朝向理解、合法率、不同动作数和重复动作；不要求模型一定改善，不把手工控制路线计为模型成绩。
- 适配器/差分器生成的事实文本中，未移动/碰墙等断言若无公开依据不得生成；不能用旧CPU诊断的隐藏坐标作在线解释。此约束不新增对模型hypothesis的语义过滤，模型假设仍按未验证假设保留，避免污染观测单因素对照。

## 修改范围与验证入口

- 任务branch：`codex/ar-p14e-public-observations`。
- 任务worktree：`/mnt/nas-tianwen/home/ssy/JevGO/.dev/worktrees/P14e`。
- owner范围：jevgo/envs/minigrid.py的公开表达；jevgo/research.py的白名单/转换接口；本任务开发探针和相关测试。涉及共享契约/CLI时由主控唯一owner协调；正式代码不依赖开发目录。

在该worktree扩展以下已有测试文件覆盖本卡验收场景，然后执行：

```bash
cd /mnt/nas-tianwen/home/ssy/JevGO/.dev/worktrees/P14e
JEVGO_ASSETS_ROOT=/mnt/nas-tianwen/home/ssy/JevGO /mnt/nas-tianwen/home/ssy/JevGO/.venv/bin/python -B -m pytest -q -p no:cacheprovider \
  .dev/tests/autoresearch/test_minigrid.py .dev/tests/autoresearch/test_research.py .dev/tests/autoresearch/test_runner.py
```

以上为本卡回归入口；实际受测文件集合、准确命令及退出码见文末验证记录。CPU 通过与真实模型结果分别报告，不累加重复测试数。

## 基线、工作环境与交接

状态：complete_with_negative_results。2026-10-19 已完成本卡范围的实现、独立审查及有限验证；以下保留原验收约定，实际记录见文末。证据审查基线 `305fed8c4a1698604eb5699a8afac898ffe4b3d5`；启动时复核最新集成HEAD并记录实际基线，继承已有AutoDL等变更，不回退历史推理版本。

项目与资源根 `/mnt/nas-tianwen/home/ssy/JevGO`，解释器 `/mnt/nas-tianwen/home/ssy/JevGO/.venv/bin/python`；worktree使用显式 `JEVGO_ASSETS_ROOT`，不用会切回主项目的env.sh。读取AGENTS及autoresearch/长期记忆/release-layout；真实GPU前再读取gpu-access/autodl技能。原checkpoint/revision/精度、`learning_mode=none`保持。只读CPU准备不触发GPU、push或远端流程。

本卡分支/worktree 已创建并保留，实际基线与交付提交见文末；子Agent优先gpt-6-astra / ultra，按实际工具能力记录。可复用开发源码和小fixture进Git，运行产物独立保存于本卡worktree的 `.dev/artifacts/` 并在回收清单记录绝对路径/哈希。原P14证据只读；其他任务的notification目录不清理、不提交。必要GPU命令需在首次启动前补入本卡执行记录，包含实际硬件、模型路径、服务/客户端窗口、预算和唯一输出目录，不能将未实现的探针写成已经可运行。

交付：实现commit、受测commit、准确命令/退出码、CPU与真实模型验证范围、逐条件结果/费用/失败、已知局限和清理记录；主控审查后合入 `codex/autoresearch`。本地提交不自动push；确需AutoDL时按当前技能同步已审查代码与指定commit。所有任务遵循[修正总计划](P14-REMEDIATION.md)，通过、失败、未覆盖和资源阻塞分开报告。

## 本轮实际交付（2026-10-19）

公开 E1/E1+E2 与 no_memory 即时反馈完成；两基座各 9 个生成条件、18 个固定候选评分条件。空间误读仍存在，不宣称避障改善。

- 实际分支起点：`a16f93e2493be8623bf58293585e3fee16db764f`；任务提交 → 集成提交：`7ae4b07f886a947679e102f9d2b73c8d2c3a06e6` → `c90304dac6617e63f89e8e409ef82347a39f488d`。
- 本卡阶段 CPU：127 passed，退出码 0。最终统一代码 `9c88efce2a37a47571d30720e7d61dbabdc9582d` 的完整 HOST 集合为 527 passed，不与阶段/审查测试相加。
- 唯一运行根：`/mnt/nas-tianwen/home/ssy/JevGO/.dev/artifacts/P14-remediation-20261019-r1`；[原任务产物](../../../.dev/artifacts/P14-remediation-20261019-r1/harvest/P14e/)、[完整产物库存](../../../.dev/artifacts/P14-remediation-20261019-r1/artifact-inventory.json)记录源绝对路径、归档目标与 SHA，工作树保留。P14h 补存的尝试标记和 P14i 非审查产物位于 `harvest/inventory-supplement/`；尝试标记不单独证明模型已执行。独立审查由主控在活动审查结束后最终归档至 `harvest/P14i-review/`。机器总入口 `report.json`，逐版本映射 `implementation-commits.json`，独立 CPU 入口 `cpu-final-summary.json`。
- 真实条件清单、受测提交和命令见各批 `execution-record.json`；原始模型输入/输出、token IDs、事件、ledger/result 和失败均保留。全部 learning_mode=none，原权重/BF16；无训练、下载或 AutoDL。
- 综合结论、剩余范围、容量留痕局限与清理记录见 [本轮结果](P14-REMEDIATION-RESULTS.md)。原 P14 历史失败未覆盖；P15 与正式矩阵未启动。
