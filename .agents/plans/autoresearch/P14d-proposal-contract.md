# P14d 严格候选协议与有界格式修复

## 需求与定位

主环节AR-2/3，关联AR-7、EVAL。输入为原始生成文本和公开合法动作，输出为严格合法候选或可追溯拒绝。依赖P14结果及行为诊断；接口以CONTRACTS.md和现有GenerationBackend/ResearchPolicy为准。

确定缺陷：普通json.loads接受重复键，现有字段校验允许提示要求的字段缺失并由数据模型补默认值。修复此契约不等于解决全forward；行为审计已证明大量严格合法原文也全是forward。

## 实施大纲

1. 严格约束限定于模型生成文本入口，不收紧内部Candidate/Action构造默认值而破坏无关接口。将正式候选解析收敛到单一入口：递归拒绝重复键、非标准JSON常量，严格检查action/hypothesis/rationale及action内部name/arguments字段、类型与公开合法动作。保留已有完整json fence的明确支持规则；不补括号、不补动作、不从坏文本提取可执行片段。
2. 第一条件仅修解析，保留现有最多一次repair、只有零合法候选才repair、部分合法保留的控制流，费用照实记录。严格策略成为所有方法共同的版本化基础。
3. 第二独立条件才比较现有通用repair后缀与“有界原错误/坏输出+精确格式要求”修复提示。预声明选择哪份坏样本、字数上限、固定seed及次数，完整请求长度预检；不要同时修改输出上限、采样参数或候选覆盖。
4. 记录逐样本拒绝阶段和原因，生成调用、合法候选、repair调用分别统计。保留旧宽松解析结果作为历史，不重写原成功/费用。

## 验收标准

- 原P14重复键样本必须被正式解析拒绝；嵌套重复键、缺字段、额外字段、错误类型、非有限常量、残缺JSON和非公开动作均有覆盖，合规旧样本仍接受。
- 坏样本本身不产生环境step；独立合法修复可产生行动并保留来源。多样本部分合法、全坏、有界repair失败和预算中断均可追溯，不漏失败费用/未知usage。
- 开发严格校验与正式入口同一契约；不要让运行器继续last-wins而仅离线报告严格通过率。
- 小型真实配对以P14保存的坏请求为起点，保持checkpoint/runtime/seed及输出256条件一致；输出变好不是CPU协议验收的前提。该样本已用于开发，不能作为独立效果测试集。

## 修改范围与验证入口

- 任务branch：`codex/ar-p14d-proposal-contract`。
- 任务worktree：`/mnt/nas-tianwen/home/ssy/JevGO/.dev/worktrees/P14d`。
- owner范围：jevgo/research.py的解析/repair；必要正式解析辅助模块；本任务开发探针和相关测试。涉及共享契约/CLI时由主控唯一owner协调；正式代码不依赖开发目录。

在该worktree扩展以下已有测试文件覆盖本卡验收场景，然后执行：

```bash
cd /mnt/nas-tianwen/home/ssy/JevGO/.dev/worktrees/P14d
JEVGO_ASSETS_ROOT=/mnt/nas-tianwen/home/ssy/JevGO /mnt/nas-tianwen/home/ssy/JevGO/.venv/bin/python -B -m pytest -q -p no:cacheprovider \
  .dev/tests/autoresearch/test_research.py .dev/tests/autoresearch/test_runner.py .dev/tests/test_p14_semif_pretrial.py
```

以上为本卡回归入口；实际受测文件集合、准确命令及退出码见文末验证记录。CPU 通过与真实模型结果分别报告，不累加重复测试数。

## 基线、工作环境与交接

状态：complete_with_negative_results。2026-10-19 已完成本卡范围的实现、独立审查及有限验证；以下保留原验收约定，实际记录见文末。证据审查基线 `305fed8c4a1698604eb5699a8afac898ffe4b3d5`；启动时复核最新集成HEAD并记录实际基线，继承已有AutoDL等变更，不回退历史推理版本。

项目与资源根 `/mnt/nas-tianwen/home/ssy/JevGO`，解释器 `/mnt/nas-tianwen/home/ssy/JevGO/.venv/bin/python`；worktree使用显式 `JEVGO_ASSETS_ROOT`，不用会切回主项目的env.sh。读取AGENTS及autoresearch/长期记忆/release-layout；真实GPU前再读取gpu-access/autodl技能。原checkpoint/revision/精度、`learning_mode=none`保持。只读CPU准备不触发GPU、push或远端流程。

本卡分支/worktree 已创建并保留，实际基线与交付提交见文末；子Agent优先gpt-6-astra / ultra，按实际工具能力记录。可复用开发源码和小fixture进Git，运行产物独立保存于本卡worktree的 `.dev/artifacts/` 并在回收清单记录绝对路径/哈希。原P14证据只读；其他任务的notification目录不清理、不提交。必要GPU命令需在首次启动前补入本卡执行记录，包含实际硬件、模型路径、服务/客户端窗口、预算和唯一输出目录，不能将未实现的探针写成已经可运行。

交付：实现commit、受测commit、准确命令/退出码、CPU与真实模型验证范围、逐条件结果/费用/失败、已知局限和清理记录；主控审查后合入 `codex/autoresearch`。本地提交不自动push；确需AutoDL时按当前技能同步已审查代码与指定commit。所有任务遵循[修正总计划](P14-REMEDIATION.md)，通过、失败、未覆盖和资源阻塞分开报告。

## 本轮实际交付（2026-10-19）

严格文本入口及有界 repair 完成。真实 generic 1/1 非法、bounded 1/1 合法；仅一份已用于开发的坏请求配对，不是效果测试。

- 实际分支起点：`53f965515d538a34130f5081c555a378c6ddc4fc`；任务提交 → 集成提交：`67edaa9752eb7d1bcb6817ade9afc68141f98c38` → `a16f93e2493be8623bf58293585e3fee16db764f`。
- 本卡阶段 CPU：118 passed，退出码 0。最终统一代码 `9c88efce2a37a47571d30720e7d61dbabdc9582d` 的完整 HOST 集合为 527 passed，不与阶段/审查测试相加。
- 唯一运行根：`/mnt/nas-tianwen/home/ssy/JevGO/.dev/artifacts/P14-remediation-20261019-r1`；[原任务产物](../../../.dev/artifacts/P14-remediation-20261019-r1/harvest/P14d/)、[完整产物库存](../../../.dev/artifacts/P14-remediation-20261019-r1/artifact-inventory.json)记录源绝对路径、归档目标与 SHA，工作树保留。P14h 补存的尝试标记和 P14i 非审查产物位于 `harvest/inventory-supplement/`；尝试标记不单独证明模型已执行。独立审查由主控在活动审查结束后最终归档至 `harvest/P14i-review/`。机器总入口 `report.json`，逐版本映射 `implementation-commits.json`，独立 CPU 入口 `cpu-final-summary.json`。
- 真实条件清单、受测提交和命令见各批 `execution-record.json`；原始模型输入/输出、token IDs、事件、ledger/result 和失败均保留。全部 learning_mode=none，原权重/BF16；无训练、下载或 AutoDL。
- 综合结论、剩余范围、容量留痕局限与清理记录见 [本轮结果](P14-REMEDIATION-RESULTS.md)。原 P14 历史失败未覆盖；P15 与正式矩阵未启动。
