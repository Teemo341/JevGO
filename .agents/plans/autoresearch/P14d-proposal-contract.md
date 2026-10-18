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

- 计划branch：`codex/ar-p14d-proposal-contract`。
- 计划worktree：`/mnt/nas-tianwen/home/ssy/JevGO/.dev/worktrees/P14d`。
- owner范围：jevgo/research.py的解析/repair；必要正式解析辅助模块；本任务开发探针和相关测试。涉及共享契约/CLI时由主控唯一owner协调；正式代码不依赖开发目录。

在该worktree扩展以下已有测试文件覆盖本卡验收场景，然后执行：

```bash
cd /mnt/nas-tianwen/home/ssy/JevGO/.dev/worktrees/P14d
JEVGO_ASSETS_ROOT=/mnt/nas-tianwen/home/ssy/JevGO /mnt/nas-tianwen/home/ssy/JevGO/.venv/bin/python -B -m pytest -q -p no:cacheprovider \
  .dev/tests/autoresearch/test_research.py .dev/tests/autoresearch/test_runner.py .dev/tests/test_p14_semif_pretrial.py
```

这些是实施后的回归入口，当前没有执行；新开发工具/探针的准确CLI、固定条件清单与预算在实现后、GPU执行前记录。测试通过不替代本卡所列真实模型对照。

## 基线、工作环境与交接

状态：planned_not_started。本卡于2026-10-19编写，仅形成实施约定，未实施或运行新实验。证据审查基线 `305fed8c4a1698604eb5699a8afac898ffe4b3d5`；启动时复核最新集成HEAD并记录实际基线，继承已有AutoDL等变更，不回退历史推理版本。

项目与资源根 `/mnt/nas-tianwen/home/ssy/JevGO`，解释器 `/mnt/nas-tianwen/home/ssy/JevGO/.venv/bin/python`；worktree使用显式 `JEVGO_ASSETS_ROOT`，不用会切回主项目的env.sh。读取AGENTS及autoresearch/长期记忆/release-layout；真实GPU前再读取gpu-access/autodl技能。原checkpoint/revision/精度、`learning_mode=none`保持。只读CPU准备不触发GPU、push或远端流程。

本卡分支/worktree为计划名称，尚未创建；子Agent优先gpt-6-astra / ultra，按实际工具能力记录。可复用开发源码和小fixture进Git，运行产物独立保存于本卡worktree的 `.dev/artifacts/` 并在回收清单记录绝对路径/哈希。原P14证据只读；其他任务的notification目录不清理、不提交。必要GPU命令需在首次启动前补入本卡执行记录，包含实际硬件、模型路径、服务/客户端窗口、预算和唯一输出目录，不能将未实现的探针写成已经可运行。

交付：实现commit、受测commit、准确命令/退出码、CPU与真实模型验证范围、逐条件结果/费用/失败、已知局限和清理记录；主控审查后合入 `codex/autoresearch`。本地提交不自动push；确需AutoDL时按当前技能同步已审查代码与指定commit。所有任务遵循[修正总计划](P14-REMEDIATION.md)，通过、失败、未覆盖和资源阻塞分开报告。
