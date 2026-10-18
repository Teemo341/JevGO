# P14i 最小组合验收与P15准入审查

## 需求与定位

主环节EVAL，关联AR-0至7。依赖P14d/e/f/g实施及P14h对拟采用条件的容量证据，输出统一可审计版本和是否可冻结正式协议的判断。该任务不是正式benchmark，也不以必须获得正收益作为完成条件。

## 实施大纲

1. 主控合入已独立审查的修改，记录每项语义/配置版本、commit和受测版本；单因素证据先保存，再运行组合版本。未经单独对照的交互不能归因于某一改动。
2. 先用小型确定性CPU任务覆盖严格解析、信息白名单、分支隔离、记忆评分对下游的影响、多候选PUCT/回传、预算预留和失败费用。运行受影响回归及发布隔离；测试数按实际集合报告，不能累加历史重叠数。
3. 真实最小闭环以已有MiniGrid开发条件为起点，同一基座下B0–B3使用相同公开信息、解析/repair规则、环境期限、提交上限与累计预算；明确每种方法的采样/覆盖差异。先9B及SemIf，其他后端按拟用范围补协议/短闭环，不默认重跑原64次矩阵。
4. 冻结这次有限验收的任务、seed、最大尝试数和预算，失败也计入表，禁止选最好seed或删除坏输出。不同seed未必产生不同Empty地图，报告实际任务多样性；这些已用开发条件不进入独立测试集。
5. 保存根分支数、单候选率、排序区分、记忆实选率、stop实际语义、逐角色费用、客观成功与终止分类。若仍退化，明确所测方法无法支持哪些主张，可保留为负结果或将未生效模块显式关闭。

## 验收标准与准入判断

- 确定的协议/成本缺陷已经修复并有测试，或相关功能明确停用并缩小主张；模型表现差本身不阻塞提出正式协议。
- 各功能名称与实际行为一致，拟用容量/并发有证据，同一基座观测/权限一致，所有失败/repair/search/replay费用完整，环境期限与提交上限独立。
- 单候选树不能证明有效搜索收益，未选择记忆不能证明评分筛选收益，advisory stop不能当控制策略。若保留这些退化条件，应明确评测的是该具体配置的表现，不能以它代表机制已得到充分检验。
- 输出P15待冻结条目：多任务/多seed及开发测试划分、模型/模态、主预算与硬上限、失败/repair规则、消融、统计口径、纳入环境能力。ALFWorld正式数据和DiscoveryWorld恢复限制继续明确。
- 最终给出“可开始P15协议冻结”或具体剩余门槛，不自动执行正式矩阵，不训练，不把有限开发成功宣称算法增益。

## 修改范围与验证入口

- 计划branch：`codex/ar-p14i-readiness`。
- 计划worktree：`/mnt/nas-tianwen/home/ssy/JevGO/.dev/worktrees/P14i`。
- owner范围：集成文档/开发配置/验收脚本；共享正式文件仅由主控合并已审查提交，发现新算法问题回对应owner。涉及共享契约/CLI时由主控唯一owner协调；正式代码不依赖开发目录。

在该worktree扩展以下已有测试文件覆盖本卡验收场景，然后执行：

```bash
cd /mnt/nas-tianwen/home/ssy/JevGO/.dev/worktrees/P14i
JEVGO_ASSETS_ROOT=/mnt/nas-tianwen/home/ssy/JevGO /mnt/nas-tianwen/home/ssy/JevGO/.venv/bin/python -B -m pytest -q -p no:cacheprovider \
  .dev/tests/autoresearch/test_research.py .dev/tests/autoresearch/test_runner.py .dev/tests/autoresearch/test_mcts.py .dev/tests/autoresearch/test_minigrid.py .dev/tests/autoresearch/test_evaluation.py .dev/tests/autoresearch/test_release.py
```

这些是实施后的回归入口，当前没有执行；新开发工具/探针的准确CLI、固定条件清单与预算在实现后、GPU执行前记录。测试通过不替代本卡所列真实模型对照。

## 基线、工作环境与交接

状态：planned_not_started。本卡于2026-10-19编写，仅形成实施约定，未实施或运行新实验。证据审查基线 `305fed8c4a1698604eb5699a8afac898ffe4b3d5`；启动时复核最新集成HEAD并记录实际基线，继承已有AutoDL等变更，不回退历史推理版本。

项目与资源根 `/mnt/nas-tianwen/home/ssy/JevGO`，解释器 `/mnt/nas-tianwen/home/ssy/JevGO/.venv/bin/python`；worktree使用显式 `JEVGO_ASSETS_ROOT`，不用会切回主项目的env.sh。读取AGENTS及autoresearch/长期记忆/release-layout；真实GPU前再读取gpu-access/autodl技能。原checkpoint/revision/精度、`learning_mode=none`保持。只读CPU准备不触发GPU、push或远端流程。

本卡分支/worktree为计划名称，尚未创建；子Agent优先gpt-6-astra / ultra，按实际工具能力记录。可复用开发源码和小fixture进Git，运行产物独立保存于本卡worktree的 `.dev/artifacts/` 并在回收清单记录绝对路径/哈希。原P14证据只读；其他任务的notification目录不清理、不提交。必要GPU命令需在首次启动前补入本卡执行记录，包含实际硬件、模型路径、服务/客户端窗口、预算和唯一输出目录，不能将未实现的探针写成已经可运行。

交付：实现commit、受测commit、准确命令/退出码、CPU与真实模型验证范围、逐条件结果/费用/失败、已知局限和清理记录；主控审查后合入 `codex/autoresearch`。本地提交不自动push；确需AutoDL时按当前技能同步已审查代码与指定commit。所有任务遵循[修正总计划](P14-REMEDIATION.md)，通过、失败、未覆盖和资源阻塞分开报告。
