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

- 任务branch：`codex/ar-p14i-readiness`。
- 任务worktree：`/mnt/nas-tianwen/home/ssy/JevGO/.dev/worktrees/P14i`。
- owner范围：集成文档/开发配置/验收脚本；共享正式文件仅由主控合并已审查提交，发现新算法问题回对应owner。涉及共享契约/CLI时由主控唯一owner协调；正式代码不依赖开发目录。

在该worktree扩展以下已有测试文件覆盖本卡验收场景，然后执行：

```bash
cd /mnt/nas-tianwen/home/ssy/JevGO/.dev/worktrees/P14i
JEVGO_ASSETS_ROOT=/mnt/nas-tianwen/home/ssy/JevGO /mnt/nas-tianwen/home/ssy/JevGO/.venv/bin/python -B -m pytest -q -p no:cacheprovider \
  .dev/tests/autoresearch/test_research.py .dev/tests/autoresearch/test_runner.py .dev/tests/autoresearch/test_mcts.py .dev/tests/autoresearch/test_minigrid.py .dev/tests/autoresearch/test_evaluation.py .dev/tests/autoresearch/test_release.py
```

以上为本卡回归入口；实际受测文件集合、准确命令及退出码见文末验证记录。CPU 通过与真实模型结果分别报告，不累加重复测试数。

## 基线、工作环境与交接

状态：complete_with_negative_results。2026-10-19 已完成本卡范围的实现、独立审查及有限验证；以下保留原验收约定，实际记录见文末。证据审查基线 `305fed8c4a1698604eb5699a8afac898ffe4b3d5`；启动时复核最新集成HEAD并记录实际基线，继承已有AutoDL等变更，不回退历史推理版本。

项目与资源根 `/mnt/nas-tianwen/home/ssy/JevGO`，解释器 `/mnt/nas-tianwen/home/ssy/JevGO/.venv/bin/python`；worktree使用显式 `JEVGO_ASSETS_ROOT`，不用会切回主项目的env.sh。读取AGENTS及autoresearch/长期记忆/release-layout；真实GPU前再读取gpu-access/autodl技能。原checkpoint/revision/精度、`learning_mode=none`保持。只读CPU准备不触发GPU、push或远端流程。

本卡分支/worktree 已创建并保留，实际基线与交付提交见文末；子Agent优先gpt-6-astra / ultra，按实际工具能力记录。可复用开发源码和小fixture进Git，运行产物独立保存于本卡worktree的 `.dev/artifacts/` 并在回收清单记录绝对路径/哈希。原P14证据只读；其他任务的notification目录不清理、不提交。必要GPU命令需在首次启动前补入本卡执行记录，包含实际硬件、模型路径、服务/客户端窗口、预算和唯一输出目录，不能将未实现的探针写成已经可运行。

交付：实现commit、受测commit、准确命令/退出码、CPU与真实模型验证范围、逐条件结果/费用/失败、已知局限和清理记录；主控审查后合入 `codex/autoresearch`。本地提交不自动push；确需AutoDL时按当前技能同步已审查代码与指定commit。所有任务遵循[修正总计划](P14-REMEDIATION.md)，通过、失败、未覆盖和资源阻塞分开报告。

## 本轮实际交付（2026-10-19）

9B 与 SemIf 各 B0–B3 一次，八个组合条件均完成且无客观成功；SemIf B0 格式失败、两个 B3 预算耗尽均保留；可开始 P15 协议冻结，范围限本轮已测 9B/SemIf 条件，保留负结果与容量限制

- 实际分支起点：`53f965515d538a34130f5081c555a378c6ddc4fc`；任务提交 → 集成提交：`62f13da311835424666a391bbd31e7d7f75d9a13` → `16d82d593576c7f20232f83dc7cd83c582e6a190`；`46bc7f2e1193f66f8c99cf0bede23c7335a099dc` → `9cb83ba7aaed19e2c910b1f757a071f1709bc7ce`；`26316fe33e222c8e773f0ce9d106a9a11e88cb80` → `22814e450ab433fd08bee78dccdce799f9c7c514`；`1bcef8e0286e14da4895be2bc4ab2aff1d7eadd1` → `57ecd2ea3d5ca503b296f91d932725a5dc858a5d`；`52152d677d5a904dfa9855bef126331b04750a71` → `a5b89cc5b81f92f2eeab2074c96781d989ee58b1`；`462f248ddb7d3b7f323dd3e5711c6c073691cba8` → `65dee504dca614745575358a0f006e79b0d1a755`；`d959862d63f2de2c8e5d753a43b21508cce22457` → `01ab7a5aaf8792f5ec043bd85d827177bc41a747`；`7465d25e02669c26d857139cc739710e31e20fbd` → `11c92c27906c7ec9922c1be4d00aeb318f679608`。
- 本卡阶段 CPU：527（最终完整集合） passed，退出码 0。最终统一代码 `9c88efce2a37a47571d30720e7d61dbabdc9582d` 的完整 HOST 集合为 527 passed，不与阶段/审查测试相加。
- 唯一运行根：`/mnt/nas-tianwen/home/ssy/JevGO/.dev/artifacts/P14-remediation-20261019-r1`；[原任务产物](../../../.dev/artifacts/P14-remediation-20261019-r1/harvest/inventory-supplement/P14i-own/)、[完整产物库存](../../../.dev/artifacts/P14-remediation-20261019-r1/artifact-inventory.json)记录源绝对路径、归档目标与 SHA，工作树保留。P14h 补存的尝试标记和 P14i 非审查产物位于 `harvest/inventory-supplement/`；尝试标记不单独证明模型已执行。独立审查由主控在活动审查结束后最终归档至 `harvest/P14i-review/`。机器总入口 `report.json`，逐版本映射 `implementation-commits.json`，独立 CPU 入口 `cpu-final-summary.json`。
- 真实条件清单、受测提交和命令见各批 `execution-record.json`；原始模型输入/输出、token IDs、事件、ledger/result 和失败均保留。全部 learning_mode=none，原权重/BF16；无训练、下载或 AutoDL。
- 综合结论、剩余范围、容量留痕局限与清理记录见 [本轮结果](P14-REMEDIATION-RESULTS.md)。原 P14 历史失败未覆盖；P15 与正式矩阵未启动。
