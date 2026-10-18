# P14g 记忆评分实效与停止语义

## 需求与定位

主环节AR-5/7，关联EVAL。确定缺陷是P14中pool<=保留上限却逐条评分；stop只作advisory符合首版契约，应明确标注而非直接改成阈值控制。逻辑依赖P14d公共基础；共享research.py按总计划串行接线，实验与候选策略分开。

## 实施大纲

1. G0最小修正：当全部评分池都将保留时跳过无取舍评分，保留ID/顺序不变并记录skip理由；memory_selection保存pool IDs、K/M、池生成规则、分数或未调用原因及最终ID，供独立重算。覆盖正常memory4/32、candidate4；不只测试memory1的特殊情形。
2. G1独立策略：明确池容量K和保留M，解除对动作候选数的隐式依赖；验证K>M时JEV才可能选择。先选择一种有界池生成规则，并与同池末尾截取比较。“有界历史尾部”与“旧保留记忆+新记录”是不同召回策略，不能同时切换后把结果只归于评分器。保留稳定同分处理、来源/类型/分支和全部原始证据。
3. 池尚未超M不调用JEV；池超M的逐item调用、重复上下文和最终真实生成输入全部计费。M只限制最终记录条数，不是提示token上限，评分前新增记忆和动作历史也占窗口。
4. Stop明确off/advisory语义：当前judge_stop=false可关闭，开启只记录未校准完成意见，不控制动作、搜索或环境success；消融称“停止意见记录/费用开关”，不能称有效停止控制。结果元数据和文档写明角色是否生效。
5. 本卡不要求实现control。若后续要引入模型请求停止/提交，应另设独立实验、冻结阈值/消费边界，不能把模拟分支意见用于结束整个episode，也不能将未成功变成success；其缺席不阻塞本卡验收。

## 验收标准

- CPU受控反事实：对固定state、pool、预算交换两条证据分数，保留ID和下一次生成请求里的实际公开记忆随之变化；覆盖低分新证据、旧反证、稳定同分、原始记录不删除、兄弟分支隔离。
- 池<=M时选择与旧代码相同、evidence judge请求/token新增为0，日志记录未调用原因；独立账本重算通过。
- 成功返回的stop评分数值不直接控制终止或成功判定；100%高分仍不能把未完成环境标成功。开关会影响费用/剩余预算，后端ModelFailure也可能触发原有错误终止，这两种间接作用分别记录，不能写成开关完全不影响轨迹，也不自动吞掉评分错误；覆盖已提交动作及证据在后继stop失败时仍保留。
- 真实G0/G1开发对照分别保存。跳过费用会释放预算并可能改变后续轨迹，不能直接从旧总费用减去4078项就声称新闭环已验；扩大K也可能更贵，必须实测。

## 修改范围与验证入口

- 任务branch：`codex/ar-p14g-memory-stop`。
- 任务worktree：`/mnt/nas-tianwen/home/ssy/JevGO/.dev/worktrees/P14g`。
- owner范围：jevgo/research.py的select_memory/角色元数据；必要评测标签与docs；本任务开发工具和相关测试。涉及共享契约/CLI时由主控唯一owner协调；正式代码不依赖开发目录。

在该worktree扩展以下已有测试文件覆盖本卡验收场景，然后执行：

```bash
cd /mnt/nas-tianwen/home/ssy/JevGO/.dev/worktrees/P14g
JEVGO_ASSETS_ROOT=/mnt/nas-tianwen/home/ssy/JevGO /mnt/nas-tianwen/home/ssy/JevGO/.venv/bin/python -B -m pytest -q -p no:cacheprovider \
  .dev/tests/autoresearch/test_research.py .dev/tests/autoresearch/test_evaluation.py .dev/tests/autoresearch/test_runner.py .dev/tests/autoresearch/test_core.py
```

以上为本卡回归入口；实际受测文件集合、准确命令及退出码见文末验证记录。CPU 通过与真实模型结果分别报告，不累加重复测试数。

## 基线、工作环境与交接

状态：complete_with_negative_results。2026-10-19 已完成本卡范围的实现、独立审查及有限验证；以下保留原验收约定，实际记录见文末。证据审查基线 `305fed8c4a1698604eb5699a8afac898ffe4b3d5`；启动时复核最新集成HEAD并记录实际基线，继承已有AutoDL等变更，不回退历史推理版本。

项目与资源根 `/mnt/nas-tianwen/home/ssy/JevGO`，解释器 `/mnt/nas-tianwen/home/ssy/JevGO/.venv/bin/python`；worktree使用显式 `JEVGO_ASSETS_ROOT`，不用会切回主项目的env.sh。读取AGENTS及autoresearch/长期记忆/release-layout；真实GPU前再读取gpu-access/autodl技能。原checkpoint/revision/精度、`learning_mode=none`保持。只读CPU准备不触发GPU、push或远端流程。

本卡分支/worktree 已创建并保留，实际基线与交付提交见文末；子Agent优先gpt-6-astra / ultra，按实际工具能力记录。可复用开发源码和小fixture进Git，运行产物独立保存于本卡worktree的 `.dev/artifacts/` 并在回收清单记录绝对路径/哈希。原P14证据只读；其他任务的notification目录不清理、不提交。必要GPU命令需在首次启动前补入本卡执行记录，包含实际硬件、模型路径、服务/客户端窗口、预算和唯一输出目录，不能将未实现的探针写成已经可运行。

交付：实现commit、受测commit、准确命令/退出码、CPU与真实模型验证范围、逐条件结果/费用/失败、已知局限和清理记录；主控审查后合入 `codex/autoresearch`。本地提交不自动push；确需AutoDL时按当前技能同步已审查代码与指定commit。所有任务遵循[修正总计划](P14-REMEDIATION.md)，通过、失败、未覆盖和资源阻塞分开报告。

## 本轮实际交付（2026-10-19）

G0 两模型真实复测各减少 38 物理请求；G1 同池选择改变实际下一提示。SemIf tail 初始及 repair 失败保留。stop 仅 off/advisory，无阈值控制。

- 实际分支起点：`48b367f5e4a8b2a1ba8dbcb3656451ad48fea3a2`；任务提交 → 集成提交：`86ddfac4ae8a1fc87b4187f0e4ad6aafab0582c2` → `9c88efce2a37a47571d30720e7d61dbabdc9582d`。
- 本卡阶段 CPU：224 passed，退出码 0。最终统一代码 `9c88efce2a37a47571d30720e7d61dbabdc9582d` 的完整 HOST 集合为 527 passed，不与阶段/审查测试相加。
- 唯一运行根：`/mnt/nas-tianwen/home/ssy/JevGO/.dev/artifacts/P14-remediation-20261019-r1`；[原任务产物](../../../.dev/artifacts/P14-remediation-20261019-r1/harvest/P14g/)、[完整产物库存](../../../.dev/artifacts/P14-remediation-20261019-r1/artifact-inventory.json)记录源绝对路径、归档目标与 SHA，工作树保留。P14h 补存的尝试标记和 P14i 非审查产物位于 `harvest/inventory-supplement/`；尝试标记不单独证明模型已执行。独立审查由主控在活动审查结束后最终归档至 `harvest/P14i-review/`。机器总入口 `report.json`，逐版本映射 `implementation-commits.json`，独立 CPU 入口 `cpu-final-summary.json`。
- 真实条件清单、受测提交和命令见各批 `execution-record.json`；原始模型输入/输出、token IDs、事件、ledger/result 和失败均保留。全部 learning_mode=none，原权重/BF16；无训练、下载或 AutoDL。
- 综合结论、剩余范围、容量留痕局限与清理记录见 [本轮结果](P14-REMEDIATION-RESULTS.md)。原 P14 历史失败未覆盖；P15 与正式矩阵未启动。
