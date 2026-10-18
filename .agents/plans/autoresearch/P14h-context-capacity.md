# P14h 原生窗口、实际请求与并发容量

## 需求与定位

主环节EVAL，关联AR-0/2/5/7。输入为本地模型配置、目标硬件/runtime及完整请求，输出为窗口/长度/并发的实测能力表，不按预分配显存猜上限。依赖P14已有长度工具；容量工具可并行开发，最终请求形状需在P14d/e/g后复核。

9B本地声明262144；P14只验证8192服务窗口以及实际约6311的最大输入+输出预留。8192不等于原生容量、推荐上限，也不等于满8192输入已经验收。96GiB卡服务采样峰含预分配KV池，不代表最低需求；服务的理论并发也不是实测并发。

## 实施大纲

1. 四层独立报告：本地配置声明；指定硬件/runtime成功启动且公布的窗口；真实长输入prefill+有效decode；目标并发下时延/吞吐/费用/显存。每层单独通过或失败，不能相互替代。
2. 优先9B，在原精度/同checkpoint下验证声明的原生窗口；允许有记录的分级定位，例如8192、32768、65536、131072、262144。不得因旧小窗口数据跳过原生窗口评估；若实测资源不足，记录具体阶段、请求形状、错误与可用硬件，不能泛化为模型架构限制。每级次数/总请求与token预算先固定，失败后按可验证原因选择下一项，不无限重启。
3. 在候选服务窗口附近构造可审计长输入，按完整模板精确分词并预留真实输出；使用受控新内容/prefix规则记录缓存状态，分别保存prefill、decode和端到端时延。合成长度压力测试只说明容量，不说明长上下文推理质量。每个条件预声明实际decode token目标并核验实际输出；仅设置max_new_tokens或EOS提前结束只支持实测长度，不能宣称目标长decode通过。
4. 预声明并发1及目标运行并发，区分单HTTP n=4产生的序列与4个并行HTTP请求；保存max_num_seqs、TP、KV/runtime相关参数、硬件/driver/kernel、输入长度分布、峰值测量范围。必须记录请求重叠及可观测的实际活跃序列/并发证据；若仅排队顺序完成，标记为排队服务而非目标并发已验。自有进程清理并保留失败请求费用/未知项。
5. 最终使用新的generation/repair/candidate/evidence/stop真实请求形状复核。SemIf及其他准备纳入正式评测的后端分别验证拟用窗口与并发；不得把9B结果迁移为其他checkpoint的容量证明。声明最大窗口不要求成为正式配置默认值，最终预算由目标工作负载确定。

## 验收标准

- 离线长度与实际模板/token IDs一致；服务配置、客户端约束与响应长度一致，无隐式裁切。原失败usage仍未知，不用离线长度补账。
- 能给出每个“模型+硬件+runtime+窗口+并发”条件的已测/未测/失败清单；单纯服务ready、短协议成功或nvidia-smi高占用都不能作为完整容量通过。
- 总请求/输入输出tokens/端到端耗时和插桩耗时分别记录；扩大窗口与累计预算独立。
- 可以因实际资源限制结束某条件，但正式选用的窗口/并发必须有真实请求证据。262144未通过时明确归因边界，不强行量化/换模型；本机资源不足按当前AutoDL技能处理，不在105运行远端生命周期脚本。

## 修改范围与验证入口

- 任务branch：`codex/ar-p14h-context-capacity`。
- 任务worktree：`/mnt/nas-tianwen/home/ssy/JevGO/.dev/worktrees/P14h`。
- owner范围：独立 .dev/tools/ 容量探针与 .dev/analysis/ 报告、相关开发测试；复用正式payload，不随意改服务默认值。涉及共享契约/CLI时由主控唯一owner协调；正式代码不依赖开发目录。

在该worktree扩展以下已有测试文件覆盖本卡验收场景，然后执行：

```bash
cd /mnt/nas-tianwen/home/ssy/JevGO/.dev/worktrees/P14h
JEVGO_ASSETS_ROOT=/mnt/nas-tianwen/home/ssy/JevGO /mnt/nas-tianwen/home/ssy/JevGO/.venv/bin/python -B -m pytest -q -p no:cacheprovider \
  .dev/tests/autoresearch/test_p14_context.py .dev/tests/autoresearch/test_p14_context_summary.py .dev/tests/autoresearch/test_autotrust.py .dev/tests/autoresearch/test_semif.py
```

以上为本卡回归入口；实际受测文件集合、准确命令及退出码见文末验证记录。CPU 通过与真实模型结果分别报告，不累加重复测试数。

## 基线、工作环境与交接

状态：capacity_verified_with_stated_limits。2026-10-19 已完成本卡范围的实现、独立审查及有限验证；以下保留原验收约定，实际记录见文末。证据审查基线 `305fed8c4a1698604eb5699a8afac898ffe4b3d5`；启动时复核最新集成HEAD并记录实际基线，继承已有AutoDL等变更，不回退历史推理版本。

项目与资源根 `/mnt/nas-tianwen/home/ssy/JevGO`，解释器 `/mnt/nas-tianwen/home/ssy/JevGO/.venv/bin/python`；worktree使用显式 `JEVGO_ASSETS_ROOT`，不用会切回主项目的env.sh。读取AGENTS及autoresearch/长期记忆/release-layout；真实GPU前再读取gpu-access/autodl技能。原checkpoint/revision/精度、`learning_mode=none`保持。只读CPU准备不触发GPU、push或远端流程。

本卡分支/worktree 已创建并保留，实际基线与交付提交见文末；子Agent优先gpt-6-astra / ultra，按实际工具能力记录。可复用开发源码和小fixture进Git，运行产物独立保存于本卡worktree的 `.dev/artifacts/` 并在回收清单记录绝对路径/哈希。原P14证据只读；其他任务的notification目录不清理、不提交。必要GPU命令需在首次启动前补入本卡执行记录，包含实际硬件、模型路径、服务/客户端窗口、预算和唯一输出目录，不能将未实现的探针写成已经可运行。

交付：实现commit、受测commit、准确命令/退出码、CPU与真实模型验证范围、逐条件结果/费用/失败、已知局限和清理记录；主控审查后合入 `codex/autoresearch`。本地提交不自动push；确需AutoDL时按当前技能同步已审查代码与指定commit。所有任务遵循[修正总计划](P14-REMEDIATION.md)，通过、失败、未覆盖和资源阻塞分开报告。

## 本轮实际交付（2026-10-19）

9B TP2：262112+32 单请求，以及 8192+32 的四序列/四 HTTP；SemIf：8160+32 单请求，均有实际 token 证据。原生四路、八路、其他模型及最低显存未验证。

- 实际分支起点：`53f965515d538a34130f5081c555a378c6ddc4fc`；任务提交 → 集成提交：`7ebdd13db7cc73d8dcd973e28f0d0d377f89d9f0` → `60cfb75460b7f5a66eec132b36529da7ed8666f2`；`e2510f5499a551ca21d82779fc0fbe2ab2b5db4f` → `156a26146be4edb6a1ea61603be8990cbfd490aa`。
- 本卡阶段 CPU：17（最终修正后的定向集合） passed，退出码 0。最终统一代码 `9c88efce2a37a47571d30720e7d61dbabdc9582d` 的完整 HOST 集合为 527 passed，不与阶段/审查测试相加。
- 唯一运行根：`/mnt/nas-tianwen/home/ssy/JevGO/.dev/artifacts/P14-remediation-20261019-r1`；[原任务产物](../../../.dev/artifacts/P14-remediation-20261019-r1/harvest/P14h/)、[完整产物库存](../../../.dev/artifacts/P14-remediation-20261019-r1/artifact-inventory.json)记录源绝对路径、归档目标与 SHA，工作树保留。P14h 补存的尝试标记和 P14i 非审查产物位于 `harvest/inventory-supplement/`；尝试标记不单独证明模型已执行。独立审查由主控在活动审查结束后最终归档至 `harvest/P14i-review/`。机器总入口 `report.json`，逐版本映射 `implementation-commits.json`，独立 CPU 入口 `cpu-final-summary.json`。
- 真实条件清单、受测提交和命令见各批 `execution-record.json`；原始模型输入/输出、token IDs、事件、ledger/result 和失败均保留。全部 learning_mode=none，原权重/BF16；无训练、下载或 AutoDL。
- 综合结论、剩余范围、容量留痕局限与清理记录见 [本轮结果](P14-REMEDIATION-RESULTS.md)。原 P14 历史失败未覆盖；P15 与正式矩阵未启动。
