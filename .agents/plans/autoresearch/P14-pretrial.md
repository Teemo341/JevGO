# P14 正式实验前预试（待执行）

## 需求与定位

主环节 EVAL，关联 AR-2 候选、AR-5 记忆及 AR-7 预算。输入为 P13 统一代码与已有失败证据，输出为可复现的上下文、输出预留、记忆和累计预算建议。只做开发预试，不冻结正式评测协议、不训练、不替换或量化 checkpoint。

前置条件：P13 组合 CPU/GPU 验收及自有进程清理完成；源代码基线 `89214798653679b143006c4ce0e99b1c36ad71f4`（后续只含文档的集成提交可一并继承）。开始时记录实际 HEAD 和实时 GPU 占用。

项目/共享资源：`/mnt/nas-tianwen/home/ssy/JevGO`；解释器为该目录 `.venv/bin/python`。读取 AGENTS、autoresearch、release-layout、gpu-access skill。使用现有冻结资源，不复制/重装环境。

## 已有证据

详细 [只读诊断](../../../.dev/artifacts/gpu-integration-20261015-ng8z3gda/pilot-notes.md) 和 [机器可读证据](../../../.dev/artifacts/gpu-integration-20261015-ng8z3gda/pilot-readonly.json) 含来源哈希、原事件与预试命令形状。本次没有执行以下预试。

- 9B 两个独立 smoke 默认 memory=32、窗口4096，矩阵 memory=4。旧 smoke 累计预算为3000请求/2000环境步，没有token硬上限；不能直接套用矩阵1000/1000预算后用更早停止宣称修复。
- B0旧诊断重放得到的服务错误只证明输入至少3841加256预留超4096；B3的4192+1是离线重建。两项原失败请求usage仍未知。
- SemIf两个无效候选运行均为B0，四个坏输出都在JSON解析阶段失败；三次达到256输出且未闭合，一次74-token修复也有语法错误。日志没有finish_reason，不能将全部失败归因为长度。
- SemIf旧16次运行的95.5%底层模型请求在搜索阶段，68.1%环境步为重放。逻辑generation/judge调用与内部请求数必须分开。

## 实施计划表

| 子任务 | 需求与实施大纲 | 验收标准 | 依赖 |
| --- | --- | --- | --- |
| P14a 9B上下文 | 独立保存候选4096/memory4和8192/memory32条件，generation输出预留256、9B judge保持原1 token，其余沿用旧smoke；先离线计算generation/judge/repair完整提示与输出余量，再做容量/协议检查和B0/B3有限闭环。8192是待验证候选。 | 没有静默截断；客户端/服务窗口一致；记录真实峰值、成本与终止分类；memory32条件覆盖旧失败边界或更早客观成功。新条件不能抹去旧失败。 | P13 |
| P14b SemIf格式 | 用已保存的四个坏请求保持prompt/seed不变，比较256与512输出；保存原输出及EOS/长度证据，严格校验JSON/schema/公开动作。若测简短提示，作为独立单因素条件。 | 每次尝试计费；完整合法候选才通过；74-token语法错单列；不通过人为补全残缺JSON伪造合法输出。必要短B0闭环采用同一开发任务seed0/1。 | P13 |
| P14c 搜索成本 | 格式问题明确后，SemIf短B3比较8与2次simulation，每次至多2步提交；固定候选数4、深4、rollout2、memory4和窗口。 | 分开报告每次提交对应的生成、各评分角色、实际请求和replay；提交前预算停止不能算闭环通过，不把短预试当效果对比。 | P14b |

P14a/P14b 可以并行，分别使用计划分支 `codex/ar-p14a-context`、`codex/ar-p14b-format` 和 `.dev/worktrees/P14a`、`.dev/worktrees/P14b`（尚未创建），从同一已验代码基线开出。gpt-6-astra / ultra 优先；主控分配不重叠GPU，P14c随后排队。

允许修改开发验证源码 `.dev/tools/`、`.dev/analysis/`、`.dev/tests/` 中各自独占文件及本轮开发配置；实际模型后端、ResearchPolicy或CLI如需变更，先单列修复需求和对应AR定位，由owner在独立分支完成。避免把一次预试配置直接改成默认配置。

## 执行与证据

每个条件新建 `.dev/artifacts/P14-<唯一标识>/`，保留配置、实际git版本、模型revision、设备/TP、服务和客户端命令、退出码、提示长度、原输出、账本、事件及清理记录。使用绝对解释器和 JEVGO_ASSETS_ROOT，不用会切回主项目的env.sh。

命令沿用 `GPU-HANDOFF.md` 的启动/model-check，以及诊断文档的现有 `python -m jevgo run --config ... --method ...`。运行CLI的上下文在YAML `model.max_model_len` 设置；run子命令没有同名命令行参数。启动前在任务卡填入实际新输出路径和GPU映射。

预试累计预算须显式声明：9B保留旧smoke限制以覆盖原边界；SemIf格式/成本短试的候选限制见诊断文档，执行前落入独立配置并记录。扩大单次窗口不会自动扩大累计预算。四模型声明262144上下文不是本机256K容量验收。

## 交接与完成条件

当前状态：ready_for_pretrial_not_executed；未执行、无结果commit。下一位执行者无需聊天历史，读取本卡、P13、诊断证据与GPU-HANDOFF后即可准备新条件。

完成时提交开发工具/必要修复及测试commit，回传每个条件的配置与结果索引，明确通过、失败或资源阻塞。P14只提出依据充分的参数候选；之后另立P15冻结多任务、多seed、开发/测试划分、主预算、失败分类及同基座对比协议，再执行正式矩阵。
