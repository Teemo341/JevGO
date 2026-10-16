# P14 阶段结果：工具已验收，真实 GPU 预试待资源

日期：2026-10-17（Asia/Shanghai）。**P14 尚未完成**；当前状态 `tools_accepted_gpu_resource_blocked`。P14a 的四次真实闭环、P14b 的八次真实探针、P14c 的两组真实成本预试均未执行，不能以离线检查替代。尚不具备依据 P14 冻结 P15 参数的条件；未启动 P15 或正式矩阵。

## 版本、范围与工作区

- 实际启动 HEAD：`523ea16da07b8b5fa67d286c15df2a26220979b0`；长期集成分支 `codex/autoresearch`。未回退到历史受测 `8921479`。后续确有 AutoDL 实现变更；独立比较确认正式 `jevgo/`、`configs/`、`scripts/`、`pyproject.toml`、`uv.lock` 相对历史受测版无变化。
- 已审查、合入并为 AutoDL 同步的运行源码提交：`c9ed2eb389300cb227b8364bffc39361e8df6897`，GitHub 同名分支 SHA 已核对；正常 push，无强推。
- P14a 实现提交 `4975e994edb78cb88646c0682fd7a51bf2b126b1`；P14b/c 实现提交 `742d59616bbfb3954c7e02418cd31cc651c66482`。各自独立分支/worktree 保留。
- 启动时已有的 `.dev/tests/codex_notifications/`、`.dev/tools/codex_notifications/` 未跟踪目录保留，未纳入本任务提交。
- 定位 EVAL，关联 AR-2 候选、AR-5 记忆、AR-7 预算。模型冻结，`learning_mode=none`，原 checkpoint/revision/精度；未使用其他任务训练权重。未改正式后端、ResearchPolicy、MCTS、CLI、成功标准或默认配置。
- 两名实现和一名独立审查按四槽执行。P14a/审查为 gpt-6-astra / ultra，P14b/c 为 gpt-6.1-sol / ultra。

## 已实现与通过的检查

| 能力 | 验收证据与范围 |
| --- | --- |
| 完整请求 token 检查 | 使用正式 AutoTrust payload builder 与本地真实 tokenizer，涵盖 system/chat template、generation、repair 后缀、逐项 JEV bare-v1 提示和输出预留；不截断提示。 |
| 运行期窗口与 usage 审计 | 要求服务/client max_model_len 明确一致；保存原 HTTP 请求/响应；配对预检/结束记录，独立核对真实请求数、raw usage、事件、ledger、result、预算状态。离线长度不能填补未知 usage。 |
| 旧失败边界验收 | memory32 需要 B0 generation 或 B3 evidence 成功处理同一请求中至少9动作/19记忆，或更早客观成功；预算先停不算覆盖。 |
| SemIf 可复用固定请求重放 | 四个原请求的 prompt、seed、n、temperature 和其余参数保留，256/512仅变上限；保存 raw text、响应、token IDs、有效 EOS 配置，finish_reason 没提供就留空。 |
| 严格格式与动作校验 | 精确三字段 schema、Action 字段与公开 Action.key 合法性；只剥离完整外层 fence，不补残缺 JSON。原74-token语法错单列。 |
| 搜索成本分析 | 初始化恢复验证、逐实际提交、未完成尾段分别核算；generation、candidate/evidence/stop、搜索环境操作、restore/replay 分开；按请求区间核对逻辑调用与内部请求，保留预算预拒绝的0实际请求。 |

测试：原 AutoTrust/SemIf 后端 **116 passed，82.10 s，rc0**；集成提交上的新增工具 **33 passed，11.39 s，rc0**（P14a17、P14b/c16）。独立审查又分别复跑17/16项；这些复跑不与33项相加。工具测试包含真实 CPU MiniGrid + mock HTTP 的20步审计链，不属于真实模型成功证据。

独立审查认可两个实现提交，无剩余执行前代码阻断；[审查报告](../../../.dev/artifacts/P14-20261017-pretrial/review/tool-review-final.md)。28个历史来源哈希全部吻合。

## P14a 离线发现

最终离线检查3598条长度记录；包含重复窗口、反事实记忆、修复探针及展开的 judge items，**不能按记录条数计真实模型请求或累计成本**。

| 请求 | 输入 + 输出预留 | 4096余量 | 8192余量 | 证据性质 |
| --- | ---: | ---: | ---: | --- |
| 原 B0 失败 generation | 4008 + 256 | -168 | 3928 | 当前固定 tokenizer/template 的完整离线重建；旧服务只给至少3841下界，原失败usage仍未知。 |
| 原 B3 首个失败 evidence item | 4192 + 1 | -97 | 3999 | 与历史离线诊断一致；不改写成服务返回usage。 |
| 历史 generation 加 repair 后缀峰值 | 4022 + 256 | -182 | 3914 | 合成格式修复长度探针，不是新增真实请求。 |
| 展开全部历史 evidence items 峰值 | 4199 + 1 | -104 | 3992 | 包含旧失败后未发送items；4096下10条超限不等于10次HTTP400。 |
| memory4 反事实样本最紧 evidence | 2120 + 1 | 1975 | 6071 | 按后缀保留记忆的长度探针，不是实际JEV筛选结果或新轨迹。 |

8192容纳这批历史完整请求与repair样本；4096容纳这批memory4反事实样本。两者都不保证新轨迹始终可容纳，不证明8192的显存容量、服务协议或闭环通过。新工具会在真实执行时逐请求检查。

[最终离线摘要](../../../.dev/artifacts/P14-20261017-pretrial/a/offline-final/summary.json)、[配置与完整命令索引](../../../.dev/artifacts/P14-20261017-pretrial/a/handoff.json)。

## P14b/P14c 状态与配置

| 项目 | 已准备条件 | 实际状态 |
| --- | --- | --- |
| P14a | 4096/memory4、8192/memory32，各B0/B3；seed0，gen256/judge1；旧smoke3000请求/2000环境步，无token硬上限 | 四个条件未运行，GPU资源阻塞；memory32原边界未验证。 |
| P14b | 四坏请求 × 输出256/512；真实请求seed2/3/7/8，n1、temperature0.7 | 八探针未运行；512不能认定修复。 |
| P14b可选短B0 | seeds0/1，max_steps6，memory4/window4096；100环境步/64请求/10万输入/1.6万输出；256/512同预算 | 配置已备，是否需要由八探针结论决定。 |
| P14c | B3 seed0，max_steps2，sim8/2；候选4/树4/rollout2/memory4/window4096；500环境步/500请求/100万输入/10万输出 | 等待P14b确定同一生成条件；两组未运行。若统一512需先记录b证据和偏离原因。 |

四个历史坏样本均严格重现JSON解析失败，原输出tokens为256/256/256/74。原EOS与finish_reason未知；74-token修复错误不能归因于上限。没有新增输出，所以本轮没有“512通过”或格式失败率改善的结论。

旧 full B3 seed0 的998个实际请求已由新成本工具重算；初始化、每次提交、尾段总数一致，两个请求事件因预算预检而0收费被单列。这是**旧日志审计**，不代替新sim8/2成本比较，也不支持算法效果判断。

## 成本、失败、资源阻塞与未验证

**本轮真实模型请求0、真实输入/输出tokens均0**；没有尝试GPU推理，因此也没有新增模型实验失败。CPU fixture的模拟账本和旧日志重算不计入新模型成本。

旧失败完整保留：9B B0尝试11请求、已知输入/输出22891/1408；B3尝试1079请求、已知3256590/33689。两次最终HTTP失败usage仍未知，数值仅为已知下界。旧SemIf四次格式失败及所有原输出保留。

本机沙箱 nvidia-smi rc9，必要主机侧复核rc0；现有PyTorch2.13.0+cu129、CUDA12.9可见4卡。两次主机准入检查均为4张A40全部被其他用户进程占用，最后每卡约6060MiB剩余，不满足本轮4B/9B至少一张空闲卡要求。未终止其他任务，未把沙箱隔离判为驱动故障。

按现行AutoDL技能，已完成可公开源码审查/提交/push并申请本轮免密SSH。当前没有本轮有效端点，未连接历史实例。建议复用已保存基础镜像与同一模型NAS、1张96GiB卡顺序运行9B/4B；8192仍须实测容量。预留约5GiB系统盘小日志、10GiB高速缓存；不训练、不输出新权重、不下载替代模型。23个输入文件约5.5MB已逐个生成SHA256清单。

尚未验证：服务实际窗口/显存峰值、四次9B真实闭环和旧边界、八次SemIf格式探针、必要B0短闭环、P14c真实逐提交成本、真实usage完整性及远端清理。这些项目保持未完成。

## 参数候选、续跑与交接

当前只保留**待GPU验证的候选**：9B8192/memory32与4096/memory4（gen256/judge1）。SemIf512仅为待比较输出上限，不能推荐为已验证修复；sim2/8尚无本轮成本依据。不得据此冻结正式协议。

收到有效SSH后，先核实AutoDL身份、GPU、NAS、branch/commit与环境，按现行远端入口prepare/monitor；先容量/协议，再执行有限闭环。只有一张卡时顺序运行；每条件新目录，退出码与结果分别判定，预算停止不自动中断其他独立条件。取回小文件时按相对路径和哈希核对，再做独立结果审查与报告更新，按技能清理/关机。绝不在105运行生命周期脚本或关机。

- [P14任务状态](P14-pretrial.md)
- [机器可读阶段报告](../../../.dev/artifacts/P14-20261017-pretrial/report.json)
- [条件对比表](../../../.dev/artifacts/P14-20261017-pretrial/condition-comparison.csv)
- [远端明确输入文件与哈希](../../../.dev/artifacts/P14-20261017-pretrial/remote-input-manifest.json)
- [P14a复现说明](../../../.dev/tools/p14_context_README.md)
- [P14b/c复现说明](../../../.dev/tools/p14_semif_pretrial.md)

当前清理状态：本轮未启动任何GPU服务/模型进程，无自有GPU资源待清理；未连接AutoDL、未发关机命令；其他用户进程保留。此阶段报告不是P14完成交接；P15与正式benchmark继续留给主对话后续决定。
