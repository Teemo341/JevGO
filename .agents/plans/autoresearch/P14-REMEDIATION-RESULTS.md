# P14d–P14i 实施、有限验证与交接

2026-10-19。起点 `53f965515d538a34130f5081c555a378c6ddc4fc`，集成分支 `codex/autoresearch`。本轮完成状态与 P15 准入结论：`complete_with_negative_results`；可开始 P15 协议冻结，范围限本轮已测 9B/SemIf 条件，保留负结果与容量限制。原 P14 仍为 `executed_with_failures`，旧结果未覆盖。没有启动 P15、正式 benchmark、训练、量化或权重下载。

唯一运行根：`/mnt/nas-tianwen/home/ssy/JevGO/.dev/artifacts/P14-remediation-20261019-r1`。机器入口 `report.json`；完整条件与费用 `policy-summary.json`；分批聚合 `analysis/aggregate-results.json`；独立审查 `harvest/P14i-review/`；最终版本与产物哈希见 `finalization.json` 和 `artifact-sha256.json`。文档提交自身的 SHA 由提交后机器记录给出，避免文件自引用。

## 实现与版本

`research.py` 按 d→e→f→g 独占修改、审查后串行集成。P14h 的工具和主控 P14i 工具并行开发。使用一个主控、两名实现和一名独立审查，最大四个槽位；protocol 为 `gpt-6.1-sol / ultra`，capacity 与 review 为 `gpt-6-astra / ultra`。未使用不支持 ultra 的模型，未启动二级 Agent。

| 任务 | 原任务提交 | 集成提交 | 实际范围 |
| --- | --- | --- | --- |
| P14d | `67edaa9752eb7d1bcb6817ade9afc68141f98c38` | `a16f93e` | 严格生成入口；generic/bounded repair 分离；CPU 与单份历史坏请求真实配对 |
| P14e | `7ae4b07f886a947679e102f9d2b73c8d2c3a06e6` | `c90304d` | legacy/E1/E1+E2；公开信息边界和 no_memory 即时反馈；真实固定状态对照 |
| P14f | `c8fe0fb4413c5b66806d502033beaf3a45af76ad` | `48b367f` | 有界补采、频次先验、来源和访问边遥测；固定候选 JEV 探针独立于在线生成 |
| P14g | `86ddfac4ae8a1fc87b4187f0e4ad6aafab0582c2` | `9c88efc` | G0 跳过无取舍评分；G1 同池 judge/tail；stop off/advisory；实际下游记忆反事实 |
| P14h | `7ebdd13db7cc73d8dcd973e28f0d0d377f89d9f0`、`e2510f5` | `60cfb75`、`156a261` | 模板/token IDs、真实输出、HTTP/序列并发及失败 usage 的容量证据工具 |
| P14i | `62f13da`、`46bc7f2`、`26316fe`、`1bcef8e`、`52152d6`、`462f248`、`d959862`、`7465d25` | `16d82d5` 至 `11c92c2`（见机器映射） | 有限清单执行、原始请求与分词捕获、独立成本重算、组合验收与交接 |

最终运行代码为 `9c88efce2a37a47571d30720e7d61dbabdc9582d`。后续只更新说明与计划，不把文档 HEAD 冒充真实推理版本。各任务分支/worktree 保留，产物已回收，未触及两个无关的 `codex_notifications` 未跟踪目录。

| 真实批次 | 实际客户端/执行器提交 |
| --- | --- |
| d repair 配对、两模型 G0 旧条件 | `9cb83ba7aaed` |
| 两模型 E/JEV 诊断；9B 首次本地预算拒绝及修正清单 | `65dee504dca6` |
| SemIf 8192 容量 | `48b367f5e4a8` |
| 两模型 f/G0/G1 机制对照及组合 | `9c88efce2a37` |
| 9B 原生单请求容量 | `156a26146be4` |
| 9B 四 HTTP、n=4 容量 | `c90304d` |

9B 服务在 `53f965515d538a34130f5081c555a378c6ddc4fc` 启动，服务/模型后端源码本轮未改；后续客户端版本和实际文件 SHA 各自保存在执行记录。主控分析使用最终代码中的 summary 工具，独立分析脚本另附 SHA。 四 HTTP/n4 的 runtime metadata 继承了准备时的 `probe_executor_commit=156a261` 字段；实际 CLI 执行版本为 `c90304d`，两版本容量工具 SHA 相同，以 CLI receipt 与实际源码 SHA 判定执行身份。

## CPU、环境与发布边界

最终完整 HOST CPU 集合在 `9c88efce2a37a47571d30720e7d61dbabdc9582d` 上 **527 passed，143.30 s，退出码 0，0 failure/error/skip**，104 份受测源码 SHA 未变。集合为 `.dev/tests/autoresearch` 加 `.dev/tests/test_p14_semif_pretrial.py`，包括真实可用环境、mock/受控模型与发布隔离；准确 argv/env、JUnit 和输出见 `cpu-final-host-command.json`、`cpu-final-host-result.json`、`cpu-final-host.xml`。这一个完整集合是本轮主计数，不加各任务/独立审查的重叠测试数。

第一次同集合在默认沙箱中因禁止本机 socket bind，卡在 ScienceWorld Java 网关端口读取。只中断了已核实的本轮 pytest；中断还暴露 fixture finally 的 `None.poll()` 次生错误，原运行 rc=2、439 passed 和 3 setup errors/部分 JUnit 原样保留，明确 `sandbox_blocked`，不冒充产品失败或完整通过。只读主机复核证明同一 socket bind 可用后，以 600 s 上限在 HOST 运行上述完整集合并通过；无源码修补或测试屏蔽。

这些测试支持严格协议、公开信息边界、原始证据/分支隔离、真实下一次请求中的记忆、PUCT/回传、预算预留、失败和 stop 间接影响。受控 CPU 多候选与成功 fixture 不能代替真实模型机制收益。

发布验证是从受测 HEAD 的 Git archive 排除 `.dev`、`.agents` 后，用现有解释器/资源运行导入、CLI 及真实 MiniGrid 的 B0/B3 fixture；同时检查 shell 语法和开发目录依赖。没有重装环境，未执行 bootstrap 安装，不能称全新离线安装验证。准确子命令、退出码和发布副本保存在 `harvest/release/`。

## 单因素结果

所有真实调用均保持 `learning_mode=none`、原 checkpoint/revision/BF16。9B 为 `autotrust/JEV-9B` revision `7ac669b8fb476ba926491e3d1c39f9330d68dbed`，生成用基座、评分用原 decision adapter；SemIf 为 Qwen3.5-4B revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`，direct scorer revision `23cf1f39fc9534fe81437200959b6dfc7106e45a`。完整身份、adapter/calibration SHA 在每批 `model-identity.json`。

- **d**：正式入口拒绝历史重复键样本和全部受测协议违例；部分合法候选保留，坏输出不推进环境。真实修复对照使用一份已用于开发的历史 SemIf 坏请求，仅改变 repair 后缀，seed=5、输出上限=256。generic 为 1 请求、1545 输入/256 输出，非法；bounded 为 1 请求、1771 输入/115 输出，合法 forward。历史初始请求没有重跑，其旧费用不混入新费用；一对样本不证明普遍修复增益，更不证明 512 有效。
- **e**：每模型 3 种表达×3 个固定状态，每条件 n=4。SemIf 合法样本 legacy/E1/E1+E2 为 12/12、8/12、7/12，全部合法动作 forward。9B 为 10/12、12/12、11/12；仅 E1+E2 的 forward2 状态出现一个 left，其他合法样本为 forward。模型仍会在公开前方墙格上提出 forward；不能声称表达修正改善了避障。公开未变化与未移动没有等同；私有状态没有用于提示或纠正动作。
- **固定候选 JEV**：测试者提供 forward/left/right，正反顺序分别评分，没有注入在线策略。SemIf 18 次最高分均 forward，9 对完整排名全部变化；9B 旧表达的 3 对最高分由 right 变 forward，E1/E1+E2 的 12 次最高分均 forward。最大单动作分差分别 0.179665/0.381943。是单次顺序敏感性观察，不是总体统计估计或校准成功概率。
- **f**：两个后端、两个固定状态均从 4 增至 8 个实际样本，仍只有 forward 一种动作。9B 合法数分别 4→8、3→6；SemIf 均 3→6。补采真实增加一轮生成；每轮最多两调用/八样本，补采不再追加 repair。目标覆盖没有达到，未硬塞转向或低分删边。
- **G0**：同配置真实 B3 两提交复测，两个模型均保持 forward/forward，仍 max_steps、未成功。9B 64→26 请求、109917→38769 输入、4027→3812 输出；SemIf 88→50 请求、152923→76485 输入、5446 输出不变。各自 evidence 逻辑评分 10→0、物理请求 38→0，candidate 8 和 stop 10 次调用不变。旧新实际生成请求逐条相等；SemIf 响应也相等，9B 7/8 生成调用响应变化，因此仅报告重测费用，不简单从旧总量扣除评分。
- **G1**：固定相同 K=6/M=4 池、相同原始证据 ID，tail 保留 e3/e4/e5/e6，两个后端的 judge 均保留 e2/e3/e4/e6，实际下一次生成请求中的完整记忆随之改变；原始档案不删。9B tail/judge 为 1/7 请求、1521/14636 输入、199/91 输出；SemIf 为 2/7 请求、3056/15305 输入、512/147 输出。SemIf tail 初始与唯一 repair 均非法，保留 InvalidCandidate；其余三条件产生合法 forward。这支持选择已接入下游，不是任务收益证明。
- **stop**：CPU 验证高意见分不能制造成功、费用会改变剩余预算、错误沿原路径传播且保留已提交动作/证据。G0 实测仍开启 advisory；组合条件显式关闭。未新增阈值控制，不能称停止控制有效。

单因素原始证据在组合开始前保存。每个条件最多一次正式尝试，没有换 seed 或追到成功；下述预算清单修正和保留失败单列。相同 prompt/seed 不保证后端每次生成文本完全相同，成本使用真实重测值。补采沿用原 seed+attempt 规则；SemIf 批内 seed 偏移与补采存在部分重叠，八样本不作为八次独立实验，本轮没有另行改写种子协议。状态多样性只有一个 Empty-5x5 seed-0 布局、3 个连续研究状态、2 种当前公开观测；这些已用开发条件不构成独立测试集。私有几何审计独立保存，没有进入模型请求。

## 有限 B0–B3 组合

每个基座 4 个方法，各一次。共同 Empty-5x5 seed=0、环境期限 20、真实提交上限 6、环境步预算 500、模型请求预算 256、输入 2000000、输出 100000；E1+E2、bounded repair/coverage、M=4/K=6 history_tail，stop off。配置文件除方法/id 外逐字段相同，`prepared/combined-fairness-check.json` 保存核对。B0 实际单样本，其余初始最多 4、补采最多 4；仅 B1/B3 启用 JEV、仅 B2/B3 启用搜索；B0 单样本且不补采，这些方法差异明确保留。比较限同一基座；SemIf 多样本内部串行调用与 9B n=4 共享提示不能按一次包装调用混为相同成本。

| 基座 | 方法 | 真实提交动作 | 终止 | 物理请求 | 输入 / 输出 token | 环境步 / 重放 |
| --- | --- | --- | --- | --- | --- | --- |
| 9B | B0 | F / F / L / F / F / F | max_steps | 7 | 14718 / 1193 | 8 / 1 |
| 9B | B1 | F / F / F / F / F / F | max_steps | 46 | 106061 / 8156 | 8 / 1 |
| 9B | B2 | F / F / L / F / F / F | max_steps | 48 | 98732 / 32315 | 68 / 37 |
| 9B | B3 | F / F / F / F / F | budget_exhausted | 173 | 438095 / 27224 | 51 / 26 |
| SemIf | B0 | F / F | invalid_candidates | 5 | 10750 / 1022 | 4 / 1 |
| SemIf | B1 | F / F / F / F / F / F | max_steps | 83 | 186400 / 8811 | 8 / 1 |
| SemIf | B2 | F / F / F / F / F / F | max_steps | 188 | 381568 / 35621 | 68 / 37 |
| SemIf | B3 | F / F / F / F | budget_exhausted | 256 | 587007 / 25393 | 38 / 17 |

F=forward，L=left；八条件客观成功均为 false。环境步包含能力验证、搜索和恢复重放，重放列是其中的子计数，不能再相加；真实提交数只取动作列。生成/评分逻辑调用、各角色/阶段费用及操作累计时间见 `analysis/combined-analysis.tsv` 和 JSON。

9B B2 的 24 组候选中 21 组单例；六个根搜索有一次候选边 2，但实际仍只访问 1 条（left 的 P=2/3、N=2，forward 的 P=1/3、N=0），所有根 Q=0。SemIf B2 为 24 组中 23 组单例，最后一组 rollout 初始已有 left/forward，所以实际 24 initial+23 coverage=47 逻辑调用、188 样本；六个根均 1 边/1 访问、Q=0。该 rollout 双动作不冒充根搜索比较。

9B B3 的 19 组候选全为单例，五个根均 1/1、Q=0。SemIf B3 有 16 组完成候选、15 组单例，另有 1 组在评分预算拒绝时未完成；双动作只出现在 rollout。四个正常完成根均 1/1、Q=0，末次根因预算拒绝为 0/0。所有 34 次逻辑生成均完整返回四样本（136 文本、78 合法），最后失败不是部分生成。

组合记忆确实发生取舍：两模型 B1 各 5 次完成评分且均不同于同池 tail，其中各 4 次被后续生成消费；9B B3 为 20 次完成且改变、11 次被后续生成消费，另有 2 次评分预留拒绝；SemIf B3 为 18 次完成且改变、10 次被后续生成消费。这些统计只证明接线和实际选择，不证明任务收益。stop 在八条件中均关闭。

9B B3 保留第五次提交后，search 与 commit 的两次六项 evidence 评分预留均拒绝：实际输入 438095 + 6×262144 = 2010959 > 2000000。两次均未发出推理 HTTP；这是保守窗口上界的累计预算准入，不是实际超窗或已花费 token 超额。SemIf B3 已实际使用 256 个物理请求，末次候选评分还需 1 个请求、剩余 0，预检拒绝且未 forward；四次提交和全部费用保留。没有事后抬高预算重跑。

真实模型结果与 CPU/mock 通过分开；搜索候选边数、实际访问边数、Q 和记忆取舍分别报告。仍退化的配置只能代表该具体配置的表现，不能证明 MCTS/记忆机制已充分检验，也不能据此否定长期方法。没有正式成功率、置信区间或算法增益结论。

## 容量与运行资源

本机四张 A40，每卡总量 46068 MiB，开始时约 45486 MiB 空闲；实际使用 GPU1/2 跑 9B TP=2，GPU3 跑 SemIf。driver 575.64.03，torch 2.13.0+cu129，vLLM 0.31.0+cu129，Transformers 5.17.0。9B 为 BF16/KV auto、max_num_seqs=8、max_num_batched_tokens=2048、显存利用率配置 0.85、chunked prefill 开启、prefix caching 关闭；模型/JIT 已预热。SemIf 保留当前本地参考实现 fallback，未补装加速内核。完整命令、版本、采样显存及服务日志已保存；不与历史其他硬件做速度归因。

| 条件 | 实际输入/输出 | 实际并发证据 | 结论 |
| --- | --- | --- | --- |
| 9B 原生窗口单请求 | 262112 + 32 | 1 HTTP/1 scheduler sequence | 总实际长度 262144，通过 |
| 9B 四 HTTP | 各 8192 + 32 | 重叠 HTTP=4 且同期 scheduler sequence=4 | 通过所测短输入形状 |
| 9B 一 HTTP、n=4 | 8192 共享输入、4×32 输出 | 1 HTTP、同期 scheduler sequence=4 | 通过，不能叫四 HTTP |
| SemIf 配置窗口 | 8160 + 32 | 单请求/单序列 | 总实际长度 8192，通过 |

9B 原生单请求端到端约 81.09 s，其中 HTTP 69.33 s、服务 prefill 67.84 s、decode 0.75 s；四 HTTP/n4 分别约 15.03/14.80 s。SemIf 容量条件端到端约 4.52 s（不含整批模型加载）。实际返回 token IDs 和 EOS/length 全部核对，绝不以 max_new_tokens 代替实际输出。9B scheduler 指标语义通过本地源码和 SHA 证明，与请求重叠窗口联合核对。

这些不证明最低显存、原生长度下四路并发、未测输入/输出组合或长上下文推理质量。配置 max_num_seqs=8 也不等于测过八路。SemIf 8192 不能继承 9B 的原生窗口结论；27B/VL 本轮未执行，未来纳入的窗口/并发需各自补证。原生容量通过后未执行备用降档条件。

前两条 9B 容量探针的具体 CLI 在工具调用中执行，但独立 CLI receipt 文件于执行后补存，明确 `recorded_after_execution=true`；请求清单、边界、模型/资源配置已预写。n4 和后续批次的解释器、脚本参数、设备环境及推理清单在模型加载前保存；普通批次未同时记录完整 shell/解释器前缀（如仅影响缓冲和字节码写入的 -u/-B），部分完整命令按工具调用事后转录到 `postlaunch-shell-command-receipts.json`。独立审查认为不影响已声明推理条件；此留痕局限保留，不倒签、不为此重跑模型。

## 成本、失败与审查

| 新批次 | 条件数 | 物理请求 | 输入 token | 输出 token |
| --- | --- | --- | --- | --- |
| 9b-combined | 4 | 274 | 657606 | 68888 |
| 9b-diagnostics | 27 | 0 | 0 | 0 |
| 9b-diagnostics-budget-corrected | 27 | 27 | 33393 | 5321 |
| 9b-g0-legacy | 1 | 64 | 109917 | 4027 |
| 9b-mechanisms | 7 | 40 | 62990 | 8213 |
| semif-capacity | 1 | 1 | 8160 | 32 |
| semif-combined | 4 | 532 | 1165725 | 70847 |
| semif-d-g0-legacy | 3 | 90 | 156239 | 5817 |
| semif-diagnostics | 27 | 54 | 66634 | 5978 |
| semif-mechanisms | 7 | 83 | 127102 | 9549 |
| h-native-single | 1 | 1 | 262112 | 32 |
| h-native-http4 | 1 | 4 | 32768 | 128 |
| h-native-n4 | 1 | 1 | 8192 | 128 |

**策略实验：1164 请求、2379606 输入、178640 输出；独立容量：7 请求、311232 输入、320 输出；全部新增模型推理合计：1171 请求、2690838 输入、178960 输出。** 全部实际 usage 已知，原始账本/响应与汇总逐项一致，未知量没有补零。

执行器清单为 108/108 完成，另有三个独立 9B 容量条件，共 111 个记录条件；108 中 80 个状态为 executed、28 个 execution_failed（原 9B 诊断预留拒绝 27 + SemIf G1 tail 1）。executed 表示条件执行完毕，不代表模型/任务成功；非法文本和 episode 的 invalid_candidates/budget_exhausted 仍在结果中。不同修正版本、固定状态诊断和容量条件不是 111 个独立任务样本。

策略账本另计 generation_calls=271、judge_calls=225、环境步=317、重放=141、rollouts=57，操作 wall_seconds 累计 5504.286 s。该时间不含所有模型加载/编排且各批存在并行，不当作总实验墙钟或等成本性能结果。包含加载/关闭的批次 end_to_end_seconds 在各 summary.json 保存；9B 常驻服务启动及 GPU 采样另留原始记录。模型推理计数不含监控/metadata HTTP、CPU fixture 或历史推理。

`generation_calls/judge_calls` 与实际 `model_requests` 分开，initial/repair/coverage、candidate/evidence/stop、search/commit/replay 全部保留账本；未知 usage 不填零。操作累计 wall_seconds 与包含加载/编排的批次端到端时间分别保存；CPU 验证也曾并行运行，时延仅为描述值，不能用于公平等时间预算或速度增益结论。

保留的失败包括：历史 P14；d generic 修复非法；E/JEV 诊断中的全部非法样本；9B 原诊断清单 27 条在本地输入预算预留阶段拒绝（0 推理 HTTP/0 模型 token），之后只将累计输入预算 250000 调为 2000000 执行一次相同条件；SemIf G1 tail 的初始与唯一 repair 非法；SemIf 组合 B0 末次 initial+repair 两条非法（全程 5 样本中 3 条非法、两次 repair）、9B B3 两次无费用输入预留拒绝、SemIf B3 一次无费用候选评分请求预算拒绝。原条件、修正理由和各自费用分别保存，不把已执行失败归为未执行或资源阻塞。

开发期间独立审查发现并修正：容量非 SSE 失败的已知 usage、输出目标判断、同期并发证据和畸形响应处理；执行器的预算覆盖和 bounded repair 来源标记；分析器的 result/ledger 对账、未完成批次、容量成本分组、直接 repair 样本计数和未知 usage。原审查/失败 fixture 保留。SemIf 容量 CPU 准备曾因 BatchEncoding 辅助比较错误失败，修正 helper 后通过，两次均未调用模型。沙箱隐藏主机 PID 曾导致主控误判首个 9B 服务已退出并启动重复进程；经主机复核，只终止核实过的重复进程树，保留原服务，纠正记录见 `h-native-service/controller-correction.json`，不能称按结果重启模型。

独立审查直接核查实现、测试、原始请求/响应、模板及 token IDs、模型分数重算、HTTP usage、ledger/result 和实际下一次记忆请求，没有只转述实现者报告。最终结论见 `harvest/P14i-review/`。

## P15 待冻结项与交接

可开始 P15 协议冻结，范围限本轮已测 9B/SemIf 条件，保留负结果与容量限制。确定的协议、信息边界、成本和方法语义修正已实现并独立复核；有限真实条件的负结果不阻止开始制定协议。此准入是开始冻结工作的依据，不代表协议已冻结、正式评测已准入所有模型/环境，或任何算法收益已被证实。

待后续调度决定：多任务/多 seed 的真实布局多样性和开发/测试划分；模型/模态及各自容量；主预算和硬上限（含 9B 保守窗口预留语义）；失败/repair/coverage 规则；同池记忆消融与 stop 意见费用开关；候选顺序与随机种子协议；搜索多边覆盖与未校准 JEV 的解释；统计口径和环境能力。ALFWorld 正式数据尚缺，DiscoveryWorld 当前仅 B0/B1 在线；CPU 成功不能扩展这些权限。若拟纳入更多模型或更长/更高并发条件，应先补相应验证，而非从本轮外推。

本轮只使用本机 GPU，没有 AutoDL 或远端结果待回收。9B 原服务及已核实重复进程树已退出，SemIf 最后批次正常 rc=0，GPU 采样器收到本轮 stop 文件后 rc=0；主机侧最终检查本轮残留进程为空、GPU compute apps 为空、四卡利用率均 0%，空闲约 45485/45486 MiB。已删除本轮创建的服务临时目录，环境、缓存、权重与其他任务进程未清理，未关闭本机。详细时间/命令/退出码及显存快照在 `resource-cleanup-final.json`、`service-cleanup.json` 和各 process-exit 记录。

本轮没有需要外部授权才能继续的实施项；未来上述研究扩展属于下一轮任务，不以未写代码作为外部阻塞。各 worktree 的有效产物已按路径和 SHA 回收，未删除工作树或其他任务文件。
