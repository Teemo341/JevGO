# P14 后续修正与最小验证总调度

2026-10-19。状态：complete_with_negative_results。P14d–i 已完成实现、独立审查、527 项最终完整 CPU 验证和两基座有限真实 GPU 验证；负结果与失败原样保留。综合交接见 [本轮结果](P14-REMEDIATION-RESULTS.md)。主环节EVAL，关联AR-2/3/4/5/6/7。长期目标仍是在冻结参数、可核算成本和公平条件下检验LLM+JEV+MCTS的贡献；不预设正收益。

## 已接收的证据与更正

- 当前审查基线 `305fed8c4a1698604eb5699a8afac898ffe4b3d5`。P14真实GPU受测版本是955df143e7e2db059cb870e8d4fed9e9741621a2，后续分析版本不能冒充推理版本。P14保持executed_with_failures；P15及正式矩阵未启动。
- [P14结果](P14-RESULTS.md)和[行为诊断](../../../.dev/analysis/P14-BEHAVIOR-DIAGNOSIS.md)表明：执行/观测/评分接线通过其已述审计，真实生成误读空间关系，搜索根退化为单动作；这既不是算法增益证据，也不足以判定方法在合理条件下无效。
- 严格JSON契约与无取舍记忆评分是确定的修正项；更好的公开表达、有界补采、不同记忆池和stop控制是需要独立验证的策略。stop现状是advisory，先明确语义，不强行添加阈值控制。
- 公开“action_success”不能直接搬进提示：MiniGrid实现由私有世界状态比较产生。仅用公开Observation构造“可见观测未变化”，不可普遍推断“未移动”。
- 8192只是旧失败回归的已测服务条件，非推荐上限或最优窗口。声明262144、服务启动、真实长输入decode、并发容量四层分别验证；预分配VRAM不能推断最低容量。

## 任务表

P14a/b/c保持原历史状态，新增六个后续子任务，不重编号已约定的P15。

| 子任务 | 需求与实施重点 | 主要验收 | 状态 |
| --- | --- | --- | --- |
| [P14d 严格候选协议](P14d-proposal-contract.md) | 统一重复键/字段/类型规则，修解析与改repair分开 | 坏输出不能直接执行，合法样本/失败费用可追溯 | complete_with_negative_results |
| [P14e 公开观测](P14e-public-observations.md) | 局部轴说明、仅可见差分分别对照 | 无隐藏信息泄漏、各方法信息一致 | complete_with_negative_results |
| [P14f 候选覆盖/JEV](P14f-candidate-coverage.md) | 有界补采与固定状态forward/转向评分探针分开 | 真实来源、费用、根可选边和排序实测 | complete_with_negative_results |
| [P14g 记忆/stop](P14g-memory-stop.md) | 跳过无取舍评分；有取舍选择独立比较；stop标advisory | 分数反事实改变下次提示，费用正确，无假成功 | complete_with_negative_results |
| [P14h 窗口/并发](P14h-context-capacity.md) | 原生窗口及真实工作负载容量分层验证 | 实际prefill/decode/并发证据，资源局限准确 | capacity_verified_with_stated_limits |
| [P14i 组合验收](P14i-readiness.md) | 合并后有限开发闭环及P15准入审查 | 实现与比较可信；不以必须成功或增益为门槛 | complete_with_negative_results |

每项独立文档含需求、实施大纲、验收、修改范围、branch/worktree、CPU命令和交接约定。GPU命令/设备/预算需在任务启动前落为实际可执行记录，不能提前假造资源和新工具CLI。

## 分工、依赖与公平性

主控+两名实现+一名审查，上限服从实际并发槽位。子Agent优先gpt-6-astra / ultra；不同任务独立branch/worktree/产物，GPU分配由主控按实时资源统一决定。

解析、候选、公开状态、记忆均涉及jevgo/research.py。即使逻辑独立，也由唯一owner按P14d→P14e→P14f→P14g顺序接线并合入，后任务从已合入commit继续；不让多个Agent同时写共享主文件。P14h的独立CPU工具、MiniGrid局部表达和固定状态JEV探针在接口冻结后可并行准备。实际执行按上述依赖推进；最终工作负载的真实请求与独立容量证据分别记录，P14i 在单因素证据保存后执行组合。

第一阶段执行 d 的契约修复及 h 的 CPU 准备；后续e/f/g分别保留单因素条件和自己的受测版本。跳过无效评分改变可用预算，补采增加调用，扩大记忆池增加评分与上下文，均需记录实际费用和方法语义，不能将多项一起修改后的成功归于某一个因素。环境期限与真实提交上限始终分开。

GPU按现行AGENTS/gpu-access/autodl分流；历史AutoDL授权与关机记录不等于当前实例仍可用或任何新实例已授权。代码/原始结果分开保存，旧未知usage不填零，未跟踪notification文件不纳入本任务。

## 与P15及长期目标的关系

可开始 P15 协议冻结，范围限本轮已测 9B/SemIf 条件，保留负结果与容量限制；本轮没有冻结协议或启动正式 benchmark。P14i准入要求是确定缺陷被修复或相关功能明确停用、条件可复现、费用可信、方法主张与实际行为一致、拟用容量有证据；**不要求先看到成功率提升**。如果模型仍失败，可以正式报告该冻结配置的负结果，但单动作和无取舍配置不足以代表相应机制已被充分检验。

正式数据和环境边界继续保留：ALFWorld仅fixture，DiscoveryWorld当前只可在线B0/B1。训练扩展、通用命令行环境和SemIf缓存优化不自动纳入这轮修正。本次不会把临时参数写成长期目标或默认配置。

完成本轮后，由主控回收commit、逐条件报告/费用/失败、独立审查与资源清理，更新本总表和各卡，再决定P15冻结范围。

## 实际完成与保留项

最终运行代码 `9c88efce2a37a47571d30720e7d61dbabdc9582d`；各任务提交和实际 GPU 执行版本见结果报告与 `implementation-commits.json`。默认沙箱 CPU 测试受 socket 权限阻塞后，保留部分记录并在主机完成同集合；主计数仅 527。各任务 worktree 与有效产物保留并回收，两个 notification 目录未动。

9B 与 SemIf 各 B0–B3 一次，八个组合条件均完成且无客观成功；SemIf B0 格式失败、两个 B3 预算耗尽均保留。容量仅支持报告内已测形状；其他模型/并发与正式任务多样性待 P15 讨论。完整成本、失败、未知 usage 和资源清理见唯一运行根的 `report.json`。
