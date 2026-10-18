# JevGO autoresearch

JevGO 第一版实现冻结参数的显式研究状态、LLM 候选、JEV 辅助评估、真实环境操作和单智能体 PUCT。模型参数不更新；`learning_mode` 目前仅接受 `none`，其他训练模式明确报错。

截至 2026-10-19，已完成首版 CPU 算法与环境验收，以及四个 checkpoint 的真实生成/JEV 评分开发验收；GPU修复已合入，统一代码版本8921479的四模型协议复核也已通过；27B/VL 使用原精度 TP=2，VL 协议包含真实图片。64 次开发矩阵仅覆盖一个 MiniGrid 任务和两个 seed，正式 benchmark 与算法效果验证尚未完成。9B 独立 B0/B3 smoke 因 4096 上下文超限而失败；矩阵采用不同记忆配置，不能覆盖这些失败。

2026-10-19 的后续修正已完成严格候选入口、公开观测选项、有界补采、无取舍记忆评分跳过与同池记忆选择；最终完整 CPU 集合 527 项通过，9B/SemIf 有限真实开发闭环的负结果保留。它们不代表正式 benchmark 或算法增益。

## 运行一个闭环

在已经安装环境的项目目录执行，输出目录必须尚不存在：

```bash
.venv/bin/python -m jevgo run \
  --config configs/autoresearch-smoke.yaml \
  --method B3 --output /tmp/jevgo-smoke-b3
```

`--method` 支持 B0（顺序 LLM）、B1（JEV 贪心重排）、B2（无 JEV 的 MCTS）、B3（JEV + MCTS）。示例使用明确标注的 MiniGrid 确定性 fixture；真实模型替换 `--model jev_9b` 等，并提供对应服务 `--base-url`。在独立工作树运行时用共享解释器，并传 `--assets-root /实际资源根目录` 或设置 `JEVGO_ASSETS_ROOT`。

每次运行保存 `config.json`、`versions.json`、`events.jsonl`、`ledger.json`、`result.json`。`result.json` 的 `final_state.success` 是环境客观信号；CLI 退出 0 只表示执行正常结束，不能代替读取 `final_state.success` 和 `termination_reason`。预算耗尽、环境终止、截断、模型错误分开记录。搜索 transition 和实际 commit 使用不同事件 phase，最终状态只包含提交轨迹证据。公开提示使用稳定的证据别名和图片占位符；原始 ID、真实图像路径及调用前请求均保留在日志中，失败请求也可追溯。

## 循环及搜索语义

1. 固定任务、seed、观测条件、预算，reset 得到初始显式状态。
2. LLM 输出带假设的原子动作 JSON；非法格式/公开合法动作集之外的提议被拒绝。最多一次格式修复，失败调用仍计费。
3. 相同动作去重，以采样频次形成先验。启用 JEV 时与归一化偏好等权混合，不按低分永久删除候选。B1 按 JEV 分数贪心选择。
4. B2/B3 使用 PUCT 的选择、扩展、真实 rollout 和回传；单智能体不翻转符号。树深与 rollout 深分别受限，终局奖励只在进入该状态的边计算一次。截断后额外价值为明确标注的零启发式，不冒充学习出的长期价值。
5. 搜索结束按根访问数、Q、稳定顺序选择行动，恢复实际根后再提交。预算中断时已观测回报标记 partial 并准确回传。
6. 原始观测追加保留，假设、推导、JEV 意见与实测事实分开。JEV 只选择已有证据进入工作记忆；失败和反证不会从原始档案删除。
7. 环境成功、失败、截断或预算停止；JEV 完成意见留作辅助，不改变客观成功。

默认验收参数为初始 4 个采样样本（B0 为 1 个）、8 次模拟、树深 4、rollout 深 2、`c_puct=1.5`、`gamma=1`。这些是功能验收参数，并非效果最优设置。

所有分支共享一个单调账本。搜索、真实提交、重放、reset、snapshot、观察刷新、失败模型请求及内部请求均计成本；缓存 observe 不收费。规划前预留恢复根状态、重放前缀、一次提交和后继快照额度。无法事先精确预测的操作耗时按实际值记录，超额后停止新增工作。

`wall_seconds` 是 adapter/backend 已插桩操作及去重后的 MCTS 搜索开销之和，不完整覆盖模型加载、编排、预处理、日志及清理，并非端到端时延。结果 metadata 的 `wall_seconds_scope` 明确此范围。当前四方法不按它比较总运行速度或等时间预算；配置了有限 `wall_seconds` 的矩阵不会被标为公平比较。正式按时间比较需先统一所有方法的计时边界。

## 候选、公开观测和记忆配置

以下配置位于 `run.options`，所有方法使用相同的生成文本解析契约：模型必须返回包含 `action`、`hypothesis`、`rationale` 的单个 JSON 对象；`action` 必须包含非空字符串 `name` 和对象 `arguments`。递归重复键、非有限数、缺失或额外字段、错误类型及公开合法动作集之外的动作均拒绝。支持完整的外层 JSON fence：忽略首尾空白后，开头必须为三个反引号、小写 `json` 紧接 LF 换行，结尾为三个反引号；不从坏输出补全或提取动作。未提供公开合法动作集时保持合法性未知，不推测环境动作。内部 `Candidate`/`Action` 构造默认值保持原有接口。

| 配置 | 默认值与含义 |
| --- | --- |
| `proposal_repair` | `generic`；可选 `bounded`，附有界坏样本和解析错误。仅零合法候选时修复一次，部分合法候选保留。 |
| `proposal_repair_text_chars` / `proposal_repair_error_chars` | `1024` / `256`；正整数，分别限制 bounded repair 引用的坏样本文本和解析原因字符数。 |
| `candidate_coverage` | `off`；可选 `bounded`。非 B0 初始有合法候选但动作种类不足时补采一次；初始全坏仅修复一次。补采后不再修复，最多两次生成调用、八个样本。B0 保持单样本加一次修复。 |
| `public_observation` | `legacy`；`e1` 添加 MiniGrid 局部数组轴和编码说明；`e1_e2` 另加相邻公开观测差分。未变化仅指公开观测相同，不等于没有移动；不读取私有 `action_success`。即时反馈在 `run.use_memory=false` 时仍可用。 |
| `memory_records` | `32`，最终工作记忆条数 M，不是 token 限制。 |
| `memory_pool_policy` | `legacy`，池容量为 `max(M,candidate_count)`；可选 `history_tail`，从当前分支原始档案取有界尾部，显式指定 `memory_pool_records=K>=M`，排除 judge opinion。 |
| `memory_selection` | `judge`；可选 `tail`，对同一个池保留最后 M 条。池不超过 M 时均跳过无取舍评分，保持原 ID 和顺序。评分选择使用稳定同分规则，原始证据和反证不删除。 |

所有非 B0 方法均要求 `run.candidate_count<=4`，包括 `candidate_coverage=off`；策略拒绝超限配置，不隐式截断。B0 的实际初始采样数为 1。

`run.judge_stop=false` 关闭停止意见；对启用 JEV 的 B1/B3，设为 true 时仅为 `advisory`；B0/B2 即使该标志为 true 仍关闭停止评分。这类意见没有阈值控制，不直接结束行动或搜索，也不能更改客观成功。停止评分的费用仍会消耗预算，后端错误仍可能按正常错误路径终止运行。结果的 `metadata.policy_mechanisms` 记录角色是否启用以及是否实际发生评分或取舍；准确费用以账本为准。

采样数、不同动作数、搜索根候选边数和实际访问边数分别记录。有界补采不保证得到多个动作，记忆发生选择也不证明任务收益。这些选项须在对比条件中显式保存，不能把开发设置视为最佳默认值。

## 模型

| backend | 生成与评分 | 验证状态 |
| --- | --- | --- |
| `jev_9b` | 同一 vLLM 服务的基座生成、LoRA decision head；客户端 logits/bias/温度路径 | 真实协议通过；A40 TP2 已测原生窗口单请求，旧 4096 上下文失败保留 |
| `jev_27b` | 基座生成及本地 `/v1/decide` | 原精度 TP=2 的真实生成/评分协议通过 |
| `jev_27b_vl` | 文本/图像生成及 typed decide；最大并发序列 8 | 原精度 TP=2 的真实图片协议通过；开发矩阵使用文本 |
| `semif` | 同一冻结 Qwen3.5-4B 实例的生成与 direct scorer | 真实 4B 生成/评分通过；开发矩阵仍有无效候选与预算失败 |

模型目录、基座及 revision 读取 `configs/jevgo.yaml`，不由简称猜测。AutoTrust 不自动启动服务；SemIf 在进程内加载本地 CUDA 权重。客户端不下载权重、不隐式重试或量化。SemIf 的偏好未校准；JEV 的候选、证据、停止及启发式状态评分语义分别记录。

服务启动入口先检查命令：

```bash
.venv/bin/python -m jevgo.models.serve_autotrust \
  --model jev_9b --assets-root "$PWD" --port 8009 --max-model-len 4096 --dry-run
```

GPU 主机移除 `--dry-run` 启动，再执行真实协议检查：

```bash
.venv/bin/python -m jevgo model-check \
  --model jev_9b --base-url http://127.0.0.1:8009 \
  --max-model-len 4096 --output /tmp/jevgo-9b-check
```

已验证的 A40 多卡启动示例（先检查实际可用卡并按需替换设备号）：

```bash
CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python -m jevgo.models.serve_autotrust \
  --model jev_27b --assets-root "$PWD" --port 8027 \
  --max-model-len 4096 --tensor-parallel-size 2
```

27B-VL 使用 `--model jev_27b_vl --port 8028`，同样显式设置 TP=2；其最大并发序列仍不超过 8。TP 默认 1，不会自动选择或增加设备。

VL 检查另传 `--image /绝对路径/公开观测.png`。检查包含生成与三个评估角色，保存请求、响应及成本；成功不代表 benchmark 效果。客户端和服务的 max_model_len 必须一致。

## 环境及能力边界

| environment | episode.task / options | 当前范围 |
| --- | --- | --- |
| `minigrid` | `MiniGrid-Empty-5x5-v0`；seed | 公开局部 7×7 文本观测，五步成功及分支恢复通过；当前不提供 VL 图像 |
| `alfworld` | 本地 `.tw-pddl` 路径，或 `options.game_file` | 四步小型 fixture 及 reset/replay 通过；正式数据尚缺 |
| `scienceworld` | `find-living-thing`；`variation: 0, simplifications: easy` | 真实 JVM 成功/失败/超限/重放通过；每次 reset 的内部 look 计费 |
| `discoveryworld` | `Tutorial`；`difficulty: Normal, modality: text` 或 `vl` | 独立进程、缓存观测、图像隔离通过；当前真实场景为 online-only |

DiscoveryWorld 新进程的私有状态/RNG 与原实例不一致，不能声称分支恢复可靠，因此当前 B2/B3 会明确拒绝。B0/B1 可在线运行。恢复成功路径仅经过协议 fixture 测试。未修改 benchmark 的随机机制来掩盖问题。

所有环境的 `options.verify_replay: false` 显式关闭恢复验证，只允许在线方法。默认验证也消耗预算，并记录 capability_verification。私有事实、全局地图、oracle scorecard 和恢复句柄不进入模型提示。

## 固定条件对比

CPU 验收矩阵包含两 seed、四方法和 B3 的四个单项消融：

```bash
.venv/bin/python -m jevgo.evaluation matrix \
  --config configs/autoresearch-cpu-matrix.yaml \
  --assets-root "$PWD" --output /tmp/jevgo-cpu-matrix --execute
.venv/bin/python -m jevgo.evaluation summarize \
  --manifest /tmp/jevgo-cpu-matrix/manifest.json \
  --output /tmp/jevgo-cpu-summary.json
```

`configs/autoresearch-matrix.yaml` 是四个真实模型的开发验证矩阵。默认命令只生成计划；显式 `--execute` 才执行。可用 `--model jev_9b` 选择已配置模型，按资源逐个验收。

分析重新加总事件中的成本，并核对 ledger/result、预算、任务和 seed。失败、未运行与跳过不算成功；成本未知时保留客观成功，但不算已证实的预算内成功。不同实际 backbone/revision/adapter、模态、环境与预算分开比较；报告覆盖率、成本、Wilson 区间及比较是否满足条件，不把不同环境的原始得分直接平均。

长期价值训练、策略微调、SemIf 缓存优化、通用命令行实验环境均为后续范围。

## 正式评测前的预试

4096/8192 是历史开发配置，不是推荐上限。四模型本地配置声明 262144 tokens。本轮 A40 TP2/BF16 的 9B 实测 262112 输入加 32 实际输出；四序列并发仅验证 8192 输入加每序列 32 输出，分别核对单 HTTP n=4 与四 HTTP。SemIf 单卡 BF16 实测 8160 输入加 32 输出，客户端配置窗口 8192。配置窗口、实际长度、并发和推理质量分别判断；不从预分配显存推断最低资源，也不外推原生四路或其他后端容量。扩大条件需另行验证。

历史 64 次开发矩阵每任务累计预算为 1000 次模型请求、1000 次环境步、200 万输入 tokens、20 万输出 tokens；搜索、评分和恢复均消耗预算。扩大单次窗口不会增加这些累计预算。本轮已复验 9B/SemIf 有限闭环并保留格式失败、预算终止与完整费用；后续可开始冻结多任务、多 seed 的正式协议，仍需确定任务划分、容量、预算和统计口径；同一生成基座内比较 B0–B3，模型与模态分组报告。
