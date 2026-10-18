# P14 行为追查：重复 forward 与候选退化

2026-10-18。EVAL，主环节 AR-2，关联 AR-4 真实执行、AR-5 公开状态与记忆、AR-6 搜索回传、AR-7 成本。原推理代码为 `955df143e7e2db059cb870e8d4fed9e9741621a2`；本次启动 HEAD 为 `8d2e3b260f07c37cda1ecb6267b0ff1682d837a6`、分支 `codex/autoresearch`，两者 `jevgo/` 正式源码无差异。保留并发任务两个未跟踪 notification 目录。本次未改变正式代码、提示、模型、成功标准或 P14 条件；没有新增 GPU 推理。

结论：观测转换、动作映射、分支恢复、候选去重和 JEV 数值传递没有发现导致本次异常的实现错误。原始生成确实误读空间关系、反复提出 forward，JEV 也给碰墙候选高分；同时，当前公开反馈和候选覆盖设计不能纠正这种错误，搜索退化为单一路径。因此不能将其概括成“模型不行，代码没有问题”，也不能把 B3 运行完称为有效搜索。

## 1. 观测和动作执行没有错位

对 P14a 四条件的 **537 条 search/commit 转移、1988 条逻辑生成／评分请求、5350 份实际 HTTP 请求中的观测**，按原 seed、完整动作历史及环境期限，用已安装 MiniGrid 3.1.0 / Gymnasium 1.4.0 独立 CPU 重放，逐字段核对全部通过。原记录和本次诊断均保留 SHA256。

- 本任务起点 `(1,1)`，全局朝东 `direction=0`；两次 forward 后依次到 `(2,1)`、`(3,1)`。从第 3 次 forward 起，正前方是墙，位置和公开观测均不变。
- 4096/B3 与 8192/B0 都提交 20 次 forward，其中第 3–20 次原地碰墙；8192/B3 的 9 次提交中，第 3–9 次同样如此。
- 局部视图随朝向旋转，agent 始终在 `(x=3,y=6)`，前方为 `(3,5)`；“局部向上”与“全局朝东”同时正确。适配器按上游 `image[x,y]` 逐行打印，没有转置错误。
- 初始目标在局部 `(5,4)`，两步后在 `(5,6)`，位于右侧，并非正前方。
- 控制动作 `forward, forward, right, forward, forward` 在第 5 步客观成功；从该成功分支恢复后，原根再 forward 仍碰墙，恢复后的指纹、步数和观测一致。这个控制用于验证环境可达与恢复，**不是模型成功成绩，也未输入模型**。

全局坐标仅用于离线诊断；原模型请求只含公开局部观测。详见 [观测报告](../artifacts/P14-followup-20261018-behavior/observation/summary.json)、[逐转移重放](../artifacts/P14-followup-20261018-behavior/observation/transition-replay-audit.jsonl)、[坐标与可见性检查](../artifacts/P14-followup-20261018-behavior/observation/coordinate-and-visibility.json)。

## 2. 模型原文已经误读；不是解析器把转向删掉

4096/B3 第一次根候选（原 `events.jsonl:33`）写道：

> facing North (direction 0)
>
> at the bottom center (row 6, col 5)
>
> The green goal is visible directly ahead

这与同一请求明示 `0=east`、7 列视图底部中央及目标列位置不符。它混淆全局方位与局部朝向，同时把 agent 的列误读成目标所在列。8192/B0 原文还把 east 与 facing upward 解释为矛盾，说明相同误解在没有 JEV／MCTS 的顺序基线里也存在。

4096/B3 的 334 次生成共 1336 个样本，1315 个严格合法样本全部是 forward，另外 21 个先发生 JSON 失败。正式解析器没有从这些合法输出里丢弃 left 或 right。候选数 4 是“采样 4 份”，不是“保证 4 个不同动作”；合并同一动作后，334 次候选事件全只剩 forward。即使只保留一个动作，当前策略也不补采，只有零合法候选才尝试一次格式修复。

SemIf 的 env20/sim8 条件提供了反证：40 次候选事件中，35 次唯一 forward，5 次同时保留 forward 和 left；sim2 的 12 次均唯一 forward。不同合法动作确实能通过解析和去重。这 5 次双候选都出现在 greedy rollout 中，未在根节点形成选择；实际仍取 forward，left 没被执行。

底层采样实现也未发现“4 次重置为同一个 seed”：SemIf 每样本用 `request.seed+i`，固定 vLLM 0.31 的 parallel sampling 也对子样本加 index。保留限制：没有采集 GPU 每个子样本的实际 RNG 状态，不能据此宣称样本统计独立。上层 seed 仅由动作历史长度与 repair 次数决定，同深度分支会复用 seed，相邻深度的四样本 seed 集合也有重叠；这是采样相关性问题，尚不能证明它导致了全部 forward。

来源：`jevgo/research.py:145–193`；[候选审计](../artifacts/P14-followup-20261018-behavior/candidates/summary.json)保存原文、实际采样参数、正式解析复算和 distinct-action 计数。

## 3. JEV 确实高估碰墙候选，分数没有接反

4096/B3 第 3 次实际提交前，原候选假设仍为 “move one step north into the empty cell directly ahead”；同一公开视图的前方格已经是墙。原评分为 **0.928361**。随后第 3–20 次提交前，这些碰墙候选的 JEV 分数在 **0.913578–0.960503**。这些是模型合理性评分，不是成功概率。

独立审查从原 HTTP 响应的 false/true token logprob、head bias、temperature 重算概率，核对 option/token 顺序、adapter、候选 item 顺序及后处理。没有发现 true/false 反转、错取 token、分数错配或把另一分支状态交给 judge。SemIf 则从保存的 option logits 重算 softmax，并用实际 tokenizer/template 核对评分 prompt 哈希。

但评分没有纠正能力，原因分两层：

1. 候选只有一个时，频次归一化是 1，JEV 分数归一化也是 1，两者混合仍是 prior=1。CPU 调用实际 ResearchPolicy，把单候选分数设为 0、1e-9、0.01、0.5、1，结果都一样。当前 JEV 是候选内排序器，既不会补出转向，也没有拒绝全部候选后重新提议的机制。
2. 本轮所有实际搜索转移的环境 reward 都是 0，叶节点使用明确标注的 zero bootstrap，所以根 Q/W 一直为 0。所审四个 B3 条件的 33 次根搜索均只有 forward；增加 simulation 只是继续探索这条动作路径。31 次非预算中断搜索的候选／转移调用顺序、P/N/W/Q 和最终选择，均能由原记录驱动当前正式 MCTS 在 CPU 上精确重现；最后两次预算中断保持原证据，没有冒充完成相同重放。

SemIf 5 次双候选的评分为 forward `0.992423`、left `0.007577`，结合采样频次 3:1 后，先验为 `0.871211/0.128789`；greedy rollout 按先验取 forward，符合当前实现。

来源：`jevgo/research.py:198–208`、`jevgo/search/mcts.py`、`jevgo/runner.py:120–125`；[逐候选评分链](../artifacts/P14-followup-20261018-behavior/judge/9B4096-candidate-chain.json)、[正式 MCTS 重放](../artifacts/P14-followup-20261018-behavior/judge/search-replay.json)。

## 4. 已证实的实现局限和独立成本缺陷

| 项目 | 本轮证据与边界 |
| --- | --- |
| 输入表达容易混读 | 文本同时说明全局 compass 与局部 facing-up；数字 `data.image` 仍按 x-major 提供，却没有明确数组轴／编码 legend。转换正确，模型原文证明它没有正确理解；尚未用替代提示验证因果改善。 |
| 失败反馈没有直接公开 | adapter 正确记录碰墙 `action_success=false`，但 `public_state` 只公开 observation、动作历史及简化记忆，不公开此标志和完整 StepResult。模型能看到观测不变，却没有显式“本步未移动”。不能说完全没有反馈，也不能说模型看到了 action_success=false。 |
| 无有效动作多样性保障 | n=4 经同动作合并可退化为 1，剩余候选不足时不补采；相同动作的不同假设也只保留第一份。与当前契约一致，但不足以保证有用搜索。 |
| 单候选 JEV 无法改变动作 | 分数再低也归一化为 prior=1；JEV 不生成新动作，当前策略没有全候选失败后的重新提议机制。 |
| 记忆评分在本配置下不改变取舍 | `pool=records[-max(memory_records,candidate_count):]`。memory4/32 且候选4时，pool 长度不大于保留上限，按分数选完仍把整个 pool 按时间顺序保留，与末尾截取相同。所有分数都不能改变这次选择，但逐条评分仍计费。这是独立的 AR-5 实效／成本缺陷，不能当成已验证的 JEV 记忆筛选。 |
| stop 只记录意见 | 当前仅保存 completion advice，且该类记录不进工作记忆，不用于动作选择或实际停止。它没有阻止本次重复碰墙，也不能将其费用写成有效停止控制。 |

上述机制与长期记忆中的首版约定核对；不以修改算法、隐藏状态、成功判据或人工补动作消除旧失败。重复 JSON 键等 P14 已知解析缺口仍独立保留，不是本次 9B 全 forward 的来源。

## 5. 对下一步的约束

本轮已定位行为链，足以反对直接将 P14 B3 视作有效算法评测。进入正式效果比较前，应先在开发任务分别验证：仅由公开观测生成的明确局部坐标／动作含义及前后差分；固定状态下包含 forward/left/right 的 judge 对照；候选不足与低效重复动作的可记录策略；真正有取舍的记忆评分池。各项应为独立条件，同基座 B0/B1/B2/B3 保持可见信息一致，模型与 learning_mode=none 不变。

这些是待验证建议，未证明改提示、扩大上下文、提高候选数、换 seed 或增加模拟次数能解决问题。本次没有更改提示／算法或新增模型费用，没有启动 P15、正式矩阵或训练。

## 6. 产物与复现

新产物根：`.dev/artifacts/P14-followup-20261018-behavior/`。分别保存 observation、candidates、judge 审计及机器摘要、原证据行号和 SHA256；旧 P14 目录未改写。主控完成链路整合；三名子 Agent 分工观测、候选和独立评分审查，分别沿用 gpt-6-astra / gpt-6.1-sol / gpt-6-astra、ultra。

独立审查通过；主控将四个脚本保存到 `.dev/analysis/p14_behavior_{observations,candidates,judge,search}.py`，改为显式新输出目录后重新完整执行，退出码均 0，结果与各 Agent 审计一致。检查覆盖 386 次生成／1544 个原样本、5167 项评分／1638 次逻辑 judge、2181 次分支公开状态对齐、31 次非中断 MCTS 完整重放及 5 个单候选分数反事实。549 次记忆选择中的 4078 项证据评分均未改变选择结果。检查粒度不同，不把这些计数相加当成独立实验数。

[机器结论](../artifacts/P14-followup-20261018-behavior/diagnosis.json)、[候选对比表](../artifacts/P14-followup-20261018-behavior/candidate-comparison.csv)、[精确复现命令](../artifacts/P14-followup-20261018-behavior/reproduce.sh)及[最终审查回执](../artifacts/P14-followup-20261018-behavior/judge/synthesis-review.json)均保留。运行命令要求原始 P14 产物、已安装环境与本地 tokenizer/head 文件；不加载模型权重。全部为 CPU 重放／静态证据计算，新增模型请求与 token 成本均为 0，GPU 与已关闭的 AutoDL 未重新启动。
