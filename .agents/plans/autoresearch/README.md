# JevGO autoresearch 实施总计划

## 目标与范围
冻结参数，贯通显式状态、LLM候选、JEV评估、真实执行、MCTS和终止。先MiniGrid，再ALFWorld、ScienceWorld、DiscoveryWorld；沿用当前三个AutoTrust checkpoint和SemIf。
默认 learning_mode=none；其他模式显式报未实现。首轮交付CPU算法/真实环境验收及GPU交接，不将mock、下载成功或小fixture当作模型效果。
用户已授权独立branch/worktree、并行agent、本地任务提交，以及开发源码入Git。GPU推理与完整效果评测待外部资源。

## 任务表
| 任务 | 环节 | 依赖 | 状态 |
| --- | --- | --- | --- |
| [P00 文档与开发基线](P00-baseline.md) | 协作 | 无 | planned |
| [P01 共享契约与账本](P01-core.md) | AR-0/4/5/7 | P00 | planned |
| [P02 MiniGrid适配](P02-minigrid.md) | AR-0/4 | P01 | planned |
| [P03 AutoTrust后端](P03-autotrust.md) | AR-2/5/6/7 | P01 | planned |
| [P04 SemIf后端](P04-semif.md) | AR-2/5/6/7 | P01 | planned |
| [P05 研究状态转换](P05-research.md) | AR-2/3/5/7 | P01 | planned |
| [P06 MCTS引擎](P06-mcts.md) | AR-1/6 | P01 | planned |
| [P07 运行器与四种方法](P07-runner.md) | AR-0至7 | P02,P03,P04,P05,P06 | planned |
| [P08 ALFWorld适配](P08-alfworld.md) | AR-0/4 | P01 | planned |
| [P09 ScienceWorld适配](P09-scienceworld.md) | AR-0/4 | P01 | planned |
| [P10 DiscoveryWorld适配](P10-discoveryworld.md) | AR-0/4 | P01 | planned |
| [P11 评测与分析](P11-evaluation.md) | EVAL | P07 | planned |
| [P12 集成验收与GPU交接](P12-integration.md) | EVAL | P08,P09,P10,P11 | planned |

## 依赖与调度
P00 → P01；P02/P03/P04/P05/P06/P08/P09/P10随后可独立开发。P07等待P02至P06；P11等待P07；P12等待P08至P11及传递依赖。
优先队列：P02/P03，P05/P06，P04/P08，P07/P09，P10/P11，P12。按依赖动态调度，不人为等待同组全部结束。
主控1＋实现2＋审查1；子agent首选gpt-6-astra ultra。各实现任务独立codex/ar-pXX-*与.dev/worktrees/PXX。共享契约、CLI、公共配置由owner改。
主控集成分支codex/autoresearch。记录base和交付SHA、命令与退出码；仅本地提交，不推送。

## 已固定算法
单智能体同步PUCT：P/N/W/Q，真实rollout回传，不交替符号。基础先验为候选采样频次，JEV开启后与归一化偏好等权混合。仅剔除非法候选，不因低分永久剪枝。
每边一个原子动作，根按访问数、Q、稳定顺序选择。smoke默认4候选、8模拟、树深4、rollout2、c_puct=1.5、gamma=1，属于验收参数。
预算预留真实提交额度。搜索、重放、失败、重试全部计费；恢复不回滚账本。原始证据追加，工作记忆可压缩，分支默认隔离；恢复句柄不进模型。
B0顺序LLM；B1 JEV重排贪心；B2无JEV MCTS；B3完整方法。候选/工作记忆/停止辅助可消融；客观成功独立。

## 目录与运行
正式代码在根jevgo包，以python -m jevgo运行，不依赖.dev。保留当前offline环境，worktree用主项目解释器和JEVGO_ASSETS_ROOT，不用会切回主项目的env.sh。
.dev/analysis、tests、tools和小型稳定fixtures源码进入Git；logs/artifacts/cache/tmp/worktrees忽略；发布仍排除整个.dev与.agents。

## 四层验收
1. 算法：状态、证据、PUCT、回传、预算和错误。
2. 环境：真实reset/step、成功语义、恢复能力实测。
3. 模型：指定checkpoint真实生成评分；硬件、配置、token及内部请求。
4. 研究：固定任务seed/权限/预算的对比与消融，报告成功率/成本，不预设增益。
当前交付前两层及后两层工具/交接。ALFWorld正式数据缺失，小fixture只作接口验收。GPU先9B再其他；资源不足不擅自量化/换模型。

## 外部交接
任务卡包含背景、分支/worktree/base SHA、可改范围、环境与资源、准确命令、验收和回收格式。GPU专用文档在P12用最终CLI填充。超出工具或长期资源能力时单列阻塞，其余任务继续。
训练、SemIf缓存优化、通用命令行实验环境进入后续任务池。
