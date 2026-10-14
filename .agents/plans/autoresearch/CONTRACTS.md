# 共享接口约定 v1

P01唯一owner在派发下游任务前实现并冻结准确Python签名，保持本文件与core一致。

公共名称：EpisodeSpec、RunConfig、ResearchState、EvidenceRecord、Candidate、ExperimentPlan、StepResult、EnvironmentAdapter、GenerationBackend、JudgeBackend、SearchDomain、BudgetLedger、RunResult。

要求：JSON可序列化；状态新版本/证据来源；JEV意见与实际成功分离；opaque恢复句柄不进模型；同一总账覆盖全部分支和失败调用；不能恢复则拒绝MCTS；模型返回usage及身份；非none训练拒绝。

资源从JEVGO_ASSETS_ROOT解析，代码从当前worktree读取。
