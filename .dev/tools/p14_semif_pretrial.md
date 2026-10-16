# P14b/P14c 开发探针

本工具仅观察和重放已有协议，不修改模型、ResearchPolicy、MCTS 或 CLI。默认 `learning_mode: none`，原 bfloat16 和模型 revision 均保留。GPU 必须由主控明确分配；不能在本机占用卡上启动，也不自行使用历史 SSH 地址。

CPU 准备（105，在 P14b 工作树执行）已生成四请求可传输 corpus、B0/B3 独立配置。`prepare` 要求目标 prepared/conditions 不存在，防止覆盖证据：

```bash
cd /mnt/nas-tianwen/home/ssy/JevGO/.dev/worktrees/P14b
/mnt/nas-tianwen/home/ssy/JevGO/.venv/bin/python -B .dev/tools/p14_semif_pretrial.py prepare \
  --evidence-root /mnt/nas-tianwen/home/ssy/JevGO/.dev/artifacts/gpu-20261014-w1d2iqko \
  --output-root /mnt/nas-tianwen/home/ssy/JevGO/.dev/artifacts/P14-20261017-pretrial
```

主控仅需同步 `.dev/artifacts/P14-20261017-pretrial/b/prepared/`、`b/conditions/`、`c/conditions/` 到远端相同相对路径。corpus 已内含四个完整请求、原输出、原 usage、公开动作及来源哈希/行号；不需要传全部历史日志。

以下是已完成 AutoDL 身份/环境/资源核验并分配一张卡之后的命令。设备选择由主控设置并单独保存 UUID、实时显存/进程和执行上下文。每次输出目录必须全新；运行命令、stdout/stderr 和 shell 退出码由外层主控保存。工具自身也写 provenance、版本、命令退出码、模型身份、原响应、token IDs 和账本。

```bash
cd /root/JevGO
/root/JevGO/.venv/bin/python -B .dev/tools/p14_semif_pretrial.py replay \
  --corpus /root/JevGO/.dev/artifacts/P14-20261017-pretrial/b/prepared/replay-manifest.json \
  --assets-root /root/JevGO \
  --output /root/JevGO/.dev/artifacts/P14-20261017-pretrial/b/replay \
  --execute
```

四请求 seed 原样保留为 **2/3/7/8**，n=1、temperature=0.7，其余采样默认通过现有后端执行并捕获有效 generation config。每请求比较256/512，共8次；不增加自适应修复。旧74-token坏repair单列，512能否修复须看新证据。EOS依据返回 token IDs 与模型有效 EOS IDs；Transformers 未提供 finish_reason 时明确为空，不能从长度或解码文本捏造它。严格校验只移除完整首尾 json fence，不补括号、引号、字段或动作。

若固定请求不足以判定可用的短闭环，再执行 B0 配置。下面示范 output256、seed0，另三条件分别替换 reserve=256/512、seed=0/1，所有短循环限6步、memory4、window4096，同100env/64req/100k输入/16k输出累计预算：

```bash
/root/JevGO/.venv/bin/python -B -m jevgo run \
  --config /root/JevGO/.dev/artifacts/P14-20261017-pretrial/b/conditions/output256-seed0-sim8.json \
  --assets-root /root/JevGO --method B0 \
  --output /root/JevGO/.dev/artifacts/P14-20261017-pretrial/b/loops/output256-seed0
```

P14b结论后再执行P14c。若256持续妨碍格式，依据b写明统一512的选择理由；两条c条件必须使用相同输出上限，只改变simulations。下面以256示范，sim8与sim2分别运行：

```bash
/root/JevGO/.venv/bin/python -B -m jevgo run \
  --config /root/JevGO/.dev/artifacts/P14-20261017-pretrial/c/conditions/output256-seed0-sim8.json \
  --assets-root /root/JevGO --method B3 \
  --output /root/JevGO/.dev/artifacts/P14-20261017-pretrial/c/loops/output256-sim8
/root/JevGO/.venv/bin/python -B .dev/analysis/p14_semif_costs.py \
  --run /root/JevGO/.dev/artifacts/P14-20261017-pretrial/c/loops/output256-sim8 \
  --output /root/JevGO/.dev/artifacts/P14-20261017-pretrial/c/loops/output256-sim8-cost.json
```

P14c固定4采样、树深4、rollout2、memory4、window4096、seed0，`run.max_steps=2` 最多2实际提交，同时显式 `episode.options.max_steps=20` 保留历史环境步数上限；两条条件同500env/500req/1M输入/100k输出累计预算。配置4采样不保证4个不同候选，实际去重候选数和逐item评分请求须从事件另查。

`factory` 默认将 `run.max_steps` 传作MiniGrid环境上限。首次P14c的env2/run2两臂因此截断搜索；原产物完整保留，单列为horizon2诊断，不能据此诊断树深4/rollout2。已执行的纠正条件env20/run2保存在独立 `horizon20-*` 目录，仍使用当时实际提交955df14；之后的开发工具修正仅使新的 `prepare` 自动生成env20/run2，不改写历史执行版本，也不要求重跑已执行条件。不要覆盖已有prepared/conditions目录。

分析逐条核对events/ledger/result，明确初始化验证、每实际提交和未完成tail；每段生成/各judge逻辑调用与内部请求、input/output、搜索环境操作、restore/replay单独报告。replay已含在env_steps，不能再加一次。预算挡住提交只算诊断，不能称闭环通过；两条短run不证明算法效果。`wall_seconds`仍非端到端时延。

CPU验证：

```bash
/mnt/nas-tianwen/home/ssy/JevGO/.venv/bin/python -B -m pytest -q -p no:cacheprovider .dev/tests/test_p14_semif_pretrial.py
```

日志和结果立即由主控取回105相同相对路径并核对哈希；本工具不管理SSH、prepare、monitor或关机。
