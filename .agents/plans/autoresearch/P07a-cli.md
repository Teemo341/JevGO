# P07a CLI、工厂与确定性验收后端

## 需求

定位 AR-0～7 的装配与 EVAL 记录；把 P07 运行器接到统一入口，并提供显式标记的 CPU fixture。不是训练或真实模型效果任务。

前置：P01～P06；运行器 API 由 P07 owner 提供。共享契约 v1。基线 `8e35eb6`，追加 runner `7792aeb`、`1ac0c9b`。Agent 为 gpt-6-astra ultra。

## 实施大纲

- 分支 `codex/ar-p07a-cli`，工作树 `/mnt/nas-tianwen/home/ssy/JevGO/.dev/worktrees/P07a`。
- 只修改 `jevgo/factory.py`、`jevgo/__main__.py`、`jevgo/models/deterministic.py`、`configs/autoresearch-smoke.yaml`、`.dev/tests/autoresearch/test_cli.py`。
- `run_from_dict` 装配环境/模型，调用 `run_episode`；`model-check` 实际调用生成及三个评分角色，保存部分失败和费用。
- 构造器失败关闭已建资源；运行目录必须新建；真实模型标记 real_model_attempt，fixture 标记 cpu_fixture。
- 资源根 `/mnt/nas-tianwen/home/ssy/JevGO`，解释器为该根 `.venv/bin/python`；不调用 env.sh 切回主工作区。

## 验收

```bash
cd /mnt/nas-tianwen/home/ssy/JevGO/.dev/worktrees/P07a
JEVGO_ASSETS_ROOT=/mnt/nas-tianwen/home/ssy/JevGO /mnt/nas-tianwen/home/ssy/JevGO/.venv/bin/python -B -m pytest -q .dev/tests/autoresearch/test_cli.py
```

要求：四种 module CLI 都能在真实 MiniGrid 与确定性替身闭环；检查不覆盖已有输出；失败保存已知费用并清理；测试不得启动 GPU。交付 `dd0d962`，21 passed，退出 0；合入 runner 修复后四方法再次通过。集成补充拒绝把无图片的环境标为 VL，最终验收见 ACCEPTANCE.md。

## 交接

已合入 `codex/autoresearch`；后续从集成分支新建工作树。GPU 使用 GPU-HANDOFF.md；代码不依赖开发目录。P07 与 P07a 在不同分支修改不同文件，依赖运行器 commit 后才执行 CLI 集成测试。
