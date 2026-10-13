# JevGO 环境使用

所有组件共用项目根目录的 `.venv`：AutoTrust、SemIf，以及 MiniGrid、ALFWorld、ScienceWorld、DiscoveryWorld。
Python、uv、Java 随 `.tools/` 提供，无需全局安装或手动激活环境。

## 使用

在项目根目录执行：

```bash
# 首次使用或复制到新目录：校验离线材料并创建环境
bash scripts/bootstrap.sh

# 运行自己的脚本
bash scripts/env.sh jevgo python your_script.py

# 检查环境位置、离线文件、依赖锁和已安装包
bash scripts/env.sh jevgo --check
```

普通运行入口在环境缺失或项目路径改变时自动离线重建；`--check` 只检查，缺失时先运行 `bootstrap.sh`。
安装使用 `.cache/offline/` 中的本地包，不自动下载模型。

## 复制与运行条件

目标系统为 Linux x86_64、glibc ≥2.35、GCC 11 级 libstdc++；GPU 功能需要兼容 CUDA 12.9 的 NVIDIA 驱动。
`/tmp` 需可写；ScienceWorld 需要允许本机 Java 通信。音视频解码另需系统 FFmpeg 共享库。

复制时保留权限、符号链接及隐藏目录，尤其是 `.tools/`、`.cache/offline/`、`.third-party/autotrust/`、`configs/`、`scripts/`，以及根目录的 `pyproject.toml`、`uv.lock`、`.python-version`。
发布副本排除 `.dev/`、`.agents/`、`AGENTS.md`；迁移还可排除 `.venv/` 和 `.cache/uv/`。仅 Git clone 不包含完整离线材料。
离线材料约 6 GB，重建后的 `.venv` 约 11 GB。复制后在新项目根目录执行 `bootstrap.sh` 和 `env.sh jevgo --check`。

未包含预训练模型权重或全量任务数据；需按实际任务另行准备。已验证最小组件运行和同机异路径离线重建，尚未做跨机实测或完整 benchmark。
