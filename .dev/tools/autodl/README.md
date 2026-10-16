# JevGO AutoDL 基础镜像与实验工具

本目录是开发运维工具，不属于离线发布包。自动分流、GitHub push、申请实例、结果分析和关机决策见
[jevgo-autodl 技能](../../../.agents/skills/jevgo-autodl/SKILL.md)。

**当前 `prepare.sh`、`monitor.sh`、`shutdown.sh` 全部在 AutoDL 执行。105 只通过 SSH 调用它们，绝不能在 105 直接运行。**
入口部署到 `/opt/jevgo-autodl`，不会被项目 Git 清理删除。三个入口不接收 `--host`/`--port`，
会在产生副作用前拒绝非 AutoDL 主机。旧的本地控制器已移除，不沿用 `.sh --host ...` 调用方式。

当前仅保留 `remote-control-files.txt` 列出的 21 个远端入口及依赖、本地明确文件打包器 `pack_overlay.py`、
部署清单和本说明。模型校验模块只负责已有文件的清单、哈希和分片索引校验。
一次性验收脚本、旧控制器与本地验收产物已清理；历史源码可从 Git 提交 `a50efca` 查看。

## 存储与镜像约定

| 位置 | 内容 |
| --- | --- |
| `/root/JevGO`，系统盘约 30 GiB | GitHub 代码、`.tools` 内 uv/Python/Java、`.venv`、`.autodl` 安装/校验记录 |
| `/root/autodl-tmp/jevgo`，高速盘约 50 GiB | 安装临时 wheel、运行/JIT 缓存、临时数据；不能依赖关机或新实例后的持久性 |
| `/root/autodl-fs/JevGO/models`，NAS 总容量约 200 GiB | 4B、9B、两个 27B 全部原始模型，持续保留并计费 |
| `/root/JevGO/models` | 指向上述 NAS 模型目录的整体软链接 |
| 项目内任务相对路径，如 `.dev/artifacts/experiments/TASK/` | 小日志、JSON、stdout/stderr；及时取回 105 的相同相对路径，不写 NAS |
| 用户为本次训练指定的 NAS 目录 | 大型 checkpoint/模型权重；提前建立项目软链接，直接通过链接保存 |

容量以本次真实挂载和剩余量为准。四模型清单合计约 128.84 GiB，不能依赖 NAS 标称容量推断剩余空间。
NAS 关闭后保留并持续计费；不要删除共享原始模型。解释器和已安装包留系统盘，不能链接到高速盘。
训练输出位置由具体任务确认，prepare 不再默认创建 checkpoint 目录。

## 远端入口部署与身份核对

先取得用户本轮提供的 SSH 主机、端口、用户名。下面 `HOST`、`PORT`、`TASK`、`BRANCH`、`COMMIT` 均须替换为本轮值；
不沿用历史实例端口。建议本地使用 SSH 参数数组，避免把用户整段输入传给 `eval`：

```bash
# 以下变量和 ssh/rsync 在 105 执行；引号内命令运行在 AutoDL。
autodl_target='root@HOST'
autodl_port='PORT'
autodl_ssh=(ssh -p "$autodl_port" -o BatchMode=yes -o ConnectTimeout=20 -o ForwardAgent=no)
autodl_local_hostname="$(hostname)"
"${autodl_ssh[@]}" "$autodl_target" 'hostname; hostname -I; tr "\0" " " < /proc/1/cmdline'
```

核对远端为 `autodl-container-*`，不是本地主机名、地址不含 `10.200.48.105`，PID 1 包含 `/init/boot/boot.sh`。
确认 SSH 主机密钥；首次可用 `StrictHostKeyChecking=accept-new`，已知主机使用已记录密钥，不能禁用检查或忽略变化。
同时检查 GPU 和 `/root/autodl-tmp`、`readlink -f /root/autodl-fs` 的真实挂载。
实例身份不符立即停止，不能只因路径叫 `/root/JevGO` 就信任目标。

新基础镜像应已有当前控制文件。旧镜像升级时，在身份核对通过后，从 105 只传 `remote-control-files.txt` 列出的源码到 `/opt`：

```bash
"${autodl_ssh[@]}" "$autodl_target" 'install -d -m 755 /opt/jevgo-autodl'
rsync -rt --checksum --files-from=.dev/tools/autodl/remote-control-files.txt \
  -e "ssh -p $autodl_port -o BatchMode=yes -o ForwardAgent=no" \
  .dev/tools/autodl/ "$autodl_target:/opt/jevgo-autodl/"
```

传输后按同一清单比较两端 SHA256；不传 `.git`、凭据、环境或整个工作树，不加 `--delete`。
部署只复制文件；接下来的 prepare/monitor/shutdown 必须放在 SSH 的远端命令内。
调用可用 `bash /opt/jevgo-autodl/prepare.sh ...`，无需依赖复制后的可执行位。

## 远端 prepare

先在 105 提交并 push 本次可公开修改，记录本地分支及已推送完整 commit。
远端只使用公开 `https://github.com/Teemo341/JevGO.git`，push URL 为 `/dev/null`，不传写入凭据或 SSH agent。
prepare 支持实验分支，不固定跑 main；必须用 `--expected-commit` 核对测试版本。

```bash
# 从 105 发起 SSH，prepare 进程实际在 AutoDL。
ssh -p PORT -o BatchMode=yes -o ForwardAgent=no root@HOST \
  'bash /opt/jevgo-autodl/prepare.sh --run-id TASK --branch BRANCH --expected-commit COMMIT --min-free-gib 70 --forbid-hostname LOCAL_HOSTNAME'
```

顺序为：

1. 核验 AutoDL 身份、GPU/剩余显存、高速盘和 NAS 真实挂载。
2. 从公开 GitHub fetch 指定分支，核对完整 commit；备份远端源码修改后，`git reset --hard` 到该 commit，按精确清单清理新增源码。
   这让每轮源码以 GitHub 为准，再叠加本轮明确文件。单独 `git pull --force` 不具有这种恢复语义。
   GitHub 联网可使用 `/etc/network_turbo`。有 GPU 进程或保护路径冲突时停止；不使用 `clean -x/-X`。
3. 应用明确 `--overlay` 源码包（如有），记录本地/远端 commit、原文件备份和哈希。
4. `models ensure` 检查 NAS 模型并建立整体链接；环境 plan/check 与训练扩展 ABI 检查通过且依赖未变则零安装。
   仅缺失/变化时补包，成功安装后删除远端 wheel；不会重装驱动或下载模型。
5. 进行当前设备 CUDA 与训练内核的微型检查；成功报告 `ready`，不启动真实实验或关机。

`--min-free-gib` 默认 0，`--device` 为可见设备限制后的逻辑编号，默认 0。
示例 70 GiB 是已验证 27B 最小 LoRA 测试使用的余量，不保证所有序列长度/batch/训练模式都够用。
多卡任务仍须核实每张卡的显存、设备映射和实验代码的并行方式。
`--inspect` 只读检查身份、GPU、挂载；`--full-model-check` 强制重读全部模型计算 SHA256。
报告写远端项目 `.dev/artifacts/autodl-prepare/TASK/`，后续也按相同相对路径取回。

### 仅补传明确列出的源码

大部分源码已 push 时不需要 overlay。后续迭代确有未发布文件时，在 105 创建每行一个项目相对路径的 UTF-8 清单，
只打包本次任务所需文件：

```bash
# 这个打包工具只读本地源码、写 JSON，不执行 prepare 或连接服务器。
.venv/bin/python .dev/tools/autodl/pack_overlay.py \
  --files .dev/tmp/autodl-task-files.txt --output .dev/tmp/autodl-task-overlay.json
```

也可重复 `--file`、`--delete-file`；删除项必须已在本机删除。工具拒绝保护路径、凭据、符号链接和超限文件
（单文件 10 MiB，合计 64 MiB）。先核实高速盘挂载，再通过 SSH 创建高速盘的 overlay 子目录并 rsync 这个 JSON；
远端 prepare 加 `--overlay /root/autodl-tmp/jevgo/overlays/TASK.json`。
每次开始新一轮 prepare，先恢复为已核对的 GitHub commit，再覆盖这一轮文件。因此首轮已覆盖的文件不会阻碍下一轮，
无需给出旧 overlay 编号；每轮使用新的 run-id 和新的 JSON。`--resume` 仅用于同一轮补齐材料，不接受不同源码包。

强制恢复只作用于 AutoDL 源码：备份 tracked/staged、可清理的新增源码与已登记覆盖，核对后 reset/精确清理。
保护环境、模型入口、`.autodl` 审计、缓存、`.dev/artifacts`、`.dev/logs` 和 NAS；未知 ignored 文件保留。
若 Git 当前或目标版本涉及这些保护路径，会停止，不能用强制更新破坏运行环境或权重。
小日志统一放受保护的项目产物路径；重新 prepare 前先取回有价值的远端调试修改和结果。
容器内源码备份只用于本实例恢复，不是 105 上的持久副本。

### 缺少 wheel 时继续

`status=needs_materials`、退出码 3 表示缺少已锁定材料。报告 `materials` 每项提供 `filename`、`size`、`sha256` 和 `destination`。
基线 wheel 从 105 的 `.cache/offline/wheels/` 取；训练扩展从 `.dev/cache/autodl/training-kernels/` 按哈希取。
只将匹配材料 rsync 到报告指定的高速盘目录：

- 基线：`/root/autodl-tmp/jevgo/install/wheels/`
- 训练扩展：`/root/autodl-tmp/jevgo/install/optional/TASK/`

然后在 AutoDL 用相同 run-id、branch、commit、overlay 参数加 `--resume` 继续。
resume 仅接受本次已记录基线和完全相同的 overlay，不重复 fetch/reset；新一轮代码使用新的 run-id。
不重新下载/替换不匹配版本，不删除本机离线材料。Python/PyTorch ABI 变化需先准备匹配扩展，不能只看包能导入。

## 远端 monitor

prepare 成功后，先在 AutoDL 读取当前空闲功率，确认没有 GPU 任务，再选择阈值：

```bash
# 这些命令都在 AutoDL shell 内执行，或作为 ssh 的远端命令执行。
bash /opt/jevgo-autodl/monitor.sh --idle-watts 45 --check --forbid-hostname LOCAL_HOSTNAME
```

RTX PRO 6000 Blackwell 96GB 的一次实测空闲 P8 约 36.43 W，对该实例用 45 W；这不是其他显卡的默认值。
确认本次阈值后，在 **AutoDL** 后台运行：

```bash
mkdir -p /root/JevGO/.dev/logs/autodl
nohup bash /opt/jevgo-autodl/monitor.sh --idle-watts 45 --run-id TASK \
  --forbid-hostname LOCAL_HOSTNAME \
  > /root/JevGO/.dev/logs/autodl/TASK.log 2>&1 < /dev/null &
```

默认每 30 秒采样所有 GPU；所有卡不高于阈值且没有 GPU 计算进程连续 30 分钟，才进入远端 shutdown 流程。
有进程、超功率、读数缺失/异常、采样间隔超过两倍设定值均重新计时。实例启动身份或 GPU UUID 集合变化则停止监控。
计时用单调时钟；这是按采样判断的空闲，不能观测采样间隔内的所有短暂活动。

监控进程位于 AutoDL，断开 105 的 SSH 不会停止它。检查后台 PID 与首条成功采样，不能只凭 nohup 返回就认为已启动。
它会在清理前、关机前再次核验功率与进程；发现活动不强制关闭。单实例文件锁阻止重复监控。
报告/采样在 `/root/JevGO/.dev/artifacts/autodl-monitor/TASK/`。
`--dry-run` 仅观察到条件满足并报告 `would_shutdown`，`--check` 只读一次；均不清理或关机。
需要只读验证计时时可以 `--idle-minutes 0.5 --interval 5 --dry-run`；正式使用仍是 30 分钟。

监控**不会把日志传给 105**。实验结束后立即取回小日志再本地分析，避免空闲关机导致后续新实例无法读取容器内结果。
GPU 阶段后的长时间 CPU 分析应取回所需文件后在 105 完成；不能依赖低 GPU 功率判断整个实验已经结束。
只有用户明确要求远端继续后处理并保持开机时，才临时停止本任务 monitor，完成取回后按该轮关机约定处理。
105 断连时监控仍可关机，容器内尚未取回的小文件不保证能在后续新镜像中恢复。
需要保持开机时，只在 AutoDL 给本任务记录的监控 PID 发 `TERM`，不要宽泛 `pkill` 或停止其他人的进程。

## 实验、权重输出与结果取回

在 AutoDL `/root/JevGO` 用 `jevgo-autodl run COMMAND ...`（或 `/opt/jevgo-autodl/jevgo-autodl run ...`）运行真实实验，
复用现有环境。记录退出码、代码版本、覆盖哈希、GPU、精度、配置和预期结果。

训练前先取得用户指定的 NAS 输出路径、估算容量。在挂载检查通过后创建独立输出目录，
**先**在项目内建立软链接，再让训练直接保存到链接中；禁止先写系统盘再移动。
不要用原始 `models/JEV-*` 目录作可覆盖训练输出，不自动更改 LoRA 的 FP32 精度。
如果用户明确选用已有布局 `JevGO/runs/TASK/checkpoints`，远端可复用：

```bash
jevgo-autodl storage prepare --run-id TASK --expected-bytes 1073741824
jevgo-autodl storage check --run-id TASK --expected-bytes 1073741824
```

它建立 `.dev/artifacts/autodl/TASK/checkpoints` 到 NAS 对应目录的链接并预留 5 GiB。
**此固定布局不是默认授权的新权重目的地**；用户指定其他目录时按其目录建立并核验专用链接，不误用该固定布局工具。
NAS 挂载丢失、余量不足或真实非空输出目录冲突时必须停止。权重保存后做哈希/索引及任务所需的重载验证，留 NAS 待后续安排取回。

小日志写项目内受保护的 `.dev/artifacts/<任务>/` 或 `.dev/logs/<任务>/`，不放 NAS；已有历史高速盘日志也应取回。以明确小文件清单从 105 取回到同一相对路径：

```bash
# 清单每行相对于 /root/JevGO 的日志路径，不包含目录、权重或任何符号链接。
rsync -rt --checksum --files-from=.dev/tmp/autodl-result-files.txt \
  -e 'ssh -p PORT -o BatchMode=yes -o ForwardAgent=no' \
  root@HOST:/root/JevGO/ ./
```

先核对清单中的远端文件及每级目录不是链接、确为小文件；不要用 `-L` 或递归下载整个产物树。
核对两端大小和 SHA256，包括失败实验的 stdout/stderr、退出码和诊断结果。
本地同路径已有其他内容时先按 run-id 留存，不覆盖旧实验。权重通过 NAS 路径及哈希关联，不混入小日志清单。

## 远端 shutdown

先在 105 分析取回结果：符合预期或接下来超过半小时不用 GPU，就关闭 AutoDL；
若短时间 debug 后仍会复测则保留实例。明确“暂不关机”的镜像联调要求优先，并停用自动关机监控。

关机前取回小日志、有价值的远端调试改动，确认 NAS 权重完整，并结束本任务 GPU 进程。
在 **AutoDL** 执行：

```bash
bash /opt/jevgo-autodl/shutdown.sh --dry-run --forbid-hostname LOCAL_HOSTNAME
bash /opt/jevgo-autodl/shutdown.sh --forbid-hostname LOCAL_HOSTNAME
```

脚本核实主机/PID 1、公开 Git origin、HEAD 已包含在记录的公开远端分支中、NAS/models 链接、空闲 GPU 和平台关机脚本。
先备份源码与补丁到容器项目 `.dev/artifacts/autodl-shutdown/<run-id>/`，
再核对计划未变化，撤销 tracked/staged 修改并按精确清单清理新增源码（包含有记录的 ignored overlay）。
环境、缓存、模型链接、NAS 输出和审计保留，不使用 `clean -x/-X`。

**容器内源码备份不是已回传 105 的持久备份**；有价值的修改须提前取回。脚本不把小文件写 NAS。
存在 GPU 进程、未发布远端 commit、未知源码符号链接/凭据路径、超限备份或身份变化时停止，不擅自杀进程。
清理后再次检查，再执行已核验的 AutoDL `/usr/bin/shutdown` 平台关机流程。
从 105 观察关机请求输出和后续 SSH 不可达；不能把一次断连单独当作关机成功，不关闭 105。
平台脚本发生变化时先只读核实新实现，不能绕过守卫执行普通系统 poweroff。

## 首次补齐共享模型

本工具面向已配置好的基础镜像，正常使用时不重复上传模型或安装环境。
保留 `.tools`、`.cache/offline` 和 `.dev/cache/autodl/training-kernels` 中的离线 wheel、构建来源及复现记录。
Git clone 不含这些二进制材料；缺包时从 105 按 prepare 清单传输，不能换成未锁定网络包。

模型只从 105 已准备的整个 `models/` 经 SSH/rsync 上传。先确认 NAS 真实挂载，复制不加 `--delete`：

```bash
rsync -rlt --partial --partial-dir=.rsync-partial --info=progress2 \
  -e 'ssh -p PORT -o BatchMode=yes -o ForwardAgent=no' \
  models/ root@HOST:/root/autodl-fs/JevGO/models/
```

上传后在 AutoDL `jevgo-autodl models link`，它完整校验模型文件 SHA256 与分片索引，再链接整个目录；
`models check` 只校验、`models list` 只列清单、`models ensure` 可复用有效完整校验记录。
元数据、清单或 NAS 身份变化时 ensure 重新完整校验。拒绝覆盖真实非空模型目录。
所有 4B/9B/27B 均存 NAS；`models download` 已停用，不调用 Hugging Face 或其他镜像下载。

## 镜像环境与已完成验收

当前镜像保留 CUDA 12.8，默认选已验证的 CUDA 12.9 编译器；不是重新安装驱动。
`runtime.sh` 配置 nvcc/FlashInfer 工具链，所有 JIT 缓存放高速盘，新实例可能重新编译。
基线版本以 `uv.lock`/清单为准；训练附加包以 `.autodl/training-kernels.json` 记录的版本、wheel 哈希和 Python/PyTorch ABI 为准。
已验证 `causal-conv1d==1.7.0`、`flash-linear-attention==0.5.2`、`fla-core==0.5.2`。
改变 Python/PyTorch/GPU 时重新核实兼容性，导入成功不代替真实训练。

2026-10-16 已完成新实例闭环（提交 `a50efca`）：旧实例关机、本地资源检查、GitHub 同步、
用户开启新实例、远端 prepare/monitor、JEV-9B 最小 LoRA 训练、结果取回、本地分析和远端 shutdown。
prepare 37.30 秒，245 项基线依赖无缺失、零安装；首次 GitHub 请求遇到临时 503，重试后恢复。
BF16 骨干与 FP32 LoRA 的三步训练、保存及重载通过，峰值分配 17.03 GiB；总耗时 466.52 秒，
其中模型冷加载 400.52 秒。两个训练快速内核实际参与前后向，重载后输出完全一致。
NAS checkpoint 共 4 文件、39.84 MiB，重读哈希一致；33 个小结果文件取回 105 后逐个校验通过。
监控实际识别 GPU 进程并重置计时；本轮主动 shutdown，未等待 30 分钟空闲触发。
平台关机请求后两次 SSH 均不可达，105 正常运行。

此前 JEV-27B 三步 LoRA 保存重载及协议调用也已通过，峰值分配 50.50 GiB。
这些是运行流程验收，不代表模型效果、训练收敛或其他 batch/序列长度的容量保证。
清理时当前流程的 265 项 CPU 回归通过；验收测试源码保留在上述 Git 历史中。

保存基础镜像仍由用户按 AutoDL 控制台流程完成；SSH 环境配置/关机不等于镜像已保存。
新镜像应包含当前 `/opt/jevgo-autodl` 远端入口。
