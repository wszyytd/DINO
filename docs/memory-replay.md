# 车辆记忆全量回问训练

训练沿用旧车记忆模型和冻结的 DINOv2 前端。每条路线从空记忆开始，新图的 16 个 token 写入一次。有效写入完成后，把截至当步所有已写过车辆裁图的 64 个查询 token 批量拿来问。A、B 的历史都会被问到。裁图的身份只用于路由和交叉熵标签。查询不写进记忆。

每步先平均该步的查询损失，再对整条路线的有效写入步取平均。梯度沿整条记忆链传播。每批完整路线累计一次梯度，再用 AdamW 更新。没有新写入的返程重访跳过训练。冻结特征会在一次运行内缓存，并供所有实验组复用。

| 实验组 | 训练目标 |
| --- | --- |
| `initial` | 冻结原检查点，直接验收 |
| `fixed` | 每步用采集时的固定参考照片查询，作为同数据的固定参考对照 |
| `replay` | 每步写完后，用曾写入的全部裁图查询 |
| `replay_independent` | 全量回问，加上当前写入之前的独立照片查询。该候选必须已入档，查询照片来自路线外，且与任何路线写入的原图摘要不同 |

三组微调从相同权重开始，用相同训练背景、路线顺序、训练轮数和优化器更新数。留出背景不参与损失与选轮，统一验收最后一轮。独立题主要来自与当前视角邻近的另一张照片，距离之前所有观察都较远的视角只有少量样本。两种车辆资产在不同布局中复用。

## 服务器运行

复用已在服务器跑过的诊断计划。该计划含 8 条训练路线和 3 条验证路线。

```bash
set -euo pipefail
conda activate dino-probe
git switch codex/evidence-memory-v1
git pull --ff-only origin codex/evidence-memory-v1

PLAN=out/memory_retention_20261117_110524/plan.json
python run_memory_replay.py audit --plan "$PLAN"
RUN_MEMORY_REPLAY_CUDA_TESTS=1 python -m unittest tests.test_memory_replay_server -v

OUT="out/memory_replay_$(date +%Y%m%d_%H%M%S)"
python run_memory_replay.py train --plan "$PLAN" --output "$OUT" --device cuda:0
cat "$OUT/report.md"
```

如需重建计划，运行 `python run_memory_retention.py plan --output out/memory_retention_new` 并把 `PLAN` 指向新生成的 `plan.json`。默认三组微调各跑 30 轮、种子 7、每批 2 条路线。先跑通可加 `--epochs 1 --variant replay_independent`。扩到三个种子可加 `--seed 7 17 27`。固定参考图从训练场景的采集记录加载并校验文件摘要。

`report.md` 报告首次写入原图、历史原图保留率、独立照片写入前及写入后的表现。`summary.json` 标识运行是否完整。`<组名>-s<种子>/validation.json` 保存逐路线逐步分数；`last.pt` 保存微调后的模型权重。

## 车辆前景与记忆更新对照

在服务器仓库中激活现有 `dino-probe` 环境，运行 `bash run_foreground_memory.sh`。脚本用已有的 11 月 17 日诊断计划，先审计数据划分并运行 CUDA 数值测试，再执行下表的四组微调。默认 30 轮、种子 7。只检查完整流程可用 `bash run_foreground_memory.sh 1 7`，三种子实验用 `bash run_foreground_memory.sh 30 7 17 27`。也可以通过 `MEMORY_PLAN` 指向重建的同格式计划。

| 裁图 | 实验组 | 和上一组相比，改变了什么 |
| --- | --- | --- |
| bbox | replay_independent | 原始 RGB 框、原学习式更新、旧图回问＋写入前独立查询 |
| foreground | replay_independent | 同一裁框内仅保留 CARLA 实例掩膜的车身像素，背景填黑 |
| foreground | replay_prepost | 写入后再对当步未写入的独立照片计算身份交叉熵，包括首次有效观察 |
| foreground | ema_prepost | 将学习式 Updater 和门替换为固定槽 EMA，两组的初始 Writer、Reader 和评分头权重相同 |

需要单独验证是否只记住第一张或最近一张时，使用 `--variant ema_prepost`，分别设置 `--ema-decay 1`（每车保留第一次有效观察）或 `--ema-decay 0`（每车保留最近一次有效观察），其余训练条件保持相同。

观察和查询两侧采用同一抠图规则，再走各自原有的冻结编码分支。空间位置、裁框和池化分辨率一致。每种裁图在一次运行内只编码一次，特征供各组和各随机种子共用。预处理会校验原图、全分辨率实例掩膜的摘要与尺寸。前景用仿真真值给出，实验衡量的是理想分割条件下的身份匹配。

三项损失各自先对单步查询、再对有该项监督的路线步平均。两个 prepost 组均使用历史回问、独立 pre、独立 post 三项，权重默认为 1、1、1。独立照片来自路线外，始终只查询。EMA 首次直接保存 Writer 的 8 个槽，之后使用旧槽 × 0.8 ＋ 新槽 × 0.2，无观察时保持旧状态。这里保留原 Reader 所需的槽向量尺度。EMA 的 Writer 和 Reader 仍参加相同监督下的微调，固定的是更新公式。两个系数可通过单独训练命令的 `--ema-decay` 和 `--postwrite-weight` 设置。

脚本打印新建的 `out/foreground_memory_*` 目录。其 `bbox/report.md` 和 `foreground/report.md` 中第二张表比较双方车辆均已建档时，**同一张独立照片**在整帧写入前后的正确数、由错变对、由对变错，以及旧角度独立照片的保留情况。`paired_independent.by_candidate` 保留 A/B 分项；`groups.new_view.help/harm` 使用只跳过目标车本次写入的另一种干预。不同步数可重复查询同一张照片。每个组的 `last.pt` 记录模型配置及裁图模式，恢复时使用 `dinoexp.replay_matcher.build_replay_matcher(checkpoint["config"])`，再严格加载 `checkpoint["model_state"]`。
