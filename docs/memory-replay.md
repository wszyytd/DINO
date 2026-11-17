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
