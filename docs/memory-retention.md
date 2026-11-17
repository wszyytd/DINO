# 车辆记忆保留诊断

这项实验复用已有区域时序训练的冻结权重。仿真车辆身份用来分配记忆槽，真值框外扩 4% 后截取原始 RGB。编码器只接收 RGB，写入沿用 16 个 token，查询沿用 64 个 token。这样可以在候选对应正确时单独测试更新器。这里的多布局仍使用相同两种车辆资产。

每条往返路线开头清空记忆，按原始顺序处理。主线对同车、同 RGB 摘要、同裁剪框去重。每一步正常写入后，依次检查以下问题。

| 报告行 | 含义 |
| --- | --- |
| `self` | 新图第一次写入后，立即用同一裁剪读一次。 |
| `old_exact` | 后续每一步重新问所有已写过的原图，并保留它在首次写入时的判断。 |
| `new_view` | 路线之外另一个相机位置拍的独立照片，默认和当前车辆相对方位相差不超过 25°。 |
| `novel_view` | 当前观察和独立查询都与该车之前写入的所有方位相差至少 45° 的子集。 |
| `old_view` | 曾作为独立新视角查询的照片，之后继续用于检查旧视角。 |
| `repeat_1`、`repeat_8` | 从正常写入后的状态出发，只给指定车强制额外写 1 或 8 次，用同一组照片比较分数和槽位变化。 |

正常写入和跳过写入从同一历史比较。后一分支只跳过指定车辆，另一辆车仍照常写入。强制重复和跳过的状态不会接回正常路线。首次写入使车辆从无记录变为有记录，因此只报告当时正确率，不把它计作两车都已建档时的配对收益。身份对错含 Null，分差单独计算正确车 logit 减去另一辆车 logit。旧查询首次正确、后来错误的统计按读取事件计数，多个时刻可能读同一张图。报告另给不同查询图数量。新视角查询另按写入次数 1、2–3、4+ 汇总。

## 在 CUDA 服务器运行

在现有 DINO 仓库和 `dino-probe` 环境下执行。默认使用现存区域时序训练产物里的 3 个 `memory` 检查点，以及它们训练时使用的老师适配器和 DINOv2。可通过 `--source-run`、`--teacher`、`--model-repo`、`--backbone` 指定搬迁后的位置。程序校验权重和训练来源的文件摘要。

```bash
set -euo pipefail
git switch codex/evidence-memory-v1
git pull --ff-only origin codex/evidence-memory-v1

OUT="out/memory_retention_$(date +%Y%m%d_%H%M%S)"
python run_memory_retention.py plan --output "$OUT" --allow-incomplete
python run_memory_retention.py evaluate --output "$OUT" --device cuda:0
cat "$OUT/evaluation/report.md"
```

`--allow-incomplete` 会在计划和报告中列出因预期 RGB、标注或采集日志缺失或为空而跳过的布局。它不会放过所用 RGB 或掩膜的摘要不符。要严格要求这些布局齐备，可去掉该参数。本地副本中两个验证布局有空文件，完整验证目前只有一个背景。计划阶段只读数据，核对已选图片和掩膜的采集摘要，不运行模型。先只看一个种子可加 `--seed 7`，需要对照可加 `--method memory last mean`。

`evaluation/summary.json` 保存分组统计，`evaluation/memory-s7.json` 等文件保存每一步的图片标识、分数、判断、原始首次判断和重复写入槽位漂移。用同目录的 `plan.json` 对照原图、车辆裁剪框和角度。若没有 `summary.json` 的 `status=complete`，该次评估还未完整输出。
