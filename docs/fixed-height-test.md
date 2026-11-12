# 固定高度旧采集：固定权重识别与融合测试

本入口将 `data/fixed_height_similar_20261104` 接入已经训练好的车辆区域时序模型。它只提取新图片的特征、加载旧权重并测试，不训练、不调整 Null 阈值、不从新数据挑选种子。

入口：[run_fixed_height_test.py](../run_fixed_height_test.py)。代码使用现有冻结编码器及 `advance/score`，旧网格实验、训练代码和结果保持原样。

## 这次能够回答什么

1. 给定采集记录中的车辆区域，原模型在这批照片上能否选对参考车。
2. 普通白车、相近颜色、相近车型分别表现如何。
3. 沿同一组 24 张照片逐步更新后，是否胜过同权重每张图单独判断。
4. 最近一次有效观察（last）、不同图均值（mean）、可训练记忆（memory）三种旧模型的差异。

本次车辆区域由采集时的三维车辆框投影得到，跨照片身份由采集记录提供。没有运行老师定位，也没有测试自动跟踪。投影框不是实例掩膜，采集记录的 `feature_visibility` 是 `unknown`；因此不报告“物理可见率”或把供框结果解释为整个搜索系统的成功率。

目标在每组的两辆采集候选中，数据没有目标缺席样例。选 Null 计错，但无法衡量目标缺席时是否会正确拒绝。这组采集最初用于目视检查，同场景和车型可能与早期研究有关；文件哈希不重叠只排除了完全相同文件，不能证明严格的场景或车型独立性。

## 数据核对结果

本机只进行了 CSV、图片、文件哈希和纯数据检查，没有运行模型：

| 项目 | 数量 |
|---|---:|
| 参考固定的采集组 | 12 |
| 每组查询角度 | 24（0°～345°，每次 15°） |
| 每步候选车辆 | 2，同一张整图的 A/B |
| 顺序判断数 | 288 |
| 整图 | 300（288 查询 + 12 参考） |
| 采集保存的裁图 | 588（576 查询 + 12 参考） |
| 查询特征键 | 576 |
| 同色不同车型组 | 4 |
| 同车型相近颜色组 | 8 |
| 与源实验集合完全相同的图片文件 | 0 |

采集高度 8 米，环绕半径 18 米，整图 2048×1536，视场角 70°。同一个角度的 A/B 必须使用同一帧、同一相机位置和同一 RGB；适配器会检查它们。refA 与 refB 是两次采集，实际 RGB 哈希不同，按各自序列独立测试。

每个采集组固定自己的参考图，按记录顺序走完 24 步，开始时清空两车状态。查询输入从原始整图按保存的投影框重新裁出，避免把另存裁图的压缩差异混入查询编码。参考输入采用采集保存的参考裁图。查看器中的查询裁图也按原整图重新生成。

## 固定使用哪些检查点

源实验默认为 `out/region_sequence_mask_20261112_152736`：

- last × 种子 7、17、27；
- mean × 种子 7、17、27；
- memory × 种子 7、17、27。

适配器要求源实验已经完整结束，校验源计划、特征清单、代码版本和全部九个检查点的 SHA256。加载时检查方法、种子、模式、轮次和计划来源，严格加载全部参数。测试前后再次检查权重，禁止静默换编码器或检查点。

原区域时序代码来自提交 `747c800e858815ba72874ff2891707c1c1fa7340`。本入口增加独立文件，不修改该模型实现。源实验在 Linux 上生成的源码哈希允许与本机 CRLF 文件在统一为 LF 后相同；不会忽略实质代码变化。

## 两种运行方式和主要指标

每个检查点都计算：

- **正常顺序**：逐张照片更新，保留同组此前的状态。完全相同查询图、车辆和框只写一次。
- **同权重单图**：每张图都重新开始，只用当前两辆车的裁图判断，不保留前面照片。它与另行训练的 last 模型不同。

报告记录正确车、错车、Null 三种去向；另报忽略 Null 后的两车排名、每组最后一步的正确率、历史救回和历史伤害次数。历史救回是“单图错、顺序对”，历史伤害是“单图对、顺序错”，比较的检查点和当前照片相同。

按采集组、车型颜色条件、参考身份和角度分别汇总。每个种子分别保留结果，再报告三个种子的均值和标准差。288 次判断来自相关的 24 步序列，不等于 288 个独立场景；最后一步仅有 12 组。

每一步保存三个分数及概率、所选车辆、单图对照、两车状态哈希、写入次数和状态变化大小。每个检查点重新加载后重放全部查询，并检查输出和选择一致；还会验证反转候选顺序后，归回相同身份的分数误差不超过 `2e-5`。这些检查由测试入口自动执行。

## 服务器执行

遵守本项目此前约定：本机不运行神经网络，不提特征，不训练，不做模型评估，不自行 SSH。以下命令由用户在已有授权 CUDA 服务器执行。

先更新代码，确认数据、源实验九个检查点、老师适配器权重和官方 DINO 权重都已经在服务器。程序不会自动下载资源。

```bash
cd /mnt/fast18/sunbo/DINO
conda activate dino-probe
git switch codex/evidence-memory-v1
git pull --ff-only origin codex/evidence-memory-v1

# 此命令只运行纯数据测试。
python -m unittest tests.test_fixed_height_data

OUT="out/fixed_height_test_$(date +%Y%m%d_%H%M%S)"
python run_fixed_height_test.py \
  --stage run \
  --data data/fixed_height_similar_20261104 \
  --checkpoint-run out/region_sequence_mask_20261112_152736 \
  --teacher out/teacher_grid_fit_e20_retry/last.pt \
  --output "$OUT" \
  --device cuda \
  --threads 4 \
  --view-batch 1 \
  --batch-size 4

printf '%s\n' "$OUT/report/report.md" "$OUT/report/index.html"
```

`run` 顺序执行 audit → prepare → evaluate → report，不包含训练。

如果原环境不能自动定位已有 DINO 资源，在同一条命令中增加 `--model-repo` 和 `--backbone`，或沿用原来的 `DINO_REPO`、`DINO_BACKBONE` 环境变量。所提供的文件必须与源训练使用的版本和 SHA256 一致。

本机生成的预览目录只用于核对图片，不要把 Windows 计划文件继续用于服务器提特征。服务器应使用新输出目录重新执行 audit，冻结它自己对应的代码和数据哈希。

## 分阶段执行

如果需要先看图片再占用 GPU，可以分阶段执行。同一输出目录在 audit 时必须尚不存在。

```bash
OUT="out/fixed_height_test_$(date +%Y%m%d_%H%M%S)"

python run_fixed_height_test.py --stage audit \
  --data data/fixed_height_similar_20261104 \
  --checkpoint-run out/region_sequence_mask_20261112_152736 --output "$OUT"
python run_fixed_height_test.py --stage preview \
  --data data/fixed_height_similar_20261104 \
  --checkpoint-run out/region_sequence_mask_20261112_152736 --output "$OUT"

python run_fixed_height_test.py --stage prepare \
  --data data/fixed_height_similar_20261104 \
  --checkpoint-run out/region_sequence_mask_20261112_152736 \
  --teacher out/teacher_grid_fit_e20_retry/last.pt \
  --output "$OUT" --device cuda --threads 4 --view-batch 1
python run_fixed_height_test.py --stage evaluate \
  --data data/fixed_height_similar_20261104 \
  --checkpoint-run out/region_sequence_mask_20261112_152736 \
  --output "$OUT" --device cuda --threads 4 --batch-size 4
python run_fixed_height_test.py --stage report \
  --data data/fixed_height_similar_20261104 \
  --checkpoint-run out/region_sequence_mask_20261112_152736 --output "$OUT"
```

不会覆盖已有特征、评价或查看器；中途失败会保留记录，修正代码后使用新输出目录重新开始。不要在测试期间修改原始数据、源码或旧权重。

## 输出与检查

- `plan.json`：数据、代码、九个检查点和读取顺序的来源记录。
- `prepared.json`、`features/`、`references/`：冻结特征和 SHA256；提取时间与显存仅对应编码阶段。
- `evaluation/run.json`：九组完成状态、分组指标、重载误差、候选顺序检查、权重前后哈希和缓存匹配耗时。
- `evaluation/{method}_seed{seed}.json`：逐步结果、同权重单图对照和状态变化。
- `report/report.md`、`report/metrics.csv`、`report/summary.json`：九组结果及三种方法的均值。
- `report/index.html` 和 `report/assets/`：固定参考、逐步播放、同一步三方法对比、截至当前的历史。换采集组回到第一步，换种子或方法保持当前步；播放不跨组。

回传结果时保留输出目录的相对结构。阅读完整查看器需要同时保留 `report/assets/`。若仅需分析数字，保留计划、prepared、evaluation 和 report 的 JSON/Markdown/CSV 即可，原始权重仍保留在源实验。

只有 `evaluation/run.json` 为 `complete`、包含九组检查点，且所有来源及重载检查通过，才能解释模型效果。本机数据预览没有模型准确率，不应把它当成已完成的效果测试。

## 代码地图

| 文件 | 作用 |
|---|---|
| [fixed_height_data.py](../dinoexp/fixed_height_data.py) | CSV/图片/来源核对、因果批次、统计 |
| [fixed_height_eval.py](../dinoexp/fixed_height_eval.py) | CUDA 冻结编码、旧权重测试、单图对照和重载检查 |
| [fixed_height_report.py](../dinoexp/fixed_height_report.py) | 只读取保存分数的报告和逐步查看器 |
| [run_fixed_height_test.py](../run_fixed_height_test.py) | 分阶段入口 |
| [test_fixed_height_data.py](../tests/test_fixed_height_data.py) | 不导入 torch 的纯数据测试 |

本次接口不包含训练选项。
