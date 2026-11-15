# LiVOS 第二步：已有车辆数据上的预训练分割测试

这一步回答：给 LiVOS 第一张图中的车辆位置后，它能否在后续观察中持续分出同一辆车，遇到图边、目标离开画面、重新出现及相似车辆时是否出错。

使用官方预训练模型，不训练，不改现有 DINO 记忆网络。每段路线从空记忆开始，只在第 0 步输入目标掩膜；第 1 步开始只给照片。另一辆车的掩膜和后续目标掩膜只用于预测后的评分。输出是像素区域；这里没有参考照片匹配，也没有 last/mean/8 槽方法比较。

## 1. 数据和样本

用 `data/search_grid_smoke_v1`，而不是没有像素掩膜的 fixed-height 数据。

默认设置：路线长 48 步、起点间隔 16、最多 12 段，初始化时目标至少 1,024 个原图像素且不接触图边。先在完整拍摄图上生成相邻路线，再按标注选择可初始化的车辆，兼顾两辆车和压力情况，选择过程不看模型预测。

本机真实数据审计结果：

| 项目 | 数量 |
|---|---:|
| 拍摄图片 | 200 |
| 符合初始化条件的测试段 | 9 |
| 每段步数 | 48 |
| 总步数 | 432 |
| 排除初始化后的评分步数 | 423 |
| 评分中目标可见 | 147 |
| 评分中目标不在画面 | 276 |
| 可见目标接触图边 | 76 |
| 以 A 车初始化的测试段 | 4 |
| 以 B 车初始化的测试段 | 5 |

9 段都包含目标离开后重新出现、接触图边、两车同时可见。它们覆盖 200 张原始图片，路线有重复和交叠，不能把 432 步称为 432 张独立图片。

这是一个场景、两辆静止车辆，A 是普通白车，B 是白色警车。相机动作是 4 米平移或 45 度转向的离散图边，并非平滑视频；不能直接给画面标注车头/车尾，也不能把目标离开视野说成真实遮挡。车辆之间的类别差异也比真正同款车辆大。此实验可筛查适用性，不足以证明跨车辆泛化或实际飞行效果。

## 2. 服务器准备

沿用服务器目录 `/mnt/fast18/sunbo/DINO` 与环境 `dino-probe`。LiVOS 是独立仓库，放在同级 `/mnt/fast18/sunbo/LiVOS`。

```bash
conda activate dino-probe
cd /mnt/fast18/sunbo/DINO
git switch codex/evidence-memory-v1
git pull --ff-only origin codex/evidence-memory-v1
```

如果服务器没有 LiVOS，第一次执行：

```bash
cd /mnt/fast18/sunbo
git clone https://github.com/uncbiag/LiVOS.git LiVOS
```

本入口对照的 LiVOS 版本为 `8c26cb03f570942a0c9a26f5bf7a8d378fc9ac60`。在没有本地改动的源码仓库中可固定到该版本：

```bash
git -C /mnt/fast18/sunbo/LiVOS checkout 8c26cb03f570942a0c9a26f5bf7a8d378fc9ac60
```

使用已有 CUDA PyTorch/torchvision。LiVOS 官方 README 使用 Python 3.10、PyTorch 2.4.0、torchvision 0.19.0；这次没有在本机验证服务器运行环境。此入口涉及的附加依赖可安装为：

```bash
python -m pip install "hydra-core>=1.3.2,<1.4" "pycocotools>=2.0.7" "tqdm>=4.66.1" "Pillow>=9.5" "numpy>=1.21" "gdown>=4.7.1"
```

不需要安装训练用的数据增强包或 GUI。`hickle` 是官方结果工具的可选依赖，此入口不使用多尺度测试。

## 3. 官方权重

显式使用 `livos-nomose-480p.pth`，不要使用默认配置里遗留的 Cutie 权重。Google Drive 地址来自 LiVOS 的 `download.py`。

如果服务器已经有该权重，直接使用已有文件；否则下载这一份即可：

```bash
mkdir -p /mnt/fast18/sunbo/LiVOS/weights
python -m gdown 'https://drive.google.com/uc?id=1ZjhVMUppZ5OZipCALU9ZYwiVF5DEPBxX' \
  -O /mnt/fast18/sunbo/LiVOS/weights/livos-nomose-480p.pth
```

入口要求完整的官方扁平 `state_dict`，用 `strict=True` 加载全部权重。缺项、额外项或尺寸不符都会报错，不能用部分加载替代。构造网络时只关闭两个 ResNet 的额外 ImageNet 下载；随后严格加载完整 LiVOS 权重，模型的结构和 `InferenceCore.step` 不变。

模型推理代码不会自动下载权重。模型只在 CUDA 上运行。

## 4. 一次运行

```bash
cd /mnt/fast18/sunbo/DINO
conda activate dino-probe
OUT="out/livos_vehicle_$(date +%Y%m%d_%H%M%S)"

python run_livos_vehicle_test.py \
  --stage run \
  --data data/search_grid_smoke_v1 \
  --livos-repo /mnt/fast18/sunbo/LiVOS \
  --weights /mnt/fast18/sunbo/LiVOS/weights/livos-nomose-480p.pth \
  --output "$OUT" \
  --device cuda \
  --size 480 \
  --length 48 \
  --stride 16 \
  --max-cases 12 \
  --threads 4

echo "$OUT"
```

旧输出不可覆盖。权重或环境出错后重新运行时使用新的 `OUT`，保留失败目录用于排查。`--size 480` 是短边 480，不是强行变成正方形；原图 1024×768 会缩成 640×480。RGB 先转成 0～1 张量，再用 torchvision 双线性、抗锯齿缩放；首帧掩膜用最近邻缩放，和官方读取器的顺序一致。输出类别概率先恢复原图尺寸，再取概率最大的类别。

推理使用 float32，不使用混合精度。每段只跟踪一个目标，临时编号为 1，另一辆车不额外初始化；A、B 分别测试。每张图均正常更新记忆，没有根据后续真值纠正、重置或跳过写入。

## 5. 分阶段运行

如需先核对图片，在新输出目录执行以下两步。它们不导入 torch，不执行模型：

```bash
python run_livos_vehicle_test.py --stage audit \
  --data data/search_grid_smoke_v1 --output "$OUT"
python run_livos_vehicle_test.py --stage preview \
  --data data/search_grid_smoke_v1 --output "$OUT"
```

之后在同一个 `OUT` 执行推理和结果页面生成：

```bash
python run_livos_vehicle_test.py --stage evaluate \
  --data data/search_grid_smoke_v1 --output "$OUT" \
  --livos-repo /mnt/fast18/sunbo/LiVOS \
  --weights /mnt/fast18/sunbo/LiVOS/weights/livos-nomose-480p.pth
python run_livos_vehicle_test.py --stage report \
  --data data/search_grid_smoke_v1 --output "$OUT"
```

预览或推理前会验证计划里的源码和输入哈希。改动了数据或实现，就需要新建输出和计划。推理结束还会再次核对模型源码、权重和数据。

## 6. 如何读结果

所有效果统计排除首帧。每辆车、每段路线和总结果分别汇总，汇总中的总结果按评分步数加权。

| 指标 | 含义 |
|---|---|
| `mean_visible_iou` | 目标可见时，预测区域和真实区域的交集÷并集；越高越好 |
| `visible_success_rate` | 目标可见且 IoU ≥ 0.5 的比例 |
| `absent_false_positive_rate` | 目标完全不可见时，仍预测至少 32 个有效像素的比例；越低越好 |
| `two_visible_wrong_rate` | 两车同屏时，预测更像另一辆车的比例；越低越好 |
| `boundary_mean_iou` | 目标接触图边时的平均 IoU |
| `tiny_mean_iou` | 目标只剩 1～31 个像素时的平均 IoU；此类本来可能无法可靠辨认 |
| `recovery_rate` | 从不可见变为可见后，在这次连续可见期间曾恢复 IoU ≥ 0.5 的事件比例 |
| `recovery_within_3_steps_rate` | 重现后的前三个可见步骤内恢复的事件比例 |
| `mean_recovery_delay_successful` | 仅恢复成功的事件，从重现到恢复相差多少步；同一步恢复记 0 |

“跟错车”是区域诊断：另一辆车的 IoU ≥ 0.1，且比目标 IoU 高出 0.05。它不是参考图识别正确率。没有目标的帧不计 IoU=1，而是单独检查误报；无某类样本时输出 `null`。

不把上述数值叫作 DAVIS J&F。时延只统计同步后的 `InferenceCore.step`，不含加载、图片缩放、读取和评分。每步记录 5 个主要持久状态的尺寸和字节数，并记录 PyTorch 峰值显存；前者不包括模型权重和瞬时激活，也不是 LiVOS 只有那一个 64×256 矩阵。

输出目录：

- `plan.json`：固定路线、初始化、所有图片/掩膜和源码校验信息。
- `inference.started.json`：推理启动记录；不能据此认定完成。
- `inference.json`：成功结束标记、权重/源码哈希、设备、时延和资源统计。
- `predictions/`：每步预测掩膜，恢复到原图尺寸。
- `records.json`、`summary.json`：逐步评分和总体/逐车/逐段汇总。
- `report/index.html`：可直接打开的查看页面，附原图、真值和预测区域；历史只显示当前及以前画面。
- `report/metrics.csv`：逐步评分表。

生成结果页面时，会从保存的预测掩膜重新计算每一步和全部汇总，不能只读一个保存的正确/错误字段。查看页面和它的 `assets` 文件夹需一起保留。

## 7. 这一步结束后能判断什么

先看可见时是否稳定分出目标、是否被另一辆车带跑、目标离开时是否大面积误报，以及重现后能否恢复。对照逐帧图确认错误发生在哪一步。

若表现较好，说明官方完整 LiVOS 在这批车辆画面上有适用性，值得再研究其更新方式。它仍不能单独证明我们的 8 槽记忆有效，也不能证明跨大视角身份匹配已解决。

若表现较差，应区分初始化目标过小、离散转向跳变、分割边缘误差、跟错另一辆车和重现失败；不要仅凭这一组结果就认定固定大小记忆不可行。

本机阶段只运行真实图片/掩膜审计、预览、纯数据测试和静态检查；未加载模型、未提取特征、未训练或推理，未自行 SSH。目前没有车辆模型效果分数。
