# LiVOS：发现两车后保持同框的短平移路线

这次按“已经发现两辆车，接下来尽量让两车一直留在镜头内”选路线。相机只平移，拍摄朝向固定，每条路线最多改变一次移动方向，不在原地连续旋转，不来回循环，不为凑步数重复照片。

## 1. 已选的六条路线

基于现有 `search_grid_smoke_v1` 的 200 张照片及仿真标注选择，过程不看模型预测。每一张图中，两辆车分别至少有 8,192 个可见像素，约为 1024×768 原图面积的 1.04%；每条路线的第一张，两辆车都不接触图边。后续允许部分车身截边，并在预览中逐张注明。

| 路线 | 固定拍摄朝向 | 照片顺序 | 平移距离 | 两车均不截边的照片 |
|---|---:|---|---:|---:|
| follow_00 | 0° | r00c02_d00 → r01c02_d00 → r01c03_d00 | 8 米 | 1/3 |
| follow_01 | 45° | r00c00_d01 → r00c01_d01 → r01c01_d01 → r02c01_d01 | 12 米 | 1/4 |
| follow_02 | 135° | r03c00_d03 → r04c00_d03 → r04c01_d03 | 8 米 | 2/3 |
| follow_03 | 180° | r04c02_d04 → r04c01_d04 → r03c01_d04 | 8 米 | 1/3 |
| follow_04 | 225° | r04c04_d05 → r03c04_d05 → r03c03_d05 | 8 米 | 2/3 |
| follow_05 | 315° | r00c04_d07 → r01c04_d07 → r02c04_d07 | 8 米 | 1/3 |

不同路线的朝向不同，但每条路线内部不改变朝向。每一步都沿原始相机图中存在的 4 米平移边移动，保持原来的高度、俯仰和侧倾。follow_05 是直线，其他路线只有一个折点。

共 **19 张不同照片**。每条路线分别指定 A、B 为目标独立测试，得到 **12 段单目标任务、38 次输入、排除 12 个初始化帧后的 26 个评分步骤**。这不是 38 张不同照片。每帧两车都可见；19 张中有 8 张两车均不截边，其余 11 张至少一辆车截边。

使用独立的新入口，旧巡查入口、已有计划和旧模型结果均保留。新入口复用同一套官方 LiVOS 推理、首帧掩膜初始化、输入缩放、严格权重加载和保存掩膜复算评分逻辑。

## 2. 在服务器直接测试

沿用已经安装好的 `dino-probe` 环境、同级 LiVOS 仓库和已有权重，无需重新下载。

```bash
conda activate dino-probe
cd /mnt/fast18/sunbo/DINO
git switch codex/evidence-memory-v1
git pull --ff-only origin codex/evidence-memory-v1
OUT="out/livos_follow_$(date +%Y%m%d_%H%M%S)"

python run_livos_follow_test.py \
  --stage run \
  --data data/search_grid_smoke_v1 \
  --livos-repo /mnt/fast18/sunbo/LiVOS \
  --weights /mnt/fast18/sunbo/LiVOS/weights/livos-nomose-480p.pth \
  --output "$OUT" \
  --min-view-pixels 8192 \
  --max-routes 6 \
  --size 480 \
  --threads 4

echo "$OUT"
```

`run` 依次执行数据核验、路线计划、真实图片预览、模型推理和结果页面生成。输出目录必须是新目录；不会覆盖旧结果。每条路线分别从空状态开始以 A、B 为目标测试，只在第一张提供相应车辆的掩膜，后续只输入照片。每条路线的实际长度为 3～4 张。

已有权重的 SHA256 为：

```text
9624de29038b1c8d64aa3f0ecc212aceea094abe3539666055f3c256dfca615e
```

## 3. 先看照片再运行模型

以下两步只读取照片、掩膜和路线，不导入 torch，不运行模型：

```bash
python run_livos_follow_test.py --stage audit \
  --data data/search_grid_smoke_v1 --output "$OUT"
python run_livos_follow_test.py --stage preview \
  --data data/search_grid_smoke_v1 --output "$OUT"
```

在同一输出目录中再执行推理和结果生成：

```bash
python run_livos_follow_test.py --stage evaluate \
  --data data/search_grid_smoke_v1 --output "$OUT" \
  --livos-repo /mnt/fast18/sunbo/LiVOS \
  --weights /mnt/fast18/sunbo/LiVOS/weights/livos-nomose-480p.pth
python run_livos_follow_test.py --stage report \
  --data data/search_grid_smoke_v1 --output "$OUT"
```

已经做过 `audit` 的目录应继续执行 `preview/evaluate/report`，不要再对它执行 `run`，因为 `run` 会重新建计划并拒绝覆盖。

## 4. 看哪些文件

- `routes.html`：相机平移路线图、车辆位置、每张照片状态、两车可见像素和截边情况；包含照片预览及模型结果入口。
- `preview/index.html`：真实原图与标注逐张查看，支持切换目标、播放和查看已观察历史。没有模型预测。
- `plan.json`：固定路线、A/B 配对、原始数据与共用实现哈希，以及新路线筛选代码的哈希。
- `report/index.html`：服务器推理后的原图、真值、预测区域与评分。
- `records.json`、`summary.json`：逐步评分和按路线、车辆汇总的结果。

路线图中的蓝线表示位置移动，数字表示照片顺序，绿箭头表示固定拍摄朝向，橙色 A/B 表示车辆位置。请保留完整输出目录，页面依赖相邻的图片文件。

## 5. 当前能说明什么

当前素材仍是一个场景中的两辆静止车辆，模拟相机发现车辆后保持观察。相邻照片间隔 4 米，没有中间视频帧；不能当作真实无人机平滑跟随移动车辆的长录像。路线是借助仿真真值离线挑选的，不是模型自己规划出的飞行路线。

固定朝向、纯平移且两车全程不截边的连续片段最多只有两张。保留三张以上的路线时，现有素材无法保证两车全程完整入镜。因此这里区分“仍在画面里”和“整辆车都在画面里”，不把极小的车身边条算作足够的同框观察。

本轮没有目标消失或重新出现的评分样本，相关指标将为 `null`。主要看两车同框时能否继续分出指定目标，特别注意是否把两辆车一起圈入。原有 IoU ≥ 0.5 的分割成功率不能单独当作具体车辆身份识别准确率。

本机只做纯数据核验、真实图片预览、静态检查与文件测试，不运行模型、训练或 SSH；本轮没有新的模型效果分数。
