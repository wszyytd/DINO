# 区域时序融合：掩膜辅助版

## 先说明本次实验的范围

本次把原来的“每张图独立判断”接成一条条连续观察路线，让车辆的证据真正跨步保留。

**这是使用仿真提示的受控实验，不是已经完成自动检测、自动跟踪或主动选动作的系统。**

- 老师的完整网络已经产生了参考图相关的热图。本次复用保存的原始数值热图，不用网页颜色反推结果。
- 老师只负责产生区域种子。老师的分数、排名、Null、热图数值都不作为自己的匹配网络的输入。
- 仿真掩膜辅助判断哪些碎片属于同一辆车、补全这辆车在当前图中**实际可见**的范围，并保持跨帧身份对应。
- 冻结编码器看到的是裁出的原始 RGB，不是掩膜、身份、老师分数或真值。参考图只用于读取和匹配，不写入记忆。
- 没有任何保留的老师活跃区域覆盖到车辆，就不会仅因为知道真实掩膜而添加这辆车。
- 路线根据相机图结构预先确定，不按目标出现、识别对错或老师分数挑路线。

因此，新结果不能直接和旧的、未使用掩膜补全的 A/B 数字比较，并把差值全部算成融合收益。融合是否有帮助，应比较本次相同输入下的 `last`、`mean`、`memory`。

## 这次改了什么

### 1. 放宽热图转框，并明确处理同车碎片

默认使用完整配置 [region_sequence_mask_assisted.json](../configs/region_sequence_mask_assisted.json)。

先按原有数值规则得到连通区域，阈值从 `max(峰值×0.45, 2/网格数)` 放宽到 `max(峰值×0.30, 1.5/网格数)`。其余扩框、合并和最多八个候选的规则不变。该修改只在新实验中启用，不修改旧 A/B 默认规则和旧产物。

随后：

1. 保留每个老师区域真正激活的网格，而不只看已经扩大的框。
2. 用这些网格与车辆真实掩膜的交集寻找老师确实支持的车辆。默认至少 16 个原图像素，且占该车可见掩膜的 1%。
3. 同一辆车的所有支持碎片合成一次观察，不会因为两个车身亮点而写两次。
4. 使用真实掩膜补全当前可见外框，四周增加 4% 留边。**这一步使用了真值，报告和查看器会明确标注。**没有补出被遮挡或没拍到的车身。
5. 一个老师区域同时碰到两辆车时，可由掩膜拆成两辆车，并记录这项辅助。只在扩框范围内、没有活跃网格支持的车辆不会加入。
6. 太小的车辆保留记录，但拒绝写入。默认至少 32 个可见像素，并且按整图缩放到 518 时的有效车辆面积至少 196。

背景候选会在仿真分组中被排除，这也是本实验的辅助条件，不能称为学会了自动排除背景。

[region_sequence_geometry.py](../dinoexp/region_sequence_geometry.py) 实现分组；[region_sequence_data.py](../dinoexp/region_sequence_data.py) 保存原始框、放宽后的框、分组来源和写入资格。

### 2. 让路线逐步连续

[region_sequence_routes.py](../dinoexp/region_sequence_routes.py) 在每个数据划分内部做固定巡查，先扫描相邻朝向，再沿邻接位置移动，同时加入反向巡查。

- 每条任务默认 24 次观察，每隔 8 步取一个窗口，窗口之间有重叠。
- 相邻观察一定对应图中一条真实的边，转向或移动经过的中间状态不会跳过。
- 当前数据对应相邻平移 4 米、相邻转向 45°。保存动作、相机位置和标称时间，加载时核对。
- 每条路线固定一张参考图。不同参考形成不同任务，查看器的“下一步”不会偷偷切换参考或跳到别的路线。
- 不跨训练和留出边界，不使用目标标签挑选“有利路线”。孤立位置不会被伪造成移动路线，未覆盖记录单独列出。
- 这仍是离散采样图片上的图路线，不是重新采集的视频，也没有证明可以无碰撞飞行。

旧的随机相邻路线本来也是图连续的。本次改动是固定、可审计、保留中间步骤，**不是声称已证明旧路线是失败根因**。

### 3. 三种方法在相同数据上逐步判断

每条路线一开始清空记忆，之后按顺序更新。每辆车占稳定的位置；训练时可以打乱车辆顺序，但整条路线使用同一个排列，标签随之正确映射。

|方法|每辆车保存什么|离开画面后|
|---|---|---|
|`last`|最近一次获准写入的观察，不混合多个视角|保留最近写入，不丢掉这辆车|
|`mean`|历次有效写入的平均|保留平均结果|
|`memory`|由可训练更新器维护的融合结果|保留融合结果|

默认三种方法都对同一路线内**完全相同的裁图**去重，只写一次。这不是近似图片去重：相近的新视角仍会写入。车辆再次遇到此前已经写过的完全相同图片时，`last` 也不再写入，因此这里称“最近一次写入”，而不是强行声称始终等于当前照片。

三种方法具有相同的路线、特征、标签、类别权重、公共参数初值、批次顺序和优化更新次数。每种方法训练 100 轮，随机种子 7、17、27，批次 4，学习率 `1e-4`，梯度裁剪 1。训练前缀中的“目标 A / 目标 B / Null”按训练集频率平衡，**不使用留出结果决定权重**。固定使用最后一轮，不根据 probe/test 挑最好的一轮。

[region_sequence_training.py](../dinoexp/region_sequence_training.py) 执行逐步融合和训练。默认记忆仍是每辆车 8×256 的表示，没有擅自换成一个更大的新模型。

## 两套真值必须分开

- `accepted_truth`：截至当前，目标是否曾经获得允许写入的区域。如果没有，匹配答案为 Null。
- `physical_truth`：截至当前，目标是否曾经在原图中物理出现。老师漏检或区域被拒写**不会改变这套真值**。

因此，物理上出现过目标但一直没接收到区域，回答 Null 可能符合第一套标签，但会在第二套评价中失败。报告同时给出：

- 已接收到目标时的正确率；
- 两辆车都已有证据时的正确率，以及忽略 Null 后的两车排序；
- 当前没有目标有效区域、只能依靠历史时的正确率；
- 物理上出现过目标时的成功率；
- 全体准确率及对应的“始终选 Null”基线；
- 普通白车和警车分别的结果、每项分子分母。

当前仍然只有一个布局、两辆车和固定参考图。路线窗口相互重叠，前缀不是独立样本。这些数字适合排查，不证明独立场景泛化。

## 历史有没有帮忙：附带的同权重检查

只在留出路线对 `memory` 做以下干预，不重新训练：

- `reset`：每步清空，只用本步有效区域。原有历史真值不变。
- `swap_content_before_current`：在正常的此前状态副本上只交换已见车辆的记忆内容，不交换 `seen`、`count`，然后处理本步。至少两车此前已有证据才统计。
- `pause_current_write`：在目标此前已接收到证据、且本步原本有写入时，跳过本步写入，保留此前状态。
- `repeat_bypass_dedup_extra1/8`：每条路线首次遇到有效观察时，故意绕过去重额外写同图 1 次、8 次，检查更新器本身是否漂移。

所有干预从正常前缀的副本开始，不污染下一步，不改变真实标签。报告按同一批合格样本配对比较，不混用不同分母。强制重复写入是压力测试，不代表默认流程会重复写入。

## 文件入口

- [run_region_sequence.py](../run_region_sequence.py)：`plan / preview / prepare / train / report / run`。
- [region_sequence_mask_assisted.json](../configs/region_sequence_mask_assisted.json)：完整冻结配置。
- [test_region_sequence_data.py](../tests/test_region_sequence_data.py)：纯数据测试，不导入 torch。
- [test_region_sequence_server.py](../tests/test_region_sequence_server.py)：服务器明确启用的 CUDA 模型检查，不在本机运行。
- [region_sequence_viewer_smoke.cjs](../tests/region_sequence_viewer_smoke.cjs)：用简化页面环境检查按钮、参考图和因果历史，不是截图或浏览器视觉验收。

## 服务器直接运行

**以下命令由用户在已经授权的 CUDA 服务器执行。本机不运行神经网络，不提特征，不训练，不评估，也不自行 SSH。**

```bash
set -euo pipefail
cd /mnt/fast18/sunbo/DINO
conda activate dino-probe

git switch codex/evidence-memory-v1
git pull --ff-only origin codex/evidence-memory-v1

# 只跑数据检查；不要在本机把它换成整个 tests 目录。
python -m unittest tests.test_region_sequence_data tests.test_region_geometry tests.test_region_metrics tests.test_region_data -v

# 此行仅在授权 CUDA 服务器执行，失败即停止。
RUN_REGION_SEQUENCE_CUDA_TESTS=1 python -m unittest tests.test_region_sequence_server -v

REGIONS=out/region_ab_20261111_191014
DATA=data/search_grid_smoke_v1
OUT="out/region_sequence_mask_$(date +%Y%m%d_%H%M%S)"
mkdir -p out

# 只读取已保存热图、RGB 和掩膜，不会重新运行老师定位。
python run_region_sequence.py plan \
  --data "$DATA" --regions "$REGIONS" --output "$OUT" \
  --config configs/region_sequence_mask_assisted.json
python run_region_sequence.py preview --data "$DATA" --output "$OUT"

# 原始车辆裁图 -> 冻结编码器。现有 DINO 路径可由程序自动查找，或沿用已有 DINO_REPO/DINO_BACKBONE。
# 必须是与旧 A/B 一致的官方 ViT-B/14 和同一个任务适配器，不会下载或回退 CPU。
python run_region_sequence.py prepare \
  --data "$DATA" --output "$OUT" \
  --teacher out/teacher_grid_fit_e20_retry/last.pt \
  --device cuda --threads 4 --view-batch 1 \
  2>&1 | tee "${OUT}/prepare.log"

# 自动执行模型检查和两次真实小批次试运行，再从相同初值训练九组。
python run_region_sequence.py train \
  --data "$DATA" --output "$OUT" --device cuda --threads 4 \
  2>&1 | tee "${OUT}/train.log"

# 仅用保存的分数出图与报告，不重新做模型前向。
python run_region_sequence.py report --data "$DATA" --output "$OUT"
printf '\n结果目录：%s\n查看器：%s/report/index.html\n' "$OUT" "$OUT"
```

`REGIONS` 必须是已经完整准备的旧 A/B 输出，至少保留原始 `prepared.json` 与 `proposals/records.json`；其 RGB、掩膜和参考图必须能在 `DATA` 中通过哈希校验。找不到该目录时先确认服务器上的真实位置，**不要为凑路径重新训练或改标签**。

程序会校验老师权重、官方 DINO 权重、预处理和相关 DINO 源码是否与该 A/B 来源一致。如果自动定位不到现有 DINO 文件，在 `prepare` 额外指定 `--model-repo`、`--backbone` 的现有路径；不要复制文档中的假占位路径。使用完整、匹配的官方资源，不自动下载。

输出必须是新目录。分阶段失败会保留现场，不自动删改或把部分结果标成完成。代码改变后需要重新 `plan`；不要在旧计划或旧特征上混跑新代码。Windows 预览产物不应拷到 Linux 继续训练，服务器按当前代码生成自己的新计划。

### 一条命令完成全套

确认输入和服务器资源后，也可使用下列命令，它与分阶段流程使用同一套检查（不会额外生成训练前的几何预览）：

```bash
OUT="out/region_sequence_mask_$(date +%Y%m%d_%H%M%S)"
python run_region_sequence.py run \
  --data data/search_grid_smoke_v1 --regions out/region_ab_20261111_191014 \
  --output "$OUT" --config configs/region_sequence_mask_assisted.json \
  --teacher out/teacher_grid_fit_e20_retry/last.pt \
  --device cuda --threads 4 --view-batch 1
```

## 产物和验收

下面的路径相对所选 `OUT`：

|产物|检查什么|
|---|---|
|`plan.json`、`observations.json`、`routes.json`|分组依据、放宽前后框、辅助条件、连续动作、划分、源文件和代码哈希、未覆盖记录|
|`preview/index.html`|只看区域与路线；明确没有模型结果|
|`prepared.json`、`features/`、`references/`|冻结裁图特征与来源，状态应为 `complete`|
|`training/checks.json`|CUDA 前向/反向、更新器梯度、候选顺序、不能偷看未来、无写入不变、交换只改内容、最近观察保留、两次真实试运行|
|`training/run.json`|九组结果、相同初值与批次预算、最终状态 `complete`、重载复现最大分数误差不超过 `2e-5`|
|`training/{方法}_seed{种子}/{划分}.json`|逐步选择、两套真值、每车状态摘要/变化、写入次数、同权重干预和原始分子分母|
|`report/index.html`|一条路线内前进/播放；当前裁图、最近写入、此前已写的不同图片、三方法同一步结果和历史干预|
|`report/report.md`、`report/summary.json`、`report/metrics.csv`|逐种子与平均结果，不能只看包含大量 Null 的总准确率|

查看器依赖同目录 `assets/`，拷回本机时请完整拷贝 `report` 目录。页面不联网。没有插入未来裁图；切换路线、参考图或种子时不会沿用另一任务的状态。

缓存前向计时不含老师定位、编码和图片读取，**不能当完整系统的在线时延**。

若需要简单的页面逻辑检查（不启动浏览器），在有 Node.js 的环境执行：

```bash
node --check tests/region_sequence_viewer_smoke.cjs
node tests/region_sequence_viewer_smoke.cjs "$OUT/preview/index.html"
```

## 本机已做与尚未做

本机只进行静态检查、纯 numpy/几何测试以及读取已保存热图的真实路线预览；检查中明确确认未导入 torch、未运行模型、未提取特征。

服务器模型测试、真实裁图编码、九组训练、检查点重放和识别效果尚需按上述命令实际运行。不能因为代码检查通过，就说融合已经改善准确率。
