# LiVOS：发现两车后的较长往返观察

当前默认选一条路线：**去程 15 张不同照片，沿原路返程 14 步，共 29 步**。相机以平移为主，沿途穿插四次分散的 45° 转向，不在同一位置连续旋转。折返点改变行进方向，相机继续看车辆；现有数据不包含无人机原地转头 180° 的连续视频。

## 1. 真实素材与路线

素材仍为 `search_grid_smoke_v1`：一个场景、两辆静止车辆、200 张已采集图像。路线只使用真实存在且可以反向连接的拍摄状态，不插值或生成新照片。筛选过程只看照片元数据和仿真标注，不看模型预测。

| 项目 | 当前默认路线 |
|---|---:|
| 路线数 | 1 |
| 去程照片 | 15 张，不重复状态 |
| 返程输入 | 14 步，重访去程照片 |
| 往返总步数 | 29 |
| 不同照片数 | 15 |
| 不同相机位置 | 11 |
| 往返平移 | 20 次，共 80 米 |
| 往返转向 | 8 次，每次 45°，各次之间有平移 |
| 两车都不截边 | 10/29 步 |
| 全程每车最小可见面积 | 13,145 原图像素 |

筛选门槛为每张照片中每辆车至少 8,192 个可见像素；首帧两车均不接触图边。**后续允许截边，同框不代表完整车身。** 当前路线有 19 步至少一辆车截边，预览会逐步标出。

去程顺序如下；第 15 张是折返点：

```text
01 r04c00_d03
02 r03c00_d03
03 r03c00_d02
04 r02c00_d02
05 r01c00_d02
06 r01c00_d01
07 r00c00_d01
08 r00c01_d01
09 r00c01_d00
10 r00c02_d00
11 r00c03_d00
12 r00c03_d07
13 r00c04_d07
14 r01c04_d07
15 r02c04_d07  ← 折返点
```

返程按第 14、13、……、1 张的顺序重访，不重复插入折返点。返程的相同照片经过了不同的观察历史，可以用于检查状态随历史改变后的输出，但不能当作新的独立图像或未见角度测试。

同一路线分别指定普通白车 A、白色警车 B 为目标，独立从空状态开始。因此共有 **2 段任务、58 次输入、排除两个首帧后的 56 个评分步骤**。这仍然只有 15 张不同照片。

## 2. 服务器直接运行

沿用已安装的 `dino-probe`、同级 LiVOS 仓库及已经下载的权重。

```bash
conda activate dino-probe
cd /mnt/fast18/sunbo/DINO
git switch codex/evidence-memory-v1
git pull --ff-only origin codex/evidence-memory-v1
OUT="out/livos_roundtrip_$(date +%Y%m%d_%H%M%S)"

python run_livos_roundtrip_test.py \
  --stage run \
  --data data/search_grid_smoke_v1 \
  --livos-repo /mnt/fast18/sunbo/LiVOS \
  --weights /mnt/fast18/sunbo/LiVOS/weights/livos-nomose-480p.pth \
  --output "$OUT" \
  --min-view-pixels 8192 \
  --max-routes 1 \
  --max-outward 17 \
  --beam-width 128 \
  --size 480 \
  --threads 4

echo "$OUT"
```

`run` 依次核验数据、固定路线、生成预览、执行模型推理、生成结果页面。输出必须是新目录。每段任务只在首帧提供相应车辆掩膜；之后只输入图像，包括折返点和返程，均不重置状态或再次给真值。另一辆车的掩膜只用于预测后的评分。

已有权重 SHA256：

```text
9624de29038b1c8d64aa3f0ecc212aceea094abe3539666055f3c256dfca615e
```

## 3. 分阶段查看和运行

以下两步只处理已有照片和标注，不导入 torch、不提特征：

```bash
python run_livos_roundtrip_test.py --stage audit \
  --data data/search_grid_smoke_v1 --output "$OUT"
python run_livos_roundtrip_test.py --stage preview \
  --data data/search_grid_smoke_v1 --output "$OUT"
```

在同一输出继续运行模型并生成页面：

```bash
python run_livos_roundtrip_test.py --stage evaluate \
  --data data/search_grid_smoke_v1 --output "$OUT" \
  --livos-repo /mnt/fast18/sunbo/LiVOS \
  --weights /mnt/fast18/sunbo/LiVOS/weights/livos-nomose-480p.pth
python run_livos_roundtrip_test.py --stage report \
  --data data/search_grid_smoke_v1 --output "$OUT"
```

已经完成 `audit` 的目录不要再次执行 `run`，因为 `run` 会重新建计划并拒绝覆盖。预览与结果目录也不覆盖已有页面。

## 4. 输出与检查

- `routes.html`：真实位置图、相机朝向、每一步的阶段、动作、双车像素与截边记录；返程注明重访去程第几张。
- `preview/index.html`：真实原图与真值；支持逐帧、播放、切换目标、查看已经观察过的历史，显示去程、折返点和返程。没有模型分数。
- `plan.json`：完整输入哈希、公共实现哈希、新路线实现哈希、A/B 相同观察序列、阶段及重访对应关系。
- `report/index.html`：服务器模型运行后的原图、真值、预测及评分，使用相同阶段标记。
- `records.json`、`summary.json`、`inference.json`：逐步记录、分组统计、权重及运行环境信息。

保留整个输出目录，页面依赖相邻图片。原有巡查和短平移入口继续可用；新入口复用同一套严格加载权重、输入缩放、官方逐帧推理和保存掩膜复核逻辑。新实现位于 [livos_roundtrip_routes.py](../dinoexp/livos_roundtrip_routes.py)，入口为 [run_livos_roundtrip_test.py](../run_livos_roundtrip_test.py)，约束测试为 [test_livos_roundtrip.py](../tests/test_livos_roundtrip.py)。

## 5. 如何解释结果

重点看指定车辆能否在视角改变和返程时持续被分出，以及是否出现把两辆车一起圈入、误跟另一车或输出空区域。原有 IoU ≥ 0.5 成功率可能包含混车输出，不能单独称为车辆身份识别准确率。

这条路线两车全程可见，没有目标缺席或重新出现的评分样本，相应指标为 `null`。返程经过相同照片，适合检查历史更新后的稳定性；如果去程和返程结果不同，只能先说明输出与历史有关，不能直接锁定为某个门控或矩阵的问题。

现有照片相隔 4 米或 45°，不是平滑视频。路线由真值离线选定，不是自主飞行规划，也没有新的车辆身份或布局测试。搜索保留有限候选，优先较长去程、较多不同位置和完整车身帧；当前 29 步是该配置找到的结果，不声称全局最长。

## 6. 本机验证范围

已通过 32 项纯数据测试、Python 3.10 语法解析和新增 Python 文件的 Ruff 检查；实际 200 张图及掩膜核验成功，真实 29 步路线和预览已生成，记录为 `torch_imported=False; model_ran=False; features_extracted=False`。本机不运行模型、训练或 SSH，本次没有新的模型效果分数。
