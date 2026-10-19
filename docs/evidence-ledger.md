# 高分辨率采集与部位证据记录

这一轮先验证：提高原始相机分辨率，是否改善部件定位；再保存可追溯的局部比较与人工审核。
没有新增自动排除、身份概率、自动可检验性估计、世界模型或无人机控制。

## 1. 同位置、高分辨率重拍

CARLA 启动在 Town10HD_Opt，场内没有其他车辆，停止其他控制同步 tick 的客户端。
在 CARLA 环境运行：

```bash
cd /mnt/fast18/sunbo/DINO
git pull --ff-only origin main
conda activate carla10

python capture_suite.py --suite configs/model_suite_no_poles.json \
  --width 2048 --height 1536 --dry-run

python capture_suite.py --suite configs/model_suite_no_poles.json \
  --width 2048 --height 1536 \
  --reference-source data/model_suite02/charger_variant_Aleft \
  --output data/model_suite03
```

仅当确认其他同步 tick 客户端已停止、当前由本程序负责时，才补 `--take-tick-ownership`。
同一参数也可用于 `capture_scene.py`。宽高必须一起指定，保持原宽高比。
改变的是 RGB 传感器原始像素，不是 YOLO 的 imgsz，也不是放大已有图片。
车辆、FOV、相机位置与视角配置不变。车辆重新生成后的实际位置、光照渲染仍须检查。
旧参考图保持不变，只提高查询图的分辨率；新拍的高分辨率参考图会保存在
各组的 `images/local_reference.png` 和 `crops/local_reference.png`，暂不参与比较。
因此不要将 `run.json` 的新相机尺寸误认为复用的旧参考图尺寸。

## 2. 部件分割

```bash
conda activate dino-probe
python -u segment_parts.py --data data/model_suite03 --output out/parts03 \
  --weights models/vehicle_parts_local/parts_segmentation.pt --device cpu
```

如果 GPU 环境已修复，可改为 `--device auto`。先检查原始裁剪、掩膜与叠加图。
检测分数不是定位正确率、可检验概率或身份概率。

## 3. 生成证据页

不用再次运行模型也能先建立记录：

```bash
python evidence.py --parts-run out/parts03 --output out/evidence03
```

也可以对已有 `out/parts02` 先试。默认检查 `right_front-door`、`right_back-door`，
这是当前侧面参考图的待审核部位。类别来自模型预测，不是人工确认的对应关系。
其他参考可显式指定，例如 `--parts left_front-door left_back-door`。
同类重复预测不会选最高分代替，不会用左门替换右门，缺少参考部位也不会自动补全。

增加局部 DINOv2 比较（首次可能下载官方代码和权重）：

```bash
python -u evidence.py --parts-run out/parts03 --output out/evidence03-dino \
  --compare --device cpu
```

离线使用已准备的仓库与权重：

```bash
python -u evidence.py --parts-run out/parts03 --output out/evidence03-dino \
  --compare --device cpu \
  --model-repo /path/to/dinov2 \
  --checkpoint /path/to/dinov2_vits14_pretrain.pth
```

使用同一 ViT-S/14、CLS、默认224尺寸、等比例填充和 ImageNet 归一化。
编码的是分割区域的**外接矩形裁剪**；掩膜用于展示，不是 mask token 池化。
仅唯一同类、宽高至少2像素的参考和查询预测会计算诊断分数。
这个2像素约束只是避免空区域，不表示足够清晰或可检验。
分数不改变证据状态，不设相似度排除阈值；未比较时保存 null，不填零。

每次选择新的输出目录。模型失败时 `evidence_run.json` 标记 incomplete，
已保存的证据页仍可检查，不应将其当作完成的 DINO 运行。

## 4. 人工审核与更新历史

下载整个 `out/evidence03-dino` 文件夹，浏览器打开 `index.html`。
查看参考、候选原图、分割叠加图和局部区域后，为可审核记录填写：

- 审核人和具体观察依据。
- 参考定位正确、候选定位正确、同侧对应部位、候选身份连续、清晰无遮挡、该特征在任务期间稳定。
- 待审核、仍不确定、相容或矛盾。不能区分阴影与真实涂装时选择不确定。

“相容”仅表示这个部位未形成反对证据，不表示确认身份。
相容/矛盾需全部检查成立；未检出或重复检出的部位只能待审核或不确定。
如果定位错误，需要修复或重新采集、重新分割，不能通过勾选检查强行绕过。

点击“导出审核 reviews.json”，上传该文件后：

```bash
python evidence.py --parts-run out/parts03 --reviews reviews.json \
  --output out/evidence03-reviewed
```

这个命令只更新人工证据，不重新运行 DINO；若也需要在新输出保存分数，补上
相同的 `--compare`、设备与离线模型参数。
审核绑定部位、图片内容、定位与拍摄元数据的 fingerprint；不同图像、不同运行不能默默套用旧审核。

历史在**同一分割运行内，按场景、候选ID、部位**分开整理，不跨左右位置交换的场景合并A/B。
当前没有跨采集运行的持续跟踪或记忆，也没有将运动过程当成连续轨迹。

- 未知记录不会抹去已有审核证据。
- 重复帧不会累加票数，也不相乘似然；同视点组数仅供描述，不是独立证据数量。
- 同一部位既有相容又有矛盾，保存 conflict，等待核查，不做多数表决。
- 不同部位的相容不会消除已有矛盾。
- 所有候选的 decision 始终为 retain，包括存在人工矛盾记录的候选；自动排除标准尚未校准。

建议动作仅是固定提示：检查定位、取得清楚的对应部位、复核矛盾或解决冲突。
它不会调用 CARLA，也不是学习到的世界模型。靠近后的真实照片仍需重新经过同一流程。

## 输出

| 文件 | 用途 |
|---|---|
| index.html、assets/ | 可离线下载的图片对照与审核页 |
| evidence.json、evidence.csv | 按观察与部位保存记录和可选余弦分数 |
| candidates.json | 每辆候选的分部位历史状态，全部保留 |
| reviews_template.json | 待审核或已导入的审核模板 |
| evidence_run.json | 输入摘要哈希、审核文件哈希、完成状态和可选模型设置 |
| features.npz | 启用比较且有可比较区域时的原始特征及输入路径 |

## 验证

```bash
python -m unittest discover -s tests -v
```

本地可验证记录、审核、历史规则及已下载数据的 DINO 推理；
2048×1536 的实际拍摄及定位改善仍需在服务器 CARLA 上验证。
