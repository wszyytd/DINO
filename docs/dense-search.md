# 参考图引导航拍空间搜索

这条入口修正了之前的任务偏差：参考车辆图与航拍整图输入，输出原图搜索网格热图和Null；用可见目标掩码训练，不再用整车裁剪二分类替代定位。旧的`run_local_probe.py`和`train_more_photos.py`保留作基线。

当前是根据老师设计说明自行实现，没有老师源码或任务权重。代码测试与本地真实骨干前向通过不等于定位效果得到验证；新的CARLA同步掩码采集及正式训练需要在服务器实际运行。

## 计算链路

冻结DINOv2第3/6/9/12层 → 两套独立适配器 → 参考64个局部token和1个整体查询（2层读取） → 航拍整图＋15个源窗（3×5、至多512像素） → 几何嵌入、1层视图Transformer和整图上下文 → 三种显式相关统计＋2层8头交叉注意力 → patch分数与每视图Null → 按原图几何四邻域映射、可靠性及几何权重融合 → 全局存在性与最终空间/Null分布。

补边patch不贡献位置票；部分内容patch按真实内容面积加权；相同源窗口按重复次数去重加权。整图大小决定窗口是否实际重叠，源图大于固定布局时整图视图仍保留全局覆盖。输出网格默认32×32，可在分步缓存入口配置；老师文档没有给出统一网格的精确尺寸。

`teacher`预设为ViT-B/14、518输入、256任务维度。骨干ID是根据文档768维特征作出的实现选择，未经老师原代码核实。`small`使用现有ViT-S/14、224输入、64任务维度，只用于低成本工程检查，不是同配置复现。完整默认值记录在`configs/dense_search_teacher.json`。

`DenseSearchHead`同时返回参考token、整体描述和保持位置的条件特征z，便于后续研究证据表征。它们没有车门/车窗等预设语义。10-20对话提到的4–6个特征探头、证据聚合器、认知状态、观察信息增益与飞行动作不等同于本定位器的单个整体查询，当前没有将这些尚未定型的研究模块伪装为已完成实现。

## 旧照片为什么不能直接训练

此前630张裁剪没有保存可见实例掩码，车辆投影框也不等于可见像素标签。新入口拒绝旧的裁剪配对清单和缺失掩码，不会自动填成矩形“真值”。可以人工审核标注旧RGB，但这不是当前脚本的自动行为。

## 服务器先检查一组掩码

启动健康、空闲的CARLA0.10，确认地图仍与旧数据相同。新采集器只操纵自己生成的车辆和传感器；有其他车辆时停止，不删除它们。不修改旧照片。

```bash
cd /mnt/fast18/sunbo/DINO
git pull --ff-only origin main
conda activate carla10

python capture_dense_scene.py \
  --config data/more_photos_20261028_030040_745590/_configs/train_pair0_location0_light0_refA.json \
  --views overview descend \
  --output data/dense_mask_check_20261102
```

下载该小目录，打开`preview.html`，确认绿色覆盖参考目标的真实可见部分，蓝色是干扰车，Null帧保留干扰车但没有参考目标。只做离线参数检查可加`--dry-run`，它不连接CARLA，也不证明传感器可用。

这里不使用YOLO/SAM推断部件。实例相机只生成训练标签。RGB和实例图必须具有相同frame、分辨率和相机位姿。CARLA分割图G为实例编号低字节、B为高字节，但该编号不是Python的actor.id。程序暂时移开并复原自建的冻结车辆，检查消失且复现的编号；关联不明确、恢复不一致或像素过少时停止。之后移除自建目标拍一张明确Null，所有失败都记录`run.json`并清理自建资源，恢复设置及天气。

依据：[CARLA UE5传感器文档](https://carla-ue5.readthedocs.io/en/latest/ref_sensors/#instance-segmentation-camera)。这条新采集路径尚需用户服务器上的实际CARLA运行验证。

## 掩码小规模数据集

第一组通过后，复用旧采集的车型、位置和实际修正后的测试相机配置。先用3组训练物理配置覆盖3种资产配对，验证和测试沿用原先各一组。两车轮流作参考，每个参考采2张整图与1张Null，共10个参考、30张查询整图；新输出单独保存。

```bash
python collect_dense_search.py \
  --source data/more_photos_20261028_030040_745590 \
  --output data/dense_search_pilot_20261102 \
  --max-train-groups 3 --dry-run

log="out/dense_capture_$(date +%Y%m%d_%H%M%S).log"
nohup python -u collect_dense_search.py \
  --source data/more_photos_20261028_030040_745590 \
  --output data/dense_search_pilot_20261102 \
  --max-train-groups 3 > "$log" 2>&1 < /dev/null &
tail -F "$log"
```

中断后续采：

```bash
python -u collect_dense_search.py --resume data/dense_search_pilot_20261102
```

它核对已完成子场景的配置、图片和掩码SHA，跳过成功项；部分尝试原样归档到该输出的`_failed`。所有子场景完成并通过跨划分检查后才生成`dense_manifest.json`。源数据控制清单必须确实complete；本地若只同步了部分旧数据，应在完整的服务器目录运行。

这是验证新链路的pilot，仍只有3种训练资产，不能指望它自然解决泛化。更广泛车型/外观/背景与确认的相似负例仍是后续数据要求。Null只表示参考目标在该图不可见，不是身份排除结论。

## 官方骨干与正式空间训练

新匹配模块从随机初始化开始。老师没有提供任务权重，不影响从官方冻结骨干训练。ViT-B权重缺失时先明确下载官方骨干，不会暗中把ViT-S冒充成B：

```bash
conda activate dino-probe
python download_dino_backbone.py --model dinov2_vitb14

python run_dense_search.py \
  --manifest data/dense_search_pilot_20261102/dense_manifest.json \
  --preset teacher --device cuda --dry-run

log="out/dense_train_$(date +%Y%m%d_%H%M%S).log"
nohup python -u run_dense_search.py \
  --manifest data/dense_search_pilot_20261102/dense_manifest.json \
  --preset teacher --device cuda > "$log" 2>&1 < /dev/null &
tail -F "$log"
```

默认仿真5轮、lr2e-4、batch1。先缓存冻结特征，后训练空间模块，用验证损失选`best.pt`，最后评估测试。原来的裁剪头权重和224裁剪缓存不兼容，不会复用。输出在新的`out/dense_search_*`，打开`test/index.html`看原图热图与Null，并查看`summary.json`的空间Top1/5指标。

完整518/B/16视图缓存很大：单个独立整图的4层float32 patch特征约269MB，参考约17MB，压缩后取决于图像；先控制pilot规模与磁盘空间。缓存支持同配置自动续算：中断后，用日志中的`cache_dense_features.py`原命令继续；再用对应`train_dense_search.py`命令训练到新的输出。训练阶段当前不恢复优化器状态，中断训练应重新开一个训练输出；不能把缓存可续算说成训练可以无损续跑。

可显式用`--dtype float16`把冻结特征存储减半，训练读取后转float32；这是另一个数值配置，不能与float32缓存/权重混用。`--preset small`仅用于先检查完整流程，不能当作老师同配置的结果。不会自动升级Torch/CUDA/驱动。

## 真实数据适配

真实配对需要人工审核的正例、负例掩码与明确的未知区域。负例不能由“没标注”推断。manifest中使用`supervision: confirmed_regions`、`reviewed: true`、`mask_source: reviewed_manual`或`reviewed_sam2`；每个样本至少有确认正例和负例。负例-only或全未知样本不定义本定位CE，程序拒绝拿它们做训练/验证。

先为该真实manifest用同一骨干与几何配置另建缓存，再从仿真第3轮开始：

```bash
python train_dense_search.py \
  --manifest data/real_dense/dense_manifest.json \
  --cache out/real_dense_features \
  --checkpoint out/dense_search_RUN/train/epoch_003.pt \
  --stage real --output out/real_dense_train_RUN --device cuda
```

上面的RUN和真实数据路径需要替换为实际已存在目录；当前没有自动下载或伪造老师的v8/v9数据。默认60轮、lr5e-6，冻结相机状态嵌入、局部Null和全局存在性分支。损失仅在确认正负网格中归一化，未知区域不参与，全局Null不参加定位loss；多个正例先逐实例归一化再等权。相机位姿缺失时用缺失标志，不填写假测量。

## 验证范围

```bash
python -m unittest discover -s tests -v
```

测试覆盖几何映射、补边、重复视图权重、梯度、空间监督/Null、已知/未知区域、逐实例权重、数据隔离和完整性、检查点重载与HTML导出、实例编号的移走/复原关联、失败清理、续采跳过成功项。合成测试和真实冻结ViT-S前向仅说明工程链路可运行，不能证明真实CARLA掩码质量或训练后定位效果。
