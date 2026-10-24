# 参考图与候选图的可训练局部匹配

这次实现老师思路中的第一步：冻结DINOv2，保留局部空间特征，通过可训练适配器和注意力学习参考与候选的关系。先使用现有CARLA车辆裁剪，与原有整车CLS余弦做同图对照，不需要车门车窗分割。

这是自行实现的候选匹配实验，未使用老师的源码或训练权重。尚未实现文档中的整幅航拍图加15分块、密集搜索网格融合与Null分支，也没有实现时序证据聚合、WM动作预测或候选排除。

## 模型具体做什么

1. 每张车辆裁剪等比例缩放并补边，默认224×224。读取DINOv2第3、6、9、12层的patch特征。完整补边patch不参与注意力；与真实内容部分重叠的patch保留。
2. 参考和候选各有独立适配器。各层投影到默认128维，学习层间权重，加入二维位置编码和残差MLP。DINOv2始终冻结，特征先缓存一次。
3. 参考特征在有效内容范围内池化到最多8×8。4个可训练查询从参考局部token读取描述。这些查询没有人工指定的车门、车窗语义。
4. 每个候选位置先与参考局部和整体描述比较，将最大局部相似度、平滑聚合及整体相似度经MLP加入候选特征，再做两层交叉注意力。
5. 参考描述读取更新后的候选特征，匹配头输出一个logit。训练监督为参考和候选是否属于同一目标身份，采用带类别权重的BCE。
6. 同时记录相同预处理、相同骨干最终层CLS的余弦作为基线。网络sigmoid分数尚未校准，不能当作可靠身份概率或排除阈值。注意力图仅用于检查模型行为。

默认ViT-S/14的输入维度为384，方便复用已有官方权重。可选择ViT-B/14和518输入；这只改变骨干和分辨率，不会自动成为老师完整模型的复现。

## 服务器首次运行

直接复用服务器已经能运行DINOv2的`dino-probe`环境。不升级已有Torch或驱动。

```bash
cd /mnt/fast18/sunbo/DINO
git pull --ff-only origin main
conda activate dino-probe
python -m pip install -r requirements-local-match.txt
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

模型代码和官方骨干权重需要指定实际路径。下面环境变量仅是位置示例，改成服务器现有路径；匹配`dinov2_vits14`的必须是ViT-S/14权重。无需老师的权重、SAM、Ultralytics或CARLA仿真器进程。

```bash
export DINO_REPO=/实际路径/dinov2_repo
export DINO_BACKBONE=/实际路径/dinov2_vits14_pretrain.pth

python run_local_probe.py \
  --data data/probe_20261019_020829_292286 \
  --model-repo "$DINO_REPO" --backbone "$DINO_BACKBONE" \
  --output out/local_match01 --device cuda --epochs 30 --dry-run

# 去掉dry-run，实际依次配对、提特征、训练和导出
python -u run_local_probe.py \
  --data data/probe_20261019_020829_292286 \
  --model-repo "$DINO_REPO" --backbone "$DINO_BACKBONE" \
  --output out/local_match01 --device cuda --epochs 30
```

任何阶段失败立即停止，已完成的文件保留，重试选择新输出名。`--dry-run`不会建立目录。没有CUDA时可用`--device cpu`。显存不足可以先降低`--batch-size 1`。

首次运行使用全部20组已确认配对做**训练拟合检查**，不是验证实验。高空图可能本身没有足够身份信息，监督标签并不保证该张图可辨认，不应把训练目标误读成模型必须在所有视点给出确定身份。

当前只有一张主要参考，不能据此证明模型学会了随参考变化的匹配，而非记住A/B外观。后续需要更多参考、交换参考后的配对和独立车辆场景。

## 四步分开执行

便于重复训练时复用冻结特征：

```bash
python prepare_pairs.py \
  --data-root data --dataset probe_20261019_020829_292286 \
  --group charger-location0-pilot --output out/pairs01.json

python cache_local_features.py \
  --manifest out/pairs01.json --data-root data \
  --model-repo "$DINO_REPO" --checkpoint "$DINO_BACKBONE" \
  --output out/features01 --device cuda

python train_local_matcher.py \
  --train-manifest out/pairs01.json --cache out/features01 \
  --output out/head01 --allow-train-only --epochs 30 --device cuda

python evaluate_local_matcher.py \
  --manifest out/pairs01.json --cache out/features01 \
  --checkpoint out/head01/last.pt --data-root data \
  --output out/report01 --allow-training-data --device cuda
```

`--checkpoint`在提特征脚本中指官方骨干，在评估脚本中指自己训练出的匹配头，不能互换。

## 怎样检查结果

- `train/history.json`：每轮训练loss。只有提供独立验证清单才记录val loss并导出`best.pt`。训练试跑只导出`last.pt`。
- `evaluation/index.html`及`assets/`：离线图片对照和平均注意力图，下载整个evaluation目录后打开。
- `evaluation/scores.csv`：每张候选的真值、view、匹配logit、未校准分数和CLS余弦。
- `evaluation/summary.json`：相同参考、场景和视点中，正例能否排在负例前，以及逐视点结果。没有正例或没有负例的组单独统计，不强行计算正确率。平分不计成功。
- `evaluation/attention.npz`：各学习查询的候选位置注意力。HTML中的热图按图片单独归一化，不可跨图比较红色强度。
- `features/cache.json`、`train/training_run.json`：骨干与图片哈希、尺寸、层号、配置和完成状态。

## 独立验证如何接入

目前左右交换和高低分辨率数据仍属于相关场景。不能把它们随机分两半，当作跨场景验证。所有相关采集必须使用相同`--group`。程序还检查跨划分的全部参考和候选图片哈希，换个组名也不能绕开重复图片检查。

准备独立车辆、位置、参考与采集运行的验证集，分别生成清单。提取缓存时重复给`--manifest`即可纳入两份：

```bash
python cache_local_features.py \
  --manifest out/train_pairs.json --manifest out/val_pairs.json \
  --data-root data --model-repo "$DINO_REPO" --checkpoint "$DINO_BACKBONE" \
  --output out/trainval_features --device cuda

python train_local_matcher.py \
  --train-manifest out/train_pairs.json --val-manifest out/val_pairs.json \
  --cache out/trainval_features --output out/head_trainval --device cuda
```

组ID只是人工声明，不能自动发现两次拍摄在物理场景上是否独立。真正泛化还需独立测试集，并控制车辆、参考、背景与观察几何捷径。当前任务是确认能否学到比固定CLS比较更有用的匹配关系。

## 代码验证

```bash
python -m unittest discover -s tests -v
```

未安装Torch时，模型训练测试明确跳过。安装Torch的环境应运行全部测试，包含补边不变性、梯度流、微型合成集拟合、缓存完整性、训练验证隔离和流程失败停止。合成测试仅证明接口和可训练性，不代表真实效果。

官方接口参考：[DINOv2中间层特征](https://github.com/facebookresearch/dinov2/blob/main/dinov2/models/vision_transformer.py)、[PyTorch安装选择](https://pytorch.org/get-started/locally/)。

## 本地已实际运行的检查

2026年10月24日，Windows CPU、Python3.12.14、Torch2.14.1+cpu，使用已有官方ViT-S/14权重和高分辨率采集数据：

- 20组配对包含10个正例和10个负例，共21张不同图片，四层真实空间特征已提取。
- 64维、1层交叉注意力的小配置训练12轮，训练loss从0.72870降至0.50482，权重重载和图片报告导出成功。
- 默认128维、2层交叉注意力配置通过一键入口完整执行2轮，覆盖配对、缓存、训练与评估。
- 76项自动测试通过，包含原项目回归测试和新增模型／数据完整性测试。
- 这批图中，学习头和相同预处理的CLS基线都在10组比较中将正例排在前面，不能据此声称学习头更好，也不能把训练数据上的结果作为泛化成绩。

本地匹配头只用于检查流程，不随代码推送为老师权重或正式模型。服务器从官方冻结骨干和随机初始化的任务模块开始训练。
