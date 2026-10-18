# CARLA 照片的车辆部件分割

先运行公开的 AutoInspect 部件分割权重，检查模型能否在当前 CARLA 图像中找到车门、侧窗、车顶。
本入口不训练、不控制 CARLA，也不计算身份概率或自动排除车辆。

## 1. 更新与依赖

在 Linux 服务器执行：

```bash
cd /mnt/fast18/sunbo/DINO
git pull --ff-only origin main
conda activate dino-probe
python -m pip install -r requirements-parts.txt
```

需要 Python 3.10+。Ultralytics 和 Hugging Face 依赖单独列出，不影响只运行采集脚本的环境。
如果已有可用的 GPU PyTorch，先确认 `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`。
本次新增的 `segment_parts.py` 不需要 CARLA 服务器运行；它读取此前拍照得到的文件。

## 2. 先检查输入，不下载模型

```bash
python segment_parts.py --data data/model_suite02 --dry-run
```

也可传入一个场景，例如 `--data data/model_suite02/charger_variant_Aleft`。
传入的是包含 `metadata.csv` 的场景目录或包含多个场景的 suite，不是 `out/model_suite02`。
当前这组数据应显示两场景、22张输入（每场景1张参考+10张候选）。

脚本使用 `metadata.csv` 中的 `crops/` 图片。这些裁剪来自采集时 CARLA 车辆包围盒投影，
不是新增的整车YOLO检测结果；A/B编号只用于输出和真值查看，不参与模型预测。
拒收未完成的采集、损坏或缺失的裁剪、近乎全黑的输入，并记录被采集器拒绝的行。

## 3. 运行部件分割

```bash
python segment_parts.py \
  --data data/model_suite02 \
  --output out/parts01 \
  --device auto
```

首次自动下载 `mitbersh/car-parts-segmentation` 的 `parts_segmentation.pt` 到 `models/vehicle_parts` 缓存。
模型卡标为 YOLO26-s、33类、输入768。默认只预测 door/window/roof 类；要查看全部部件，加 `--parts all`。
`auto` 优先可用CUDA，CPU可用 `--device cpu`，指定GPU可用 `--device 0`。

输出目录存在时拒绝覆盖；重跑用 `out/parts02` 等新名称。推理失败时，
`parts_run.json` 会记录 `status: incomplete` 和错误，不把半次运行报告成成功。

### 服务器无法下载时

可以自行下载权重后指定本地文件：

```bash
python segment_parts.py \
  --data data/model_suite02 \
  --output out/parts01 \
  --weights models/parts_segmentation.pt \
  --device auto
```

公开权重地址：
[AutoInspect 模型文件](https://huggingface.co/mitbersh/car-parts-segmentation/tree/main)。
普通 `yolo*-seg.pt` 的COCO权重不是部件权重，脚本会检查模型任务和类别，避免误用。
默认从HF的main版本下载，实际权重SHA256、请求版本、类别表和软件版本保存在运行记录中。
需要固定模型版本时，可用 `--revision` 指定HF提交版本，或始终使用同一个本地权重文件。

## 4. 查看与下载结果

把整个 `out/parts01` 下载到本地，打开 `index.html`，不要只下载HTML。

主要输出：

```text
out/parts01/
  index.html               # 每张参考/候选的叠加图、类别、置信度、尺寸
  parts.csv                # 每个预测实例一行；无检测也有明确记录
  parts.json               # 按输入照片组织的全部预测
  parts_run.json           # 权重哈希、参数、版本、输入来源、运行状态
  scenes/
    000_charger_variant_Aleft/
      s0000_source.png     # 原始车辆裁剪，未画框
      s0000_overlay.png    # 仅供查看的掩膜叠加图
      s0000_00_*_mask.png  # 与原始车辆裁剪等尺寸的0/255掩膜
      s0000_00_*_crop.png  # 原始局部矩形裁剪
      s0000_00_*_masked.png # 掩膜外填充中性色的局部裁剪
      s0000.json           # 该输入的预测记录
      roi/*.json           # 导出给现有match.py的同类别ROI
```

样本编号与文件的对应关系见 `parts.json`，不假设 `s0001` 永远属于某个视角。
位置和像素数量都在原始车辆裁剪坐标中计算；掩膜外接框的右、下边界为exclusive。
叠加图可能放大用于显示，但输入768不代表车辆原本有768像素，也不会增加真实细节。

重点检查：

1. 参考图的门、窗是否圈对。
2. 俯视是否出现把风挡误报为侧窗、把车顶误报为门等问题。
3. 左/右侧标签与实际车辆是否一致。模型预测标签不是几何真值。
4. 门掩膜是否混进车窗、路面，是否包含警车的涂装区域。
5. 小图中掩膜原始像素是否足够，而不只是看放大图好像很清楚。

`confidence` 是模型检测置信度，不是可检验性，也不是“不是目标”的概率。
`no_detection` 表示在本次阈值下没有所选部件预测，不表示部件不存在或车辆可以排除。
所有记录均标为 `testability_validated: false`，需要先检查泛化效果。

## 5. 检查后再接现有 DINOv2

分割脚本按精确部件类别导出 `roi/*.json`。例如参考和查询都只接受 `left_front-door`，
不把 `right_front-door` 自动作为对应部位。某图同类预测多于一个时标为歧义，
不简单选最高置信度；参考缺少唯一预测的类别不导出ROI文件。

先检查运行日志/HTML里实际生成了哪些ROI，选定且复核一个文件再执行：

```bash
python match.py \
  --data data/model_suite02/charger_variant_Aleft \
  --roi out/parts01/scenes/000_charger_variant_Aleft/roi/left_front-door.json \
  --output out/parts_dino01 \
  --device auto
python report.py --results out/parts_dino01
```

上面是路径示例，不保证该类在本次参考里一定检出；文件不存在时，应先查看 `index.html` 和实际ROI列表。
`visible=true` 为兼容旧接口，仅表示存在唯一同类模型预测，绝不是已经验证可见性或可比性。
误标左右侧、低分辨率、遮挡仍可能让比较不成立。可手工把未通过检查的条目改为 `visible=false`。

现有 `match.py --roi` 比较的是掩膜外接矩形，**并非掩膜内patch特征**。
`*_masked.png` 另行保留供下一步掩膜比较实验。当前先验证自动定位与分割，
不将检测置信度、DINO余弦或深色像素比例相乘当成身份概率。

## 来源与本地验证范围

- [模型说明与权重](https://huggingface.co/mitbersh/car-parts-segmentation)
- [对应33类数据配置](https://huggingface.co/datasets/mitbersh/car-parts-segmentation-yolo/blob/main/dataset.yaml)
- [原始HITL部件数据](https://humansintheloop.org/resources/datasets/car-parts-and-car-damages-dataset/)

代码用原生掩膜、空结果和不匹配类别的测试检查输出语义，并在现有CARLA文件上检查输入。
合成结果测试只验证坐标、文件导出与错误处理，不代表公开模型在CARLA上有效；
是否能准确找到部件，以服务器真实权重推理和图像复核为准。
