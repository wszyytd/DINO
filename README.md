# 相似车辆辨识信号实验

## 六组对照：稍大颜色差异与目标位置交换

```bash
conda activate carla10
python capture_suite.py --output data/suite01
```

默认 `configs/scene_suite.json`：同款 Charger，A 始终白色目标，B 分别为浅银灰
`205,205,205`、中银灰 `185,185,185`、浅米灰 `220,210,190`；各拍 A 在两个位置的版本。
相较 scene01 的 B=`220,220,220`，这些设置适度增加外观差异，但真实效果需看渲染。
几何、光照、出生点与五个观察视点保持相同，6 组是同地点的对照条件，不是独立场景统计样本。
位置名称 left/right 指默认相机布局下的槽位，斜视时屏幕投影不一定左右保持。

各组共用第一组的 A 参考图。后来场景自身拍摄的参考图备份为 `local_reference.png`，
正式 reference 图像及 CSV 参考行来自第一组；`run.json.reference_source` 记录来源和图像哈希。
参考帧号／位姿属于来源场景，查询帧号／位姿属于当前场景。查询图像不被替换。
这控制参考输入变化，但不能消除裁剪背景、遮挡、光照及位置等其他影响。

一共 30 个共享观察视点、60 张候选裁剪，另有每组参考图和本地备份。
下载 `data/suite01` 整个目录后打开 `suite_preview.html`，先看六组总览，再点入各组详情。
相机仍在固定视点取图，不执行连续飞行或自动动作选择；不评价排除风险。
如任一组采集失败会停止并保留已采集数据和 `suite.json` 状态，不自动忽略失败。
已有输出目录不会覆盖，重跑请使用 `data/suite02` 等新名称。
可先运行 `python capture_suite.py --dry-run` 查看条件计划，不需要启动 CARLA。

## 当前先运行：两候选同场预览

先运行新入口，检查场景是否符合任务，不运行下面的旧单车对照：

```bash
conda activate carla10
python capture_scene.py --output data/scene01
```

配置为 `configs/two_candidates.json`。两辆 Charger 同时在场：A 白色目标，B 浅灰干扰。
颜色差异只是首轮受控辨识线索，不代表已经解决车型或具体车辆身份验证，也不能保证
在高空一定无法区分。配置颜色需要以真实渲染验收，不能仅凭属性认定相似。
两车围绕第 0 个出生点沿车头方向前后间隔 8 m，均静止；位置若不可生成则明确报错。
不自动创建标记、不修改地图资产，不使用完全相同外观来伪造实例可分性。
该方案有位置和背景捷径，后续模型实验需交换目标位置、改变朝向等控制；本轮只看场景。

输出 `preview.html`、`crops/reference.png`、五张同时包含两辆车的原图、对应候选裁剪及元数据。
参考图明确属于 A；HTML 中的 A/B 标签只用于真值查看，不是模型输入。
第一张总览距场景中心 70 m、俯视角 85°，后续下降和侧视。每个视点两个候选共享帧号与相机位姿。
这些是固定视点预览，不是完整连续飞行轨迹，不执行证据融合或候选排除。
包围盒裁剪不保证无障碍遮挡，必须查看总览及原图。曝光使用前一版的 histogram 诊断设置。
需要静态空场景及唯一 tick 主控；已同步世界需停止其他 tick 客户端后加 `--take-tick-ownership`。

下载 `data/scene01` 整个目录，先打开 `preview.html`，重点检查两车真实外观、目标参考、
初始像素大小、两车是否都在画面中及是否存在遮挡。场景通过后再决定正式采集条件。

## 旧单车对照入口

给定一张参考图，检查 DINOv2 对目标及相似干扰车在不同视角下的匹配分数，
为后续证据融合提供数据。本项目没有训练策略、世界模型或排除规则。

默认一辆目标 A、一辆干扰 B，配置为同色但不同车型，6 个视角，共 12 张查询图和
1 张单独视角的参考图。先用单光照、单位置摸底，之后在配置中增加天气和位置。
默认使用 `vehicle.lincoln.mkz` 与 `vehicle.dodge.charger`，名称来自实验室 CARLA 0.10.0
服务器的实际资产清单；颜色属性支持、渲染效果与出生点仍需真实采集验证。

采集实现借鉴 `D:\Work\carla` 中同步 tick、实际相机位姿校验、车辆包围盒投影和
异常清理的做法。本项目独立运行，不需要导入旧项目，也不修改其文件。

## Linux 服务器操作

将整个项目部署到一个目录，例如 `/mnt/fast18/sunbo/DINO`，以下命令都在该目录运行。
需要 Python 3.10 或更高版本。采集与匹配可以使用两个不同环境。

### 1. 采集环境和资产检查

先启动现有 CARLA 0.10.0 服务，使用已经能连接它的环境。
CARLA 客户端 wheel 应与服务器版本匹配，不要直接安装 PyPI 上未知版本的 `carla`。

```bash
cd /mnt/fast18/sunbo/DINO
conda activate carla10
python -m pip install -r requirements.txt
python capture.py --dry-run
python capture.py --inventory > inventory.json
```

`--inventory` 只读服务器版本、当前地图、可用车辆 Blueprint、现有车辆数量和出生点位姿。
编辑 `configs/experiment.json`，确认两辆车支持颜色属性，并选择一个无遮挡出生点。
默认第一个 identity 是目标。地图由服务器预先加载；脚本不自动切换地图。

### 2. 拍摄

```bash
python capture.py --config configs/experiment.json --output data/probe01
```

如服务运行在另一台机器，加 `--host SERVER_IP --port 2000`。
如果当前世界已经是同步模式，先停止其他推进 tick 的客户端，再加 `--take-tick-ownership`。
采集期间此脚本应是唯一 tick 主控；它会暂停异步推进并临时控制仿真。

此轮是控制变量的单候选辨识实验：A/B **依次**在同一出生点、相同天气下拍摄，
降低背景与位置作为身份捷径的机会。不是多辆车同时出现的检测、关联或主动取证实验。
场景必须没有其他车辆，脚本不会删除已有车辆。摄像机直接放置在固定视点，不执行飞行路线。

每辆车先刹车等待稳定，然后冻结物理；检查传感器帧号、实际位姿和近乎全黑图像。
当前默认 `exposure_mode=histogram`、`gamma=2.2`，每视点预热 60 ticks（3 秒仿真时间），
用于排查 UE5 下手动曝光的极暗画面。不是已经校准的正式摄影设置。
正式比较前应确认自动曝光随视角变化的影响，再选择统一且验证过的曝光参数。
`camera_attributes` 可配置曝光参数，不要覆盖尺寸、FOV 和 sensor_tick；实际值保存于 run.json。
近黑定义为 RGB 均值 <1/255 且超过 99% 像素的最大通道值 <=2/255；它是采集质量诊断，
不是候选不匹配证据。保存原图和亮度统计后拒绝该视图；参考图近黑则终止采集并记录 incomplete。
匹配入口也拒绝旧数据中被误标 captured 的近黑裁剪。`probe03` 的黑帧应保留用于诊断，不用于匹配。
结束或失败时销毁本脚本创建的 Actor，恢复天气与世界设置，清理异常写入 `run.json`。
强制杀进程仍可能来不及恢复；此时检查服务器状态后重新运行。

输出：

```text
data/probe01/
  run.json             # 配置、版本、地图、车辆/相机属性、状态与清理记录
  metadata.csv         # 身份、视角、顺序、天气、位置、帧号、位姿与裁剪框
  images/              # 原始 RGB
  crops/               # 仿真真值包围盒生成的整车裁剪
  roi_template.json    # 可选局部对照标注模板，默认均未标注
```

裁剪使用仿真真值，不评价检测器。车辆被画面截断、投影异常或像素过小时标为 rejected，
不得把这些视图当作负证据。包围盒投影不能证明无遮挡，拍完先检查原图与裁剪图，
确认没有建筑、树木或背景物遮住辨识部位。相机坐标也不保证避开场景几何。
无效查询可在 `metadata.csv` 中改 `status=rejected` 并填 `reason`；参考图无效则重新采集。

参考图来自 A 的 35° 方位、20° 仰角，与查询视角不同；它不是已有查询图的复制。
仍来自同一仿真场景，第一轮结果不能代表跨场景参考图泛化。
`sequence` 只是配置中的固定视角顺序，不是实际飞行轨迹，也不保证信息单调增加。
`feature_visibility` 默认 `unknown`，可人工填写 `visible`、`not_visible`、`unclear`。

### 3. DINOv2 匹配

可以在独立模型环境中运行，不需要 CARLA 包：

```bash
conda create -n dino-probe python=3.11 -y
conda activate dino-probe
python -m pip install -r requirements-model.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python match.py --data data/probe01 --output out/probe01 --device auto
python report.py --results out/probe01
```

PyTorch 的 CUDA 构建需要与服务器驱动兼容，按 [官方安装页](https://pytorch.org/get-started/locally/)
选择安装命令；CPU 也能跑这个小实验。`auto` 在 CUDA 可用时使用 CUDA，否则使用 CPU。
首次匹配会从 [DINOv2 官方仓库](https://github.com/facebookresearch/dinov2) 及其模型地址下载代码和权重。

默认 ViT-S/14，224×224，CLS 特征、余弦相似度。
图像按比例缩放并填充，不用中心裁剪，避免把车头、车尾等辨识部位裁掉。
该预处理是本实验选择，需要记录并保持一致。
也可单独对照 `--size 448` 或 `--pooling patch_mean`，每次使用新的输出目录。
`patch_mean` 包含填充区域的 patch，不能视为自动车辆区域特征。

服务器无法联网时，预先准备官方仓库和对应 backbone 权重：

```bash
python match.py --data data/probe01 --output out/probe01 \
  --model-repo /path/to/dinov2 --checkpoint /path/to/dinov2_vits14_pretrain.pth
```

权重必须与 `--model` 对应。离线读取不使用 pretrained 自动下载路径。
输出保存模型设置、元数据哈希和可选权重哈希；需要严格复现时固定本地仓库版本及权重。

### 4. 可选人工局部对照

先查看裁剪图，复制模板为 `roi.json`，手工填写。坐标在**整车裁剪图**上归一化，
格式是 `[left, top, right, bottom]`，不是原始 RGB 的像素坐标。

```json
{
  "reference": {"part": "side_window", "visible": true, "box": [0.2, 0.2, 0.7, 0.6]},
  "A_p0_noon_02_side": {"part": "side_window", "visible": true, "box": [0.2, 0.2, 0.7, 0.6]}
}
```

这些 ROI 数字只是格式示例，必须按真实图像修改。
只有参考图和查询图都显式标为 visible=true 且 part 相同才比较。
缺少标注、不可见或部位不同输出 local_status=unavailable，局部分数留空，不填零。
同样处理所有候选，不要只标注目标的有利视图。
局部图再次编码得到分数；这是人工指定区域的辅助对照，不是自动局部匹配或自动特征发现。

```bash
python match.py --data data/probe01 --roi data/probe01/roi.json --output out/probe01-local
python report.py --results out/probe01-local
```

### 5. 看结果

`scores.csv` 保存每张图的整车与可选局部分数；`features.npz` 保存原始特征及样本 ID，
供后续历史融合实验使用。`report.html` 含参考图、分数曲线和查询图库，`scores.png` 可用于汇报。
HTML 是静态文件，无需服务器桌面或浏览器；可以下载 `out/probe01` 整个目录后查看。
若数据移动到另一位置，运行报告时用 `--data` 指定新目录。

先检查：同视角下 A 与 B/C 是否有区分信号？A 跨视角是否大幅波动？
低分是否出现在特征不可见、模糊或裁剪有问题的视图？局部对照是否改善了可比视图？
暂不凭分数阈值排除，不把余弦分数叫正确概率，不用这一小组相关图片证明误排风险。

## 离线测试

```bash
python -m unittest discover -s tests -v
python capture.py --dry-run
```

测试使用 NumPy、Pillow 和报告生成依赖；不需要 CARLA 或 PyTorch。
覆盖坐标投影、位姿／陈旧帧检查、完整裁剪、分数数值、局部缺失、数据完整性和报告输出。
本机离线验证不能替代服务器上的真实 CARLA 采集和 DINO 推理验收。

输出目录已存在时采集与匹配会拒绝覆盖，请更换实验名。数据、权重和输出不进入 Git。
