# 相似车辆辨识信号实验

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

每辆车先刹车等待稳定，然后冻结物理；相机固定曝光，检查传感器帧号与实际位姿。
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
