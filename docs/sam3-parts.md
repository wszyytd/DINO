# SAM 3 对照：同一批 CARLA 图片的车门与侧窗

目标：比较现有 YOLO 与 SAM 3 的部位定位，先检查漏检、前后门混淆和挡风玻璃误标。
不重新拍摄，不改变 DINOv2，不自动排除候选。

## 当前已知条件

- 本地数据：`data/probe_20261019_020829_292286`，两组场景，20张查询与2张重复参考。
- YOLO 对照：`out/probe_20261019_020829_292286-parts`。
- SAM 3 使用官方 `facebook/sam3` 的 `sam3.pt`，不是 SAM 3.1 视频权重。
- 2026-10-19 本机 `hf download facebook/sam3 sam3.pt --dry-run` 返回访问未获批准。
  因此本轮只能验证输入、接口与输出格式；尚无真实 SAM 3 图片推理结果，不能报告效果提升。

## 1. 申请官方权重

打开 https://huggingface.co/facebook/sam3 ，使用自己的 Hugging Face 账号申请访问。
获批后在服务器执行 `hf auth login`，交互输入自己的访问令牌，不要把令牌粘贴到聊天或写进代码。
已经获准下载的本地 `sam3.pt` 可以直接通过 `--weights` 使用。
本程序不会绕过下载访问控制，也不会自动选择第三方权重。

## 2. 独立环境

不要修改已有 `dino-probe` 或 CARLA 环境。官方安装说明推荐 Python≥3.12、PyTorch≥2.7。
先查看 `nvidia-smi`，确认 GPU 与驱动状态。此前 `dino-probe` 的 cu130 构建无法使用服务器GPU；
下面按官方当前示例安装 cu128，仍需驱动与硬件支持。

```bash
cd /mnt/fast18/sunbo/DINO
git pull --ff-only origin main
conda create -n sam3-probe python=3.12 -y
conda activate sam3-probe

python -m pip install torch==2.10.0 torchvision \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-sam3.txt \
  -i https://pypi.tuna.tsinghua.edu.cn/simple --timeout 120 --retries 5

git clone https://github.com/facebookresearch/sam3.git /mnt/fast18/sunbo/sam3
python -m pip install -e /mnt/fast18/sunbo/sam3 \
  -i https://pypi.tuna.tsinghua.edu.cn/simple --timeout 120 --retries 5

python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

`sam3` 目录已经存在时先检查是否是官方克隆，无需再次 clone。
依赖约束保留官方 `numpy<2`；`setuptools<81` 用于提供当前官方代码导入的 `pkg_resources`。
本项目不自动升级系统 NVIDIA 驱动。

## 3. 下载与输入检查

```bash
hf auth login
hf download facebook/sam3 sam3.pt --local-dir models/sam3

python segment_sam3.py \
  --data data/probe_20261019_020829_292286 --dry-run
```

dry-run 无需 SAM 3、Torch、权重或联网；读取完整性检查与提示词配置，不创建输出目录。
`--prompts configs/sam3_parts.json` 可以显式指定提示词文件。

默认分别测试：car door、front car door、rear car door、side window、windshield、rear windshield、car roof。
每张图只计算一次图像编码，每个提示前清除上一提示状态，独立保存全部区域。
front_door/rear_door 是提示名称，不是经过验证的前后门真值；不推断左右侧。
不同提示可能产生重复或重叠掩膜，不会被自动融合为独立证据。

## 4. 推理与对照

GPU可用时执行：

```bash
python -u segment_sam3.py \
  --data data/probe_20261019_020829_292286 \
  --weights models/sam3/sam3.pt \
  --output out/sam3_parts01 --device cuda

python compare_parts.py \
  --baseline out/probe_20261019_020829_292286-parts \
  --candidate out/sam3_parts01 \
  --output out/sam3_vs_yolo01
```

只有第一条成功完成后再运行第二条。CPU可用 `--device cpu`，但这类模型可能很慢。
未指定 `--weights` 时使用当前已登录账号尝试从官方仓库下载；无访问权限则在创建输出前停止。

SAM 3 使用官方处理器默认1008×1008预处理；输出必须恢复到原始车辆裁剪尺寸，
不接受直接拉伸未知坐标系的掩膜。输出使用与现有分割程序相同的原生尺寸掩膜、局部裁剪、
叠加图、`parts.json`、`parts.csv`、`parts_run.json` 和 `index.html`。
记录权重SHA256、提示词、设备、官方包路径及版本；权重哈希流式计算，不一次复制整个大权重进内存。
支持 bfloat16 的 CUDA 设备使用 autocast 降低推理内存，其余使用 float32；实际模式写入运行记录。

下载整个 `out/sam3_vs_yolo01`，打开 `index.html`，逐张对照两个模型的原图和掩膜。
比较器要求样本ID、拍摄元数据、来源哈希以及原图像素一致，否则拒绝生成对照。
数量统计只描述输出，不是准确率；两个模型的标签体系及分数也不能直接比较。

本次重点检查：

1. 警车侧面明明清楚可见、YOLO却漏检的图，SAM 3 是否能找出车门。
2. 普通车侧面是否仍把两扇门合并。
3. 前后挡风玻璃是否误作侧窗。
4. 高空车门不可见时，是否产生错误车门区域。

现有 `evidence.py` 默认寻找 right_front-door/right_back-door；SAM 3 不输出这些已确认类别，
不能直接把 front_door 重命名为 right_front-door 后套用。先检查定位与对应关系，
暂不将该轮输出接入自动身份比较或排除。

## 验证范围

离线测试覆盖提示独立性、一次图像编码、原始坐标掩膜、空结果、非法分数、通用导出、
对照输入一致性与不依赖模型的 dry-run。测试使用合成接口样例，绝不当作真实 SAM 3 结果。

官方来源：

- https://github.com/facebookresearch/sam3
- https://github.com/facebookresearch/sam3/blob/main/sam3/model/sam3_image_processor.py
- https://github.com/facebookresearch/sam3/blob/main/sam3/model_builder.py
- https://huggingface.co/facebook/sam3
