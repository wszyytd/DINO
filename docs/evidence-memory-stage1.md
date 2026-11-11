# 第一阶段：正确归属下的车辆记忆融合

本实现复用本仓库按老师说明复现的 `DenseSearchHead` 第 20 轮权重。不是老师原始任务检查点；原模型只在 4 张航拍图、8 个配对上拟合过，双车留图存在认反。

2026-11-10 已完成本机三模式 × 三随机种子训练和保存权重复算。当前结果不支持这一版学习记忆有效；具体数据见 [第一轮实验结果](evidence-memory-stage1-results.md)。

## 实现范围

输入为参考特征与当前每辆车的区域 token。真实实例掩码从整图特征中读取区域，真实身份负责跨帧送入正确候选槽；它们是 oracle 辅助，不属于最终自主运行输入。候选顺序在训练样本间随机化，模型参数不绑定 A/B，候选数 K 可变。

编码器、参考分支冻结，缓存为 float32。此次 CPU 实验只使用全图分支的参考图交互前特征：DINOv2 第 3/6/9/12 层 → 原航拍适配器 → 二维窗口几何 → 原视图 Transformer。局部 15 个窗口、相机位姿嵌入、参考条件打分及 Null 头都不进入记忆特征。这是 full-view ablation，不是老师完整 16 视图模型效果。

GT 掩码使用与原图完全相同的 letterbox 整数缩放与填充，计算每个 14px patch 的掩码覆盖。对车辆覆盖的特征区域进行加权 4×4 池化，不裁剪并重编码 RGB。参考使用冻结的 8×8 token。训练新模块：8×256 个共享初始化的逐车记忆槽、区域读取、门控跨帧更新、参考匹配读出。没有相机世界坐标、真实身份编码、未来帧或目标分数输入记忆。相机位姿曾用于训练原任务头，这一历史不等于当前输入位姿。

缺席车辆不写入记忆；当前观察模式每步重置；均值模式平均每步区域读取的槽特征；学习记忆模式按门控更新。共享场景记忆、自动分割、身份关联与动作策略不在本轮范围。

运行时记忆是调用方持有的张量字典，包含 `slots[B,K,8,256]`、`seen[B,K]`、`count[B,K]`。模型的更新网络在内部读写这些张量，训练权重与每条任务的临时记忆分开保存。调用 `state = model.update(observation, valid, state)` 接收一帧；调用 `model.score(reference, reference_valid, state)` 输出 K 辆已发现候选及 Null 的分数。开始新任务时将 state 设为 None。候选增加时可追加槽；删除或重排候选时，调用方必须同步管理已有状态。

每辆车独立更新、共享网络参数，因此一张图的两辆车可同时写入各自的记忆。本轮由标签保证分流，尚未验证模型自行把新观察关联到正确车辆的能力。Null 在这里表示“目标尚未进入已观察候选集合”，不等于“不确定”或“整个场景没有目标”。

## 实验协议

- 三种模式：current、mean、memory；相同冻结特征、区域读取结构、匹配结构、初始公共权重、路线、优化预算。memory 额外含门控更新参数，报告各自参数数。
- 每条路线用每张参考图各训练一次；候选槽在每条样本内保持一致，在不同样本间随机排列。路线不依赖目标位置或质量标签选择。
- 动作仅用于生成相邻网格状态序列。模型逐帧读取，计算图贯穿一段 8 帧；不在帧间 detach。各序列重置记忆。
- 当前帧模式监督当前候选可用性；历史模式监督至今已观察的候选可用性。“历史目标当前消失”不能变成历史 Null。完整参考目标未出现过时，标签为 observed-set Null；这不是“尚不确定”的标签。
- 默认固定 20 轮、三随机种子；不根据 probe 选 epoch 或调参。
- 如果有按组划分的独立 val/test，检查其未参与 teacher 拟合，使用这些组。只有一个布局时，用前三列拟合、后两列 probe；任何 teacher 拟合 RGB 强制归入拟合侧；两侧路线只沿各自允许的状态边生成，检查图像哈希交集为空。
- 单布局 probe 仍共享地图、资产和参考，仅为诊断，不得称为跨场景或未见车辆验证。轨迹及前缀相关，数量不代表独立样本数。

## 指标与判断

重点看 `both_visible`：目标与至少一个干扰车都在当前图中时的候选识别。`current_visible` 包含只有目标可见的容易帧。`target_hidden_after_seen` 测记忆保留，不能单独证明互补证据融合。`all_prefix` 混合了大量缺席帧，不能只看它。

学习记忆另做同权重 reset_history、故意交换候选历史的 swapped_history 消融；重复同一原图三次记录概率漂移。记录 0.9 置信阈值的错判数量，但该阈值未经校准，不用作飞行停止条件。学习记忆不天然幂等，若漂移明显必须如实报告。

## 运行

本机已发现可用 Python：`D:/Dev/Envs/Conda/dl_env/python.exe`（PyTorch 2.5.1，CPU）。系统默认 Python 没有 torch。可以在有 CUDA 的服务器使用相同入口及 `--device cuda`。

```powershell
Set-Location D:\Work\DINO
& D:\Dev\Envs\Conda\dl_env\python.exe -u prepare_evidence_memory.py `
  --data-root data/search_grid_smoke_v1 `
  --teacher out/teacher_grid_fit_e20_retry/last.pt `
  --backbone D:/Cache/dinov2_w/dinov2_vitb14_pretrain.pth `
  --model-repo D:/Cache/dinov2_repo `
  --dense-cache out/teacher_grid_fit_e20_retry/features `
  --output out/memory_fullview_cache_v1 --device cpu --threads 4

& D:\Dev\Envs\Conda\dl_env\python.exe -u train_evidence_memory.py `
  --cache out/memory_fullview_cache_v1 --output out/memory_stage1_v1 `
  --epochs 20 --routes 24 --length 8 --batch-size 8 `
  --seeds 7 17 27 --device cpu --threads 4

& D:\Dev\Envs\Conda\dl_env\python.exe -u evaluate_evidence_memory.py `
  --run out/memory_stage1_v1 --cache out/memory_fullview_cache_v1 `
  --device cpu --threads 4
```

缓存可按已完成特征续作，并核对 teacher/backbone/source/part 哈希。训练结果目录必须全新，避免覆盖。产物包含划分、固定 probe 路线、逐轮损失、每种模式与种子的最终权重、分层指标、消融及重复漂移。

已有结果复查只运行最后一条命令。重新训练时换新的 output 路径。评估脚本检查特征缓存指纹，重新加载九个权重，要求 probe 指标与训练报告完全一致，并另测训练位置上的固定路线。

## 当前限制

GT 掩码与身份提供了额外帮助，结果不代表自动检测或关联效果。全图特征仍可能包含背景和位置信息。普通白车与警车并不构成细微差异困难对。相同场景反复路线不能替代新布局。即使记忆指标提高，也需在独立布局、相似车辆及预测关联条件下继续验证。
