# 车辆记忆全量回问训练

训练沿用旧车记忆模型和冻结的 DINOv2 前端。每条路线从空记忆开始，新图的 16 个 token 写入一次。有效写入完成后，把截至当步所有已写过车辆裁图的 64 个查询 token 批量拿来问。A、B 的历史都会被问到。裁图的身份只用于路由和交叉熵标签。查询不写进记忆。

每步先平均该步的查询损失，再对整条路线的有效写入步取平均。梯度沿整条记忆链传播。每批完整路线累计一次梯度，再用 AdamW 更新。没有新写入的返程重访跳过训练。冻结特征会在一次运行内缓存，并供所有实验组复用。

| 实验组 | 训练目标 |
| --- | --- |
| `initial` | 冻结原检查点，直接验收 |
| `fixed` | 每步用采集时的固定参考照片查询，作为同数据的固定参考对照 |
| `replay` | 每步写完后，用曾写入的全部裁图查询 |
| `replay_independent` | 全量回问，加上当前写入之前的独立照片查询。该候选必须已入档，查询照片来自路线外，且与任何路线写入的原图摘要不同 |

三组微调从相同权重开始，用相同训练背景、路线顺序、训练轮数和优化器更新数。留出背景不参与损失与选轮，统一验收最后一轮。独立题主要来自与当前视角邻近的另一张照片，距离之前所有观察都较远的视角只有少量样本。两种车辆资产在不同布局中复用。

## 服务器运行

复用已在服务器跑过的诊断计划。该计划含 8 条训练路线和 3 条验证路线。

```bash
set -euo pipefail
conda activate dino-probe
git switch codex/evidence-memory-v1
git pull --ff-only origin codex/evidence-memory-v1

PLAN=out/memory_retention_20261117_110524/plan.json
python run_memory_replay.py audit --plan "$PLAN"
RUN_MEMORY_REPLAY_CUDA_TESTS=1 python -m unittest tests.test_memory_replay_server -v

OUT="out/memory_replay_$(date +%Y%m%d_%H%M%S)"
python run_memory_replay.py train --plan "$PLAN" --output "$OUT" --device cuda:0
cat "$OUT/report.md"
```

如需重建计划，运行 `python run_memory_retention.py plan --output out/memory_retention_new` 并把 `PLAN` 指向新生成的 `plan.json`。默认三组微调各跑 30 轮、种子 7、每批 2 条路线。先跑通可加 `--epochs 1 --variant replay_independent`。扩到三个种子可加 `--seed 7 17 27`。固定参考图从训练场景的采集记录加载并校验文件摘要。

`report.md` 报告首次写入原图、历史原图保留率、独立照片写入前及写入后的表现。`summary.json` 标识运行是否完整。`<组名>-s<种子>/validation.json` 保存逐路线逐步分数；`last.pt` 保存微调后的模型权重。

## 车辆前景与记忆更新对照

在服务器仓库中激活现有 `dino-probe` 环境，运行 `bash run_foreground_memory.sh`。脚本用已有的 11 月 17 日诊断计划，先审计数据划分并运行 CUDA 数值测试，再执行下表的四组微调。默认 30 轮、种子 7。只检查完整流程可用 `bash run_foreground_memory.sh 1 7`，三种子实验用 `bash run_foreground_memory.sh 30 7 17 27`。也可以通过 `MEMORY_PLAN` 指向重建的同格式计划。

| 裁图 | 实验组 | 和上一组相比，改变了什么 |
| --- | --- | --- |
| bbox | replay_independent | 原始 RGB 框、原学习式更新、旧图回问＋写入前独立查询 |
| foreground | replay_independent | 同一裁框内仅保留 CARLA 实例掩膜的车身像素，背景填黑 |
| foreground | replay_prepost | 写入后再对当步未写入的独立照片计算身份交叉熵，包括首次有效观察 |
| foreground | ema_prepost | 将学习式 Updater 和门替换为固定槽 EMA，两组的初始 Writer、Reader 和评分头权重相同 |

需要单独验证是否只记住第一张或最近一张时，使用 `--variant ema_prepost`，分别设置 `--ema-decay 1`（每车保留第一次有效观察）或 `--ema-decay 0`（每车保留最近一次有效观察），其余训练条件保持相同。

观察和查询两侧采用同一抠图规则，再走各自原有的冻结编码分支。空间位置、裁框和池化分辨率一致。每种裁图在一次运行内只编码一次，特征供各组和各随机种子共用。预处理会校验原图、全分辨率实例掩膜的摘要与尺寸。前景用仿真真值给出，实验衡量的是理想分割条件下的身份匹配。

三项损失各自先对单步查询、再对有该项监督的路线步平均。两个 prepost 组均使用历史回问、独立 pre、独立 post 三项，权重默认为 1、1、1。独立照片来自路线外，始终只查询。EMA 首次直接保存 Writer 的 8 个槽，之后使用旧槽 × 0.8 ＋ 新槽 × 0.2，无观察时保持旧状态。这里保留原 Reader 所需的槽向量尺度。EMA 的 Writer 和 Reader 仍参加相同监督下的微调，固定的是更新公式。两个系数可通过单独训练命令的 `--ema-decay` 和 `--postwrite-weight` 设置。

脚本打印新建的 `out/foreground_memory_*` 目录。其 `bbox/report.md` 和 `foreground/report.md` 中第二张表比较双方车辆均已建档时，**同一张独立照片**在整帧写入前后的正确数、由错变对、由对变错，以及旧角度独立照片的保留情况。`paired_independent.by_candidate` 保留 A/B 分项；`groups.new_view.help/harm` 使用只跳过目标车本次写入的另一种干预。不同步数可重复查询同一张照片。每个组的 `last.pt` 记录模型配置及裁图模式，恢复时使用 `dinoexp.replay_matcher.build_replay_matcher(checkpoint["config"])`，再严格加载 `checkpoint["model_state"]`。

## 实验八 · 无遮挡车辆上的 Writer 槽竞争

### 使用已有车辆框的离线预实验

服务器在 `dino-probe` 环境运行 `bash run_experiment8.sh --offline`。默认 30 轮、种子 7。数据来自 `data/fixed_height_similar_20261104` 已保存的环绕照片。脚本逐张核对入选照片与目视筛选时的文件摘要，读取车辆框内的原始 RGB，全程可在 CARLA 关闭时运行。

入选标准是整辆车入框，车身上没有明显的树枝、栏杆、广告牌或其他车辆遮挡。仅采用 refA 采集批次中的 A、B 两辆车，refB 会重复相同机位。每个地点使用三个车辆组合，包括同色不同车型、同车型相近车色。位置 42 用于微调，位置 64 用于验证，每条序列写入 8 次。共 162 张车辆裁图，来自 81 张双车原图。训练有 48 张写入图、30 张独立查询图，验证有 48 张写入图、36 张独立查询图，其中 30 张可以做写入前后的配对比较。两种 Writer 共用照片、顺序、初始化和损失。查询用另一个机位，按每辆车的真实位置计算方位差，要求不超过 25°。

输出位于脚本打印的 `out/experiment8_offline_*/bbox/`。每条序列的 `data/<采集组>/review.jpg` 展示实际输入、用途与角度。输入保留框内路面，质量依据目视筛查。验证地点只有一个，应同时查看普通 Writer 和竞争 Writer 的独立查询、旧视角回问与各序列结果。A、B 是控制实验里的两个视角，A→AB→ABB 能检查写入第二个视角后对第一个视角的保留情况。固定机位间隔限制了对照覆盖，报告中的分母按实际可选历史计算。此预实验的正确率单独解读。

验证集的 6 个“车辆与序列”组合均支持 A、B、AB、BA 和 ABB。按实验八原有的近角 25°、远角 45°阈值，成套 AA/BB 等张数对照的覆盖为 0。用这批数据可以检查增加视角后能否记住旧视角；仅凭 AB 相比单张的变化，还不能判定增益来自视角互补。

### 重新采集前景数据

服务器切到 `codex/evidence-memory-v1` 并更新，在 `dino-probe` 环境运行。首次采集需要空闲的 CARLA 会话，地图为 `Town10HD_Opt`，默认连接本机 2000 端口。

```bash
bash run_experiment8.sh           # 30 轮，种子 7
# bash run_experiment8.sh 1 7    # 流程检查
# bash run_experiment8.sh 30 7 17 27
```

首次执行会在 `data/experiment8_clear_orbits_v1` 重新采集。两辆白色车（普通 Charger、警用 Charger）分别放在同一地图锚点上方 220 米，固定车身、绕车拍摄。每个位置对两辆车使用相同的相对相机设置。每车共 16 个写入机位，覆盖四个方位，每个写入机位都配一张单独拍摄的查询图。查询机位的水平方位偏移 4°～8°、仰角增加 3°、距离增加 0.6 米。8 个位置训练，另外 3 个位置验证，共 704 张 RGB 图。模型一次收到两辆车在相应步的前景，各自来自单车拍摄。验证衡量这两种已知车辆在新机位下的识别与记忆。

每张照片同步采 RGB 和真值实例掩膜。移走车后在原机位采空场深度，检查覆盖整车三维框投影和可见掩膜的矩形。矩形内所有像素的背景都须比车辆最远角点远至少 1 米，车身完整入框且至少有 8192 个前景像素。检查覆盖掩膜内部的洞。保存 RGB、掩膜、检查区域的 16 位深度图和相机位姿，深度单位为毫米，超过 65.535 米的值饱和。训练前逐张复核。验证集前景缩略图在 `data/experiment8_clear_orbits_v1/validation_foregrounds.jpg`，每行一辆车的四个方位，写入图与对应查询图左右成对，图下标有水平方位角。单个位置采集完成后保存进度。再次运行时会复用已完成的位置，完整数据通过复核后可直接重复训练。

在 `dino-probe` 中运行脚本即可。首次采集时，当前 Python 如果缺少 `carla`，脚本会用已有的 `carla10` Conda 环境采集，随后用 `dino-probe` 训练。已采好的数据直接在训练环境复核。自定义采集解释器可设 `CARLA_PYTHON=/path/to/carla/env/bin/python`。连接其他端口或主机可设 `CARLA_PORT` 或 `CARLA_HOST`。单独采集用 `conda run --no-capture-output -n carla10 python -u capture_clear_memory.py capture`，单独复核用 `python capture_clear_memory.py audit`。旧街景实验可按历史提交 `4537018` 复现。

每个种子的普通 Writer（`replay_prepost`）和竞争 Writer（`competitive_prepost`）共用原有 region-sequence 初始检查点、前景裁剪、同一组新序列和三项 prepost 损失。竞争版只在 Writer 的每个注意力头内沿槽做 softmax，排除无效输入，再对每个槽按有效输入权重求加权平均。首次写入照常由 Writer 初始化记忆，之后调用原 Updater 和门控。`initial` 是微调前的普通 Writer。

打印的输出目录下，`report.md` 保留与实验七一致的指标。`writer_report.md` 另列每条路线的新查询、旧角度回问、独立旧查询数和遗忘次数。先看新、旧查询同时正确的数量以及旧查询从对变错的次数，再结合注意力和槽间余弦解释变化。不同种子、路线及重复回问分别解读。

控制实验的 A、B 是同一辆车相隔较远的两个观察角度，具体方位随路线变化。每条留出路线每辆车最多选一对，依据方位和可用的非重复照片确定，不使用模型分数。沿用计划的角度阈值（默认近角 25°、远角 45°）。两张独立查询不参与写入，同一对象的所有历史共用同一份干扰车记忆。A、B 单图测首次识别；AB、BA 测顺序；matched-AA、BB、AB、BA 用同时具有两种单角度双图对照的同一批案例。AB 后最多再写三张不同的 B 角度照片，每次重问原来的 A、B 两张查询。覆盖不足的路线或长序列按实际例数报告。

每组子目录的 `view_pairs.json` 保存选图、顺序、方位及逐题决策和分差。`writer_maps.json` 保存自然路线逐次写入的各头聚合权重、有效位、槽间余弦和逐槽实际更新范数，竞争组还保存第二次归一化前的分配权重。`writer_heatmaps.png` 显示每条路线第一个样本，横轴是连续索引的输入 token，纵轴是槽，灰格表示无效输入，蓝色越深聚合权重越大。各组用一致样本及色标。
