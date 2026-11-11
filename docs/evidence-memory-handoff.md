# 车辆证据记忆模型详细交接

更新日期：2026 年 11 月 11 日。面向用户本人、服务器训练执行者，以及下一位接手代码的助手。

后续进展：服务器 20/100/200 轮结果已回传，200 轮双车留出准确率为单帧 60.5%、均值 55.1%、记忆 39.1%；记忆每步清空为 56.5%。第一轮诊断代码与服务器命令见 [诊断使用说明](evidence-memory-diagnostics.md)。下文历史 CPU 数据保留原口径，不作为最新服务器结果。

**当前状态：第一阶段代码已经完成并推送，历史诊断没有证明学习记忆有效。后续只在本机编写代码、整理文档和推送；由用户同步到服务器后训练。不要自行在本机启动特征提取、训练、模型评估，也不要自行登录服务器运行。**

这份文档记录实际代码和已经发生的实验。后面的改进方向是建议，不表示已实现或已验证。

**1. 接手时先知道这几件事**

| 项目 | 当前情况 |
|---|---|
| 本机仓库 | `D:/Work/DINO` |
| 远端仓库 | https://github.com/wszyytd/DINO |
| 当前分支 | `codex/evidence-memory-v1`，服务器应同步此分支 |
| 模型代码提交 | `b3b548f6968d623fd7a4d9c084119b872fa0659c` |
| 代码所基于的提交 | `87f23c8`，老师模型本地复现与续训版本 |
| 服务器目录 | 原交接记录为 `/mnt/fast18/sunbo/DINO`；本次没有远程核实 |
| 服务器环境 | 原交接记录为 Conda `dino-probe`；本次没有远程核实 |
| 当前输入 | 参考图特征、当前每辆车的区域特征、上一时刻车辆记忆 |
| 当前输出 | 更新后的逐车记忆，以及 K 辆候选车和 Null 的匹配分数 |
| 当前额外帮助 | 真实车辆掩码负责区域读取，真实身份负责跨帧分流 |
| 还未实现 | 自动候选发现与身份关联、共享场景记忆、证据价值预测、动作推荐 |
| 主要结论 | 程序能维护记忆，但当前训练方式与模型尚未学会可靠融合 |

模型代码提交号不包含这份后来追加的交接文档；同步分支最新版本可同时获得代码与文档。权重、数据和运行产物位于被 Git 忽略的目录，推送代码不会同步它们。

**2. 用户真正要解决的问题，以及本阶段的范围**

最终任务是参考图引导的无人机车辆搜索：照片按时间逐张到达，目标可能出现，也可能缺席；一张照片可能有多辆相似车。模型需要保留不同候选的历史证据，逐渐确认或否定候选，并推荐下一次移动或观察动作。

用户希望先把证据融合做好，再接动作推荐；沿用老师的编码器，先不引入显式三维信息；航拍 RGB 保持整图输入，不先裁车辆小图重新编码。多图具有互补信息是研究出发点，需要验证的是具体模型是否学会使用它。用户也希望控制 token 开销，非必要不要调用子代理。

第一阶段选择先排除“车没分出来”和“历史送错车”两类干扰：用真实标签把观察送进正确候选记忆，单独训练记忆与参考匹配。这通常称为 oracle 设置，意思是由真实答案提供部分中间信息。它只能回答“在正确分流的条件下，融合是否有用”，不能回答自动搜索是否成功。

此前讨论过的共享记忆、候选查询、实例分割、流形约束、视角特征预测、信息价值和飞行规划，都属于更大设计。当前代码没有把这些模块全部实现，不能据此描述成完整系统。

**3. 老师模型的来源，以及本次复用了什么**

来源是用户提供的老师设计说明，本地按说明复现了 `DenseSearchHead`。DINOv2 骨干使用官方预训练权重，任务头由本项目训练。老师原始源码和原始任务检查点仍待取得，因此应称“老师设计的本地复现版”。

原模型输入参考车辆图和当前航拍图，输出 32×32 搜索网格及一个 Null，合计 1,025 个分数。完整流程包含整图加 15 个局部窗口、参考交互、相机状态嵌入及空间分数融合。

本轮取第 20 轮任务头：`out/teacher_grid_fit_e20_retry/last.pt`。它只在当前布局的 4 张航拍图、8 个参考配对上拟合过，原双车留图已有认反问题。续训到第 50 轮没有改善，不能把续训目录的 `last.pt` 当成第 20 轮。

本次复用的计算链路是：

```text
航拍整图
  → 等比缩放并补边到 518×518
  → 冻结 DINOv2 ViT-B/14，第 3/6/9/12 层
  → 每层 37×37×768 的 patch 特征
  → 冻结航拍适配器：投影到 256 维、层间加权、二维位置编码
  → 窗口几何嵌入、图内自注意力
  → 37×37×256 整图特征
```

这次只使用全图分支，未启用 15 个局部窗口，属于 full-view ablation。最初这样做是为了完成本机 CPU 诊断；迁到 GPU 后脚本也不会自动恢复完整多窗口，需要另行实现与对照。

取特征的位置在参考交互和相机状态嵌入之前，所以记忆输入没有显式相机世界坐标，也没有参考条件打分。已有二维位置编码仍保留，特征也可能包含背景。冻结权重曾如何被训练，与本次前向输入什么，是两个不同问题。

参考分支使用同一冻结 DINO 骨干和独立参考适配器，再有效区域池化成 8×8，即 64×256 token。代码调用 `_reference()`，实际取其返回的局部 token 和有效标记，不使用返回的整体参考描述。

对原模型仅做了可复用接口提取：新增 `DenseSearchHead.encode_views()`，原 `forward()` 改为调用它。此前已用相同权重与输入检查旧、新 forward，所有返回字段逐位一致；未重新训练老师模型权重。

**4. 当前模型结构：一张照片如何走完流程**

```text
当前照片 → 冻结编码 → 整图空间特征
                         │
            真实掩码读取各车辆的特征区域
                         │
            ┌────────────┴────────────┐
         A：16×256                 B：16×256
            │                         │
      writer 读成 8×256          writer 读成 8×256
            │                         │
   A 的旧记忆 + 新证据         B 的旧记忆 + 新证据
            │                         │
      updater + gate             updater + gate
            │                         │
       A 的新记忆                 B 的新记忆
            └────────────┬────────────┘
                         ↑
参考图 → 冻结编码 → 64×256 参考 token
                         │
              reader 分别读取每辆车记忆
                         │
                共享匹配网络 + Null
                         │
             [A 分数，B 分数，Null 分数]
```

真实掩码先按与 RGB 一致的整数缩放和补边对齐，再计算每个 14×14 patch 内的车辆覆盖比例。在特征图上取覆盖区域，做加权 4×4 池化，得到每车 16 个 token。这里裁的是特征区域，不是车辆 RGB；不会为每辆车额外运行一次 DINO。

掩码加权不意味着特征完全没有背景：DINO 和后续自注意力已经读过整图，一个 patch 也可能同时覆盖车和背景。空间分辨率不足时，掩码不能凭空恢复车辆细节。

候选 A/B 分别更新，共享网络参数；同一帧可同时收集两辆车的证据。当前分流来自标签，不是模型自己判断它们跨帧是不是同一辆车。

**5. 模型接口、状态与更新公式**

源码：`dinoexp/evidence_memory.py`，类名 `EvidenceMatcher`。没有把整条序列封装成单次 `forward()`，而是显式调用 `update()` 和 `score()`。

| 名称 | 形状 | 含义 |
|---|---|---|
| `observation` | `[B,K,P,D]`，本轮 P=16、D=256 | 当前每辆候选的区域 token |
| `valid` | `[B,K,P]`，bool | 哪些区域 token 有车辆覆盖 |
| `reference` | `[B,R,D]`，本轮 R=64 | 每个任务的一张参考图特征 |
| `reference_valid` | `[B,R]`，bool | 参考有效 token |
| `state['slots']` | `[B,K,S,D]`，S=8 | 每辆车 8 个记忆向量 |
| `state['seen']` | `[B,K]`，bool | 至今是否观察过该候选 |
| `state['count']` | `[B,K]` | 有效观察次数；均值基线用于计数 |
| `score()` 返回值 | `[B,K+1]` | 未归一化分数，最后一列是 Null |

B 是并行任务数，K 是候选容量。8 是每辆车的记忆向量数，不是最多 8 辆车。当前要求 K≥1，尚未支持真正的零容量空列表；可以预留未见槽并把 valid 全设 False。

运行记忆由调用方保存，但写入、更新和读取网络属于模型。模型参数保存在检查点；各任务的临时记忆不会自动永久保存到检查点中。开始新任务时 `state=None`，不能把上一个任务的状态接到下一个任务。

三个交叉注意力块各做一件事：

| 模块 | Query | Key / Value | 输出 |
|---|---|---|---|
| `writer` | 8 个共享的可学习初始槽 | 当前车辆 16 个 token | 本次证据 W，8×256 |
| `updater` | 旧记忆 M | 本次证据 W | 更新提案 U，8×256 |
| `reader` | 参考图 64 个 token | 某辆车记忆 | 与参考对应的读取结果 |

`CrossBlock` 包含多头交叉注意力、残差、LayerNorm 和两层前馈网络；本轮记忆模块使用 4 个注意力头。核心更新是：

```python
written, _ = self.writer(initial_slots, current_tokens, current_valid)
proposal, _ = self.updater(previous_memory, written, written_valid)
gate = self.gate(torch.cat((previous_memory, proposal), dim=-1)).sigmoid()
updated = previous_memory + gate * (proposal - previous_memory)
```

这是主逻辑摘录，变量名用直白含义表示，完整检查和批量处理见源码。`gate` 每个槽输出一个 0～1 系数，控制该槽的更新幅度，偏置初始为 -2。第一次看见候选，直接用 written 初始化；当前缺席，不更新原状态。代码没有显式质量门控、重复照片去重或证据价值监督。

`score()` 用参考 token 读取记忆，将参考及读取结果分别按参考有效标记池化，拼接四部分：参考、读取结果、逐元素乘积、绝对差。一个 MLP 给每辆候选打一个分。尚未观察过的槽被置为 -1e4，最后拼接一个可学习的 Null 标量。

Null 表示“目标尚未进入已观察候选集合”。它不是不确定度，也不证明整个搜索区域没有目标。softmax 概率没有校准，不能直接用 0.9 等阈值宣布找到目标或结束飞行。

车辆数增加时，`update()` 能在尾部追加槽。已有候选的索引必须保持身份一致；删除、重排、合并候选时需外部同步管理状态。训练中先按场景真实车辆数分配全部槽，再用 seen 隐藏未出现候选，因此本轮并未检验在线发现新车辆的能力。

跨照片交互是“旧记忆读取新证据”，不是保留所有历史图再做联合自注意力。历史信息经过固定 8 槽压缩；当前没有共享场景记忆，也没有显式流形损失或视角预测头。

**6. 怎样调用模型**

以下示例在服务器使用，输入已经是冻结特征。完整 RGB 到特征的转换由准备脚本负责。

```python
import torch
from dinoexp.evidence_memory import EvidenceMatcher

checkpoint = torch.load(
    "out/memory_stage1_server_v1/memory_seed7/last.pt",
    map_location="cpu", weights_only=True,
)
model = EvidenceMatcher(**checkpoint["config"]).to("cuda").eval()
model.load_state_dict(checkpoint["model_state"])

state = None
with torch.inference_mode():
    for observation, valid in sequence:
        # observation / valid / reference / reference_valid 应在同一设备。
        state = model.update(observation, valid, state)
        logits = model.score(reference, reference_valid, state)
        probabilities = logits.softmax(-1)
```

`sequence`、`reference`、`reference_valid` 需由数据管线提供，示例不是独立可运行的 RGB 推理入口。推理使用 inference_mode，避免累积训练计算图；训练一段 8 帧时不在帧间 detach，让最后的损失能传回早期观察。

**7. 数据、特征缓存和原始标签**

现有数据根目录为 `data/search_grid_smoke_v1`：5×5 采样点，每点 8 个方向，共 200 张 1024×768 航拍图，间距 4 米；只有一个独立布局、两辆车及两张参考图。车辆是普通白色 Charger 和白色警用 Charger，并不是已构造好的细微外观差异困难对。

准备脚本需要完整的数据目录，包括 `dataset_manifest.json`、每个场景的 `frames.jsonl`、`scene.json`、`graph.json`、`audit/observations.jsonl`，以及它们引用的 RGB、参考图和实例掩码。只复制照片不够。

脚本逐项核对来源：任务权重与骨干哈希、图像审计哈希、参考及掩码哈希、缓存分片和场景文件哈希。特征在提取时冻结，输出为 float32。可选 `--dense-cache` 用于复用旧骨干缓存中的全图分支，规格必须和任务权重一致。

每个场景缓存包含：

```text
observations       [N,K,16,256]
valid              [N,K,16]
references         [K,64,256]
reference_valid    [K,64]
state_ids / image_hashes / objects
graph / group / split / scene_id
```

图和状态索引用于选择下一张照片、划分数据和按身份路由，不作为记忆网络的输入。标签内的质量分级目前没有用作观察质量门控；代码将存在有效掩码覆盖的候选视为可见。数据里的其他车辆不会因此自动成为候选。

照片来自离散图像库，训练路线是沿状态图生成的相邻观察序列，不是一次真实连续飞行录像；车辆位置固定。图中存在一条移动边，也不等于已经验证碰撞、可通行性或真实飞行动态。

**8. 当前训练到底在优化什么**

训练入口为 `train_evidence_memory.py`，核心逻辑在 `dinoexp/evidence_training.py`。

三种模式共用区域读取和参考匹配结构，但各自独立训练：

| 模式 | 历史处理 | 训练标签 |
|---|---|---|
| `current` | 每一步重置，只有当前观察 | 目标当前可见则为该候选，否则 Null |
| `mean` | 对每次 writer 输出的槽特征累计平均 | 目标此前出现过则为该候选，否则 Null |
| `memory` | 交叉注意力与门控更新 | 与 mean 相同 |

mean 平均的是训练得到的逐帧槽特征，不是直接平均原始 DINO 向量。三模式公共模块初始权重相同，训练路线与预算相同；memory 额外增加更新模块，因此参数量不同，训练后公共模块权重也会不同。

当前损失只有每一步的候选分类交叉熵，再对时间步取平均。没有 SupCon、Δmatch 价值标签、动作损失、分割损失、流形平滑或重复一致性损失。这里没有把讨论过的其他监督悄悄算成已实现。

每条路线对每个参考分别构造任务；候选次序在不同训练样本间打乱，同一条序列内保持一致。模型不接收 A/B 身份编码，训练不会让“第一个槽永久代表 A”。

本轮明确使用：20 轮，24 条路线/轮，每条 8 帧，2 张参考形成 48 个任务，batch 8，种子 7/17/27；AdamW 学习率 1e-4、weight decay 0.01，梯度裁剪 1。CLI 的 routes 默认值是 48，复现历史记录必须显式写 `--routes 24`。

每轮路线种子为 `seed*1000+epoch`，epoch 从 0 开始。每一步从合法相邻边抽样，原地重复拍摄权重 0.1、其他边权重 1；不是按目标可见性选择路线，也不在“已经认对”时停止，而是固定长度。测试路线固定 seed=901，训练位置复查路线 seed=902。

**9. 划分和指标口径**

当前单布局按列划分：前三列拟合，后两列留出。老师已经拟合过的 RGB 强制进入拟合侧，因此实际为 121 张拟合、79 张留出。两侧不共享 RGB 哈希，路线不能越过各自允许状态。

同一布局的背景、车辆和参考仍共享，这只叫空间留出诊断，不能叫独立场景测试。训练照片数量、轨迹数量、序列前缀数量是不同概念；不同轨迹会重复读取同一照片。

有多布局且标注 val/test 时，代码检查 group 和老师训练图是否跨划分。但是，**现有实现将所有非 train 场景合并为 probe，没有分别报告 val 和 test**。正式实验前应先拆开验证集与最终测试集；现在不能一边在 probe 调参，一边把它叫独立测试。

| 指标 | 含义与用途 |
|---|---|
| `both_visible` | 目标和至少一个干扰候选当前都可见；主要看能否区分两车 |
| `current_visible` | 目标当前可见，也含只有目标可见的容易情况 |
| `target_hidden_after_seen` | 目标以前出现，现在缺席；主要看历史保留 |
| `target_not_seen` | 目标从未出现；正确标签为 Null |
| `all_prefix` | 所有时间步，容易被大量 Null 帧影响 |
| `final` | 固定长度路线的最后一步，不是主动停止成功率 |

报告使用共同的“至今是否看过目标”真值。current 在目标消失后无法保留候选，会在该层吃亏，不能单独用这个指标证明融合收益。both_visible 更适合比较识别，但仍不是只有一个因素不同的严格因果实验。

另有三类检查：每步清空历史；每步故意轮换整个候选状态（含 seen/count）；从空历史重复同一照片三次。最后一项本轮仅抽了 6 张留图、只用参考 A，应扩充后再作稳健结论。

**10. 已经完成的历史实验：结果为负**

以下实验发生在用户明确要求后续不在本机运行之前。它们是本机 CPU 的历史记录，本次交接没有新增训练。详细记录另见 `docs/evidence-memory-stage1-results.md`。

| 模式 | 训练位置双车准确率 | 留出位置双车准确率 | 留出目标当前可见 | 留出目标出现后消失 |
|---|---:|---:|---:|---:|
| current | 93.1% | 22.1% | 26.9% | 0.0% |
| mean | 97.5% | 18.0% | 24.1% | 32.0% |
| memory | 85.5% | 4.4% | 16.6% | 24.7% |

表格为三个种子平均值。每个种子的留出双车分母 98、当前可见 151、出现后消失 73、全部前缀 384、最终任务 48；训练位置双车分母 106。它们是相关前缀，不是同等数量的独立试验。

| 种子 | current 双车正确数 | mean 双车正确数 | memory 双车正确数 |
|---|---:|---:|---:|
| 7 | 33/98 | 10/98 | 8/98 |
| 17 | 3/98 | 16/98 | 1/98 |
| 27 | 29/98 | 27/98 | 4/98 |

同一 memory 权重每步清空历史，双车平均从 4.4% 升为 8.8%；故意轮换历史升为 36.1%。后者是异常信号，不是可采用策略，不能仅凭此认定标签错误或锁定根因。

重复同一照片检查中，memory 概率分布的最大分量变化，三个种子各自均值为 9.3、13.0、10.7 个百分点；检查中最大为 26.4 个百分点。current 和 mean 在这个“从空历史只读同一帧”的特例中变化为 0；mean 对混合历史中的重复观察不天然幂等。

可以确认训练位置上能拟合、换位置后明显变差，当前记忆更新没有改善这一问题。尚不能确定是冻结特征、局部细节、背景偏差、匹配头、更新结构、训练量还是数据覆盖导致；不能把负结果解释成多图融合这个方向本身不成立。

历史代码验证为 179 项测试通过、17 项子测试通过，Ruff 通过；9 个保存检查点重新加载后，指标与原报告一致。测试覆盖缺席保留、候选追加和排列等变、早期帧梯度、未来帧不影响过去、掩码对齐、路线划分等。它们验证程序性质，不证明准确率达标。本轮文档工作不重新运行模型或测试。

**11. 文件与产物在哪里**

| 文件或目录 | 接手时用途 | Git 同步 |
|---|---|---|
| `dinoexp/dense_model.py` | 复用冻结的 `encode_views()` | 是 |
| `dinoexp/local_attention.py` | `SpatialAdapter`、`CrossBlock` | 是 |
| `dinoexp/evidence_memory.py` | 模型、状态、三种历史处理 | 是 |
| `dinoexp/evidence_data.py` | 冻结提取、掩码池化、缓存校验 | 是 |
| `dinoexp/evidence_training.py` | 划分、路线、训练、评估、消融 | 是 |
| `prepare_evidence_memory.py` | 准备冻结特征的 CLI | 是 |
| `train_evidence_memory.py` | 三模式、多种子训练 CLI | 是 |
| `evaluate_evidence_memory.py` | 重载权重和训练侧复查 CLI | 是 |
| `tests/test_evidence_memory.py`、`tests/test_evidence_data.py` | 新增行为测试 | 是 |
| `docs/evidence-memory-stage1.md` | 实现说明，包含历史本机命令 | 是 |
| `docs/evidence-memory-stage1-results.md` | 历史详细结果 | 是 |
| `data/search_grid_smoke_v1/` | 原始网格数据，需完整目录 | 否 |
| `out/teacher_grid_fit_e20_retry/last.pt` | 冻结任务头，第 20 轮 | 否 |
| `D:/Cache/dinov2_w/dinov2_vitb14_pretrain.pth` | 本机官方骨干，服务器需对应文件 | 否 |
| `D:/Cache/dinov2_repo/` | 本机 DINOv2 源码，服务器需自己的路径 | 否 |
| `out/memory_fullview_cache_v1/` | 本机历史特征缓存 | 否 |
| `out/memory_stage1_v1/` | 本机历史九组训练结果 | 否 |

运行目录包含 `run.json`、`split.json`、`probe_routes.json`、`report.md`；各 `{mode}_seed{seed}` 子目录含 `last.pt`、`history.json`、`probe.json`。memory 另有 `reset_history.json`、`swapped_history.json`。复查生成 `checkpoint_recheck.json`。`probe.json` 有分层指标及每条路线最终判断，但没有保存完整逐步 logits。

原始本机 `docs/teacher-grid-model-handoff.md` 目前未跟踪，不能假定服务器拉代码后一定有该文件；本交接已摘要其中影响记忆实验的关键信息。其他既有未跟踪诊断脚本和老师文档没有被一起推送，不应随手 `git add .`。

历史关键指纹：

```text
teacher SHA256:
a8eb9e1e41ce380a12a26f76d602a7dc9fd9b33ad35f026cd00f5ccef559ed35
backbone SHA256:
0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73
historical cache.json SHA256:
394d9799df31ce84936f3996559b1cac08f397c69827f3f46f57e3e6d257e2db
```

**12. 服务器同步与运行步骤**

下面命令交给用户在服务器执行。当前助手没有执行这些服务器命令，不宣称服务器已具备对应环境、已同步文件或已完成 GPU 验证。

**12.1 同步代码，确认分支**

```bash
cd /mnt/fast18/sunbo/DINO
git status --short
git fetch origin
git switch codex/evidence-memory-v1
git pull --ff-only origin codex/evidence-memory-v1
git log -3 --oneline
```

服务器若有未提交改动，应先检查并保留；不要用 reset --hard、强制 checkout 或清理 data/out 来解决同步问题。首次 switch 通常会跟踪同名 origin 分支；若已有其他同名分支，应先核对提交。

**12.2 核对环境和资产**

沿用已有 `dino-probe` 环境，先检查 CUDA，不要盲目重装 PyTorch：

```bash
conda run --no-capture-output -n dino-probe python -c \
  "import torch, numpy, PIL; print(torch.__version__); print('CUDA:', torch.cuda.is_available()); assert torch.cuda.is_available(), '需要可用 CUDA，暂不开始训练'"

conda run --no-capture-output -n dino-probe python -c \
  "from pathlib import Path; from dinoexp.local_paths import resolve_backbone_paths; print(*resolve_backbone_paths(None, None, 'dinov2_vitb14', Path.cwd()), sep='\n')"
```

后一条仅查找已有 DINOv2 源码和权重并打印两个路径，不下载资产。若查找失败，先定位或同步已有文件。准备脚本要求源码目录包含 `hubconf.py`；只有 `.pth` 不够。项目依赖可参考 `requirements-model.txt`，DINOv2 源码自身依赖也需满足；不要仅为运行缓存后的记忆头安装 CARLA 或分割模型。

设置本轮路径。**下面前两个值必须用上一条实际输出替换**，不要把占位文本当成路径：

```bash
MEMORY_DINO_REPO='/替换为已存在的DINOv2源码目录'
MEMORY_BACKBONE='/替换为dinov2_vitb14_pretrain.pth的完整路径'
MEMORY_TEACHER='out/teacher_grid_fit_e20_retry/last.pt'
MEMORY_DATA='data/search_grid_smoke_v1'
MEMORY_CACHE='out/memory_fullview_cache_server_v1'
MEMORY_RUN='out/memory_stage1_server_v1'

test -f "$MEMORY_DINO_REPO/hubconf.py" || exit 1
test -f "$MEMORY_BACKBONE" || exit 1
test -f "$MEMORY_TEACHER" || exit 1
test -f "$MEMORY_DATA/dataset_manifest.json" || exit 1
sha256sum "$MEMORY_TEACHER" "$MEMORY_BACKBONE"
```

这些变量与后续命令在同一 shell 执行。确认任务头确实是第 20 轮，且哈希和历史一致；若要用新任务头，明确登记新实验，重新准备特征。

**12.3 在服务器提取冻结特征**

```bash
conda run --no-capture-output -n dino-probe python -u prepare_evidence_memory.py \
  --data-root "$MEMORY_DATA" \
  --teacher "$MEMORY_TEACHER" \
  --backbone "$MEMORY_BACKBONE" \
  --model-repo "$MEMORY_DINO_REPO" \
  --output "$MEMORY_CACHE" \
  --device cuda --threads 4
```

这一条不更新编码器权重，但会执行模型提取特征，因此也只在服务器运行。这里没有传旧 dense-cache，避免服务器不存在旧缓存或规格不一致；有匹配缓存时可以另外加 `--dense-cache`。复用与重算可能存在数值差异，服务器重新提取不保证与历史 CPU 缓存逐位一致。

完成后检查 `cache.json` 的 status 为 complete。不要先手动创建一个空的 MEMORY_CACHE 目录；脚本会创建目录，已经存在但没有合法 cache.json 会被拒绝。

**12.4 小规模检查后，再跑完整对照**

先用独立目录检查数据到训练的服务器链路，不将一轮结果作为有效性结论：

```bash
conda run --no-capture-output -n dino-probe python -u train_evidence_memory.py \
  --cache "$MEMORY_CACHE" \
  --output out/memory_stage1_server_smoke_v1 \
  --epochs 1 --routes 4 --length 8 --batch-size 4 \
  --lr 0.0001 --seeds 7 --device cuda --threads 4
```

此入口每个种子都会训练 current、mean、memory 三种模式，没有“只训 memory”的 CLI 开关。随后运行与历史相同预算的完整对照：

```bash
conda run --no-capture-output -n dino-probe python -u train_evidence_memory.py \
  --cache "$MEMORY_CACHE" --output "$MEMORY_RUN" \
  --epochs 20 --routes 24 --length 8 --batch-size 8 \
  --lr 0.0001 --seeds 7 17 27 --device cuda --threads 4
```

训练不自动选择最佳轮次，保存固定末轮。本入口没有 AMP、DDP、调度器、早停或 resume 参数；当前检查点也没有优化器和随机数状态，不能当作精确断点续训文件。中断后请保留旧记录，换新 output 重跑；不要覆盖历史目录。

**12.5 重新加载权重复查**

```bash
conda run --no-capture-output -n dino-probe python -u evaluate_evidence_memory.py \
  --run "$MEMORY_RUN" --cache "$MEMORY_CACHE" \
  --device cuda --threads 4
```

复查读取 run.json 和固定路线，要求缓存指纹一致、重载权重后的分层指标与训练报告一致，并另外评估训练位置上的固定路线。训练和复查尽量使用相同设备与环境；跨 CPU/GPU 数值差异可能改变接近决策边界的 argmax。若报不一致，应保留差异调查，不要改报告数值让它过检查。该脚本尚未在服务器 CUDA 环境验证。

服务器完成后优先回传：run.json、report.md、checkpoint_recheck.json、split.json、probe_routes.json，以及各组 history/probe/消融 JSON。需要进一步诊断时再同步权重，避免一开始把整个骨干缓存搬来搬去。

**13. 缓存、迁移和错误处理的边界**

| 现象 | 含义与处理 |
|---|---|
| 找不到数据或参考图 | 检查是否同步了完整数据目录及相对路径 |
| teacher/backbone mismatch | 骨干与任务头不配套；核对文件和哈希，不绕过检查 |
| cache contract changed | 权重、清单、标签或特征规格变化；新建缓存目录 |
| need complete stage-one cache | 特征准备未完成或缓存类型不对 |
| output exists | 训练不允许覆盖；换新输出目录，不删旧结果 |
| held-out scene overlaps teacher pretraining | 留出组或图被老师模型拟合过；修正数据划分 |
| multiple layouts need val/test | 多布局不能继续使用单布局空间切分，需明确独立 group/split |
| saved checkpoint did not reproduce metrics | 查缓存、设备数值差异、版本与文件变化；不直接取消断言 |

准备阶段支持在相同来源与规格下复用已完成分片；训练阶段不支持续训。两者不要混淆。

本机缓存 contract 的来源路径键用 Windows 分隔符，Linux 重建 contract 时会使用 `/`。因此不要在 Linux 对复制来的 Windows 缓存直接继续 prepare；新建服务器缓存更稳妥。完整历史缓存可用于读取诊断，但跨系统迁移本轮没有端到端验证。

训练记录绑定 cache.json 的完整文件哈希。训练后不要在同一目录重新 prepare、编辑 JSON 或重保存场景缓存，再拿旧 run 做复查。即使语义相同，序列化文件改变也可能使校验失败。

数据集内部路径应为相对路径。原始采集清单使用 `/`；不要手工替换成服务器绝对路径或把 Windows 路径写入数据引用。

**14. 下一步怎么推进，先做什么**

第一步是在服务器确认相同代码、权重与数据链路能运行，并保留三种模式的对照。换 GPU 只是运行环境变化，不应预期它自动修复结构与泛化问题。

第二步先检查基础实例匹配。用新布局建立独立 train/val/test；将代码中的 val/test 合并逻辑拆开。训练侧检查同图换参考是否选对、掩码与参考是否对应；再在验证侧比较全图分支与完整多窗口特征。单帧都认不稳时，不宜直接增加飞行策略。

第三步用更容易检查的历史保存方案作对照，例如保留有容量限制的原始证据 token，再由注意力读取，同时明确定义重复照片处理。与当前门控压缩比较时，保持数据、读取预算、参考匹配和种子尽量一致，记录容量与参数差异。这个方案当前未编码。

第四步在融合收益成立后接预测分割和身份关联，分别量化真实路由与预测路由的性能差距。之后再补共享场景记忆，保存已经看过哪些区域、哪些候选仍需观察等搜索信息。

最后才训练下一动作推荐：需要动作成本、候选视角的预期证据收益、停止规则及可行移动约束，并以达到识别标准所需的观察次数或路径成本评价。当前模型没有动作输出，当前随机路线也不是学到的策略。

流形相关想法仍是后续研究方向：可在视角条件特征预测上研究局部连续性，并用于候选视角评估；目前没有实现预测头、位姿扰动正则或相应训练标签。不要把一般交叉注意力本身说成已经显式利用流形训练。

**15. 接手后的验收与交接原则**

代码层面，保留候选独立性、缺席不写入、候选追加、因果性与早期帧梯度检查。实验层面，学习记忆应在多个种子与独立场景上稳定超过单帧、均值；清空历史应失去有用信息，错误历史应造成合理退化，重复照片不应明显漂移。具体达标数值需要根据新验证集事先确定，不能看完测试结果再调整。

当前阶段不要用历史留出 79 张图不断挑超参数，也不要把仅两车、固定背景上的训练准确率当成任务完成。候选匹配、空间定位、身份关联、动作效率是不同指标，报告时分别说明。

下一位助手先读本交接，再读模型文件和训练逻辑；改动限于用户授权范围。本机不启动模型运行，代码完成后推送当前分支，由用户安排服务器训练。没有实际看到服务器产物时，只能说明命令已准备，不能声称训练通过。新增实现、建议和验证结论应继续分开记录。
