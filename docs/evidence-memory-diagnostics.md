# 车辆证据记忆：第一轮诊断使用说明

日期：2026-11-11。对应 `codex/evidence-memory-v1` 分支。

代码已实现原方案 P0–P2 的最小诊断。本机仅进行源码静态检查，没有执行模型前向、训练或新测试。下面的测试与诊断由用户在服务器运行；在拿到服务器输出之前，不宣称诊断重放已经通过。

## 本轮依据与范围

主要对象为 `out/memory_e200_20261111_090849` 的九组检查点，以及训练时绑定的 `out/memory_fullview_cache_server_v1`。服务器 200 轮双车留出准确率：current 60.5%、mean 55.1%、memory 39.1%；同一 memory 权重每步重置为 56.5%。三种方法在训练位置双车均为 100%。不再用本机 20 轮的 4.4% 作为最新结论。

新增诊断不训练，不改模型结构，不更改旧训练入口，不覆盖旧缓存、权重或报告。也没有实施新数据采集、质量写入门控、历史库、特征方案更换或动作训练。

## 服务器怎么运行

先同步代码；如果服务器有未提交改动，先检查保留，不强制覆盖。

```bash
cd /mnt/fast18/sunbo/DINO
git status --short
git fetch origin
git switch codex/evidence-memory-v1
git pull --ff-only origin codex/evidence-memory-v1
```

先在服务器运行新测试。它们使用小型张量和模型前向，不加载 DINO、不训练正式模型：

```bash
conda run --no-capture-output -n dino-probe python -m unittest discover \
  -s tests -p test_evidence_diagnostics.py -v
```

测试全部通过后，可以先做小范围检查。此命令只读 memory seed=7，每个划分取前 4 个参考任务，明确标记 partial，不能当作全部重放通过：

```bash
conda run --no-capture-output -n dino-probe python -u diagnose_evidence_memory.py \
  --run out/memory_e200_20261111_090849 \
  --cache out/memory_fullview_cache_server_v1 \
  --data-root data/search_grid_smoke_v1 \
  --output out/memory_diagnosis_e200_smoke_v1 \
  --modes memory --seeds 7 --max-routes 4 \
  --repeat-samples 1 --examples 4 --device cuda --threads 4
```

正式诊断九组检查点，包含训练位置与留出位置：

```bash
conda run --no-capture-output -n dino-probe python -u diagnose_evidence_memory.py \
  --run out/memory_e200_20261111_090849 \
  --cache out/memory_fullview_cache_server_v1 \
  --data-root data/search_grid_smoke_v1 \
  --output out/memory_diagnosis_e200_v1 \
  --device cuda --threads 4
```

输出目录必须不存在，脚本会创建。再次运行请换新目录名。只有缓存后的区域特征参加模型计算，不会重新编码 RGB；原图和掩码用于来源核对、质量/位置统计和有限样例展示。

## 输入检查与重放规则

- 核对 run 完成状态、cache.json 指纹、场景特征文件哈希、检查点配置、epoch/seed/mode，以及已有 checkpoint_recheck 中的权重哈希。
- 原始数据清单与来源文件必须符合缓存契约；再核对 RGB、参考和实例掩码哈希。哈希一致不等于语义标签一定正确，HTML 样例仍需人工检查。
- 正常重放沿用原 batch size、分组顺序、probe_routes 与训练位置 seed=902，保持旧评价标签。完整 probe 指标必须与 run.json 相同；若有旧 checkpoint_recheck，训练位置指标也必须相同。
- 检查失败时，新目录中的 manifest 标记 failed，保留不一致指标和逐步输出；不修改旧报告。设备数值差异、文件或版本差异应调查，不能直接删掉一致性检查。
- `--max-routes` 限制的是配好参考后的任务数，不是物理路线数；任何非零限额都按 partial 处理。
- 如果没有历史训练侧复查记录，仍会导出训练侧数据，但其 verified 为 false。`all_selected_replays_verified` 只覆盖本次实际选中的模式、种子与划分。

## 能拿到什么

| 文件 | 内容 |
|---|---|
| `diagnosis_manifest.json` | 完成/失败、输入与代码指纹、设备、参数、是否 partial、已选重放是否全通过 |
| `data_audit.md` | 来源核对范围、物理可见与缓存可见不一致的数量；不伪称已经人工审图 |
| `candidate_quality.csv` | 每个原始帧每辆候选的掩码面积、框中心、掩码重心、质量与两种可见性 |
| `steps.jsonl` | 正常历史逐步分数、概率、真值、候选身份映射、更新前输出、门控与状态变化 |
| `error_types.csv` | 双车同框中选对、选错候选、选 Null，按模式/种子/参考/划分统计 |
| `position_quality.csv` | 双车结果按质量和左右关系分组；不作为位置捷径的因果证明 |
| `reference_switch.jsonl` / `.csv` | 同一状态不写入、换参考的完整输出及统计 |
| `reference_switch_pairs.csv` | 同一前缀下至少两个已见目标的参考切换结果，记录是否都认对、预测是否变化 |
| `history_controls.jsonl` / `.csv` | 清空、暂停、只交换内容的成对结果及前后变化 |
| `repeat_checks.jsonl` / `.csv` | 重复后漂移、选择翻转、去重对照和再接新图结果；CSV 汇总均值/中位数/最大值 |
| `replay_checks.json` | 正常重放与旧指标的逐组核对 |
| `train_routes.json` / `probe_routes.json` | 本次实际使用的固定路线 |
| `*_repeat_selection.json` | 重复检查选样及各类限额，没有样例的类别不会被补造 |
| `summary.md` | 主要错误表与解释边界 |
| `examples/`、`example_selection.json` | 有限原图/参考/掩码、完整前缀预测及选样规则 |

没有记录的 CSV 会包含 `no_records` 空表头；对应 manifest/选样 JSON 可确认数量为零。

回传时建议打包整个诊断输出目录。这里没有复制模型权重和骨干缓存，图片最多按 `--examples` 限额收集。

## 怎样理解字段，避免误读

**旧真值与物理真值分开。** `truth_index` 仍基于缓存 valid 的累积出现历史，确保能复现旧报告；`physical_truth_index` 来自原始掩码是否有正像素。物理可见是分割标签定义，不代表人一定能识别。最后一个输出索引为 Null。

**两个分数差。** `identity_margin` 为目标分数减其他已见候选最高分；没有可比候选或目标未见时为 null。`candidate_null_margin` 为已见候选最高分减 Null。它们是未归一化分数差，不是校准置信度。双车表才是当前重点比较场景，不能把所有上下文混在一起解释。

**更新统计来自外部观察。** hooks 只读取 updater/gate 的结果，不改变返回值或权重。每个已写入候选有记忆变化量、相对变化量；memory 另有 gate 均值/最大值及更新提案变化量。第一次出现时直接写入 written，gate 虽被计算但不被采用，因此 `gate_applied=false`。

**历史干预的标签不跟着变。** reset 清空所有候选历史，但正确答案仍基于原始前缀。pause 只在目标以前出现过时统计。swap 只轮换已见候选的 slots，保留 seen/count；两辆车时就是互换，多车时为已见候选循环轮换。每个时间点都从正常历史克隆，干预不会累积污染下一次正常重放。

**swap 分为两个时点。** `before_current` 先在接收当前新图前比较，记录的 step/frame 为历史末帧，另记 next_frame_id；`after_current` 再给两份状态相同的新图。两个时点的可见性和真值各自对应它们的真实前缀。

**参考切换不更新状态。** 原路线已经为每张参考重复一次，导出时只从原 target=0 的那份路线状态切换所有已见参考，避免 A/B 任务重复统计。同样预测不变可能来自线索不足，需结合原图和质量判断。

**单帧历史对照。** `any_single_frame_correct_so_far` 记录用同一检查点逐张从空状态读取时，历史中是否曾在目标可见时选对目标；早期正确 Null 不算。`correct_without_any_correct_single` 记录当前正常历史答对、但这些单帧均未答对的前缀。它是定位案例的离线统计，不是证明融合机制的最终指标，也不是用真值挑照片的部署策略。

**重复的次数。** extra_reads=1/2/4/8 表示首次读取后的额外重复次数。每个次数从同一个首次读取后的状态独立开始。空历史、一张 usable 历史、usable 与非 usable 混合历史分别选样；每种类型、参考、场景默认最多 2 个前缀。不是每类都有样例。

重复后接的新图，是原路线里首个不同 RGB、目标有有效区域且质量标记 usable 的后续帧。它由标签作诊断选样，不输入未来特征来决定当前输出；也不保证该视角提供互补信息。可能跳过中间帧，所以它是受控特征实验，不是飞行路径。缺少符合条件的新图时明确留空。

**去重对照仅跳过写入。** 相同状态再次评分的最大概率差应不超过 1e-6；这验证确定性读取，不代表已解决相机产生的近似重复图像。此对照没有加入在线近似去重模块。

**图像选样有范围。** HTML 按划分、错误类型、质量、模式和种子轮流取第一批样例，默认最多 24 条，包含正确与错误；每条展示完整前缀输出、原图和独立掩码。它不覆盖全部情况，也不是人工筛选后的最佳展示。

## 文件职责与验证状态

- `diagnose_evidence_memory.py`：CLI，允许选模式/种子/划分与小范围检查。
- `dinoexp/evidence_diagnostics.py`：只读检查点、逐步重放、历史干预、重复与更新观察。
- `dinoexp/evidence_diagnostic_data.py`：来源核对、原始掩码质量与位置、物理真值旁路信息。
- `dinoexp/evidence_diagnostic_reports.py`：错误计数、CSV/JSONL 与 HTML。
- `tests/test_evidence_diagnostics.py`：新增服务器测试，包含旧 evaluator 对齐、observer 透明性、真值不随状态干预改变、内容交换不改标记、重复不修改原状态、报告写出和输入目录保护。

本机静态检查不能证明服务器数值重放正确。只有服务器测试通过、正式输出 status=complete、partial=false 且 all_selected_replays_verified=true 后，才能说这一轮已完成正常重放核对。诊断结果出来前不新增“位置捷径已确认”或“更新公式已确定有错”等结论。
