# 独立查询监督车辆证据记忆：实现与服务器交接

交付日期：2026-11-14。设计与计划日期：2026-11-13。基线分支 `codex/evidence-memory-v1`，HEAD `cf19b96fceb477b3b6630908fcc52493360fc37b`。

本轮已经编写实验代码。本机仅执行纯数据、缓存文件读取、人工指标、CLI、静态检查与报告生成；没有运行神经网络，没有训练，也没有登录服务器。**尚无证据说明新的 Memory 已经超过 Last。**

## 1. 任务交付状态

|任务|代码与本地检查|尚未完成的验收|
|---|---|---|
|1 Manifest 与独立查询|已实现；源图、父图、裁图、候选标签、Null、padding、随机清单检查通过|真实多车辆分组数据仍需采集/标注|
|2 CUDA 特征缓存|已实现并完成静态检查；旧缓存格式在本地读取验证|真实冻结编码、扰动编码和 CUDA 缓存准备|
|3 Reader 与损失|已实现；对应 CUDA 测试已编写|多查询等价、梯度、空组损失、重复 KL 的服务器运行|
|4 短序列与五组训练|已实现；公共初值、批次表、最终检查点、重载核验均有代码|smoke 和正式训练、数值/显存/速度验证|
|5 单车反事实评估|已实现；人工指标通过|Full/Skip/Repeat 的真实 CUDA 状态检查与效果评估|
|6 CLI 与报告|五个子命令已实现；人工记录可生成 JSON/Markdown/静态 HTML|完整服务器流水线验收|

因此，Task 1 与 Task 6 的本地工程部分完成；Task 2–5 的代码与测试交付完成，CUDA 验收仍待执行。不能把“有测试代码”写成“模型测试已通过”。

## 2. 模型实际如何工作

复用原 `EvidenceMatcher` 和 `region_sequence_training.advance`，没有修改旧网络。

```text
既有固定参考定位产生的车辆区域（仿真身份辅助路由）
    → RGB 裁图，可先做固定的 blur / cover
    → 冻结 DINOv2 + independent + pool → 16×256
    → Writer → 8×256
    → Updater + Gate + 该车旧状态 → 该车新记忆 8×256

未写入的独立 RGB 查询
    → 冻结 reference 编码分支 → 64×256
    → Reader 读取各车辆记忆
    → 各车得分 + Null
```

每条 episode 清空状态；缺席车保留旧状态。身份 ID 只用于路由和标签，外观组只用于抽样与统计，均不作为神经网络输入。没有新增 Teacher、Evidence Bank、可靠性/新颖性预测头、动作控制、三维信息或自动身份关联。

## 3. 数据协议与当前覆盖

从旧 `plan.json / observations.json / routes.json` 生成新清单。先固定路线、候选顺序、定位参考和扰动，再独立抽取查询。查询变化不会重跑老师定位。

- 取原路线连续 4/8 步窗口，默认步长 4，不删除中间照片来凑独立查询。
- 查询与**整个窗口**的写入源图、增强父图、裁图像素哈希隔离；同一图里的另一辆车也不能作为查询。
- 同图不同裁框依靠源图隔离；完全相同裁图另有像素哈希检查；外层划分也检查源图/父图/裁图交集。
- 整段路线固定候选排列。查询目标此前未获得允许写入的区域，或不在候选集合中时，真值为 Null。补齐候选不开放，补齐查询不计损失。
- 同车同源同裁图在正常轨迹只写一次；Repeat 诊断故意绕过去重。
- `grouped` 需要每个可用观察的外观组及来源。训练集没有有效历史查询会失败；不能用 probe 中有历史查询来冒充训练监督有效。
- `query_all` 明确关闭历史分组。C 与 D 在此模式下是同一个有效训练目标，不能解释为成功的历史消融。
- `identity_disjoint` 拒绝外层身份重叠；当前数据只能显式使用 `smoke_shared_identity`。

现有数据本地审计（旧来源 `out/region_sequence_mask_20261112_152736`）：

|项目|数量|
|---|---:|
|可用短窗口|1694|
|独立查询记录|19712|
|其中 Null 标签|7028|
|实际写入记录|5920|
|进入可用车辆观察池的独立源图|104|
|车辆身份|2|
|外观组 / 历史查询|0 / 0|

104 指可用车辆裁图池的源图数，不是原始图片库总数；大量全空图仍可作为路线步骤。窗口重叠，1694 条窗口不能视为1694个独立场景。train/probe 使用相同两辆车；现有 Null 主要来自尚未建立目标状态，没有真正外部身份负例。只有固定前端允许的区域进入查询池，不能据此宣称包含了所有遮挡/漏检情况。

外观注释为 JSON，key 使用旧 observations 中的 observation `key`：

```json
{
  "实际的源观察key": {"group": "side_left", "source": "manual-annotation-v1"}
}
```

配置 `view_group_manifest` 指向此文件；改 `group_mode` 为 `grouped`。仅有相机 yaw 不足以伪造相对车身方位。真正评估新车泛化时，另采车辆身份不重叠的 train/validation/test，重新生成来源路线和新输出目录。

## 4. 缓存和训练约定

缓存文件位于 `features/<encoder_signature>/...npz`。编码器命名空间包含权重、预处理、编码角色和源码哈希；条目另存样本摘要、父图、扰动、SHA256。加载时校验角色、样本、编码器、数组形状与文件哈希。

clean 观察只在旧 `region_sequence_features_v1` 来源完全匹配时复用。旧数组名为 `observation`；新缓存统一为 `tokens`。新查询一定走 reference；受损观察先修改 RGB 后再编码。blur 在不超过256像素尺度使用 sigma 1/2；cover 为15%/30%面积，保存 seed 和实际覆盖位置。没有 Token 加噪声冒充图像扰动。

编码器冻结，无 CPU 回退。源码校验允许 Windows CRLF/Linux LF 的换行差异，不允许内容差异。已开始准备的目录拒绝覆盖；异常留下 running 状态。改变代码后新建实验目录，不混入旧缓存。

训练损失：

```text
L_current + history_weight × L_history + repeat_weight × L_repeat
默认 history_weight=1.0, repeat_weight=0.1
```

- 每组先平均有效查询，再平均有效步骤，再平均有效 episode；空组贡献0并记录查询/步骤/episode分母。
- target/Null 权重只从训练清单统计；按实际训练查询总体均值归一到1。某类缺失时不伪造样本。
- 每步正常更新后计算监督；短序列结束只调用一次 backward 和 optimizer.step，正常递归路径不 detach。
- Repeat 从**正常更新之后**克隆状态，只额外写一辆本步出现且此前已建档的车；参照概率 stop-gradient，计算 KL(p_normal || p_repeat)。仅正常答对且非Null的合法查询参加该训练项。
- Repeat 状态不接入下一步；正常状态仍保留完整 BPTT。
- A_last/B_mean 使用 current+history；C_memory_current 使用 current；D_memory_history 使用 current+history；E_memory_repeat 使用全部三项。
- 各方法使用相同公共参数初值、episode抽样表、候选排列、查询和扰动。日志记录实际前向开销，不声称计算量相同。
- AdamW lr=1e-4、weight_decay=.01、clip=1、batch=4；smoke seed7，默认200步；full种子7/17/27，每组2300步，固定最后检查点。
- Writer/Updater/Gate/Reader/Matcher 联合训练，不能把全部改善归功于 Updater 单独学得更好。

每组保存 initial.pt、final.pt、优化器/RNG状态、损失日志、计数、schedule摘要、初始化摘要、时长和显存峰值。保存最终模型后，用固定训练输入做重载核验，最大logit误差超过2e-5则失败。full 要求同一清单的对应 smoke 和 smoke 评估已经完成；不设置准确率门槛，不依据 probe 挑检查点。

## 5. 评估怎样避免虚假的收益

对当前写入的车辆 i，从相同 before 状态产生：

- Full：本步所有车正常更新。
- Skip_i：仅关闭 i 的 valid，其他车继续更新。

两个分支读同一批查询，候选ID排列不变。主收益只计算 i 以前已有状态的配对，此时可用候选也相同。首次建档的 Skip 没有该车可用状态，**单列首次建档准确率，不进入 help/harm/net_gain**。下一正常步骤只接 Full。

help/harm 根据身份判断对错计算；conditional_harm 的分母是 Skip 原本答对的配对。margin 在可用车辆上重新归一化后计算，少于两个可用车或真值为Null时为 N/A；Null概率变化另报。概率上升不等于正确辨识增加。

Repeat 额外写同图1次和8次，正常答对/答错分别统计。所有查询的重复结果都报告，不只展示训练正则筛选过的查询。结果按方法、种子、车辆、外观组、质量、首次/后续、潜在互补组、场景分层，原始行保存Full/Skip/Repeat分数。

默认更新配对单位是“被干预车辆 × 当前独立查询”，因此更新A也会检查对B查询的影响；这些记录高度相关。总体跨方法汇总只作诊断，正式比较看各方法/种子及其分层。三个种子的均值和标准差不等于独立车辆泛化的置信区间。

## 6. 本地验证与未执行项

实际执行的核心命令：

```powershell
python -m unittest tests.test_evidence_query_data tests.test_evidence_query_metrics -v
python -m unittest tests.test_region_sequence_data tests.test_region_geometry tests.test_region_metrics tests.test_region_data -q
python run_evidence_query.py --help
python run_evidence_query.py plan --help
python run_evidence_query.py prepare --help
python run_evidence_query.py train --help
python run_evidence_query.py evaluate --help
python run_evidence_query.py report --help
```

实际结果：24项新纯数据/指标/报告测试通过，37项旧纯数据回归通过，共61项。另做10个Python文件的AST/compile静态检查、六次CLI帮助检查、纯数据/报告/CLI import 不加载torch、真实旧NPZ读取、真实清单重新生成摘要一致、人工HTML报告生成及14个相对资源链接检查。结果保存在 `docs/evidence-query-implementation-ledger.md` 及本地 `out/evidence_query_delivery_20261114/local-verification.json`。

未执行：`tests.test_evidence_query_server`中的12项CUDA测试、真实编码、训练、网络评估、重载数值核验、OOM/速度评估。没有把 skip 或静态解析计作 CUDA 测试通过。

## 7. 服务器可复制命令

本轮通过 `origin/codex/evidence-memory-v1` 分支交付代码，服务器先按下面命令拉取。原始数据、权重与本地人工报告不随 Git 上传。以下服务器路径来自现有工程记录，未登录重新核实。命令只使用已实现CLI。

```bash
set -euo pipefail
cd /mnt/fast18/sunbo/DINO
conda activate dino-probe
git switch codex/evidence-memory-v1
git pull --ff-only origin codex/evidence-memory-v1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

SOURCE=out/region_sequence_mask_20261112_152736
DATA=data/search_grid_smoke_v1
TEACHER=out/teacher_grid_fit_e20_retry/last.pt
OUT="out/evidence_query_$(date +%Y%m%d_%H%M%S)"

python -m unittest tests.test_evidence_query_data tests.test_evidence_query_metrics -v
python run_evidence_query.py plan \
  --source-run "$SOURCE" --data "$DATA" \
  --config configs/evidence_query_v1.json --output "$OUT"

# 只有此开关开启时，服务器模型测试才运行；失败就停止。
RUN_EVIDENCE_QUERY_CUDA_TESTS=1 python -m unittest tests.test_evidence_query_server -v \
  2>&1 | tee "$OUT/server-tests.log"

python run_evidence_query.py prepare \
  --data "$DATA" --source-run "$SOURCE" --output "$OUT" \
  --teacher "$TEACHER" --device cuda:0 --threads 4 --view-batch 1 \
  2>&1 | tee "$OUT/prepare.log"

# 不会下载模型；自动定位失败时为 prepare 添加已有 --model-repo / --backbone 路径。
python run_evidence_query.py train --output "$OUT" \
  --device cuda:0 --threads 4 --budget smoke --variant all \
  2>&1 | tee "$OUT/train-smoke.log"

python run_evidence_query.py evaluate --output "$OUT" \
  --device cuda:0 --threads 4 --budget smoke --split probe \
  2>&1 | tee "$OUT/evaluate-smoke.log"
python run_evidence_query.py report --output "$OUT" --data "$DATA"

# 2300步 × 五组 × 三种子。先检查上面的工程试验；当前query_all仍不提供历史组对照。
python run_evidence_query.py train --output "$OUT" \
  --device cuda:0 --threads 4 --budget full --variant all \
  2>&1 | tee "$OUT/train-full.log"
python run_evidence_query.py evaluate --output "$OUT" \
  --device cuda:0 --threads 4 --budget full --split probe \
  2>&1 | tee "$OUT/evaluate-full.log"
python run_evidence_query.py report --output "$OUT" --data "$DATA"
```

如先想用更短预算排查，只修改一份新配置的 `smoke_updates` 为1～200，并使用新OUT；不要改原输出中的manifest。正式完整历史实验应先准备真实外观组，用新配置/新输出执行。新增验证/测试划分时将 evaluate 的 `--split` 改为实际存在的 `validation`（也接受 `val`）或 `test`。

CUDA检查日志保留在 `server-tests.log`；plan生成的checks.json中的“NOT RUN”是计划时快照，不会因为手工执行unittest自动变成通过，必须结合服务器日志判断。

## 8. 文件与产物

|文件|职责|
|---|---|
|dinoexp/evidence_query_data.py|纯数据采样、审计、RGB扰动、缓存读取、权重统计|
|dinoexp/evidence_query_prepare.py|冻结CUDA编码、旧clean缓存复用与严格来源校验|
|dinoexp/evidence_query_training.py|多查询Reader、三项损失、BPTT、五组训练|
|dinoexp/evidence_query_eval.py|单车Full/Skip与Repeat1/8干预、原始预测|
|dinoexp/evidence_query_metrics.py|纯数据指标、分母与各方法/种子分层|
|dinoexp/evidence_query_report.py|仅从保存记录生成Markdown/JSON/HTML及图片资源|
|run_evidence_query.py|延迟导入的五子命令入口|
|configs/evidence_query_v1.json|预先固定实验设置，默认query_all烟雾模式|
|tests/test_evidence_query_data.py|数据/缓存/分割/标签测试|
|tests/test_evidence_query_metrics.py|人工指标与报告测试|
|tests/test_evidence_query_server.py|必须显式启用的CUDA模型测试|

服务器产物：

```text
OUT/
  manifest.json, checks.json
  prepared.json, features/<encoder_signature>/*.npz
  runs/smoke|full/
    run.json, schedule-<seed>.json
    <variant>-s<seed>/initial.pt, final.pt, log.json, reload_logits.npz
  evaluation/smoke|full/<split>/records.json, summary.json, <variant>-s<seed>.json
  report/index.html, report.md, summary.json, assets/, <budget>_<split>/*.html
```

下载查看至少保留完整 `report/`（包括assets和子页面）、`evaluation/`、两个run.json、manifest和日志。复现实验另需prepared、features、checkpoints及原始来源文件。报告可通过 `--data` 指向本地RGB重新生成；不会重新运行网络。

## 9. 已知限制与后续验收

1. 本机没有神经网络数值验证，CUDA环境兼容、显存和速度仍有待验证。任何失败保留日志与running产物，不删改真值或放宽来源检查。
2. 两车单场景和重叠窗口只能做机制排查；完整外观保持和独立车辆泛化尚缺数据。
3. 评估为逐候选、逐查询的离线干预，会比正常推理慢；不代表无人机实时速度。首次建档、重复1/8次、空分母都单独保存。
4. 窗口内没有未来信息进入记忆；查询源图可以来自库中其他时刻，只用于读出监督。查询抽样在训练前固定。
5. 静态报告复用保存分数与RGB裁图；人工示例标记 `synthetic=true`，不能当成实验效果。
6. 没有自动拒写策略；可靠性和新颖性通过输入扰动、历史/重复损失和干预结果研究。尚未证明其学会了正确的信息选择。

Git交付沿用 `codex/evidence-memory-v1`，实现提交基于 `cf19b96`。用户于2026-11-14明确授权提交并推送本轮相关文件；具体发布提交以 `git log` 和远端核验为准。本轮旧模块未修改，原有无关未跟踪材料保留。
