# 独立查询监督的车辆证据记忆 Implementation Plan

> **For agentic workers:** 使用 superpowers:executing-plans 在当前任务中逐项实施。用户要求非必要不使用子代理，沿用此偏好。步骤用复选框记录，不能把本文当作已完成实现。

**Goal:** 在不改动现有 8×256 记忆核心的情况下，增加独立查询、历史保持、重复稳定性训练与单车更新收益评估。

**Architecture:** 保留冻结编码器、EvidenceMatcher 和 advance。新增独立的纯数据清单、CUDA 缓存／训练／评估模块及只读报告入口。先完成直接监督；不实现 Teacher、额外锚点、多原型或动作规划。

**Tech Stack:** Python、NumPy、Pillow、PyTorch/CUDA、unittest、静态 HTML/Markdown。沿用项目现有依赖，不为本轮更换编码器或引入新训练框架。

**Spec:** [仓库内完整设计与交接](../../evidence-query-memory-design.md)。该文件与 [Obsidian 主稿](D:/Data/Notes/Obsidian/计算机/科研/01-当前工作/设计与交接-独立查询监督的车辆证据记忆-20261113.md)内容一致。跨机器交接时同时携带仓库设计与本计划；后续修改应同步主稿和副本。

## 2026-11-14 实施状态更新

用户已经明确授权本次实施。以下原计划检查项保留作验收清单，不能把“已写好 CUDA 测试”当成“已在服务器通过”。实际交付见 [实现与服务器交接](../../evidence-query-memory.md) 和 [实施记录](../../evidence-query-implementation-ledger.md)。

|任务|本轮实现状态|剩余验收|
|---|---|---|
|Task 1|数据模块、配置、测试已实现；本地纯数据检查通过|独立车辆/外观组数据仍欠缺|
|Task 2|缓存代码、来源校验、RGB扰动、旧格式测试已实现|CUDA编码待服务器执行|
|Task 3|多查询读取、损失、空组分母和CUDA测试已实现|前向/梯度测试待服务器执行|
|Task 4|BPTT、五组实验、初始化/批次一致性、重载检查已实现|训练及重载数值验收待服务器执行|
|Task 5|单车Skip、Repeat、指标已实现；人工指标测试通过|真实网络配对评估待服务器执行|
|Task 6|五项CLI、Markdown/JSON/HTML报告、服务器命令已实现|完整服务器链路待执行|

## Global Constraints

- 文档基线 HEAD：cf19b96fceb477b3b6630908fcc52493360fc37b；核心实现来自 747c800。
- 原设计交接时仅授权文档；本次用户已明确授权实现及本地纯数据/静态/CLI检查，仍不授权在本机执行模型。
- 本机不运行神经网络前向、不提特征、不训练、不评估、不自行 SSH。
- 原分支、旧脚本、旧配置和旧实验产物保持可用；新实验使用独立入口与新目录。
- 写入 16×256、查询 64×256、记忆 8×256；编码器冻结，ID 只供路由和监督。
- 无相机／三维输入，无自动关联，无 Teacher，第一轮无独立可靠性／新颖性头。
- 独立查询的源图及增强父样本与整个 episode 写入集合分离。
- 区域选择保持 source_reference_id 固定，更换查询不触发重新定位。
- 外层 split 不混用，旧 probe 不调参；正式新车泛化必须使用独立车辆实例。
- 缺失数据报错或明确标记 query_all 降级，不能复制图像或伪造外观组。
- 所有模型测试仅在用户授权的 CUDA 服务器启用；本地纯数据测试不得导入 torch。
- 非必要不使用子代理，不触碰仓库已有未跟踪资料。

## Review Focus

1. 同一源图的另一裁框、另一车辆或增强版本进入查询，造成隐蔽泄漏。
2. 更换查询重新选区域，导致两份记忆的输入不再一致。
3. 额外重复／skip 分支覆盖正常状态，污染下一步训练或评估。
4. 新候选首次写入与旧候选更新混合，或停止了其他车的更新，造成虚假收益。
5. 历史／重复查询为空、候选补齐或全部未见时出现 NaN、错标签或错误分母。

这些风险分别由任务 1、3、4、5 的具体验收覆盖。

---

## Task 1：清单、角色分离与可用数据审计

**Files**

- Create: dinoexp/evidence_query_data.py
- Create: configs/evidence_query_v1.json
- Create: tests/test_evidence_query_data.py
- Read/reuse: dinoexp/region_sequence_data.py、region_sequence_routes.py、region_ab_data.py

**Interfaces**

- build_manifest(source_dir: Path, data_root: Path, config: dict) -> dict
- validate_manifest(manifest: dict) -> dict
- pack_query_batch(episodes: list[dict], prepared: dict, feature_root: Path) -> dict[str, np.ndarray]
- 输入旧 plan/observations/routes 与可选 view_group 注释清单；输出模型无关的 JSON 清单和审计报告。
- 数组字段、形状、查询组编码完全遵守 Spec 第 7 节。
- manifest 中记录 schema_version=1、config、source_hashes、episodes、split_audit、coverage、status。
- 每个 episode 记录固定候选顺序、source_reference_id、顺序步骤、查询及各自角色、固定标签、view_group 来源和扰动。
- config.group_mode 仅允许 grouped / query_all。grouped 缺少可用组或独立查询时必须失败，query_all 明确关闭 history。

- [ ] 写 test_source_frame_disjoint：同图不同裁框、不同车辆和增强父样本重叠均拒绝。
- [ ] 写 test_query_changes_do_not_change_context：改变查询列表不改变写入清单的哈希。
- [ ] 写 test_view_group_missing：grouped 明确失败，query_all 成功但 history 计数为零。
- [ ] 写 test_permutation_padding_null：候选置换映射标签；未出现查询目标为 K；补齐位置不参与损失；不依赖当前预测决定标签。
- [ ] 写 test_seeded_manifest：相同输入与 seed 相同；不同外层 split 不被混合。
- [ ] 本机运行上述纯数据测试确认尚未实现时失败；然后实现接口并验证通过。
- [ ] 固定默认配置：长度 4/8、current/general=2、history=1、扰动 clean/blur/cover=0.70/0.15/0.15、repeat_probability=0.25、history_weight=1、repeat_weight=0.1。
- [ ] 输出覆盖报告：可用 episode、独立源图、车辆、外观组、Null、实际写入和缺失原因；不要只报告生成了多少前缀。

验收命令：python -m unittest tests.test_evidence_query_data -v。测试增加检查 sys.modules 中未加载 torch。

## Task 2：服务器特征缓存

**Files**

- Create: dinoexp/evidence_query_prepare.py
- Extend planned tests: tests/test_evidence_query_data.py、tests/test_evidence_query_server.py
- Reuse: dinoexp/region_ab_prepare.py 的 FrozenRuntime、cuda_device、pool

**Interfaces**

- prepare(args: argparse.Namespace) -> dict
- 输入 data、source_run、output、teacher、device、model_repo、backbone、threads、view_batch。
- 正常观察复用来源完全匹配的 16×256 缓存；受损观察走 independent + pool。
- 新查询走 reference，输出 tokens[64,256]、valid[64]。
- prepared.json 保存 status、manifest_sha256、encoder_hashes、code_hashes、observations、queries、environment。
- 每个缓存条目记录角色、父样本、扰动、路径和 SHA256；不能只依据 shape 判断角色。

- [ ] 纯数据测试 test_cache_role_and_hash_rejection：改角色或源哈希必须拒绝加载。
- [ ] 服务器测试 test_query_encoder_branch：64-token 查询来自 reference；16-token 写入来自 independent + pool。
- [ ] 实现服务器端 RGB 扰动；按 Spec 第 9 节固定尺度、sigma、覆盖比例与 seed。
- [ ] 验证相同扰动及 seed 可复现，不在 token 上加噪声冒充图像扰动。
- [ ] 编码前后校验权重／源图／manifest；状态按 running → complete 转换，异常保持未完成。
- [ ] 输出目录已经开始编码则拒绝覆盖；CUDA 不可用时失败，不回退 CPU。
- [ ] 验证冻结编码器无梯度且不进入后续优化器。

本机只检查代码与缓存元数据；此任务所有编码和模型测试由用户在授权服务器执行。

## Task 3：多查询读取与损失

**Files**

- Create: dinoexp/evidence_query_training.py
- Create/extend: tests/test_evidence_query_server.py
- Reuse: dinoexp/evidence_memory.py 的 EvidenceMatcher

**Interfaces**

- score_queries(model, state: dict, queries: Tensor, valid: Tensor, active: Tensor) -> Tensor
- queries 为 B×J×64×256，输出 B×J×(K+1)；inactive 查询输出占位零，损失侧排除。
- query_losses(normal_logits: Tensor, repeat_logits: Tensor | None, batch: dict, config: dict, class_weights: Tensor) -> dict
- 序列 logits 为 B×T×J×(K+1)；返回 total、current、history、repeat、counts。
- batch 保留 query_label、query_group、query_active、step_valid、repeat_active；标签只用于损失。
- 当前／历史按 Spec 约定分组归一；Null/target 权重仅由训练 manifest 生成。
- repeat 为 KL(stopgrad(p_normal) || p_repeat)，温度 1；只在正常非 Null 且判断正确的合法查询上训练。

- [ ] 写 test_score_query_does_not_mutate_state：多次查询后 slots/seen/count 不变。
- [ ] 写 test_multiple_queries_equal_single_reads：批量多查询与逐个调用旧 score 一致。
- [ ] 写 test_loss_empty_groups：无 history／repeat、padding 步骤、无活动查询不产生 NaN。
- [ ] 写 test_repeat_target_detached：参照分布不接收梯度；重复预测分支接收梯度。
- [ ] 写 test_loss_denominators：增加 padding 或复制被屏蔽查询不改变损失；合法复制的权重按约定平均。
- [ ] 服务器确认测试失败后实现接口，再验证通过。
- [ ] 记录每组有效数和加权前后数值；目标为零的项不得显示为“已有效训练”。

不新增可学习参数来实现 query 展开。不得把全无效查询直接传给旧 model.score。

## Task 4：短序列训练与公平对照

**Files**

- Extend: dinoexp/evidence_query_training.py
- Extend: tests/test_evidence_query_server.py
- Reuse: region_sequence_training.make_model、advance、parameter_hash

**Interfaces**

- rollout_queries(model, method: str, batch: dict, config: dict) -> dict
- 返回 normal_logits、repeat_logits、batch 中 repeat_active 的有效版本、final_state。
- train(args: argparse.Namespace) -> dict
- variant 允许 A_last / B_mean / C_memory_current / D_memory_history / E_memory_repeat / all。
- 共享初始化、manifest、批次和扰动次序；C 不计算历史项，E 才计算重复训练项。
- 每步正常状态只来自 advance；重复副本从正常更新后的状态建立，按 repeat_candidate 只额外写入一个候选。

- [ ] 写 test_temporal_gradient_reaches_updater：至少两次有效写入后，早期步骤与 Updater 有有效梯度。
- [ ] 写 test_repeat_branch_no_pollution：开启诊断分支不改变正常后续状态。
- [ ] 写 test_absent_candidate_unchanged：其他候选更新时缺席车的 slots、seen、count 保持。
- [ ] 写 test_no_future_influence 与 test_episode_reset。
- [ ] 实现每短序列一次 backward / optimizer.step，不在主路径 detach。
- [ ] 明确冻结编码器；AdamW lr=1e-4、weight_decay=0.01、clip=1、batch=4、seeds=7/17/27。
- [ ] smoke 固定 seed 7、最多 200 次更新；正式固定 2300 次更新，不在 probe 挑检查点。
- [ ] 保存初始化与数据 schedule 哈希、各方法实际前向次数、训练时长和峰值显存。
- [ ] 加载最终检查点复现固定输入，最大分数误差不超过 2e-5。

此处 joint-training 的结果归因于整个下游训练系统；不宣称只改变 Updater 参数。若需要只训练 Updater，另立配置和对照。

## Task 5：单车更新反事实与指标

**Files**

- Create: dinoexp/evidence_query_eval.py
- Create: dinoexp/evidence_query_metrics.py
- Create: tests/test_evidence_query_metrics.py
- Extend: tests/test_evidence_query_server.py

**Interfaces**

- single_vehicle_control(model, method: str, before: dict, observation: Tensor, valid: Tensor, queries: dict, candidate_index: int) -> dict
- 输入同一正常前缀，返回 full_state、skip_state、full_logits、skip_logits、eligible。
- skip_state 只关闭 candidate_index 的 valid，其余车辆本步写入保持。
- eligibility 要求该候选此前 seen 且本步确实写入；第一次写入另报。
- repeat_control(model, method: str, after: dict, observation: Tensor, valid: Tensor, queries: dict, candidate_index: int) -> list[dict]
- 故意绕过去重，报告额外 1/8 次；只作用于副本。
- summarize_updates(rows: list[dict]) -> dict
- evaluate(args: argparse.Namespace) -> dict

- [ ] 纯指标测试给四对结果：两次帮助、一次伤害、一次不变；断言 help=0.5、harm=0.25、net=0.25，并验证 conditional_harm 的分母。
- [ ] 写 test_first_write_excluded：新增候选不进入旧记忆更新收益。
- [ ] 写 test_candidate_i_only：A 被暂停，B 仍执行当前更新；两分支候选可用集相同。
- [ ] 写 test_controls_leave_prefix_unchanged：任何控制分支不覆盖传入历史。
- [ ] 写 test_ranking_vs_null：车辆排名正确但选 Null 时，两项指标不同。
- [ ] 写 test_no_competitor_margin：少于两辆可用车时车间 margin 标 N/A。
- [ ] 纯数据模块实现各指标、分子分母、分层及元数据；服务器模块仅产出原始记录。
- [ ] 报告输入质量、车辆、外观组、first/subsequent、旧／新组；不能只输出总准确率。

## Task 6：CLI、报告与交接

**Files**

- Create: run_evidence_query.py
- Create: dinoexp/evidence_query_report.py
- Create: docs/evidence-query-memory.md
- Extend: configs/evidence_query_v1.json、tests/test_evidence_query_data.py

**Interfaces**

- CLI 子命令：plan / prepare / train / evaluate / report。
- plan：--source-run、--data、--config、--output。
- prepare：--data、--output、--teacher、--device；可选既有模型 repo/backbone 路径和线程／批次参数。
- train：--output、--device、--budget smoke|full、--variant；数据位置由 manifest 解析并验证，可用 --data 明确重定位根路径。
- evaluate：--output、--device、--split；不得接收用于调阈值的测试参数。
- report：--output，仅读取保存分数，不导入模型运行模块。
- report(args: argparse.Namespace) -> dict，生成 report.md、summary.json、index.html、assets。

- [ ] 写 test_report_from_saved_rows：人工结果生成报告，不加载 torch，不进行模型前向。
- [ ] 检查 CLI 模块延迟导入，plan/report 不因顶层导入而加载 CUDA 运行组件。
- [ ] 报告明确身份辅助、固定定位参考、查询来源、状态容量和实验版本。
- [ ] 页面显示写入图、独立查询、历史列表和单车干预对照，不能显示未来照片为已见。
- [ ] 实现后实际检查 --help，更新 docs/evidence-query-memory.md 中可复制服务器命令。
- [ ] 写清下载结果需要报告 assets、原始评估 JSON 和 run.json；复现实验还需 manifest/prepared/checkpoint。
- [ ] 完成一次短服务器试验后核验全链路状态与检查点复现，再允许启动正式矩阵。

## 验证命令的边界

下列测试文件现已创建。实际本地运行记录见实施日志；服务器测试仍待用户执行：

本机只执行纯数据测试：

    python -m unittest tests.test_evidence_query_data tests.test_evidence_query_metrics -v

授权 CUDA 服务器执行模型测试，环境开关拟定为：

    RUN_EVIDENCE_QUERY_CUDA_TESTS=1 python -m unittest tests.test_evidence_query_server -v

服务器命令由用户执行，本任务不自行 SSH。训练 CLI 的最终调用须等 Task 6 实现并验证后再交付，不能把计划接口冒充现成程序。

## 里程碑与停止条件

- M1：数据审计通过，独立查询真实存在。缺数据先输出缺口，不凑样本。
- M2：缓存角色、冻结编码、批次和损失通过服务器检查。
- M3：一个种子的短试验完成，配对指标可以从原始记录重算。
- M4：正式矩阵完成，按预先协议汇总所有种子。
- M5：依据结果决定是否研究结构／教师；不是自动升级复杂模型。

错误、OOM、哈希不一致时保留现场，停止相关阶段；不能回退 CPU、混用旧产物或重写真值绕过失败。

## 自查记录

- [x] 与主设计对齐：直接监督、三类损失、查询隔离、无三维输入。
- [x] 各拟新增模块有明确输入输出和所属验收。
- [x] 初始模板、Teacher、多原型和动作模型没有混入第一轮。
- [x] 单车干预与旧暂停全部写入区分。
- [x] 测试与缓存角色、候选集合、空组分母逐项对应。
- [x] 本地与服务器工作边界明确。
- [x] Task 1–6 的代码与测试文件已交付，本地数据/指标/CLI/静态检查已执行。
- [ ] CUDA 模型检查、训练、效果验证仍待服务器实际执行。
