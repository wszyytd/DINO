# 参考图引导搜索的网格采集

## 服务器试采

在空闲 CARLA 会话使用 Town10HD_Opt 和现有 carla10 环境。预检只读。若存在其他会话的同步控制或活动对象，采集入口会退出。

```bash
cd /mnt/fast18/sunbo/DINO
git switch main && git pull --ff-only origin main
conda run --no-capture-output -n carla10 python -u capture_search_grid.py --preflight
conda run --no-capture-output -n carla10 python -u capture_search_grid.py \
  --output data/search_grid_smoke_v1
```

先采 1 个布局。默认 5×5 网格，间距 4 m，固定世界高度为停车锚点 Z 加 8 m，俯仰 −45°。采集 1024×768 整图和同步实例图，水平 FOV 70°。每点单独实拍 8 个方向，一共 200 张观察图，另有 A 和 B 各自独立拍摄的参考图。固定俯仰会留下正下方盲区，请检查相邻拍摄点能否补足观察。若有明显被建筑阻挡的点，可在该布局配置 `excluded_points` 为 `[[行,列], ...]`，再用新目录试采。已存在的输出目录禁止覆盖。失败时保留目录和 `audit/failures.jsonl`，修正原因后换新目录重采。

布局采用分散且不同朝向的白色 Dodge Charger 和白色警用 Dodge Charger。旧试采的原始 RGB 中 Lincoln 虽配置为白色，实际车身偏深且从上方能看到座椅。旧记录缺少生效属性与材质诊断，原因尚未区分。更换的 Charger 对在旧 `model_suite02/charger_variant_Aleft/images/overview.png` 中均呈白色，警用版车顶有可见装饰。需要确认这项区别在新采集的斜视图和 518 px 整图预处理后仍可看见。新审计文件保存请求颜色、actor 返回的属性、资产和传感器配置。车辆净距按两个实际投影包围盒外接圆计算保守下界，试点要求至少 1.5 m。

## 结构与使用边界

- `dataset_manifest.json` 保存布局、配置、代码哈希和状态。 `splits.json` 在采集和路线生成之前固定布局组的划分，同一停车背景锚点只进入一个划分。试点只有一个 train 布局，用于核实采集方法。
- 每个 `scenes/<scene>/graph.json` 保存点坐标、行列、端点采集状态、相邻点和状态边。CARLA 世界 +X 是行正向，+Y 是列正向，绝对 yaw 0° 指向 +X，90° 指向 +Y。每次平移走一个格子并保持原朝向，每次左／右转在原点旋转 45°。名义动作代价为平移 2 秒、转向 0.5 秒、保持 0.5 秒。图边的通行和碰撞字段为 unknown，运行输入的合法动作掩码和停止成功条件为 null。
- 每布局的 `rgb/` 是原始整图，`labels/` 和 `frames.jsonl` 保存按车拆开的掩码、可见像素、质量与 ignore mask。未映射车辆和未标注区域使用原有 ignore 策略。实例键通过移开／恢复探针校验，探针与观察分开。实际姿态、同步帧号、采集时间、标定和图像校验和在 `audit/`。 `references/` 含参考整图、对象区域、裁剪框、掩码和原始实例缓冲。
- `episodes/` 中每条序列从邻接图上选择 16～32 帧，覆盖搜索发现、同框后换位置、消失重现、重复后移动补充。两个模板用不同位置相对目标至少 25° 方位变化筛查候选，真实可见细节由目视确认。筛选使用标注，运行时通过 `dinoexp.search_episodes.iter_v1_inputs` 逐步解码当步整图和参考，另提供离线相对时间和上一动作编号。图状态、真实位置、身份、原始时间和标注均留在审计或监督层。
- 保持再观察是缓存重复读取。离线时间由图边代价累计，原始相机采集时间另存。预设路线和终止条件不提供最优动作或停止监督。当前支持离线定位、可见性和持续记忆实验，以及按名义代价进行离散图策略实验。连续控制、真机通行和专家动作监督需要额外验证与数据。

## 试点验收和扩采

```bash
conda run --no-capture-output -n carla10 python capture_search_grid.py \
  --output data/search_grid_smoke_v1 --validate
```

打开 `data/search_grid_smoke_v1/preview/index.html`，进入网格布局和序列页面。对照原始 RGB、掩码叠图、独立参考和 518 px 整图，记录检查的状态 ID。查看同框时两辆车的真实颜色和局部区别，核对同一辆车不同采样点的侧面与细节，检查 only A／only B／neither、全部 weak、目标身份及近处盲区。usable 表示固定像素和边界筛查通过，weak 表示有真实可见像素但触边或尺寸不足，absent 表示没有目标可见像素。

只有实际画面通过，且自动统计有完整双车同框、四种可见组合、每辆车有 weak 和不同方位的 usable 观察、两种参考均能生成四类路线时，再记录审查。请将下面的审查说明替换为具体状态 ID 和所见细节。

```bash
conda run --no-capture-output -n carla10 python capture_search_grid.py \
  --output data/search_grid_smoke_v1 --record-review '填写实际检查的状态ID、车身和车顶细节、换侧观察及518px结果'
conda run --no-capture-output -n carla10 python -u capture_search_grid.py \
  --stage pilot --reviewed-smoke data/search_grid_smoke_v1 \
  --output data/search_grid_pilot_v1
```

扩采计划为 6 个布局，train 和 val 各 3 个，计划每个划分 600 张原始观察图和 12 条派生路线。每个划分均计划覆盖四种模板和两个参考身份。使用相同两种资产，验证范围是布局和背景留出。 `validation.json` 分开报告独立布局、原始图、派生序列、可见组合、质量及生成缺口。派生路线缺失时保留缺口报告。 `complete` 表示采集文件和标注可核对，实际数据是否适合研究仍需检查图像与覆盖。
