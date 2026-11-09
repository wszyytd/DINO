# 参考图引导搜索的连续观察数据

采集入口为 `capture_search_episodes.py`。默认先采 3 个布局、3 条序列，计划共 67 张观察图。配置文件为 `configs/search_episode_pilot.json`，种子为 `20261109`。旧 `data/grid_pilot_20261109` 是离散视图库。此入口使用独立输出目录，旧图不参与拼接。

## 服务器试采

使用已有的 `carla10` 环境和 Town10HD_Opt 地图。只读预检检查版本、地图、同步模式及现有车辆、行人、控制器、传感器。采集入口重复预检。若会话已经同步或有活动对象，使用独立的空闲 CARLA 会话。

```bash
cd /mnt/fast18/sunbo/DINO
git switch main && git pull --ff-only origin main
conda run --no-capture-output -n carla10 python -u capture_search_episodes.py --preflight
conda run --no-capture-output -n carla10 python -u capture_search_episodes.py \
  --output data/search_episodes_smoke_v1
```

`--dry-run` 只打印计划。因采集失败需要续跑时，保持同一配置和代码版本，在原命令末尾加 `--resume`。已完成序列先重新验证。未完成目录移到该数据集的 `failed_attempts/`，保留现场，随后从起点重采整条序列。失败记录也在每个序列的 `audit/failures.jsonl` 中。若进程被强行终止并留下 `.capture.lock`，先确认旧进程和其 CARLA 会话已经停止，再另开新的输出目录或处理残留锁。

从 `0119509` 升级采集器后续采，使用 `--resume --resume-after-code-update`。配置、路线、动作定义和质量规则须一致。已完成序列保留原代码版本，清单按序列记录各自使用的源码。相机配对按实际旋转比较正俯视姿态。超时记录包含两个传感器的收帧、姿态匹配计数和最近帧状态。

## 输出和语义

- `dataset_manifest.json` 固定配置、代码源文件哈希、种子、布局、参考角色、序列计划和完成状态。`splits.json` 按布局组划分。此配置只使用两种车辆资产，评估只能称为布局或背景留出检查。
- `action_spec.json` 记录 `kinematic_camera_sim` 约定。0–3 是相对机头前、后、左、右 0.5 米，4–5 是左、右偏航 15 度，6 是保持再拍。动作执行 4 个 0.05 秒插值 tick，随后等待同步拍摄。观测间隔另记。每条序列第一个观察没有动作，脚本达到预算时记录 `scripted_horizon`。朝向按动作变化。
- `scenes/<scene>/episodes/<episode>/rgb/` 保存无叠色的整图。`inputs.jsonl` 保存参考和 RGB 的文件定位、相对时间、上一动作和合法动作掩码字段。`iter_v1_inputs()` 输出解码图像与这些数值，文件名留在加载层。当前合法动作掩码为 `null`，因为模拟采集未检查无人机体积碰撞。
- `frames.jsonl`、`labels/` 保存身份、分别可见的掩码、目标是否可见和标注质量。身份为 `<scene>:A/B`。实例映射由移开与恢复自建车辆确认。`ignore_mask` 覆盖未映射的车辆或未标注像素。若 API 无法提供足够的语义标签枚举，则保守地忽略全部未映射像素。
- `reference/` 保存同一布局中独立拍摄的参考整图、目标掩码、原始实例图和实际裁剪框。靠近拍摄的目标框必须长边至少 240 像素、短边至少 140 像素且不触边。参考裁剪只提供给参考分支。
- `audit/observations.jsonl` 保存 RGB 与实例相机各自的帧号、时间、尺寸、位姿、名义标定和哈希。它同时记录实际插值姿态、起终状态、动作持续时间和 `collision_check=unknown`。原始实例 PNG 保留 R 语义标签和 G/B 不透明实例键。校准探针帧不写入正式序列。车辆实际外观配置、位姿和实例映射保存在 `episode.json` 的审计字段。[CARLA 实例相机说明](https://carla-ue5.readthedocs.io/en/latest/tuto_G_instance_segmentation_sensor/)
- `quality_spec.json` 固定第一轮筛查。零像素意味着当前帧不可见。有像素但触边、原图像素少于 900、518 等比缩放后少于 225 像素或短边少于 10 像素时记 `weak`。同步或映射无法核对时记录失败与重试。触边的严重截断程度由人工检查。`usable` 是像素和边界筛查结果，仍需目视检查。

## 验收和扩采

```bash
conda run --no-capture-output -n carla10 python capture_search_episodes.py \
  --output data/search_episodes_smoke_v1 --validate
```

看 `validation_report.md`、`validation.json` 与 `preview/index.html`。逐条检查参考与目标身份，重点检查两车同框、仅干扰可见、出画重现、重复后再取得有效观察，以及全部 weak 和 unknown。预览左侧是原始 RGB，右侧是另存的叠图。同步、哈希、动作连贯和掩码都经脚本检查，视觉检查仍需查看实际画面。`complete` 表示文件与序列结构完整。

三条试采验收通过后，才记录目视结果并在新目录采 24 条。扩采计划为 6 个布局，每个布局各 4 种路线模板。模板是计划，报告中的实际可见性统计才是采集结果。若试采的发现、双车同框、出画重现、仅干扰或重复后的有效观察缺失，入口会拒绝签收，先调整试采计划。

```bash
conda run --no-capture-output -n carla10 python capture_search_episodes.py \
  --output data/search_episodes_smoke_v1 \
  --record-smoke-review '已检查参考、三条序列、身份与全部弱/未知观察'
conda run --no-capture-output -n carla10 python -u capture_search_episodes.py \
  --stage pilot --reviewed-smoke data/search_episodes_smoke_v1 \
  --output data/search_episodes_pilot_v1
```

通过验收后的数据适合离线记忆训练。相机按 `set_transform` 插值采集，所有动作的可达性都标为未知。真机飞行、碰撞可行性和动作策略闭环评价需要另外的执行环境。

## 旧网格数据核查

本地读到旧目录 119 个文件。45 张 RGB、26 张掩码、15 张目标可见和 30 张不可见。两车同框 7 张、仅目标 8 张、仅干扰 4 张、两车均不可见 26 张。参考裁剪 173×104。旧记录中最小的四个目标掩码是 65、68、184 和 207 像素。现有图像及掩码哈希与清单一致。
