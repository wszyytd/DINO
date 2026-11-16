# 多背景、随机双车姿态的往返数据采集

本入口在不同地图位置摆放两辆车，随机改变两车的位置和朝向，采集完整网格照片，再从中挑选双车持续同框的往返路线。原来的采集、短路线和29步往返入口均保留，旧数据不覆盖。

代码在本机只做纯数据、模拟和静态验证，没有连接CARLA、没有新采照片、没有运行模型。下面的采集命令应在你实际运行CARLA的机器上执行；LiVOS推理命令应在GPU机器上执行。

## 1. 默认采多少、变化哪些内容

配置：`configs/multiscene_roundtrip_v1.json`。

| 项目 | 默认值 |
|---|---|
| 地图 | `Town10HD_Opt`，必须已经加载 |
| 不同背景 | 6个相距至少60米的地图出生点附近 |
| 每个背景 | 2组独立随机摆车，共请求12个布局 |
| 数据划分 | 前4个背景用于train，后2个用于val；同一背景的布局不跨组 |
| 两车 | 白色普通Charger、白色警用Charger |
| 位置 | 两车中点在背景中心附近前后/左右各随机偏移2米；两车连线方向任意，间距9～11米 |
| 车辆朝向 | 两辆车各自独立随机转向，范围覆盖一整圈 |
| 车辆俯仰、侧倾 | 不主动随机翻滚；车辆落地、稳定后冻结，倾斜过大拒收 |
| 相机 | 1024×768、70度视野、离地约8米、向下45度 |
| 原始观察 | 每布局5×5位置、间距4米，每位置8个方向，共200张网格RGB |
| 路线要求 | 首帧两车均不截边；全程每辆车至少8192个可见像素 |
| 路线长度 | 去程至少10张不同照片，最多17张；加返程后19～33步 |
| 失败重试 | 每布局最多5次，每次重新随机摆车，失败文件保留 |

“6个背景”指同一地图中的6个不同位置，不是6张不同城镇地图。自动选择会排除过度倾斜的出生点，并尽量选择相互远离的位置；它不保证附近没有墙、树或建筑。出生点附近不能摆下车辆、参考图不可用、或者没有合格长路线，都会记录失败并尝试下一次布局。

所有布局首次成功时，保留的原始网格RGB为12×200＝2400张。最多60次尝试，网格RGB上限12000张，另有参考图、校准照片、原始实例图和掩膜。失败目录也保留，因此磁盘用量高于最终合格照片的用量。实际成功数会写进清单，不承诺一定采满12组。

两辆车仍使用配置中的两种资产。这一版扩充的是背景和观察角度，不据此声称能区分没见过的具体车辆。

## 2. 在CARLA机器上更新代码

在`DINO`仓库目录中操作，使用已安装匹配版本CARLA Python包、numpy和Pillow的Python环境：

```powershell
git pull --ff-only origin codex/evidence-memory-v1
python capture_multiscene_roundtrip.py --stage dry-run
```

默认阶段就是`dry-run`，它只打印计划，不连接CARLA、不创建照片。输出包括12个布局、2400张首次成功照片和最多12000张尝试照片，以及各布局第一次摆车的随机参数。

确保CARLA已运行，当前地图为配置中的地图。场景中不得已有车辆、行人、传感器或其他同步采集程序。入口不会擅自切换地图、清理别人的车辆或关闭别人的程序。客户端与服务端版本必须完全相同。

先检查连接、地图、资产和6个背景位置：

```powershell
python capture_multiscene_roundtrip.py --stage preflight --config configs/multiscene_roundtrip_v1.json
```

这一步不摆车、不采照片。输出每个背景的出生点编号、实际位置、分组和train/val归属。

## 3. 开始采集

使用一个还不存在的新目录：

```powershell
python capture_multiscene_roundtrip.py --stage capture --config configs/multiscene_roundtrip_v1.json --output data/multiscene_roundtrip_20261116_v1
```

每次尝试按以下顺序执行：

1. 重新检查世界空闲，按固定种子生成两车位置和各自朝向。
2. 摆车，等待车辆落稳后冻结；记录请求位置和实际位置。
3. 核对两车间距，校准仿真实例编号，采两张独立参考图。
4. 同步采200组RGB、原始实例图、两车掩膜和忽略区域。
5. 重读文件，检查同步记录、图像与掩膜摘要、相机位置和原始标注。
6. 挑选符合要求的长往返路线。没有合格路线则本次拒收，换一组随机姿态。
7. 合格布局写进可用清单。失败尝试留在原目录，保留原因。

每次只销毁本入口创建的演员和传感器，并恢复进入采集前的天气与世界设置。发现清理失败会立刻停止，不继续摆下一组车。

退出码含义：

- `0`：请求的所有布局都成功。
- `2`：部分布局达到重试上限；如果有成功布局，可用子集已发布。务必查看`validation.json`中的成功数、失败数和train/val数量。
- 其他异常：中断或环境问题；不要把目录存在当作采集成功。

如果全部失败，清单保持`incomplete`，不能导出成可用测试数据。部分成功也可能恰好缺少train或val，不能把这种子集直接当成完整的训练/验证实验。

## 4. 中断后继续

优先用一次Ctrl+C正常停止，等待当前演员和传感器清理结束。用同一配置、同一代码版本和原输出目录继续：

```powershell
python capture_multiscene_roundtrip.py --stage capture --config configs/multiscene_roundtrip_v1.json --output data/multiscene_roundtrip_20261116_v1 --resume
```

已经合格的布局会重新检查完整性，然后跳过。中断尝试的照片保留，下一次使用新的尝试目录和新种子，不补写旧照片。中断尝试计入每布局最多5次的上限。已用完5次的布局不会无限重试；需要更多尝试时应改配置并建立新数据集。

恢复要求配置、相关源码摘要、CARLA版本和地图出生点列表完全一致。修改配置或代码后，请使用新目录。

同时只允许一个采集进程。强制杀进程或断电可能留下`.capture.lock`；这时入口会拒绝恢复。确认旧采集进程已经退出、CARLA中残留演员已由你检查处理后，才能手工移除这一个锁文件，再执行恢复。不要对仍在运行的采集进程删除锁。

## 5. 检查和浏览真实照片

采集完成后执行：

```powershell
python capture_multiscene_roundtrip.py --stage validate --output data/multiscene_roundtrip_20261116_v1
python capture_multiscene_roundtrip.py --stage preview --output data/multiscene_roundtrip_20261116_v1
```

预览会新建`data/multiscene_roundtrip_20261116_v1/roundtrip_preview/`，包括：

- `routes.html`：实际路线图、去程/折返点/返程、每步可见像素与截边情况、背景分组。
- `preview/index.html`：原图、两车真值和逐步历史。此时没有预测分数。
- `plan.json`：可重读检查的全部输入、文件摘要、路线和分组。

同一输出不重复生成预览，以免覆盖旧产物。需要另一个预览目录时使用：

```powershell
python run_multiscene_roundtrip_test.py --stage audit --data data/multiscene_roundtrip_20261116_v1 --output out/multiscene_preview_v2
python run_multiscene_roundtrip_test.py --stage preview --data data/multiscene_roundtrip_20261116_v1 --output out/multiscene_preview_v2
```

查看每个背景，确认车确实位于合理地面、车辆特征清楚、画面背景有变化。自动通过只代表预先设置的检查通过，不代表已经进行人工逐图确认。

## 6. 在GPU服务器上测试LiVOS

将完整采集目录复制到服务器同名相对位置，保持目录结构，并使用相同提交的代码。采集清单的源码摘要统一按LF换行计算，支持Windows采集后到Linux验收。测试计划沿用旧接口的逐字节源码检查，因此服务器应重新audit，不要直接使用本机生成的`plan.json`。

```bash
cd /mnt/fast18/sunbo/DINO
git pull --ff-only origin codex/evidence-memory-v1
conda activate dino-probe
OUT="out/livos_multiscene_$(date +%Y%m%d_%H%M%S)"
python run_multiscene_roundtrip_test.py --stage run \
  --data data/multiscene_roundtrip_20261116_v1 \
  --output "$OUT" \
  --livos-repo /mnt/fast18/sunbo/LiVOS \
  --weights /mnt/fast18/sunbo/LiVOS/weights/livos-nomose-480p.pth \
  --size 480 --device cuda --threads 4
```

沿用已核对的LiVOS仓库版本`8c26cb03f570942a0c9a26f5bf7a8d378fc9ac60`。已下载的`livos-nomose-480p.pth` SHA256为`9624de29038b1c8d64aa3f0ecc212aceea094abe3539666055f3c256dfca615e`。

`run`顺序为audit、preview、evaluate、report。每条路线分别对A和B做两次独立单目标测试，使用完全相同的照片顺序。每次仅第0步输入目标掩膜，后续真值仅用于评分；到折返点不清空记忆。最终查看`$OUT/report/index.html`。

这里只执行预训练LiVOS分割，不训练DINO记忆，也不是参考照片匹配实验。train/val和背景分组会保留在导出计划及页面中，分组信息不会进入模型输入。原始RGB、掩膜、参考图和路线可供后续DINO记忆实验接入；这次没有新增DINO训练方法。

## 7. 怎样修改采集规模

修改配置副本并使用新输出目录：

| 配置字段 | 含义 |
|---|---|
| `sampling.backgrounds` | 需要几个不同位置的背景 |
| `sampling.train_backgrounds` | 前几个背景属于train，至少1个且小于总背景数 |
| `sampling.layouts_per_background` | 每背景随机摆几组车 |
| `sampling.max_attempts` | 每组最多尝试几次 |
| `sampling.anchor_spawn_indices` | 默认null自动选择；也可指定恰好N个出生点编号 |
| `sampling.min_background_distance_m` | 背景间最小距离，默认60米；程序还检查观察网格范围不相交 |
| `sampling.center_jitter_m` | 两车中点的前后/左右随机范围 |
| `sampling.separation_m` | 两车间距范围 |
| `sampling.yaw_offset_deg` | 每车独立朝向范围，默认覆盖整圈 |
| `route.min_outward` | 合格去程至少多少张不同照片 |
| `route.max_outward` | 最多多少张，默认17；搜索是有限搜索，不保证找到理论最长路线 |
| `route.min_view_pixels` | 每一步每辆车的最低可见像素数 |
| `seed` | 复现随机计划的总种子 |

例如先试两个背景各一组，可以在配置副本中设`backgrounds: 2`、`train_backgrounds: 1`、`layouts_per_background: 1`。保持`min_outward: 10`可继续检查长路线可行性。换地图时修改`map`、提前加载目标地图，并重新运行preflight；不会自动加载地图。

## 8. 数据记录和边界

```text
数据目录/
  dataset_manifest.json       # 请求布局、每次尝试、成功清单、随机种子、配置和代码摘要
  splits.json                 # 按背景的train/val
  action_spec.json
  quality_spec.json
  validation.json             # 最终成功/失败/每条路线长度
  attempts/
    scene_00_00/
      attempt_01/              # 失败也保留
        scene.json            # 实际车辆位置/姿态、校准和清理记录
        graph.json
        frames.jsonl
        rgb/
        labels/               # 原始实例图、两车掩膜、忽略区域，具体文件由底层采集器保存
        audit/
        references/
        roundtrip_routes.json # 仅合格尝试保存
      attempt_02/
  roundtrip_preview/           # 手动执行preview后生成
```

车辆跨帧编号来自仿真，不是自动跟踪。路线也是根据真值可见区域离线挑出的，不是已经学会的主动导航。去程沿真实离散图边平移、少量转45度，返程精确重访去程照片；总步数和不同照片数分别记录。这里没有新拍的连续飞行视频，也没有末端机身原地转180度的录像。背景距离和图边不代表已通过真实飞行避障检查。

## 9. 本机验证范围

`tests/test_multiscene.py`覆盖随机计划复现、背景隔离、失败目录保护、有限重试、普通中断与运行中记录恢复、清理失败停止、配置/版本/源码和成功文件校验、导出计划/查看器兼容。测试用模拟CARLA接口和临时小图片，不冒充新采集结果。另外使用现有`search_grid_smoke_v1`真实照片做只读路线验收回归，并校验旧29步计划仍能读取。

Python3.10语法、Ruff与旧车辆路线测试一并检查。本机没有import torch或carla，没有运行模型、训练、特征提取或真实采集。实际新背景能否摆车及长路线的成功率，需以上述CARLA机器采集结果为准。
