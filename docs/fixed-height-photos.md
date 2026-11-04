# 固定高度相似双车看图采集

本轮按用户要求先补照片供人工查看，不训练网络、不划分测试集。复用现有 capture_scene 的原图、裁剪、车辆姿态核对与清理机制；新入口为 collect_fixed_height_photos.py。现有采集配置没有 fixed_camera_height_m 时行为不变。

默认 3 种配对 × 2 个实际位置 = 6 组，每组分别以 A/B 为参考拍两次，共 12 个任务。配对是 Charger/MKZ 同色、Charger 同车型近似色、MKZ 同车型近似色。后两者主要区别是颜色，只是对照，不能宣称学到了同色同款车的实例识别。车型 ID 必须存在于服务器实际 CARLA 库；缺失会列出可用车型并停止，不随意换车。

每组相机 Z 坐标固定为锚点 Z + 8m（不是地形随动高度），水平环绕半径 18m，0–345° 每 15° 一张，共 24 角度。两车同向、间距沿用 8m；第二个位置交换 A/B 左右并改变云量。每任务有一张参考图、24 张同时包含两辆车的原图及相应裁剪。预计 300 张整图、588 条裁剪记录；A/B 重拍不是独立场景。

参考图也在同一个高度平面，但围绕目标车辆单独取景；查询图围绕双车中心拍摄。真实位姿写入 metadata.csv，固定高度模式的 elevation/distance 也记录实际值。完成校验对每张图检查固定 Z 的 3cm 容差。边缘出框、黑图等沿用原检查，失败不算成功；地面障碍物及可见部位仍需看图判断。

## 服务器命令

先启动原先可用的 CARLA 地图，确保没有其他采集客户端和场内车辆。不要启动训练环境，用 carla10。

```bash
cd /mnt/fast18/sunbo/DINO-evidence-300fcf8
git fetch origin codex/active-evidence-pilot
git switch --detach origin/codex/active-evidence-pilot
conda activate carla10

# 只读取实际车辆库和地图，预览数量，不创建采集目录。
python collect_fixed_height_photos.py \
  --output ../DINO/data/fixed_height_similar_20261104 --anchors 42 64 --dry-run

# 实际采集。
python -u collect_fixed_height_photos.py \
  --output ../DINO/data/fixed_height_similar_20261104 --anchors 42 64
```

自动选择两个间隔至少 40m 的实际 spawn 点，但不保证该处可容纳双车。若需沿用已验证的位置，可以在新采集命令指定 `--anchors 42 64`，程序会检查索引实际存在；位置是否空旷仍由 CARLA 实际生成检查。不能把索引存在当作摆放已成功。

```bash
# 中断后继续；已成功任务重新校验，失败目录保留，新尝试写新目录。
python -u collect_fixed_height_photos.py \
  --resume ../DINO/data/fixed_height_similar_20261104
```

续采严格使用保存的几何/配对配置；要改高度、车型或锚点，应使用新 output。新采集支持 `--height-m 8 --radius-m 18 --angle-step 15` 与 `--models vehicle.dodge.charger vehicle.lincoln.mkz`。改变高度时建议先预览少量任务，实际成像质量尚需服务器验证。

把整个输出目录下载，打开 `index.html`。首页按组展示缩略图，点进去的 preview.html 包括每个角度的整图及 A/B 裁剪。collection.json 记录计划、成功任务、失败尝试和错误；不覆盖旧照片、不删除失败目录、不自动移除已有车辆。

## 实现验证

验证固定相机高度不受观察中心或方位角影响、6 组/12 任务/588 裁剪计数、实际照片元数据高度漂移会被拒绝，并运行全量回归测试。本地未连接 CARLA，没有声称已经采出这些照片；最终外观、遮挡和车型相似度由服务器采集后人工检查。
