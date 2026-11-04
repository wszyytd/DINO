# 裁剪证据实验 v2：服务器执行顺序

继续使用车辆裁剪，不依赖尚未训练完成的整图注意力头。这次有三项改动：候选相对分数、冻结上游重训价值头、恢复扩采并生成分组学习曲线。先运行第 1 步，再扩采；不需要先重训表征模型。

## 更新代码

下面按现有服务器工作树路径编写。旧实验仍在它自己的 out 目录内。

```bash
cd /mnt/fast18/sunbo/DINO-evidence-300fcf8
git fetch origin codex/active-evidence-pilot
git switch --detach origin/codex/active-evidence-pilot
conda activate dino-probe
python -m unittest discover -s tests -p 'test_evidence_*.py' -q
```

## 1. 复用服务器 2000 步模型，只检查价值训练

```bash
python -u run_evidence_pilot.py \
  --data ../DINO/data/more_photos_20261028_030040_745590/train \
  --cache ../DINO/out/more_photos_20261101_135207_870565/features \
  --memory-run out/active_evidence_20261103_130010_646203 \
  --allow-pilot-location-split --skip-evaluation \
  --score-mode relative_softmax_v2 \
  --value-steps 300 --states-per-episode 12 \
  --alphas 0 0.1 --seed 7 --device cuda
```

输出自动使用新目录。`--memory-run` 会加载并复制旧运行的两个 `alpha_*/memory/last.pt`，检查特征配置、原特征缓存清单、alpha、训练任务及位置组。表征权重不更新；旧 value 权重和旧标签不加载，标签全部重新生成。新实验仍有独立的完整表征检查点，不依赖移动后的原路径。

新分数为 `s = 2 * softmax(logits / temperature) - 1`，margin 仍为目标分数减最强干扰分数。共同 logit 偏移不会改变分数；真正极大的相对差异仍可能饱和，这不是通用的消除饱和方法。分数没有经过置信度校准。

`--skip-evaluation` 不加载评测角色的特征张量，不生成该角色的标签，不运行其策略。数据清单仍用于分组和重复图片审计。查看新目录中的 `report.md` 与 `alpha_*/value/training_diagnostics.json`，重点看相对全零预测的 MSE、动作 top-1、两两排序及相对随机动作的 regret；这些是训练拟合指标。

需要严格对照旧分数时，用完全相同命令另跑一次，只将 `--score-mode` 改成 `legacy_tanh_v1`。两个模式都会新建输出并重新生成标签。两种标签的数值尺度不同，不能单独比较绝对 MSE 大小；应比较各自零预测基线和动作排序。新旧分数模式及温度均纳入上游指纹，禁止混用标签或价值检查点。

## 2. 恢复原扩采计划

先预览。此命令离线读取已存记录，不连接 CARLA、不写文件：

```bash
conda activate carla10
python collect_match_photos.py \
  --resume ../DINO/data/more_photos_20261028_030040_745590 \
  --expand-plan --dry-run
```

输出列出已完成、待采、总场景数以及新增 case。若保存的原计划是 60 个任务，应恢复到 60；它不是新发明的几千张计划。以每任务 1 参考＋20 候选裁剪计，60 个任务为 1260 条裁剪记录。实际数量以预览为准。

`--expand-plan` 要求已保存 `original_plan`，不能与 `--pilot` 同时使用。保留当前计划里的所有 case 配置，包括已经修正的相机和车辆摆放；仅恢复原计划中被省略的 case。新增 case 使用其原配置，因此可能遇到尚未验证的拍摄位置。程序会在采集失败时停止并保留记录，不会把失败算成功。

确认 CARLA 正常运行、使用原地图和空闲受控场景后执行：

```bash
python -u collect_match_photos.py \
  --resume ../DINO/data/more_photos_20261028_030040_745590 \
  --expand-plan
```

中断后可用同一命令继续，已完成场景会重新校验并跳过。原有不完整尝试沿用现有归档机制；不会覆盖已完成图片。扩采恢复的是原有资产／位置／光照安排，没有增加跨背景参考、任意新车型或更密集的中间视角；这些属于后续采集设计。

## 3. 扩采完成后建立新缓存

不要沿用旧的 630 图缓存。全部采集完成后，程序会重建 pairs_train／val／test 清单。使用新输出目录提取冻结特征：

```bash
conda activate dino-probe
python cache_local_features.py \
  --manifest ../DINO/data/more_photos_20261028_030040_745590/pairs_train.json \
  --manifest ../DINO/data/more_photos_20261028_030040_745590/pairs_val.json \
  --manifest ../DINO/data/more_photos_20261028_030040_745590/pairs_test.json \
  --data-root ../DINO/data/more_photos_20261028_030040_745590 \
  --output out/evidence_expanded_features_20261104 \
  --model dinov2_vits14 --size 224 --layers 3 6 9 12 --device cuda
```

这里只统一提取冻结、无拟合的 DINO 特征；后面的训练过程不会读取预留测试角色的特征。已有骨干权重由项目自动查找逻辑定位，不要求升级 Torch 或驱动。

## 4. 生成小／中／大训练集

```bash
python prepare_evidence_splits.py \
  --data ../DINO/data/more_photos_20261028_030040_745590 \
  --cache out/evidence_expanded_features_20261104 \
  --output out/evidence_expanded_splits_20261104 \
  --value-positions 64 --fractions 0.25 0.5 1.0 --seed 7
```

`--data` 在这里指向包含 train／val／test 的集合根目录。原 train 中的位置 64 固定给价值训练，其余 train 位置组成表征池；原 val 用于开发验证，原 test 预留。需要改变价值训练位置时在实验前明确设置，不能根据验证成绩挑选。

输出 `size_1.json`、`size_2.json`、`size_3.json`。它们按完整 A/B 物理配置组生成嵌套子集，价值训练集、验证集和预留测试集保持不变。验证／测试车型与训练车型、各角色位置及图像哈希都做隔离检查。分组不完整、没有预留测试、缺少身份字段或存在泄漏会报错。比例四舍五入到完整组，实际任务数以输出为准；少量组时不同档可能大小相同。

该曲线主要检验“增加表征训练数据”的影响，价值训练数据量被固定，不应称作所有模块的数据规模曲线。此处使用过的旧位置 105 可以进入开发训练池；它已经不作为新的独立测试。参考背景仍未独立，位置不同也不自动证明背景完全不同。

## 5. 跑配对学习曲线

先完成一个档位，再扩大到全部组合；以下命令会执行 3 个数据量 × 3 个种子，每次内部比较 alpha=0 和 0.1：

```bash
for size in 1 2 3; do
  for seed in 7 17 27; do
    python -u run_evidence_pilot.py \
      --data ../DINO/data/more_photos_20261028_030040_745590 \
      --cache out/evidence_expanded_features_20261104 \
      --split-manifest out/evidence_expanded_splits_20261104/size_${size}.json \
      --memory-steps-per-episode 200 --value-steps-per-state 10 \
      --states-per-episode 12 --budget 8 \
      --alphas 0 0.1 --seed "$seed" --device cuda || break 2
  done
done
```

每个表征任务平均分配约 200 次随机更新，数据量增加时总步数随之增加。价值头按生成的训练状态数量分配更新。这里的 200／10 是预先固定的起点，不是验证过的最优设置；程序记录实际步数且不自动选最好结果。每次自动新建输出目录，报告保存 seed、角色清单和分数版本。

比较随机、固定／覆盖、学习策略和各表征自己的贪心 oracle。首先判断辅助分支是否随数据量稳定改善，以及学习策略是否超过简单策略。不要直接比较不同分数定义的绝对收益值。最终测试还需在选定设置后单独开启明确的测试流程；本轮入口默认只使用清单中的开发验证角色，不会自动消费 reserved_test。

## 本地验证

2026-11-04 已用同步回来的服务器 2000 步检查点执行 relative_softmax_v2 的价值训练诊断，两个分支各 96 个训练状态、300 步；原表征检查点字节一致，评测被跳过。两分支训练 MSE 分别为 0.219305／0.162813，各自全零基线为 0.243488／0.180670。分数定义修正已经可运行，但这些训练结果不代表策略有效或泛化改善。

采集扩展通过模拟文件与调用链测试，尚需服务器真实 CARLA 执行；没有在本地伪造新增照片。当前扩采后的完整学习曲线也尚未运行。
