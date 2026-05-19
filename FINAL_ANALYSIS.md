# Ali-CCP 点击与转化预测项目最终报告

## 1. 项目介绍

本项目基于天池数据集 Ali-CCP: Alibaba Click and Conversion Prediction，围绕推荐广告场景中的点击率和转化率预测展开。数据来自真实电商推荐链路，样本以曝光为基本单位，包含广告侧稀疏特征、用户/上下文侧稀疏特征，以及点击和转化标签。

项目数据位于 `D:\Ali-CCP`，原始文件约 41GB：

- `sample_train/sample_skeleton_train.csv`
- `sample_train/common_features_train.csv`
- `sample_test/sample_skeleton_test.csv`
- `sample_test/common_features_test.csv`

数据不是常规宽表，而是稀疏特征格式。`sample_skeleton` 中每条样本通过 `common_feature_id` 关联 `common_features`，因此核心难点不是单纯训练模型，而是如何可靠地解析、join、评估和解释大规模稀疏数据。

## 2. 项目目标

本项目目标分为三层：

1. 建立可复现的数据处理链路，能够解析 Ali-CCP 的稀疏特征格式，并完成 `ad + common` 特征合并。
2. 建立从简单到复杂的模型体系：`LR Hashing -> DeepFM -> ESMM + DCN-V2`。
3. 判断最终模型选择：如果只做 CTR 排序，选择最强点击模型；如果做完整点击转化链路，选择能同时输出 CTR、CVR、CTCVR 的最终模型。

核心任务定义：

- CTR: 预测曝光后是否点击。
- CVR: 在点击样本中预测是否转化。
- CTCVR: 预测曝光后最终是否转化，即点击并转化。

本项目不以 Accuracy 作为核心指标，因为点击率只有约 4% 到 5%，全预测不点击也能得到约 95% 的 Accuracy。这种指标在严重类别不均衡场景下会误导判断。

## 3. 项目规划与可行性

项目采用分阶段推进，而不是直接训练复杂模型。原因是 Ali-CCP 数据规模大、格式特殊、转化极稀疏，如果直接上深度模型，很难判断问题来自数据、特征、模型还是评估。

整体规划如下：

1. 数据验收与解析器自检。
2. 建立 LR Hashing 快速基线。
3. 实现 common feature join，形成 `ad + common` 完整特征。
4. 建立 DeepFM 神经网络基线。
5. 建立 ESMM + DCN-V2 最终链路模型。
6. 扩大样本到 1M，补充 PR-AUC、Lift、校准分桶等业务指标。
7. 增加 seed、early stopping、best checkpoint 和误差分析。
8. 汇总最终报告。

可行性判断：

- 原始数据无法一次性读入内存，因此所有核心处理采用流式读取或分桶/选择性 join。
- 10k joined 文件约 130MB，100k joined 约 1.27GB，500k joined 约 7.15GB，1M joined 约 13.93GB，磁盘和训练成本可接受。
- 本机 GPU 为 NVIDIA GeForce RTX 4070 Laptop GPU，可以支撑 DeepFM 和 ESMM 的 1M 样本训练。

## 4. 数据结构与处理逻辑

### 4.1 原始数据结构

`sample_skeleton_*` 每行结构为：

```text
sample_id, click, conversion, common_feature_id, ad_feature_count, ad_sparse_features
```

`common_features_*` 每行结构为：

```text
common_feature_id, common_feature_count, common_sparse_features
```

稀疏特征分隔符：

- 特征之间使用 `\x01`
- field 与 feature 使用 `\x02`
- feature 与 value 使用 `\x03`

### 4.2 处理逻辑

第一阶段先只使用 `sample_skeleton` 的广告侧特征训练模型，目的是验证标签、解析和训练链路。随后再通过 `common_feature_id` 合并 `common_features`，形成完整样本。

由于 `common_features` 文件很大，项目实现了两种 join 策略：

- `selective_join`: 用于小到中等规模样本，先收集所需 common id，再扫描 common 文件匹配。
- `bucket_join`: 用于可扩展全量方案，按 `hash(common_feature_id) % bucket_count` 分桶后逐桶 join。

实际实验中，前 1M skeleton 需要的 common id 在 common 文件前约 73 万行内即可全部命中，因此 `selective_join` 在本项目规模下效率很高。

## 5. 阶段落地过程与发现

### 5.1 Stage 01: 数据验收与 ad-only 基线

完成内容：

- 实现 Ali-CCP 稀疏特征解析器。
- 实现数据验收脚本。
- 实现 LR Hashing、DeepFM、ESMM + DCN-V2 的最小训练入口。

自检结果：

- 前 1,000 条 train skeleton 解析错误为 0。
- click rate 为 0.032。
- common features 前 1,000 行平均特征数约 528。
- LR Hashing 前 1,000 行 AUC 为 0.5046。

阶段发现：

- 解析逻辑可靠。
- common 侧特征非常多，后续必须控制特征尺度。
- ad-only 只能验证流程，不能代表最终效果。

### 5.2 Stage 02: common feature join 与归一化

完成内容：

- 实现 `bucket_join.py` 和 `selective_join.py`。
- 实现 joined 特征流。
- 让 LR Hashing 支持 `ad` 和 `ad + common` 两种输入。

关键结果：

- 前 1,000 条 skeleton 需要 13 个 common id。
- 扫描 common 文件前 707,394 行后全部命中。
- join 覆盖率为 100%。

重要发现：

直接把 common 特征加入 LR 后，指标变差：

- ad-only no normalization: AUC 0.5046, LogLoss 0.1783
- ad + common no normalization: AUC 0.4361, LogLoss 1.6635

原因是 common 特征每行有几百个，直接相加导致在线 LR 特征尺度过大。加入 `value_clip=10` 和样本级 L2 归一化后，结果改善：

- ad-only normalized: AUC 0.5360, LogLoss 0.1774
- ad + common normalized: AUC 0.5834, LogLoss 0.1667

阶段结论：

common 特征有效，但必须做尺度控制。这个发现也影响后续 DeepFM 和 ESMM 的输入设计。

### 5.3 Stage 03: DeepFM 基线

完成内容：

- 生成 10k joined 样本验证可行性。
- 扩展到 100k joined 样本。
- 实现 train/valid 稳定切分。
- 给 PyTorch 训练入口加入 valid 指标。
- 修正 DeepFM embedding 初始化。

100k 数据：

- joined rows: 100,000
- train rows: 79,878
- valid rows: 20,122
- valid clicks: 845
- valid conversions: 8

关键结果：

- LR Hashing 100k: CTR AUC 0.6107, LogLoss 0.1722
- DeepFM 100k: CTR AUC 0.6241, LogLoss 0.2380

重要发现：

DeepFM 初期表现不稳定，原因是 PyTorch 默认 Embedding 初始化过大，在 FM 交叉项中会导致初始 logit 过大。项目将 sparse embedding 初始化为小方差正态分布，将 linear embedding 初始化为 0 后，DeepFM 成为可用的 CTR 神经网络基线。

阶段结论：

DeepFM 在排序能力上超过 LR，说明神经网络确实学到了非线性和特征交叉信息。但 LogLoss 不如 LR，说明概率校准仍有改进空间。

### 5.4 Stage 04: ESMM + DCN-V2 500k 训练

完成内容：

- 给 ESMM 增加 CTR、CTCVR、CVR-on-clicked 指标。
- 扩展到 500k joined 样本。
- 加入 CTCVR loss 权重和正例权重实验。

500k 数据：

- joined rows: 500,000
- train rows: 399,871
- valid rows: 100,129
- valid clicks: 4,513
- valid conversions: 29
- valid CTCVR positives: 27

关键结果：

| 模型 | CTR AUC | CTR LogLoss | CTCVR AUC | CVR AUC(clicked) |
| --- | ---: | ---: | ---: | ---: |
| LR Hashing | 0.6075 | 0.1813 | n/a | n/a |
| DeepFM | 0.6243 | 0.1870 | n/a | n/a |
| ESMM w1 | 0.6190 | 0.1890 | 0.5251 | 0.4947 |
| ESMM w5 | 0.6230 | 0.1839 | 0.5345 | 0.4937 |
| ESMM posw100 | 0.5996 | 0.2030 | 0.5547 | 0.4712 |

阶段发现：

- ESMM w5 的 CTR AUC 接近 DeepFM。
- posw100 提高了 CTCVR AUC，但严重损伤 CTR 和校准。
- 转化正例仍偏少，CVR AUC 不稳定。

阶段结论：

如果只看 CTR，DeepFM 更强；如果看完整链路，ESMM w5 是更合理的候选。

### 5.5 Stage 05: 1M 数据与完整指标体系

完成内容：

- 扩展到 1M joined 样本。
- 增加 PR-AUC、Lift@K、校准分桶。
- 比较 LR、DeepFM、ESMM w5、ESMM posw25。

1M 数据：

- joined rows: 1,000,000
- train rows: 799,567
- valid rows: 200,433
- valid clicks: 9,304
- valid conversions: 64
- valid CTCVR positives: 62

CTR 结果：

| 模型 | CTR AUC | CTR PR-AUC | CTR LogLoss | CTR Lift@1% |
| --- | ---: | ---: | ---: | ---: |
| LR Hashing | 0.6062 | 0.0724 | 0.1891 | 2.81 |
| DeepFM | 0.6200 | 0.0747 | 0.1873 | 2.87 |
| ESMM w5 | 0.6197 | 0.0752 | 0.1921 | 2.93 |
| ESMM posw25 | 0.6183 | 0.0722 | 0.1977 | 2.34 |

CTCVR 结果：

| 模型 | CTCVR AUC | CTCVR PR-AUC | CTCVR LogLoss | CTCVR Lift@1% |
| --- | ---: | ---: | ---: | ---: |
| ESMM w5 | 0.5804 | 0.000800 | 0.003643 | 6.45 |
| ESMM posw25 | 0.5753 | 0.000590 | 0.008045 | 4.84 |

阶段发现：

- DeepFM 是最强纯 CTR 模型。
- ESMM w5 几乎追平 DeepFM 的 CTR 表现，并且拥有最好的 CTCVR Lift。
- posw25 不如 w5 稳定，转化概率被放大后校准变差。

阶段结论：

DeepFM 适合点击排序；ESMM w5 更适合作为最终链路模型。

### 5.6 Stage 06-A/B: 早停与稳定性验证

完成内容：

- 给 `train_torch.py` 增加 `seed`。
- 增加逐 epoch 验证。
- 增加 early stopping。
- 保存 best checkpoint。
- 进行 seed=2025 稳定性实验。

DeepFM seed 2025：

- best CTR AUC: 0.6299
- best CTR PR-AUC: 0.0773
- best CTR LogLoss: 0.1835
- best CTR Lift@1%: 2.69

ESMM w5 seed 2025：

- best CTR AUC: 0.6157
- best CTR PR-AUC: 0.0745
- best CTR LogLoss: 0.1945
- best CTR Lift@1%: 3.18
- best CTCVR AUC: 0.5479
- best CTCVR PR-AUC: 0.000645
- best CTCVR Lift@1%: 6.45

阶段发现：

- DeepFM 的 CTR 结论稳定，seed2025 不弱于 seed2024。
- ESMM w5 的 CTCVR Lift@1% 稳定，但 CTCVR AUC 和 CVR AUC 对 seed 更敏感。
- 原因是 valid 中 CTCVR 正例只有 62 个，转化指标天然波动。

阶段结论：

项目报告中可以稳健地宣称 DeepFM 是强 CTR baseline；对于 ESMM，应强调其 Top-K 转化链路排序价值，而不是过度解读 CVR AUC。

### 5.7 Stage 06-C: 预测导出与误差分析

完成内容：

- 实现 `export_predictions.py`。
- 实现 `analyze_predictions.py`。
- 导出 DeepFM 和 ESMM w5 的 valid 预测。
- 生成错误样本 CSV。

输出文件：

- `reports/predictions_deepfm_seed2025_valid.csv`
- `reports/predictions_esmm_w5_seed2025_valid.csv`
- `reports/analysis_deepfm_seed2025`
- `reports/analysis_esmm_w5_seed2025`

DeepFM 分析：

- CTR AUC: 0.6299
- CTR PR-AUC: 0.0773
- CTR LogLoss: 0.1835
- CTR Lift@1%: 2.69
- bin [0.0, 0.1): avg prediction 0.0486, true CTR 0.0452
- bin [0.1, 0.2): avg prediction 0.1089, true CTR 0.1188

ESMM w5 分析：

- CTR AUC: 0.6157
- CTR Lift@1%: 3.18
- CTCVR Lift@1%: 6.45
- CVR-on-clicked Lift@1%: 4.84

阶段发现：

- DeepFM 概率校准更好，适合作为 CTR 模型。
- ESMM w5 的 Top 1% 样本更能富集点击和转化链路目标。
- 高分未点击样本可能来自曝光噪声或用户注意力缺失。
- 低分已点击样本提示模型仍漏掉部分用户/广告组合。
- 低 pCTCVR 但转化的样本最值得后续分析，因为它们代表转化漏召回。

## 6. 最终结果与模型选择

最终选择取决于业务目标。

如果目标是纯 CTR 排序：

- 推荐模型：DeepFM seed2025
- 理由：CTR AUC、CTR PR-AUC 和 CTR LogLoss 综合最好。

如果目标是完整点击转化链路：

- 推荐模型：ESMM + DCN-V2 w5
- 理由：CTR 指标接近 DeepFM，同时可以输出 pCTR、pCVR、pCTCVR，并且 CTCVR Lift@1% 达到 6.45。

最终建议：

本项目的最终模型选择为 `ESMM + DCN-V2 w5`，同时保留 `DeepFM` 作为 CTR baseline。报告中应明确说明：DeepFM 是 CTR 最强模型，ESMM w5 是更符合点击-转化链路目标的最终模型。

## 7. 我们学到了什么

### 7.1 数据处理比模型更关键

Ali-CCP 的难点首先是格式解析和特征 join。没有可靠的数据链路，复杂模型没有意义。项目中通过流式解析、选择性 join、分桶 join 和阶段自检，逐步确认了数据可靠性。

### 7.2 指标必须匹配业务问题

Accuracy 在该项目中无意义，因为点击率只有约 4.6%。真正有价值的是：

- AUC: 排序能力
- PR-AUC: 稀疏正例识别能力
- LogLoss: 概率校准
- Lift@K: 推荐业务中 top 排名样本的价值
- 校准分桶: 预测概率是否可信

### 7.3 common 特征有效，但必须归一化

common 特征显著提升模型效果，但每条样本特征数量多，直接加入会造成数值不稳定。样本级 L2 归一化和 value clipping 是必要步骤。

### 7.4 DeepFM 和 ESMM 的价值不同

DeepFM 更适合纯 CTR 预测，训练和校准相对稳定。ESMM 更适合完整链路建模，尤其是需要同时输出 CTR、CVR、CTCVR 的场景。

### 7.5 转化预测的核心瓶颈是正例稀疏

即使扩展到 1M 样本，valid 中 CTCVR 正例也只有 62 个。这导致 CVR/CTCVR 指标对随机种子敏感。因此在结论中必须谨慎，更多强调 Lift 和方向性，而不是过度宣称 CVR AUC 的绝对稳定。

## 8. 当前局限

1. 未进行全量训练，只扩展到 1M joined 样本。
2. CTCVR 正例仍然较少，转化指标不够稳定。
3. 没有进行多 seed 的完整 3 轮统计，只补充了 seed2025。
4. 没有做更细粒度的特征分组解释，例如不同 field_id 的贡献。
5. 没有实现线上推理服务，仅完成离线训练和分析。

## 9. 后续优化方向

如果继续推进，可以做以下工作：

1. 补充 ESMM seed2026，进一步验证 CTCVR 指标稳定性。
2. 扩展到 2M 或更多 joined 样本，提高转化正例数量。
3. 加入更细的 field-level ablation，判断哪些特征组最有效。
4. 进行概率校准，例如 Platt scaling 或 isotonic calibration。
5. 增加导出预测样本的人工审查和错误案例归因。
6. 生成可视化图表：Lift 曲线、PR 曲线、校准曲线、模型对比柱状图。
7. 封装推理脚本，输出 `sample_id, pCTR, pCVR, pCTCVR`。

## 10. 总结

本项目从原始 Ali-CCP 大规模稀疏数据出发，完整构建了推荐广告点击与转化预测流程。我们没有直接跳到复杂模型，而是按数据验收、基线验证、特征 join、神经网络建模、多任务链路建模、指标扩展、稳定性验证和误差分析逐步推进。

最终结果表明：

- `LR Hashing` 是必要且稳定的工程基线。
- `DeepFM` 是当前最强的纯 CTR 模型。
- `ESMM + DCN-V2 w5` 是最适合作为最终项目模型的链路模型，因为它在 CTR 表现接近 DeepFM 的同时，能给出 CVR 和 CTCVR，并在 CTCVR Top-K Lift 上表现最好。

项目最大的收获是：在真实推荐数据中，模型好坏不能只看单一准确率，而要结合任务链路、样本稀疏性、排序指标、概率校准和业务 Top-K 效果综合判断。

