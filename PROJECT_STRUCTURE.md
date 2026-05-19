# Ali-CCP 项目目录与文件说明

本文档说明 `D:\Ali-CCP` 项目的目录结构、关键脚本、数据产物、模型文件和报告文件用途，便于复现、汇报和后续维护。

## 1. 根目录总览

```text
D:\Ali-CCP
├── sample_train/              原始训练数据
├── sample_test/               原始测试数据
├── configs/                   配置文件
├── src/                       项目源码
├── scripts/                   自检脚本
├── processed/                 中间处理数据
├── models/                    训练得到的模型 checkpoint
├── reports/                   阶段报告、指标 JSON、预测结果和误差分析
├── .venv/                     早期创建的虚拟环境，目前主要使用系统 Python
├── README.md                  项目快速说明
├── PROJECT_CHECKLIST.md       项目分阶段自检清单
├── FINAL_ANALYSIS.md          最终项目报告
├── PROJECT_STRUCTURE.md       当前目录说明文档
└── requirements.txt           依赖列表
```

## 2. 原始数据目录

### `sample_train/`

训练集原始文件目录。

```text
sample_train/
├── sample_skeleton_train.csv
└── common_features_train.csv
```

- `sample_skeleton_train.csv`: 曝光样本骨架文件，包含 `sample_id`、`click`、`conversion`、`common_feature_id` 和广告侧稀疏特征。
- `common_features_train.csv`: 用户/上下文公共特征文件，通过 `common_feature_id` 与 skeleton 文件关联。

### `sample_test/`

测试集原始文件目录。

```text
sample_test/
├── sample_skeleton_test.csv
└── common_features_test.csv
```

- 用途与 train 文件一致。
- 当前主要实验集中在 train 文件的切分验证上，test 文件暂未作为最终评估输入。

## 3. 配置目录 `configs/`

```text
configs/
├── data_config.json
└── model_config.json
```

- `data_config.json`: 记录数据根目录、train/test 文件相对路径、输出目录名、特征分隔符等。
- `model_config.json`: 记录 LR、DeepFM、ESMM+DCN-V2 的默认模型参数。

## 4. 源码目录 `src/`

源码按功能分为 `data`、`baselines`、`models`、`train` 四部分。

### 4.1 `src/data/`: 数据解析、join、切分

```text
src/data/
├── ali_ccp_format.py
├── feature_stream.py
├── inspect.py
├── selective_join.py
├── bucket_join.py
├── split_joined.py
├── torch_dataset.py
└── __init__.py
```

#### `ali_ccp_format.py`

Ali-CCP 原始格式解析核心模块。

主要功能：

- 定义稀疏特征分隔符。
- 解析 skeleton 行。
- 解析 common feature 行。
- 解析 joined 行。
- 定义 `SkeletonRecord`、`CommonFeatureRecord`、`SparseFeature` 等数据结构。

这是所有数据处理脚本的基础。

#### `feature_stream.py`

将解析后的稀疏特征转为 hash 特征流。

主要功能：

- 使用稳定 hash 将高维稀疏特征映射到固定维度。
- 支持 `value_clip`。
- 支持样本级 L2 归一化。
- 生成 LR 和 PyTorch 模型共用的 hashed example。

关键发现：`ad + common` 特征必须做 L2 归一化，否则在线 LR 和深度模型都容易不稳定。

#### `inspect.py`

数据验收脚本。

主要功能：

- 扫描指定行数的 skeleton/common 文件。
- 统计解析错误率。
- 统计点击率、转化率、CTCVR 比例。
- 统计平均特征数和 top field。

用于 Stage 01 数据验收。

#### `selective_join.py`

选择性 join 脚本。

主要功能：

- 先读取前 N 条 skeleton。
- 收集所需 `common_feature_id`。
- 顺序扫描 common 文件，找到所需 id 后合并。

适合 1k、10k、100k、500k、1M 这类实验规模。当前项目主要使用它生成 joined 数据。

#### `bucket_join.py`

可扩展分桶 join 脚本。

主要功能：

- 按 `hash(common_feature_id) % bucket_count` 将 skeleton 和 common 分桶。
- 每次只加载一个 common bucket 到内存。
- 适合更大规模或全量 join。

当前已实现并通过小样本自检，但实际大样本实验优先使用了更快的 `selective_join.py`。

#### `split_joined.py`

joined 数据 train/valid 切分脚本。

主要功能：

- 按 `sample_id` hash 稳定切分。
- 输出 train.csv 和 valid.csv。
- 统计 train/valid 点击率、转化率和正例数。

保证不同模型在同一份切分上公平对比。

#### `torch_dataset.py`

PyTorch IterableDataset。

主要功能：

- 流式读取 joined 或 ad-only 样本。
- 输出 padded indices、values、mask。
- 支持 DeepFM 和 ESMM+DCN-V2 共用输入。

### 4.2 `src/baselines/`: 传统基线模型

```text
src/baselines/
├── lr_hashing.py
└── __init__.py
```

#### `lr_hashing.py`

在线 Logistic Regression + Hashing Trick 基线。

主要功能：

- 固定 hash 维度稀疏输入。
- 在线 SGD 更新。
- 作为最小工程基线，验证特征、标签和指标是否合理。

LR Hashing 的价值不是追求最终最优，而是快速发现数据处理问题。

### 4.3 `src/models/`: 深度模型

```text
src/models/
├── deepfm.py
├── esmm_dcnv2.py
└── __init__.py
```

#### `deepfm.py`

DeepFM CTR 模型。

主要功能：

- 一阶线性项。
- FM 二阶交叉项。
- DNN 高阶非线性项。

项目中它是最强 CTR baseline。最终 1M seed2025 的 CTR AUC 达到 0.6299。

#### `esmm_dcnv2.py`

ESMM + DCN-V2 多任务链路模型。

主要功能：

- 共享 embedding。
- CTR tower 输出 `pCTR`。
- CVR tower 输出 `pCVR`。
- 通过 `pCTCVR = pCTR * pCVR` 建模曝光到转化链路。
- tower 内使用 DCN-V2 风格交叉网络和 DNN。

项目中它是最终推荐的链路模型候选。

### 4.4 `src/train/`: 训练、评估、导出、分析

```text
src/train/
├── metrics.py
├── train_lr_hashing.py
├── train_lr_hashing_eval.py
├── train_torch.py
├── export_predictions.py
├── analyze_predictions.py
└── __init__.py
```

#### `metrics.py`

统一指标模块。

支持：

- ROC-AUC
- PR-AUC
- LogLoss
- Lift@1% / 5% / 10%
- rate@1% / 5% / 10%
- 校准分桶
- 二分类完整报告生成

这是后期模型选择和最终报告的重要基础。

#### `train_lr_hashing.py`

LR Hashing 快速训练脚本。

用途：

- 快速在 ad-only 或 joined 特征上训练 CTR baseline。
- 主要用于早期流程验证。

#### `train_lr_hashing_eval.py`

LR Hashing train/valid 正式评估脚本。

用途：

- 在 train.csv 上训练。
- 在 valid.csv 上评估。
- 输出完整 CTR 指标，包括 AUC、PR-AUC、LogLoss、Lift、校准分桶。

#### `train_torch.py`

PyTorch 主训练入口。

支持模型：

- DeepFM
- ESMM + DCN-V2

已实现功能：

- seed 控制
- GPU/CPU 训练
- joined 特征输入
- 每 epoch valid 评估
- early stopping
- best checkpoint 保存
- CTR/CTCVR/CVR-on-clicked 指标
- CTCVR loss 权重
- CTCVR positive weight

这是项目最核心的训练脚本。

#### `export_predictions.py`

预测导出脚本。

用途：

- 从 checkpoint 重建模型。
- 对 valid.csv 导出预测。
- DeepFM 输出 `pCTR`。
- ESMM 输出 `pCTR`、`pCVR`、`pCTCVR`。

输出文件用于误差分析和最终报告。

#### `analyze_predictions.py`

预测结果分析脚本。

用途：

- 读取预测 CSV。
- 计算 AUC、PR-AUC、Lift、校准分桶。
- 输出错误样本：
  - 高 pCTR 未点击
  - 低 pCTR 已点击
  - 高 pCTCVR 未转化
  - 低 pCTCVR 已转化
  - 点击转化但低 pCVR

## 5. 脚本目录 `scripts/`

```text
scripts/
└── self_check.py
```

### `self_check.py`

基础解析器自检脚本。

用途：

- 使用构造样本验证 skeleton/common 解析逻辑。
- 项目早期用于确认分隔符解析无误。

## 6. 中间数据目录 `processed/`

```text
processed/
├── joined_stage02_smoke/
├── joined_stage02_selective/
├── joined_stage03_10k/
├── joined_stage03_100k/
├── joined_stage04_500k/
└── joined_stage05_1m/
```

### `joined_stage02_smoke/`

Stage 02 分桶 join 小样本冒烟测试输出。

用途：

- 验证分桶 join 脚本流程。
- 发现前 1,000 skeleton 对应的 common id 不在 common 前 10,000 行中，因此覆盖率为 0。
- 这个结果帮助我们改用 selective join 做小样本验证。

### `joined_stage02_selective/`

Stage 02 选择性 join 1,000 样本输出。

用途：

- 验证 `common_feature_id` join 逻辑正确。
- join 覆盖率为 100%。

### `joined_stage03_10k/`

Stage 03 的 10k joined 数据。

用途：

- 验证 joined 文件体量、切分、LR 和 DeepFM 小规模训练。
- 不适合评估转化，因为 valid 转化正例为 0。

### `joined_stage03_100k/`

Stage 03 的 100k joined 数据。

用途：

- 建立 DeepFM 正式 CTR baseline。
- valid 有 845 个点击正例，但转化正例仍很少。

### `joined_stage04_500k/`

Stage 04 的 500k joined 数据。

用途：

- 训练 ESMM + DCN-V2。
- 对比 DeepFM、LR、ESMM w1/w5/posw100。
- valid 有 27 个 CTCVR 正例，转化评估开始有参考价值但仍不稳定。

### `joined_stage05_1m/`

Stage 05 的 1M joined 数据，是当前最重要的实验数据。

包含：

- `joined_train.csv`
- `train.csv`
- `valid.csv`

用途：

- 最终模型对比。
- 多 seed 稳定性验证。
- 预测导出和误差分析。

关键规模：

- joined rows: 1,000,000
- train rows: 799,567
- valid rows: 200,433
- valid clicks: 9,304
- valid conversions: 64
- valid CTCVR positives: 62

## 7. 模型目录 `models/`

模型目录保存各阶段 PyTorch checkpoint。

重要模型：

```text
models/
├── deepfm_stage05_1m.pt
├── esmm_dcnv2_stage05_1m_w5.pt
├── esmm_dcnv2_stage05_1m_posw25.pt
├── deepfm_stage06b_1m_seed2025.pt
└── esmm_w5_stage06b_1m_seed2025.pt
```

### 推荐保留模型

#### `deepfm_stage06b_1m_seed2025.pt`

当前最强 CTR baseline。

关键结果：

- CTR AUC: 0.6299
- CTR PR-AUC: 0.0773
- CTR LogLoss: 0.1835

#### `esmm_w5_stage06b_1m_seed2025.pt`

当前推荐的最终链路模型。

关键结果：

- CTR AUC: 0.6157
- CTR Lift@1%: 3.18
- CTCVR Lift@1%: 6.45
- 支持同时输出 `pCTR`、`pCVR`、`pCTCVR`

### 其他 checkpoint

其他模型文件是阶段实验产物，用于对比和追溯，包括：

- 100k DeepFM 实验
- 500k DeepFM/ESMM 实验
- 1M posw25 实验
- early stopping smoke test

## 8. 报告目录 `reports/`

`reports/` 是项目分析和汇报最重要的目录，包含阶段报告、指标 JSON、预测文件和误差分析结果。

### 8.1 阶段报告

```text
stage_01_self_check.md
stage_02_self_check.md
stage_03_self_check.md
stage_04_self_check.md
stage_05_self_check.md
stage_06a_self_check.md
stage_06b_self_check.md
stage_06c_self_check.md
```

用途：

- 记录每个阶段做了什么。
- 记录自检结果。
- 记录发现的问题和修正。
- 作为最终报告 `FINAL_ANALYSIS.md` 的素材来源。

### 8.2 指标 JSON

例如：

```text
stage_05_lr_1m.json
stage_05_deepfm_1m.json
stage_05_esmm_1m_w5.json
stage_05_esmm_1m_posw25.json
stage_06b_deepfm_1m_seed2025.json
stage_06b_esmm_w5_1m_seed2025.json
```

用途：

- 保存每次训练/评估的完整指标。
- 包含 AUC、PR-AUC、LogLoss、Lift、校准分桶、history、best metrics 等。

### 8.3 预测文件

```text
predictions_deepfm_seed2025_valid.csv
predictions_esmm_w5_seed2025_valid.csv
```

用途：

- 保存 valid 样本逐行预测分数。
- 用于误差分析、分桶分析和人工检查。

字段包括：

- `sample_id`
- `click`
- `conversion`
- `ctcvr`
- `common_feature_id`
- `feature_count`
- `pctr`
- `pcvr`
- `pctcvr`

### 8.4 误差分析目录

```text
analysis_deepfm_seed2025/
analysis_esmm_w5_seed2025/
```

每个目录包含：

- `summary.json`
- `high_pctr_not_clicked.csv`
- `low_pctr_clicked.csv`

ESMM 目录额外包含：

- `high_pctcvr_not_converted.csv`
- `low_pctcvr_converted.csv`
- `low_pcvr_clicked_converted.csv`

用途：

- 分析模型高分误判和低分漏判样本。
- 支撑最终报告中的误差分析章节。

## 9. 根目录关键文档

### `README.md`

项目快速说明文档。

适合快速了解：

- 项目目标
- 数据格式
- 快速运行命令
- 阶段进展

### `PROJECT_CHECKLIST.md`

项目分段自检清单。

用途：

- 作为项目推进过程中的 checklist。
- 每个阶段都能对照它检查是否完成。

### `FINAL_ANALYSIS.md`

最终项目报告。

内容包括：

- 项目介绍
- 项目目标
- 项目规划与可行性
- 每阶段落地过程和发现
- 最终结果和模型选择
- 学到的经验
- 当前局限
- 后续优化方向

这是汇报和提交项目时最重要的文件。

### `PROJECT_STRUCTURE.md`

当前文件。

用途：

- 解释项目目录结构。
- 说明每个文件夹和关键文件作用。
- 方便别人接手项目。

### `requirements.txt`

依赖列表。

当前内容：

```text
numpy
scikit-learn
torch
tqdm
```

实际运行主要使用系统 Python：

```text
C:\Users\WLC\AppData\Local\Programs\Python\Python313\python.exe
```

## 10. 推荐阅读顺序

如果第一次接触本项目，建议按下面顺序阅读：

1. `README.md`
2. `FINAL_ANALYSIS.md`
3. `PROJECT_STRUCTURE.md`
4. `PROJECT_CHECKLIST.md`
5. `reports/stage_05_self_check.md`
6. `reports/stage_06b_self_check.md`
7. `reports/stage_06c_self_check.md`

如果要复现实验，建议阅读：

1. `src/data/ali_ccp_format.py`
2. `src/data/selective_join.py`
3. `src/data/split_joined.py`
4. `src/train/train_lr_hashing_eval.py`
5. `src/train/train_torch.py`
6. `src/train/export_predictions.py`
7. `src/train/analyze_predictions.py`

## 11. 项目当前推荐结论

当前项目结论是：

- `DeepFM` 是最强 CTR baseline。
- `ESMM + DCN-V2 w5` 是最终推荐的 CTR/CVR/CTCVR 链路模型。
- 不建议用 Accuracy 判断模型，因为点击率很低。
- 评估应主要看 AUC、PR-AUC、LogLoss、Lift@K 和校准分桶。

最重要的最终模型文件：

```text
D:\Ali-CCP\models\esmm_w5_stage06b_1m_seed2025.pt
```

最重要的最终报告文件：

```text
D:\Ali-CCP\FINAL_ANALYSIS.md
```

