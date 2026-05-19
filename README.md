# Ali-CCP CTR/CVR/CTCVR Modeling Project

基于阿里天池 [Ali-CCP 数据集](https://tianchi.aliyun.com/dataset/408) 的点击率、转化率与联合转化率建模项目。项目目标不是单纯追求 `Accuracy`，而是围绕推荐广告排序更关键的指标：`AUC`、`PR-AUC`、`LogLoss`、`Lift@K` 与校准表现，建立一套可复现、可对照、可解释的建模流程。

## 项目目标

- 构建 Ali-CCP 原始稀疏特征解析、样本关联、特征归一化、训练评估的一体化流程。
- 使用 `LR Hashing` 作为快速可解释基线，验证数据解析、标签分布和指标计算是否可靠。
- 使用 `DeepFM` 作为深度 CTR 排序基线，评估稀疏交叉特征的建模收益。
- 使用 `ESMM + DCN-V2` 作为最终推荐模型，联合建模 `CTR`、`CVR`、`CTCVR`，避免只在点击样本上训练 CVR 带来的样本选择偏差。
- 输出完整项目报告、目录说明、阶段自检清单和实验结果，便于课程汇报或 GitHub 展示。

## 数据说明

原始数据默认放置在本地：

```text
D:\Ali-CCP\sample_train\
D:\Ali-CCP\sample_test\
```

由于天池数据、处理中间产物和模型权重体积较大，GitHub 仓库只保存代码、配置、轻量报告和指标 JSON，不上传：

- `sample_train/`
- `sample_test/`
- `processed/`
- `models/`
- `.venv/`
- 大体积预测 CSV 与误差分析 CSV

## 模型方案

| 模型 | 角色 | 主要用途 |
| --- | --- | --- |
| `LR Hashing` | 快速基线 | 检查解析、标签、特征有效性和指标可信度 |
| `DeepFM` | CTR 深度基线 | 建模稀疏特征的一阶、二阶和高阶交互 |
| `ESMM + DCN-V2` | 最终模型 | 联合建模 CTR / CVR / CTCVR，适合转化稀疏场景 |

最终采用 `ESMM + DCN-V2`，因为业务目标更关注转化链路，而不仅是点击排序。`DeepFM` 在 CTR 指标上表现强，但 ESMM 在 CTCVR 与头部转化召回上更符合最终目标。

## 当前关键结果

| 模型 | 数据规模 | CTR AUC | CTR PR-AUC | CTR LogLoss | CTR Lift@1% | CTCVR Lift@1% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LR Hashing | 1M | 0.6062 | 0.0724 | 0.1891 | 2.81 | - |
| DeepFM | 1M / seed2024 | 0.6200 | 0.0747 | 0.1873 | 2.87 | - |
| ESMM + DCN-V2 w5 | 1M / seed2024 | 0.6197 | 0.0752 | 0.1921 | 2.93 | 6.45 |
| DeepFM Early Stop | 1M / seed2025 | 0.6299 | 0.0773 | 0.1835 | 2.69 | - |
| ESMM + DCN-V2 w5 Early Stop | 1M / seed2025 | 0.6157 | 0.0745 | 0.1945 | - | 6.45 |

结论要点：

- `Accuracy` 不适合作为主指标，因为 Ali-CCP 点击和转化标签高度不平衡。
- `DeepFM` 是当前 CTR 排序最强的对照模型。
- `ESMM + DCN-V2` 是最终推荐模型，因为它直接服务 CTCVR / CVR 链路，并在头部转化 Lift 上有明确优势。
- 当前结果已经能支撑完整项目汇报；若继续提升，应优先做多随机种子稳定性验证、超参搜索、特征分组消融和概率校准。

## 快速开始

### 1. 环境准备

```powershell
cd D:\Ali-CCP
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

如果系统 `python` 不在 `PATH`，可以使用 Codex 自带 Python 运行环境。

### 2. 自检解析器

```powershell
python scripts\self_check.py
```

### 3. 数据检查

```powershell
python -m src.data.inspect --data-root D:\Ali-CCP --split train --max-lines 100000
```

### 4. 训练 LR Hashing 基线

```powershell
python -m src.train.train_lr_hashing --data-root D:\Ali-CCP --max-lines 100000 --hash-size 262144
```

### 5. 构建 joined 数据

```powershell
python -m src.data.selective_join --data-root D:\Ali-CCP --max-skeleton-lines 1000000 --output D:\Ali-CCP\processed\joined_1m\joined_train.csv
```

### 6. 训练 DeepFM / ESMM

```powershell
python -m src.train.train_torch --model deepfm --joined-path D:\Ali-CCP\processed\joined_1m\joined_train.csv --epochs 10 --seed 2025
python -m src.train.train_torch --model esmm_dcnv2 --joined-path D:\Ali-CCP\processed\joined_1m\joined_train.csv --epochs 10 --seed 2025 --ctcvr-loss-weight 5
```

## 项目结构

```text
D:\Ali-CCP
├── configs/                  # 数据与模型配置
├── reports/                  # 指标、阶段总结、最终报告辅助文件
├── scripts/                  # 自检脚本
├── src/                      # 数据处理、模型、训练和评估代码
├── FINAL_ANALYSIS.md         # 项目完整专业报告
├── PROJECT_CHECKLIST.md      # 分阶段自检清单
├── PROJECT_STRUCTURE.md      # 文件夹和子文件说明
├── README.md                 # GitHub 项目首页
├── requirements.txt          # Python 依赖
└── .gitignore                # 排除大数据和模型产物
```

更完整的结构说明见 `PROJECT_STRUCTURE.md`。

## 主要报告

- `FINAL_ANALYSIS.md`：项目介绍、目标、规划、落地过程、结果、分析和总结。
- `PROJECT_CHECKLIST.md`：每个阶段的可自检清单，用于后续纠错和复现。
- `PROJECT_STRUCTURE.md`：目录与文件职责说明。
- `reports/stage*_summary.md`：阶段性实验发现。
- `reports/*metrics*.json`：轻量实验指标记录。

## 复现实验的注意事项

- 推荐先跑 `LR Hashing`，确认解析和指标无误后再跑深度模型。
- 大规模 joined 数据会占用较多磁盘空间，应保存在 `processed/` 下并由 `.gitignore` 排除。
- 模型权重保存在 `models/` 下，不建议上传 GitHub。
- ESMM 的转化正例极稀疏，判断模型优劣时必须结合 `PR-AUC`、`Lift@K`、校准曲线和多 seed 稳定性。

## 后续优化方向

- 补充 `seed=2026` 等多随机种子实验，验证结论稳定性。
- 对 ESMM 的 loss weight、embedding dim、DCN cross layer 做小范围搜索。
- 加入特征分组消融，量化 ad/common/field 级特征贡献。
- 对 CTR 与 CTCVR 概率做校准，比较等频分箱、Platt Scaling、Isotonic Regression。
- 将最终结果整理成答辩 PPT 或 GitHub Releases。
