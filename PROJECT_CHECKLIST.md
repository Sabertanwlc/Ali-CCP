# Ali-CCP ESMM + DCN-V2 项目分段自检清单

## 0. 项目目标与边界
- [ ] 明确最终目标是完整曝光空间上的 CTR、CVR、CTCVR 联合建模，而不是单独购买预测。
- [ ] 最终模型确定为 ESMM + DCN-V2。
- [ ] 对照基线确定为 LR Hashing 和 DeepFM。
- [ ] 训练数据使用 `sample_train`，测试/验证数据使用 `sample_test` 或从 train 中切分出的 valid。
- [ ] 所有实验记录数据版本、样本规模、特征版本、模型版本、随机种子和指标。

## 1. 原始数据验收
- [ ] 确认目录存在：`D:\Ali-CCP\sample_train` 和 `D:\Ali-CCP\sample_test`。
- [ ] 确认四个核心文件存在：`sample_skeleton_train.csv`、`common_features_train.csv`、`sample_skeleton_test.csv`、`common_features_test.csv`。
- [ ] 记录文件大小、行数、首尾样本、异常空行数量。
- [ ] 确认 `sample_skeleton_*` 字段结构为：样本ID、click、conversion、common_feature_id、ad特征数、ad稀疏特征串。
- [ ] 确认 `common_features_*` 字段结构为：common_feature_id、common特征数、common稀疏特征串。
- [ ] 确认稀疏特征分隔符：特征之间为 `\x01`，field 与 feature 为 `\x02`，feature 与 value 为 `\x03`。
- [ ] 检查 train/test 的字段格式一致性。

## 2. 标签与任务自检
- [ ] 统计 train 中 `click=1` 比例。
- [ ] 统计 train 中 `conversion=1` 比例。
- [ ] 检查是否存在 `conversion=1` 但 `click=0` 的样本；如果存在，需要单独解释或清洗。
- [ ] 定义 CTR 标签：`y_ctr = click`。
- [ ] 定义 CTCVR 标签：`y_ctcvr = click * conversion`，或在确认数据语义后使用 `conversion`。
- [ ] 定义 CVR 评估样本：只在 `click=1` 子集上评估 `conversion`。
- [ ] 保证 ESMM 训练在完整曝光空间优化 CTR 和 CTCVR。

## 3. 数据抽样与 EDA
- [ ] 先抽样 10万行验证解析逻辑。
- [ ] 再抽样 100万行做初版 EDA。
- [ ] 统计每行 ad 特征数量分布。
- [ ] 统计每行 common 特征数量分布。
- [ ] 统计 field_id 分布、feature_id 去重数、长尾程度。
- [ ] 检查特征 value 是否主要为 1.0 和 log 类连续值。
- [ ] 检查 common_feature_id 在 skeleton 与 common 文件中的覆盖率。
- [ ] 输出 EDA 报告，包含标签分布、特征分布、异常样本示例。

## 4. 解析器自检
- [ ] 解析单行 skeleton 后能得到基础列和 ad 稀疏特征列表。
- [ ] 解析单行 common 后能得到 common_feature_id 和 common 稀疏特征列表。
- [ ] 对异常特征片段做跳过和计数，不让单条坏数据中断全流程。
- [ ] 对 field_id、feature_id、value 做类型校验。
- [ ] 输出解析失败率；失败率过高时停止训练。
- [ ] 确认同一个样本中 ad 特征和 common 特征不会错误拼接。

## 5. 大文件处理方案
- [ ] 不把 40GB CSV 一次性读入内存。
- [ ] 原始 CSV 保持只读。
- [ ] 中间产物单独放到 `D:\Ali-CCP\processed`。
- [ ] 日志和指标单独放到 `D:\Ali-CCP\reports`。
- [ ] 模型文件单独放到 `D:\Ali-CCP\models`。
- [ ] 使用流式读取或分块读取处理大文件。
- [ ] common join 使用分桶 join 或外部排序 join，避免内存爆炸。
- [ ] 处理完成后保存 Parquet、NPZ、TFRecord 或 PyTorch Dataset 索引之一。

## 6. LR Hashing 基线
- [ ] 只使用 sample_skeleton 的 ad 特征先跑通 CTR。
- [ ] 使用 hashing trick 构造固定维度稀疏向量。
- [ ] 记录 hash 维度、正负样本比例、训练样本数。
- [ ] 训练 Logistic Regression CTR 模型。
- [ ] 在 valid 上计算 AUC 和 LogLoss。
- [ ] 加入 common 特征后再次训练，确认指标是否提升。
- [ ] 若 LR 指标异常，优先检查解析、标签、数据切分，而不是调深度模型。

## 7. DeepFM 基线
- [ ] 使用与最终模型一致的特征字典或 hashing 方案。
- [ ] 明确 sparse field embedding 维度。
- [ ] DeepFM 同时包含 FM 一阶/二阶和 DNN 部分。
- [ ] 先训练 CTR 单任务版本。
- [ ] 再训练 CTCVR 或多任务版本作为中间对照。
- [ ] 记录 AUC、LogLoss、训练耗时、显存/内存占用。
- [ ] DeepFM 指标应显著优于或至少不弱于 LR；否则检查特征输入和训练稳定性。

## 8. ESMM + DCN-V2 最终模型
- [ ] 模型包含共享 embedding 层。
- [ ] CTR tower 输出 `pCTR`。
- [ ] CVR tower 输出 `pCVR`。
- [ ] CTCVR 输出为 `pCTCVR = pCTR * pCVR`。
- [ ] tower 主干使用 DCN-V2 结构学习显式特征交叉。
- [ ] 损失函数包含 CTR loss 和 CTCVR loss。
- [ ] CVR 不直接只用点击样本训练，避免样本选择偏差。
- [ ] 训练日志同时记录 CTR loss、CTCVR loss、总 loss。
- [ ] 推理阶段同时输出 pCTR、pCVR、pCTCVR。

## 9. 训练与验证切分
- [ ] 如果数据有时间字段，优先按时间切分。
- [ ] 如果没有时间字段，使用稳定随机切分并固定随机种子。
- [ ] valid/test 中保持 click 和 conversion 标签比例可比。
- [ ] 禁止把 valid/test 信息用于特征统计、目标编码或字典过滤。
- [ ] 训练、验证、测试的样本数和正例数写入报告。

## 10. 指标体系
- [ ] CTR 指标：AUC、LogLoss。
- [ ] CTCVR 指标：AUC、LogLoss、PR-AUC。
- [ ] CVR 指标：在 `click=1` 样本上计算 AUC、LogLoss。
- [ ] 排序指标：可选 Top-K Recall、NDCG、Lift。
- [ ] 分桶指标：高频/低频 common_feature_id、热门/长尾 item/ad 特征。
- [ ] 校准检查：预测分数分桶后的真实点击率和转化率。
- [ ] 所有模型必须在同一 valid/test 切分上对比。

## 11. 常见错误排查
- [ ] AUC 接近 0.5：优先检查标签列、特征解析、训练/验证是否错位。
- [ ] LogLoss 极大：检查输出概率、label 类型、样本权重、数值溢出。
- [ ] CVR 指标异常高：检查是否发生标签泄漏或只在转化相关样本上评估。
- [ ] 加 common 特征后变差：检查 join 覆盖率和 common_feature_id 是否错配。
- [ ] 深度模型弱于 LR：检查 embedding 输入、hash 冲突、学习率、batch 构造。
- [ ] 训练很慢：先降低样本量、hash 维度、embedding 维度，再优化 DataLoader。
- [ ] 内存爆炸：检查是否把 common_features 全量加载成 Python dict。

## 12. 实验对比表
- [ ] LR Hashing + ad only。
- [ ] LR Hashing + ad + common。
- [ ] DeepFM + ad + common。
- [ ] ESMM + DeepFM backbone，作为结构对照。
- [ ] ESMM + DCN-V2 backbone，作为最终模型。
- [ ] 每个实验记录：数据版本、特征版本、模型参数、训练轮数、指标、耗时、资源占用。

## 13. 项目交付物
- [ ] `README.md`：项目目标、数据说明、运行方式、实验结论。
- [ ] `configs/`：数据路径、特征配置、模型配置。
- [ ] `src/data/`：解析、抽样、join、dataset 构建。
- [ ] `src/models/`：LR、DeepFM、ESMM_DCNV2。
- [ ] `src/train/`：训练入口和评估入口。
- [ ] `reports/`：EDA、指标表、错误分析。
- [ ] `models/`：保存训练好的模型和特征字典。
- [ ] `processed/`：保存可复用中间数据。

## 14. 阶段完成标准
- [ ] 阶段一完成：能解析样本并生成 EDA 报告。
- [ ] 阶段二完成：LR Hashing baseline 指标稳定可复现。
- [ ] 阶段三完成：DeepFM baseline 指标完成并与 LR 对比。
- [ ] 阶段四完成：ESMM + DCN-V2 能稳定训练，输出 CTR/CVR/CTCVR。
- [ ] 阶段五完成：最终模型在核心指标上优于 LR 和 DeepFM，并有误差分析报告。
