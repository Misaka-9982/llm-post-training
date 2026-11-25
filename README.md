# LLM Post-training

基于 **Qwen3-0.6B-Base + GSM8K** 搭建小模型后训练实验框架，涵盖 **Synthetic Data Generation、Rejection Sampling、Supervised Fine-Tuning（SFT）、Group Relative Policy Optimization（GRPO）与 On-Policy Distillation（OPD）**，探索合成数据构建、强化学习奖励设计及 Teacher-Student 蒸馏策略对模型数学推理能力的影响。

## 项目简介

本项目以数学推理任务为切入点，基于 GSM8K 构建并筛选模型生成的合成推理数据，搭建从数据构建、模型训练到统一评测的完整后训练流程，并通过多组对照实验分析不同后训练方法的效果与局限性。

- **Student Model**：Qwen3-0.6B-Base
- **Teacher Model**：Qwen3-8B
- **Benchmark**：GSM8K
- **Framework**：PyTorch / Transformers / Datasets / TRL

## 主要工作

### 1. Baseline & Evaluation

- 实现 GSM8K 数据加载、Prompt 构造、批量推理与数学答案解析。
- 建立统一的准确率、答案解析成功率与格式合规率评测流程。
- 基于 GSM8K 完整测试集（1,319 samples）对不同训练阶段的 checkpoint 进行评估。

### 2. Supervised Fine-Tuning

- 搭建全参数 SFT 训练流程，对比不同数据来源、训练参数与 checkpoint 的表现。
- 针对直接使用原始 GSM8K 数据进行 SFT 导致的性能退化，引入模型自生成推理数据，并通过 **Rejection Sampling** 筛选正确的推理样本。
- 使用 Qwen3-8B 生成并筛选高质量推理数据，进一步开展 **Off-Policy SFT**，探索大模型向小模型的推理能力迁移。

### 3. Reinforcement Learning with GRPO

- 基于 TRL 搭建 GRPO 训练流程，设计可验证的 correctness reward 与 format reward。
- 对比不同奖励组合、学习率、Batch Size、生成长度及训练步数对模型性能的影响。
- 分析奖励提升与 benchmark accuracy 不一致、格式奖励偏置及组内有效 advantage 不足等问题。
- 探索基于 Rollout 正确率的训练样本难度筛选策略。

### 4. On-Policy Distillation（OPD，进行中）

- 采用 Qwen3-8B 作为 Teacher，Qwen3-0.6B 作为 Student，完成 Teacher 与 Student 的 Tokenizer 对齐验证。
- 通过不同长度的 Student Rollout Prefix 测试 Teacher 在学生生成轨迹上的续写能力，分析错误推理轨迹对蒸馏信号可靠性的影响。
- 探索基于 Student On-Policy Rollout、Teacher Token-Level Logits 与 Reverse KL Divergence 的蒸馏训练方法。
- 研究不同 Token 监督范围与错误轨迹处理策略对蒸馏效果的影响。

## Preliminary Results

以下为统一评测流程下的 GSM8K 全量测试集代表性结果：

| Training Method | Answer Accuracy |
|---|---:|
| Qwen3-0.6B-Base | 62.02% |
| SFT with Self-Generated Data | 67.40% |
| Off-Policy SFT with Qwen3-8B | **69.75%** |

通过合成推理数据筛选与 Teacher-Student 知识迁移，模型准确率较 Base 提升约 **7.73 percentage points**。

GRPO 已完成多组对照实验，但尚未观察到稳定超过最佳 SFT checkpoint 的收益。OPD 相关实验仍在进行中。

## 实验关注点

- 训练数据质量及分布对 SFT 的影响。
- Reward Design 与 GRPO 训练稳定性。
- Rollout Difficulty 与 Group-Relative Advantage。
- Off-Policy SFT 与 On-Policy Distillation 的差异。
- Teacher 在 Student 生成轨迹上的监督信号可靠性。
- 训练指标与实际推理能力之间的关系。

## Development Environment

- Python 3.12 / uv
- PyTorch
- Hugging Face Transformers / Datasets / TRL
- WSL / Google Colab（A100 / L4）

项目使用 uv 管理 Python 环境与依赖，本地负责开发和轻量调试，GPU 环境负责模型推理、训练与评测。
