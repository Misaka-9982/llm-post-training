# LLM Post-training

基于 **Qwen3-0.6B-Base + GSM8K** 搭建并实践小模型后训练流程，覆盖 **baseline 评测、SFT、GRPO、reward 设计与统一评测**。

项目主要目标是通过可复现的小规模实验理解 LLM post-training 中的数据、训练目标和奖励设计，并观察 SFT 与 RL 阶段对数学推理能力和输出格式的影响。

## 当前进度

### 1. Baseline 与评测

- 完成 GSM8K 数据加载、prompt 构造与批量推理。
- 实现答案解析、准确率与格式合规率评测。
- 使用固定评测流程对 Base、SFT 与 GRPO checkpoint 进行对比，减少不同实验之间的评测偏差。

### 2. Supervised Fine-Tuning

- 基于 TRL 搭建全参数 SFT 训练流程。
- 对比原始 GSM8K 训练数据与模型生成的合成数据。
- 对生成结果进行正确性筛选（rejection sampling），构造更贴近模型分布的高质量 SFT 数据。
- 在统一评测设置下，合成数据 SFT 相比直接使用原始训练集取得了明显更稳定的提升。

### 3. GRPO

- 基于 SFT checkpoint 搭建 GRPO 训练流程。
- 实现 correctness / format reward，并尝试不同 reward 组合与权重。
- 对 group size、batch size、learning rate、训练步数等因素进行了小规模对照实验。
- 目前 GRPO pipeline 已稳定跑通，但在 GSM8K 上尚未观察到稳定超过最佳 SFT checkpoint 的收益，仍在继续分析 reward、采样分布与训练动态。

## 实验关注点

目前主要关注以下问题：

- 原始数据与模型生成数据对 SFT 的影响。
- SFT checkpoint 对后续 GRPO 的影响。
- correctness reward 与 format reward 之间的权衡。
- group 内全对 / 全错样本导致 advantage 信息不足的问题。
- reward 提升、格式合规率与最终 benchmark accuracy 之间并不总是一致。
- batch size、学习率和训练步数对小模型 GRPO 稳定性的影响。

README 暂只保留项目主线，详细实验数据与分析会在项目继续整理后补充。

## 项目结构

```text
src/llm_post_training/
├── baseline_eval.py   # GSM8K baseline / checkpoint 评测
├── data.py            # 数据加载与训练数据处理
├── parsing.py         # 数学答案解析与格式判断
└── rewards.py         # GRPO reward functions

scripts/
├── setup_colab.py     # Colab 运行环境初始化
├── train_sft.py       # SFT 训练入口
└── train_grpo.py      # GRPO 训练入口
```

## 技术栈

- Python / PyTorch
- Hugging Face Transformers / Datasets
- TRL
- uv
- Google Colab（L4 / A100）

## 环境

项目使用 Python 3.12，并通过 uv 管理依赖：

```bash
uv sync --locked
```

本地主要负责开发与轻量调试，GPU 推理和训练主要在 Colab 环境完成。

---

> Work in progress. The repository is being cleaned up together with ongoing SFT / GRPO experiments.
