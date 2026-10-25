# LLM Post-training

以 **Qwen3-0.6B-Base + GSM8K** 学习并搭建后训练流程。

## 项目路线

1. GSM8K 加载与格式转换、Base 模型推理、答案解析和准确率评估。
2. 全参数 SFT，记录配置、训练指标和 checkpoint，并与 Base 比较。
3. 补齐 policy gradient、PPO clipping、KL 与 group-relative advantage 基础，再开展 GRPO 实验。

当前仅完成项目与开发环境初始化，尚未实现数据、评估或训练流程。

## WSL 开发环境

Python 3.12，使用 uv 管理独立的 `.venv` 与依赖锁文件：

```bash
cd ~/llm-post-training
uv sync --locked
uv run python -c "import llm_post_training; print('import OK')"
uv run ruff check .
uv run ruff format --check .
```

如果 uv 尚未加入 PATH，当前机器可使用已有安装：

```bash
export PATH="$HOME/miniconda3/envs/cs336/bin:$PATH"
```

这里只复用 uv 和 Python 可执行文件，项目依赖安装在本目录 `.venv`。
初始环境只包含开发工具；PyTorch、Transformers、Datasets、TRL、Accelerate 和 W&B
将在对应阶段按实际 GPU 环境添加和锁定。

## 本地与 Colab

本地 WSL 负责代码、Git 和轻量调试；Colab A100 负责推理与训练，3090 服务器恢复后可作为另一执行环境。
现有 Colab CLI 保持独立安装，不作为本项目依赖。

多文件项目在远端执行前需要显式同步源码、配置和项目依赖文件。
不能只发送训练入口文件就假定整个项目都已同步。具体同步脚本留到首次云端运行时实现并验证。
模型与公开数据优先在执行环境下载，训练结果及时持久化；运行完成后停止 Colab runtime。

## 目录约定

```text
src/llm_post_training/  Python 包，后续放数据、解析、评估和训练代码
configs/               后续实验配置
scripts/               后续运行与同步入口
```

数据、模型、checkpoint、日志、W&B 本地目录与凭据均不提交到 Git。
