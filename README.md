# model_tuning

在 **RTX 4060 Ti (8GB)** 上做本地大模型**微调 + 推理**的工作区（WSL2 / Ubuntu 24.04）。

- **微调**：Unsloth QLoRA 微调 `Qwen3.5-4B`（alpaca_zh 5.1 万条中文指令）
- **推理**：llama.cpp `llama-server` 跑 `Qwen3.5-9B-GGUF`（本地对话，OpenAI 兼容 API）

> ⭐ 排查任何报错前，先看 **[docs/踩坑记录.md](docs/踩坑记录.md)** —— 18 条实战坑（WSL / 网络 / 环境 / 数据 / 显存），每条含现象 · 原因 · 解法。

## 文件说明

| 文件 | 作用 |
|---|---|
| `setup-ml-env-uv.sh` | WSL 环境一键安装（uv 版，幂等可重跑） |
| `train_lora.py` | QLoRA 微调脚本（8GB 调优，默认 500 条冒烟，`SMOKE_TEST=0` 跑全量） |
| `infer_lora.py` | 加载 LoRA adapter，对比「微调前 vs 微调后」 |
| `run_chat.sh` | 启动 9B GGUF 聊天服务（llama-server，自带 Web UI） |
| `docs/踩坑记录.md` | 踩坑合集（持续更新） |
| `AGENTS.md` | 项目导航手册：结构 / 环境 / 命令 / 约定 |

## 快速开始

```bash
# 1) 装环境（WSL 内，用 bash 不用 sh）
bash setup-ml-env-uv.sh
source ~/venvs/ft/bin/activate
uv pip install unsloth              # 不在脚本内，需单独装

# 2) 冒烟微调（500 条，验证全链路）
python train_lora.py

# 3) 全量微调（5.1 万条）
SMOKE_TEST=0 python train_lora.py

# 4) 对比微调前后
python infer_lora.py
```

## 硬件与显存约束

RTX 4060 Ti **8GB** / 16GB RAM。8GB 显存下可行范围：

| 任务 | 可行 | 关键参数 |
|---|---|---|
| 4B QLoRA | ✅ 推荐 | seq 2048, batch 1 × 累积 8, r=16/alpha=32, adamw_8bit, bf16 |
| 9B QLoRA | ⚠️ 吃紧 | seq ≤1024~2048，容易 OOM |
| 9B Q4 推理 | ✅ | GGUF + llama.cpp，`-ngl 99`，ctx 8k~16k |
| ≥14B | ❌ | 任何方式都跑不动 |

训练在 WSL2 内运行，模型与数据放 Windows E 盘（WSL 经 `/mnt/e/...` 访问）。
注意：训练**产物**要写到 Linux 家目录，`/mnt/e` 上 `torch.save` 写 zip 会损坏（见踩坑第 18 条）。

## 国内网络

| 用途 | 镜像 |
|---|---|
| apt | `mirrors.aliyun.com` |
| pip / uv | `pypi.tuna.tsinghua.edu.cn` |
| HuggingFace | `export HF_ENDPOINT=https://hf-mirror.com` |
| 模型 / 数据集 | ModelScope（魔搭） |

---
*个人学习记录，踩坑文档持续更新。*
