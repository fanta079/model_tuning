# AGENTS.md — model_tuning

本文件是 AI Agent 在本仓库工作的操作手册：项目结构、环境、命令、约定与坑。改动结构或命令时同步更新本文件。

> ⭐ **排查任何报错前，先查 `docs/踩坑记录.md`** —— 17 条已解决的实战坑（WSL/网络/环境/数据/显存），每条含现象、原因、解法。新解决的坑应追加到该文档，保持持续更新。

## 1. 项目概述

本地大模型**微调 + 推理**工作区，面向一张 **RTX 4060 Ti (8GB 显存)**。
主机为 Windows，**所有训练/推理均在 WSL2 (Ubuntu 24.04) 内运行**；模型与数据存放在 Windows E 盘，WSL 通过 `/mnt/e/...` 访问。

两条主线：

| 主线 | 基座 | 用途 | 工具 |
|---|---|---|---|
| 微调 | `unsloth/Qwen3.5-4B` (BF16, 8.8GB) | QLoRA 指令微调 | Unsloth + TRL |
| 聊天推理 | `unsloth/Qwen3.5-9B-GGUF` (Q4_K_XL, 5.6GB) | 本地对话 | llama.cpp `llama-server` |

## 2. 硬件与环境

- GPU: RTX 4060 Ti, **8GB VRAM**, bf16 支持, CUDA 13.1 (宿主驱动), WSL 内 torch 2.14.1+cu130 (装 unsloth 时从其依赖自动升级, 已验证可用, **勿随意变动**)
- 内存 16GB；E 盘存放模型（`/mnt/e/VMOS_workspace/model_tuning`）
- 网络: **huggingface.co 直连不通**。模型/数据下载一律走 **ModelScope（魔搭）**；HF 生态操作设 `HF_ENDPOINT=https://hf-mirror.com`
- 训练框架 **Unsloth 只支持 Linux/WSL，不支持原生 Windows**

## 3. 目录结构

```
model_tuning/
├── alpaca_zh/                    中文指令数据集 (51155 条, instruction/input/output)
├── models/
│   ├── Qwen3.5-4B/               微调基座 (BF16 safetensors, 完整)
│   └── Qwen3.5-9B-GGUF/          推理模型 (Qwen3.5-9B-UD-Q4_K_XL.gguf + mmproj-F16.gguf)
├── outputs/                      (训练产物, 首次训练后生成)
├── setup-ml-env-uv.sh            WSL 环境一键安装 (uv 版, 幂等可重跑)
├── train_lora.py                 QLoRA 微调脚本 (Unsloth, 8GB 调优)
├── run_chat.sh                   9B GGUF 聊天服务启动 (llama-server)
├── unsloth_compiled_cache/       Unsloth 内核编译缓存 (自动生成, 勿提交)
├── docs/                         📚 项目文档 (新文档一律放这里)
│   └── 踩坑记录.md                17 条已解决的环境/数据/训练坑, 排查问题先查
└── .workbuddy/memory/            Agent 工作日志 (YYYY-MM-DD.md, 追加式)
```

## 4. 环境搭建 (WSL 内)

```bash
bash /mnt/e/VMOS_workspace/model_tuning/setup-ml-env-uv.sh
```

脚本内容：apt 换阿里云源 → 装 uv → 建 venv `~/venvs/ft` (python3.12) → torch 2.5.1+cu124 (阿里云 wheel) → llamafactory/bitsandbytes/accelerate/peft/trl/datasets/modelscope (清华 PyPI) → 设置 `HF_ENDPOINT=hf-mirror.com` → nvidia-smi + CUDA 自检。幂等，可重复执行。

**必须用 `bash` 运行，禁止 `sh`**（dash 不支持 `set -o pipefail`，报 `set: Illegal option -o pipefail`）。

使用环境前：`source ~/venvs/ft/bin/activate`。

**注意**：`train_lora.py` 依赖的 **Unsloth 不在一键脚本里**，环境重建后需单独执行 `uv pip install unsloth`（它会自带 triton 等依赖，可能联动升级 torch —— 装完先 `uv pip list | grep torch` 确认）。

## 5. 常用命令 (WSL 内)

```bash
# 微调（默认 500 条冒烟测试；SMOKE_TEST=0 跑全量 51k）
source ~/venvs/ft/bin/activate
python /mnt/e/VMOS_workspace/model_tuning/train_lora.py          # 冒烟
SMOKE_TEST=0 python /mnt/e/VMOS_workspace/model_tuning/train_lora.py   # 全量

# 聊天 (9B GGUF, llama.cpp llama-server, 自带 Web UI)
bash /mnt/e/VMOS_workspace/model_tuning/run_chat.sh              # 16k 上下文
CTX=32768 bash run_chat.sh                                       # 更长上下文
ENABLE_VISION=1 bash run_chat.sh                                 # 启用看图 (多占 ~900MB 显存)

# 下载模型/数据集 (ModelScope, python 版)
python -c "from modelscope import snapshot_download; \
snapshot_download('unsloth/Qwen3.5-9B-GGUF', local_dir='models/Qwen3.5-9B-GGUF', \
allow_patterns=['Qwen3.5-9B-UD-Q4_K_XL.gguf','mmproj-F16.gguf'])"   # 只拉指定文件
```

## 6. 显存约束 (8GB) 与调优参数

| 任务 | 可行 | 关键参数 |
|---|---|---|
| 4B QLoRA | ✅ 推荐 | seq 2048, batch 1 × 累积 8, r=16/alpha=32, adamw_8bit, bf16 |
| 9B QLoRA | ⚠️ 吃紧 | seq ≤1024~2048, 其余同上；容易 OOM |
| 9B Q4 推理 | ✅ | GGUF + llama.cpp, `-ngl 99`, ctx 8k~16k |
| ≥14B | ❌ | 任何方式都跑不动 |

OOM 时降 `MAX_SEQ_LEN` → 1024 或调低 `-ngl`。

## 7. 数据与格式

- 微调数据: `alpaca_zh/alpaca_data_zh_51k.json`，JSON 数组，字段 `instruction / input / output`（Alpaca 格式，中文，51155 条）
- `train_lora.py` 内置 `to_text()` 已将其转为 Qwen chat 模板文本，换数据集时改 `DATA_PATH` 并保持该字段结构
- 训练产物: `outputs/qwen35-4b-lora/lora_adapter/`

## 8. 约定与坑

1. **sh ≠ bash**：所有 `.sh` 用 `bash` 运行。
2. **GGUF 不能微调**：GGUF 仅推理（llama.cpp 的训练二进制已废弃）。微调必须用 BF16 safetensors 原版 + QLoRA。预量化权重不省微调显存。
3. **unsloth 的 Qwen3.5 没有 bnb-4bit 预量化版**（魔搭/HF 均核实），`-bnb-4bit` 只覆盖 Qwen3/Qwen3-VL/gemma 等旧系。
4. 下载中断后**重跑同一条 snapshot_download 即可断点续传**；`models/*/. ____temp`（实为 `._____temp`）是未完成分片，可删。
5. WSL 内访问 Windows 盘慢于原生盘；训练吞吐可接受，大文件读写建议放 Linux 文件系统。
6. `models/`、`outputs/`、`unsloth_compiled_cache/` 体量大，勿纳入版本控制。
7. WSL 内 **不要** 装 NVIDIA 驱动，用宿主驱动直通；`nvidia-smi` 应能看到 4060 Ti。
8. **Git LFS 指针陷阱**：clone 含大文件的仓库前确认已执行 `git lfs install`。若文件仅百余字节且开头是 `version https://git-lfs...`，那是 LFS 指针不是本体，解析会报 `Invalid value` —— 用 ModelScope `/resolve/` 直链重下并对照指针里的 sha256 校验（详见 `docs/踩坑记录.md` 第 12 条）。数据集下载一律到新目录，禁止覆盖已有文件。
