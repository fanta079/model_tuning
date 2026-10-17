# AGENTS.md — model_tuning

本文件是 AI Agent 在本仓库工作的操作手册：项目结构、环境、命令、约定与坑。改动结构或命令时同步更新本文件。

> ⭐ **排查任何报错前，先查 `docs/踩坑记录.md`** —— 19 条已解决的实战坑（WSL/网络/环境/数据/显存），每条含现象、原因、解法。新解决的坑应追加到该文档，保持持续更新。

## 1. 项目概述

本地大模型**数据生产 + 微调 + 推理**工作区，面向一张 **RTX 4060 Ti (8GB 显存)**。
主机为 Windows，**所有训练/推理均在 WSL2 (Ubuntu 24.04) 内运行**；模型与数据存放在 Windows E 盘，WSL 通过 `/mnt/e/...` 访问。

**当前主目标**：造一个**人设扮演模型**（角色 `E.V.`，性格取自灰原哀），并长期使用其 LoRA adapter。

三条主线：

| 主线 | 基座 | 用途 | 工具 |
|---|---|---|---|
| **数据生产** | — | 人设数据生成 / 微信记录清洗 / 划分 | `gen_persona_data.py`、`wechat_paste_to_json.py`、`prepare_data.py` |
| **微调** | `unsloth/Qwen3.5-4B` (BF16, 8.8GB) | QLoRA 人设 · 风格微调 | Unsloth + TRL |
| **推理** | `unsloth/Qwen3.5-9B-GGUF` (Q4_K_XL, 5.6GB) | 本地对话 | llama.cpp `llama-server` |

> 📘 做人设数据前**务必先读** `docs/人设数据生产指南.md` —— 里面记录了「同批同质」「比例靠措辞失效」两个必踩的坑。

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
├── my_data/                      🔒 个人训练数据 (含聊天记录/人设数据, 已 gitignore, 绝不入库)
│
├── 数据生产 ─────────────────────────────────────────────
│   ├── personas/                 🎭 人设卡 + 场景矩阵 + 评测题集
│   │   ├── haibara.md              人设卡 (角色 E.V., 性格取自灰原哀; 含称呼规则)
│   │   ├── haibara_scenes.json     场景矩阵 (~18000 组合) + 种子样本
│   │   └── haibara_eval.json       25 题固定评测集 (改了就失去可比性)
│   ├── gen_persona_data.py       人设数据批量生成器 (人设卡 + 场景矩阵 + 种子锚定)
│   ├── wechat_to_weclone.py      微信导出 -> WeClone CSV (9 列, 自动识别列名/类型)
│   ├── wechat_paste_to_json.py   微信 PC 复制文本 -> conversations JSON (安全, 免解密工具)
│   └── prepare_data.py           通用: 校验/清洗/去重/划分 (支持 Alpaca / ShareGPT / 对话)
│
├── 训练与使用 ───────────────────────────────────────────
│   ├── train_lora.py             QLoRA 微调 (Unsloth, 8GB 调优; DATA_FORMAT=chat 支持多轮)
│   ├── infer_lora.py             加载 LoRA 对比「微调前 vs 微调后」
│   ├── chat_lora.py              多轮对话 + 流式输出 终端 REPL (加载 LoRA)
│   ├── memory.py                 Markdown 外挂长期记忆模块 (纯标准库)
│   ├── memory.md                 外挂记忆本体 (🔒 个人内容, 已 gitignore)
│   ├── eval_persona.py           人设评测: 跑固定题集, 自动标注「崩人设」
│   ├── check_model.py            模型体检: 3 句话定「模型坏 vs 推理后端坏」
│   ├── serve_hf.py               PyTorch+GPU 4bit 的 OpenAI 兼容服务 (替代 llama-server)
│   ├── export_gguf.py            合并 LoRA 并导出 GGUF (⚠️ qwen35 混合架构下输出乱码, 已弃用)
│   └── run_chat.sh               9B GGUF 聊天服务启动 (llama-server)
│
├── 语音助手 App (P1) ────────────────────────────────────
│   ├── app/
│   │   ├── server.py             后端: STT -> LLM -> TTS (FastAPI, 端口 8090)
│   │   ├── index.html            前端: 半双工语音界面 (回完才能说下一句)
│   │   ├── system.txt            语音专用 system prompt (禁括号动作/Markdown)
│   │   ├── voice.json            🎭 音色配置 (voice / rate / pitch)
│   │   ├── try_voices.py         音色试听台 (生成 8 组候选 + 可点击的试听网页)
│   │   └── voice_samples/        试听产物 (已 gitignore, 可随时重新生成)
│   ├── setup_app.sh              独立 venv ~/venvs/evapp + 装依赖 + 预下 whisper
│   ├── run_app.sh                启动语音助手（前端，端口 8090）
│   ├── start_ev.sh               一条命令起「模型服务 + 前端」，支持 --adapter / --stop
│   ├── backup.sh                 打包「数据+配置+报告+adapter」→ artifacts/*.tar.gz
│   └── make_data_repo.sh         导出独立数据仓（data/ + reports/ + provenance README）
│
├── setup-ml-env-uv.sh            WSL 环境一键安装 (uv 版, 幂等可重跑)
├── setup_weclone.sh              WeClone 独立环境一键搭建 (⚠️ 勿与 ~/venvs/ft 混装, 依赖冲突)
├── unsloth_compiled_cache/       Unsloth 内核编译缓存 (自动生成, 勿提交)
├── docs/                         📚 项目文档 (新文档一律放这里)
│   ├── 踩坑记录.md                19 条已解决的环境/数据/训练坑, 排查问题先查
│   ├── WeClone-数字分身落地.md      聊天记录微调数字分身方案 (独立 venv, 含微信导出格式)
│   ├── 人设数据生产指南.md          从人设卡到可对话模型的完整流水线
│   ├── 训练与推理脚本.md            8 个脚本逐个讲: 超参含义 / 环境变量 / 数据格式 / 坑
│   ├── train_lora脚本导读.md        零基础逐段走读: 名词解释 + 为什么这个参数值
│   ├── P1-语音助手.md               语音助手 App 的架构 / 音色方案 / 路线图 (含安卓目标)
│   └── 仓库组织.md                  代码 / 数据 / 权重 三仓分离方案与版本 tag 约定
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

### 数据生产

```bash
# 人设数据: 人设卡 + 场景矩阵 -> 批量生成 (需 OpenAI 兼容 API)
python gen_persona_data.py --dry-run                 # 先看提示词, 不花钱, 不需 key
export OPENAI_API_KEY=sk-xxx                         # key 只走环境变量, 不写入文件
python gen_persona_data.py \
    --api-base https://api.deepseek.com/v1 --model deepseek-chat \
    --seed-file my_data/ev_manual_40.json --num-seeds 20 \
    --n 360 --per-call 12 --out my_data/ev_api.json  # 中断后重跑同一条即续传

# 微信记录 (免解密工具, 最安全): PC 端复制粘贴 -> txt
python wechat_paste_to_json.py my_data/chat1.txt --me "你的微信昵称" --probe  # 先探测
python wechat_paste_to_json.py my_data/ --me "你的微信昵称"                   # 整目录批量转换

# 通用: 校验 / 去重 / 划分 (支持 Alpaca / ShareGPT / 对话 三种格式混用)
python prepare_data.py my_data/ev_all.json --out-dir my_data/ev --keep-dropped
```

### 训练

```bash
source ~/venvs/ft/bin/activate
python train_lora.py                                  # 冒烟 500 条
SMOKE_TEST=0 python train_lora.py                     # 全量 (默认 alpaca_zh 51k)

# 训多轮人设数据 (换 OUTPUT_DIR 版本号, 别覆盖旧 adapter):
DATA_PATH=/mnt/e/VMOS_workspace/model_tuning/my_data/ev/train.json \
DATA_FORMAT=chat RESPONSES_ONLY=1 NUM_EPOCHS=5 \
OUTPUT_DIR=/home/fanta/ft_outputs/ev-v1 SMOKE_TEST=0 \
nohup python train_lora.py > ~/train_ev_v1.log 2>&1 &
```

### 评测 / 使用

```bash
# 人设评测: 跑固定题集, 并排对比「底座 vs 微调后」, 自动标注崩人设
python eval_persona.py --compare --adapter /home/fanta/ft_outputs/ev-v1/lora_adapter

# 模型体检: 3 句话 + 渲染后的 prompt + tok/s, 30 秒判定「模型坏 vs 推理后端坏」
python check_model.py                            # 测 ev-v2
python check_model.py --base                     # 只测底座
python check_model.py --no-system                # 排除 system 提示的影响

# OpenAI 兼容服务 (PyTorch + GPU 4bit, 用于 P1 语音助手; 比 CPU 版 llama.cpp 快 5~8 倍)
python serve_hf.py --adapter /home/fanta/ft_outputs/ev-v2/lora_adapter

# 终端多轮对话 (流式; 会话内 /reset 清空上下文, /remember 记一条, /exit 退出)
python chat_lora.py
ADAPTER=/mnt/e/VMOS_workspace/model_tuning/models/Qwen3.5-4B python chat_lora.py  # 跑底座

# 导出 GGUF 供 llama.cpp 日常使用 (启动快、显存低、能接任意前端)
python export_gguf.py --adapter /home/fanta/ft_outputs/ev-v1/lora_adapter --out ~/gguf/ev

# 9B GGUF 聊天服务 (自带 Web UI)
bash run_chat.sh                                     # 16k 上下文
CTX=32768 bash run_chat.sh                           # 更长上下文
ENABLE_VISION=1 bash run_chat.sh                     # 启用看图 (多占 ~900MB 显存)

# 下载模型/数据集 (ModelScope)
python -c "from modelscope import snapshot_download; \
snapshot_download('unsloth/Qwen3.5-9B-GGUF', local_dir='models/Qwen3.5-9B-GGUF', \
allow_patterns=['Qwen3.5-9B-UD-Q4_K_XL.gguf','mmproj-F16.gguf'])"   # 只拉指定文件
```

### 语音助手 App (P1)

```bash
# 建独立环境（不碰 ~/venvs/ft），含预下载 whisper 模型 ~3.1GB，走魔搭
bash setup_app.sh

# 调音色: 生成 8 组候选 + 可点击的试听网页 -> 听完填 app/voice.json
source ~/venvs/evapp/bin/activate
python app/try_voices.py

# 起助手（另开终端先起 llama-server）
bash run_app.sh                                       # 打开 http://localhost:8090
PORT=8091 LLM_BASE=http://127.0.0.1:8080/v1 bash run_app.sh   # 换端口 / 换模型服务

# 一条命令起「模型服务 + 前端」（推荐，自动等模型就绪）
bash start_ev.sh                                      # 默认 adapter = ft_outputs/ev-v2
bash start_ev.sh --adapter ~/ft_outputs/ev-v3/lora_adapter
bash start_ev.sh --no-llm                             # 只起前端（模型服务已自己起好）
bash start_ev.sh --stop                               # 全停，释放显存
```

> P1 是**半双工**（回完才能说下一句），纯对话、无工具调用。
> 完整架构 / 音色三层方案 / 路线图（P4 是**安卓 App**）见 `docs/P1-语音助手.md`。

## 6. 显存约束 (8GB) 与调优参数

| 任务 | 可行 | 关键参数 |
|---|---|---|
| 4B QLoRA | ✅ 推荐 | seq 2048, batch 1 × 累积 8, r=16/alpha=32, adamw_8bit, bf16 |
| 9B QLoRA | ⚠️ 吃紧 | seq ≤1024~2048, 其余同上；容易 OOM |
| 9B Q4 推理 | ✅ | GGUF + llama.cpp, `-ngl 99`, ctx 8k~16k |
| ≥14B | ❌ | 任何方式都跑不动 |

OOM 时降 `MAX_SEQ_LEN` → 1024 或调低 `-ngl`。

## 7. 数据与格式

两条数据来源，最终都归一到 **`conversations(role/content)`** 格式喂给 `train_lora.py`：

| 来源 | 工具 | 说明 |
|---|---|---|
| **人设生成** | `gen_persona_data.py` | 人设卡 + 场景矩阵 + 种子锚定；完整流程见 `docs/人设数据生产指南.md` |
| **微信记录** | `wechat_paste_to_json.py` | PC 端复制粘贴，免解密工具（PyWxDump 原项目已删库，现存同名仓库多为钓鱼）|

`prepare_data.py` 统一做校验 / 去重 / 划分，**支持三种输入格式混用**：

- 单轮 Alpaca：`{"instruction", "input", "output"}`
- 对话：`{"conversations":[{"role","content"}]}`
- ShareGPT：`{"conversations":[{"from","value"}]}`（LLaMA-Factory / WeClone 的输出）

产物：`train.json` / `val.json`。**val 是 holdout，绝不混入训练**。

其他：

- 通用指令数据：`alpaca_zh/`（51155 条，Alpaca 格式）
- 训练产物：由 `OUTPUT_DIR` 决定，**建议按版本号区分**（`ev-v1` / `ev-v2`），训崩了能回滚
- 评测题集：`personas/haibara_eval.json`（25 题）—— **不要修改**，改了就失去纵向可比性
- 🔒 `my_data/`、`wechat_data/`、`memory.md` 含个人内容，**已 gitignore**

## 8. 约定与坑

1. **sh ≠ bash**：所有 `.sh` 用 `bash` 运行。
2. **GGUF 不能微调**：GGUF 仅推理（llama.cpp 的训练二进制已废弃）。微调必须用 BF16 safetensors 原版 + QLoRA。预量化权重不省微调显存。
3. **unsloth 的 Qwen3.5 没有 bnb-4bit 预量化版**（魔搭/HF 均核实），`-bnb-4bit` 只覆盖 Qwen3/Qwen3-VL/gemma 等旧系。
4. 下载中断后**重跑同一条 snapshot_download 即可断点续传**；`models/*/. ____temp`（实为 `._____temp`）是未完成分片，可删。
5. WSL 内访问 Windows 盘慢于原生盘；训练吞吐可接受，大文件读写建议放 Linux 文件系统。
6. `models/`、`outputs/`、`unsloth_compiled_cache/` 体量大，勿纳入版本控制。
7. WSL 内 **不要** 装 NVIDIA 驱动，用宿主驱动直通；`nvidia-smi` 应能看到 4060 Ti。
8. **Git LFS 指针陷阱**：clone 含大文件的仓库前确认已执行 `git lfs install`。若文件仅百余字节且开头是 `version https://git-lfs...`，那是 LFS 指针不是本体，解析会报 `Invalid value` —— 用 ModelScope `/resolve/` 直链重下并对照指针里的 sha256 校验（详见 `docs/踩坑记录.md` 第 12 条）。数据集下载一律到新目录，禁止覆盖已有文件。
9. **比例类要求必须做成「维度」，不能靠 prompt 措辞**。实测：人设卡写「约 10% 会叫名字」，生成 240 条结果**叫名字 0 条**——模型把"稀有"理解成了"不用"。改成场景矩阵里的 `call_names` 维度（9 项不叫 + 1 项叫）后，比例由结构保证。
10. **数据生成要「一次调用下发 N 个不同任务」**。一次调用只给一个场景，会导致整批 N 条**全是同一场景**（只是换同义词）。prompt 里要写明"第 i 条对应任务 i，场景必须真正不同"。
11. **生成完先体检，再训练**。花 10 秒统计能省几小时：assistant 回复**重复率 < 15%**、平均长度符合人设、违规数（emoji/感叹号/可爱语气词）**= 0**、关键特征比例符合设计。
12. **`--seed-file` 是质量关键**。用手写精品当 few-shot 锚点，质量明显高于纯生成。推荐配比：**手写 10% + 机造 90%**。
13. **`.sh` 必须保持 LF**。Windows 侧 git `autocrlf` 会在检出时转 CRLF，导致 WSL 报 `$'\r': command not found` / `bad interpreter`。仓库已用 `.gitattributes` 固定 `*.sh text eol=lf`。
14. **API key 只走环境变量**，绝不写入任何文件。`.env` / `*.key` 已 gitignore。
15. **训练前换 `OUTPUT_DIR`**（如 `ev-v1` / `ev-v2`）。复用同一个目录会覆盖旧 adapter，训崩了无法回滚。
16. **个人数据目录已 gitignore**（`my_data/` / `wechat_data/` / `memory.md`）。提交前用 `git status` 确认，**不要用 `git add -f` 强加**。
