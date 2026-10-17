# -*- coding: utf-8 -*-
# Qwen3.5-4B + alpaca_zh  QLoRA 微调脚本
# 为 RTX 4060 Ti (8GB 显存) 调优
#
# 运行环境: 请在 WSL 内运行 (Unsloth 官方支持 Linux/WSL, 不支持原生 Windows)
#
# 首次安装 (WSL 内执行一次):
#   sudo apt update && sudo apt install -y python3-venv python3-pip build-essential
#   python3 -m venv ~/venvs/ft && source ~/venvs/ft/bin/activate
#   pip install -U pip
#   pip install torch --index-url https://download.pytorch.org/whl/cu124
#   pip install "unsloth[colab-new]" trl transformers datasets accelerate peft bitsandbytes
#
# 运行:
#   source ~/venvs/ft/bin/activate
#   python /mnt/e/VMOS_workspace/model_tuning/train_lora.py
#
# 说明:
#   - 如果 SFTConfig 报错, 说明 TRL 版本较旧, 见文件底部注释的兼容写法
#   - 默认数据: alpaca_zh (51k 中文指令, 单轮)
#
# 常用环境变量:
#   DATA_PATH=/path/train.json     换数据集
#   DATA_FORMAT=chat               多轮对话数据(客服场景), 字段 conversations=[{role,content}]
#   SYSTEM="你是..."               多轮时统一注入的 system 人设
#   RESPONSES_ONLY=1               只对 assistant 回复算 loss
#   NUM_EPOCHS=2                   训练轮数
#   SMOKE_TEST=0                   跑全量 (默认 1 = 只用 500 条冒烟)
#
# 客服微调示例:
#   DATA_PATH=/mnt/e/VMOS_workspace/model_tuning/customer_service/train.json \
#   DATA_FORMAT=chat RESPONSES_ONLY=1 NUM_EPOCHS=2 SMOKE_TEST=0 \
#   SYSTEM="你是「星辰科技」的客服小林，语气亲切专业，不确定的信息绝不编造。" \
#   python train_lora.py

import os

from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

# ---------------- 路径 (WSL 通过 /mnt 访问 Windows 盘) ----------------
BASE       = os.environ.get("BASE", "/mnt/e/VMOS_workspace/model_tuning")
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(BASE, "models/Qwen3.5-4B"))
# 默认用 alpaca_zh；换自己的数据: DATA_PATH=/path/train.json
DATA_PATH  = os.environ.get("DATA_PATH", os.path.join(BASE, "alpaca_zh/alpaca_data_zh_51k.json"))
# 输出写进 WSL 原生 ext4 家目录，避免 /mnt/e 上 torch.save 写 zip 损坏（9P 文件系统 bug）
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/home/fanta/ft_outputs/qwen35-4b-lora")

# ---------------- 关键超参 (8GB 场景) ----------------
MAX_SEQ_LEN = 2048      # 4B 在 8GB 上可到 2048; 爆显存就降到 1024
SMOKE_TEST  = os.environ.get("SMOKE_TEST", "1") == "1"   # 默认冒烟(500条); SMOKE_TEST=0 跑全量

# ---------------- 数据格式开关 ----------------
# DATA_FORMAT=alpaca (默认) 单轮, 字段 instruction / input / output
# DATA_FORMAT=chat          多轮, 字段 conversations=[{role, content}]  ← 客服场景用这个
DATA_FORMAT    = os.environ.get("DATA_FORMAT", "alpaca").lower()
SYSTEM         = os.environ.get("SYSTEM", "").strip()      # 可选: 统一注入的 system 人设
RESPONSES_ONLY = os.environ.get("RESPONSES_ONLY", "0") == "1"   # 只对 assistant 回复算 loss
NUM_EPOCHS     = int(os.environ.get("NUM_EPOCHS", "1"))

# ---------------- 1. 加载模型 (4bit) ----------------
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name      = MODEL_PATH,
    max_seq_length  = MAX_SEQ_LEN,
    dtype           = None,          # 自动选择 bf16
    load_in_4bit    = True,
)

# ---------------- 2. 挂 LoRA ----------------
model = FastLanguageModel.get_peft_model(
    model,
    r                       = 16,
    lora_alpha              = 32,
    lora_dropout            = 0,
    target_modules          = ["q_proj", "k_proj", "v_proj", "o_proj",
                               "gate_proj", "up_proj", "down_proj"],
    bias                    = "none",
    use_gradient_checkpointing = "unsloth",
    random_state            = 42,
)

# ---------------- 3. 数据格式化 ----------------
def to_text_alpaca(ex):
    """单轮 Alpaca: instruction(+input) -> user, output -> assistant"""
    inst = ex["instruction"].strip()
    inp  = (ex.get("input") or "").strip()
    user = inst + ("\n" + inp if inp else "")
    msgs = [
        {"role": "user",      "content": user},
        {"role": "assistant", "content": ex["output"].strip()},
    ]
    return {"text": tokenizer.apply_chat_template(msgs, tokenize=False,
                                                  add_generation_prompt=False)}


def to_text_chat(ex):
    """多轮对话: conversations=[{role, content}] 整体渲染（客服场景）"""
    msgs = [{"role": m["role"], "content": str(m["content"]).strip()}
            for m in ex["conversations"]]
    if SYSTEM and not any(m["role"] == "system" for m in msgs):
        msgs = [{"role": "system", "content": SYSTEM}] + msgs
    return {"text": tokenizer.apply_chat_template(msgs, tokenize=False,
                                                  add_generation_prompt=False)}


to_text = to_text_chat if DATA_FORMAT == "chat" else to_text_alpaca
print(f"[数据] {DATA_PATH}")
print(f"[数据] 格式={DATA_FORMAT} | epochs={NUM_EPOCHS} | 只训回复={RESPONSES_ONLY}")

ds = load_dataset("json", data_files=DATA_PATH, split="train").map(to_text)
if SMOKE_TEST:
    ds = ds.select(range(min(500, len(ds))))
print(f"[数据] 样本数 = {len(ds)}")

# ---------------- 4. 训练 ----------------
trainer = SFTTrainer(
    model           = model,
    tokenizer       = tokenizer,
    train_dataset   = ds,
    args = SFTConfig(
        dataset_text_field          = "text",
        max_seq_length              = MAX_SEQ_LEN,
        packing                     = True,
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 8,     # 等效 batch = 8
        warmup_steps                = 20,
        num_train_epochs            = NUM_EPOCHS,
        learning_rate               = 2e-4,
        bf16                        = True,
        logging_steps               = 10,
        optim                       = "adamw_8bit",
        weight_decay                = 0.01,
        lr_scheduler_type           = "linear",
        save_steps                  = 500,
        output_dir                  = OUTPUT_DIR,
        report_to                   = "none",
        seed                        = 42,
    ),
)
# 只对 assistant 回复计算 loss（人设/风格微调推荐：不让模型去学"用户说的话"）
if RESPONSES_ONLY:
    from unsloth.chat_templates import train_on_responses_only
    trainer = train_on_responses_only(
        trainer,
        instruction_part = "<|im_start|>user\n",
        response_part    = "<|im_start|>assistant\n",
    )
    print("[训练] 已开启 train_on_responses_only（只对 assistant 回复算 loss）")

trainer.train()

# ---------------- 5. 保存 adapter ----------------
ADAPTER = os.path.join(OUTPUT_DIR, "lora_adapter")
model.save_pretrained(ADAPTER)
tokenizer.save_pretrained(ADAPTER)
print("训练完成, LoRA adapter 保存在:", ADAPTER)

# ---------------- 6. 可选: 合并并导出 GGUF (用于 llama.cpp 推理) ----------------
# model.save_pretrained_gguf(os.path.join(OUTPUT_DIR, "gguf"),
#                            tokenizer, quantization_method="q4_k_m")

# ---------------- 旧版 TRL 兼容写法 (SFTConfig 不可用时替换上面的 args 部分) ----------------
# from transformers import TrainingArguments
# trainer = SFTTrainer(
#     model=model, tokenizer=tokenizer, train_dataset=ds,
#     dataset_text_field="text", max_seq_length=MAX_SEQ_LEN, packing=True,
#     args=TrainingArguments(per_device_train_batch_size=1,
#         gradient_accumulation_steps=8, warmup_steps=20, num_train_epochs=1,
#         learning_rate=2e-4, bf16=True, logging_steps=10, optim="adamw_8bit",
#         output_dir=OUTPUT_DIR, report_to="none", seed=42),
# )
