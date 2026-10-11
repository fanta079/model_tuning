# -*- coding: utf-8 -*-
# infer_lora.py - 加载 LoRA adapter, 对比"微调前 vs 微调后"的回答
#
# 用法 (WSL 内):
#   source ~/venvs/ft/bin/activate
#   python /mnt/e/VMOS_workspace/model_tuning/infer_lora.py
# 自定义:
#   ADAPTER=/path/to/lora_adapter MAX_NEW=512 python infer_lora.py
#
# 原理: 同一份权重, 用 peft 的 disable_adapter() 临时关掉 LoRA 就是"原始底座",
#       打开就是"微调后", 一次加载两种行为对比, 8GB 显存刚好够用。

import os
from unsloth import FastLanguageModel

ADAPTER = os.environ.get("ADAPTER", "/home/fanta/ft_outputs/qwen35-4b-lora/lora_adapter")
MAX_NEW = int(os.environ.get("MAX_NEW", "256"))

# 指令风格的问题 (alpaca_zh 是单轮指令数据, 不用 system prompt)
PROMPTS = [
    "我们如何在日常生活中减少用水？",          # 来自训练集, 看贴合度
    "用简单的语言解释什么是通货膨胀。",          # 解释类
    "给我三个提高工作效率的建议。",              # 建议类
    "帮我写一段 50 字左右的运动会加油稿。",      # 创作类
]

print("加载底座 + LoRA adapter:", ADAPTER)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name     = ADAPTER,   # adapter_config.json 里记录了底座路径, 会自动带上
    max_seq_length = 2048,
    dtype          = None,
    load_in_4bit   = True,
)
FastLanguageModel.for_inference(model)   # 切到推理模式 (比训练模式快)


def generate(prompt):
    msgs = [{"role": "user", "content": prompt}]
    # 第一步: 只渲染成 chat 模板文本 (返回 str)
    text = tokenizer.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=False
    )
    # 第二步: 文本 -> 张量 (返回 BatchEncoding, 才有 .to())
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens = MAX_NEW,
        do_sample      = True,
        temperature    = 0.7,
        top_p          = 0.9,
    )
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                            skip_special_tokens=True).strip()


for i, p in enumerate(PROMPTS, 1):
    print("=" * 70)
    print(f"[{i}] 提问: {p}")
    print("-" * 70)
    if hasattr(model, "disable_adapter"):
        with model.disable_adapter():      # 临时关掉 LoRA = 原始底座
            print("【微调前】", generate(p))
    else:
        print("【微调前】(当前 peft 版本不支持 disable_adapter, 跳过)")
    print("-" * 70)
    print("【微调后】", generate(p))          # LoRA 生效
    print()

print("对比完成。关注: 微调后是否更贴近'直接回答问题'的指令风格。")
