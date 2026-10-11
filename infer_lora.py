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

# Qwen3.5 是多模态模型, from_pretrained 返回的 "tokenizer" 实为 Processor;
# 取其底层纯文本 tokenizer, 否则 tok(text) 会把文本当作 images 参数去解码成图片。
# Processor 有 .tokenizer 属性; 若本就是普通 tokenizer 则 getattr 返回自身。
tok = getattr(tokenizer, "tokenizer", tokenizer)

# 终止符: Qwen 系用 <|im_end|> (部分版本还有 <|endoftext|>), 显式给上防止续写
EOS_IDS = [tok.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>")]
EOS_IDS = [i for i in EOS_IDS if isinstance(i, int) and i >= 0]
PAD_ID  = tok.pad_token_id if tok.pad_token_id is not None else (EOS_IDS[0] if EOS_IDS else 0)

# 兜底截断标记: 一旦出现说明模型开始编造下一轮对话, 只保留第一轮回答
STOP_MARKS = ("<|im_end|>", "\nuser", "\nassistant", "\nUser", "\nAssistant")


def render(msgs):
    """渲染 chat 模板; 优先关闭思考链, 模板不支持 enable_thinking 时回退。"""
    try:
        return tokenizer.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=False,
            enable_thinking=False)
    except Exception:
        return tokenizer.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=False)


def generate(prompt):
    text = render([{"role": "user", "content": prompt}])
    if os.environ.get("DEBUG_PROMPT") == "1":
        print("---- 实际送入模型的文本 ----")
        print(repr(text))
        print("---------------------------")
    # 用纯文本 tokenizer 编码 -> 张量 (BatchEncoding 才有 .to())
    inputs = tok(text, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens = MAX_NEW,
        do_sample      = True,
        temperature    = 0.7,
        top_p          = 0.9,
        eos_token_id   = EOS_IDS or None,   # 关键: 显式终止符, 否则会一直续写
        pad_token_id   = PAD_ID,
    )
    ans = tok.decode(out[0][inputs["input_ids"].shape[1]:],
                     skip_special_tokens=True).strip()
    for m in STOP_MARKS:                    # 兜底: 只保留第一轮回答
        if m in ans:
            ans = ans.split(m)[0].strip()
    return ans


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
