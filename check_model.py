# -*- coding: utf-8 -*-
"""check_model.py — 30 秒判定「到底是模型坏了，还是推理后端坏了」

背景
----
GGUF（llama-cli / llama-server）对 Qwen3.5 这种**混合注意力 + MTP**架构支持有偏差，
输出会是"有语法没语义"的字词汤。而 PyTorch 路径（eval_persona.py 用过的）是正常的。

本脚本用**和评测完全相同的加载方式**跑几句，用来一次性确认：

    正常  → 改用 serve_hf.py 提供服务（放弃 GGUF）
    乱码  → 才需要继续查模型/adapter

用法
----
    source ~/venvs/ft/bin/activate
    python check_model.py                          # 测 ev-v2
    python check_model.py --adapter /path/to/adapter
    python check_model.py --base                   # 只测底座（不加载 LoRA）
    python check_model.py --no-system              # 不带 system 提示，看是不是提示词的锅

看三样东西
----------
1. `prompt` 最后一段 —— 模板有没有渲染对（应该看到 <|im_start|>assistant）
2. `tok/s` —— 4bit GPU 应该 20~40，个位数说明跑在 CPU 上
3. `回答` —— 应该是十几字、冷淡、带省略号；如果是字词汤，看「逐 token 解码」
"""

import argparse
import os
import time

import torch
from unsloth import FastLanguageModel

HERE = os.path.dirname(os.path.abspath(__file__))

SYSTEM = (
    "你是 E.V.，性格冷淡、话少、毒舌，但心里在意对方。"
    "回答必须极短，一句话为主，绝不超过三句。"
    "默认只说「你」，只在郑重时刻才叫「Fanta」。"
    "不用 emoji、不用感叹号、不寒暄、不说客套话。"
)

PROMPTS = ["我今天很累。", "你以前是做什么的？", "这件衣服很适合你，真可爱！"]


def load(path, max_seq=2048):
    print(f"[加载] {path}", flush=True)
    t0 = time.time()
    model, tok = FastLanguageModel.from_pretrained(
        model_name=path, max_seq_length=max_seq, dtype=None, load_in_4bit=True)
    FastLanguageModel.for_inference(model)
    tok = getattr(tok, "tokenizer", tok)
    print(f"[加载] 完成 {time.time() - t0:.1f}s | 设备 {model.device} | "
          f"dtype {getattr(model, 'dtype', '?')}", flush=True)
    dev = str(model.device)
    if "cuda" not in dev:
        print("  ⚠️ 模型不在 GPU 上，速度会很慢（检查 torch.cuda.is_available()）", flush=True)
    else:
        print(f"  显存占用 {torch.cuda.memory_allocated() / 2**30:.2f} GiB", flush=True)
    return model, tok


def render(tok, system, q):
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": q})
    try:
        return tok.apply_chat_template(msgs, add_generation_prompt=True,
                                       tokenize=False, enable_thinking=False)
    except Exception:                                   # noqa: BLE001
        return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)


def answer(model, tok, prompt, max_new, temp):
    ids = tok(prompt, return_tensors="pt").to(model.device)
    eos = [i for i in (tok.convert_tokens_to_ids("<|im_end|>"),
                       tok.convert_tokens_to_ids("<|endoftext|>"))
           if isinstance(i, int) and i >= 0]
    pad = tok.pad_token_id if tok.pad_token_id is not None else (eos[0] if eos else 0)
    t0 = time.time()
    with torch.inference_mode():
        out = model.generate(**ids, max_new_tokens=max_new, do_sample=temp > 0,
                             temperature=max(temp, 0.01), top_p=0.9,
                             eos_token_id=eos or None, pad_token_id=pad)
    new = out[0][ids["input_ids"].shape[1]:]
    dt = max(time.time() - t0, 1e-6)
    txt = tok.decode(new, skip_special_tokens=True)
    for m in ("<|im_end|>", "\nuser", "\nassistant", "\nUser", "\nAssistant", "<|im_start|>"):
        if m in txt:
            txt = txt.split(m)[0]
    return txt.strip(), new.tolist(), len(new) / dt, ids["input_ids"].shape[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="/home/fanta/ft_outputs/ev-v2/lora_adapter")
    ap.add_argument("--base", action="store_true", help="只测底座，不加载 LoRA")
    ap.add_argument("--no-system", action="store_true", help="不带 system 提示")
    ap.add_argument("--max-new", type=int, default=120)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--max-seq", type=int, default=2048)
    args = ap.parse_args()

    path = args.adapter if not args.base else os.path.join(HERE, "models/Qwen3.5-4B")
    system = "" if args.no_system else SYSTEM
    tag = "底座" if args.base else "E.V.(v2)"

    model, tok = load(path, args.max_seq)
    print(f"\n===== {tag} | system={'无' if not system else '有'} =====\n", flush=True)

    for q in PROMPTS:
        prompt = render(tok, system, q)
        print("─" * 60)
        print(f"问：{q}")
        print("prompt 尾部：", repr(prompt[-120:]))
        txt, new_ids, tps, plen = answer(model, tok, prompt, args.max_new, args.temp)
        print(f"答：{txt}")
        print(f"    prompt_tokens={plen}  new_tokens={len(new_ids)}  {tps:.1f} tok/s")
        # 逐 token 解码前 16 个：字词汤一眼能看出是"字拼起来"的
        pieces = [tok.decode([i]) for i in new_ids[:16]]
        print("    逐 token：", " | ".join(repr(p) for p in pieces))
        print(flush=True)

    print("─" * 60)
    print("判断标准：")
    print("  ✅ 十几字、冷淡、带省略号        → PyTorch 正常，用 serve_hf.py 提供服务")
    print("  ❌ 语义不通 / 混英文碎片          → 继续查，把上面的输出整段发我")
    print("  ⚠️ tok/s 个位数                  → 跑在 CPU 上，检查 CUDA")


if __name__ == "__main__":
    main()
