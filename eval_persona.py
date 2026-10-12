# -*- coding: utf-8 -*-
"""eval_persona.py — 人设微调效果评测

做什么
------
跑一批**固定题目**，把模型的回答打出来，并自动标注肉眼容易漏的"崩人设"信号：
  · 回复过长（灰原哀不写小作文）
  · 出现 emoji / 网络流行语 / 感叹号
  · 出现「呀」「啦」「哒」这类可爱语气词
  · 冒出"作为AI""有什么可以帮您"的助手腔

用同一批题对比「微调前 vs 微调后」，才知道这次训练到底有没有用。

用法（WSL 内）
--------------
    source ~/venvs/ft/bin/activate

    # 只看微调后
    python eval_persona.py --adapter /home/fanta/ft_outputs/qwen35-4b-haibara-v1/lora_adapter

    # 对比微调前 / 微调后（推荐）
    python eval_persona.py --compare \
        --adapter /home/fanta/ft_outputs/qwen35-4b-haibara-v1/lora_adapter

    # 只看底座（基线）
    python eval_persona.py --base-only

产出: eval_result_<tag>.md —— 直接打开看，或贴给别人看
"""

import argparse
import datetime
import json
import os
import re
import sys

import torch
from unsloth import FastLanguageModel

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------- 自动检查规则 ----------
CHECKS = [
    (re.compile(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]"), "emoji"),
    (re.compile(r"(qwq|QAQ|orz|233+|hhh{2,}|哈哈{2,})", re.I), "网络流行语"),
    (re.compile(r"(绝绝子|yyds|破防|栓Q|emo了)"), "网络流行语"),
    (re.compile(r"(作为(一个)?(AI|人工智能|助手))"), "AI 自曝"),
    (re.compile(r"(请问有什么(可以|能)帮(您|你))"), "助手腔"),
    (re.compile(r"[!！]"), "感叹号"),
    (re.compile(r"[呀啦哒哟嘞]"), "可爱语气词"),
]
MAX_OK = 220          # 超过这个字数就算"太长了"


def check(txt):
    hits = [why for pat, why in CHECKS if pat.search(txt)]
    if len(txt) > MAX_OK:
        hits.append(f"过长({len(txt)}字)")
    return hits


def load(path, max_seq=2048):
    model, tok = FastLanguageModel.from_pretrained(
        model_name=path, max_seq_length=max_seq, dtype=None, load_in_4bit=True)
    FastLanguageModel.for_inference(model)
    tok = getattr(tok, "tokenizer", tok)
    eos = [tok.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>")]
    eos = [i for i in eos if isinstance(i, int) and i >= 0]
    pad = tok.pad_token_id if tok.pad_token_id is not None else (eos[0] if eos else 0)
    return model, tok, eos, pad


def render(tok, system, question):
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": question})
    try:
        return tok.apply_chat_template(msgs, add_generation_prompt=True,
                                       tokenize=False, enable_thinking=False)
    except Exception:                                   # noqa: BLE001
        return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)


def answer(model, tok, eos, pad, system, question, max_new):
    ids = tok(render(tok, system, question), return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model.generate(**ids, max_new_tokens=max_new, do_sample=True,
                             temperature=0.7, top_p=0.9,
                             eos_token_id=eos or None, pad_token_id=pad)
    txt = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
    for m in ("<|im_end|>", "\nuser", "\nassistant", "\nUser", "\nAssistant"):
        if m in txt:
            txt = txt.split(m)[0]
    return txt.strip()


def run(model, tok, eos, pad, questions, system, max_new, label):
    rows = []
    for i, q in enumerate(questions, 1):
        a = answer(model, tok, eos, pad, system, q, max_new)
        rows.append({"q": q, "a": a, "flags": check(a)})
        print(f"  [{label}] {i}/{len(questions)} {q[:24]}…", flush=True)
    return rows


def summarize(rows):
    n = len(rows)
    if not n:
        return {}
    bad = sum(1 for r in rows if r["flags"])
    avg = sum(len(r["a"]) for r in rows) / n
    from collections import Counter
    c = Counter(f for r in rows for f in r["flags"])
    return {"n": n, "bad": bad, "bad_rate": bad / n, "avg_len": avg, "flags": c}


def write_md(path, tag, results, system):
    L = [f"# 人设评测报告 · {tag}", "",
         f"- 时间：{datetime.datetime.now():%Y-%m-%d %H:%M}",
         f"- system 提示：{'有' if system else '无（仅靠权重）'}", ""]
    for label, rows in results.items():
        s = summarize(rows)
        L += [f"## {label}", ""]
        if s:
            L += [f"- 题数 **{s['n']}** ｜ 有问题的 **{s['bad']}** 条"
                  f"（{s['bad_rate']*100:.0f}%）｜ 平均回答 **{s['avg_len']:.0f}** 字", ""]
            if s["flags"]:
                detail = "、".join(f"{k}×{v}" for k, v in s["flags"].most_common())
                L += [f"- 问题分布：{detail}", ""]
        L += ["| # | 提问 | 回答 | 标注 |", "|---|---|---|---|"]
        for i, r in enumerate(rows, 1):
            a = r["a"].replace("\n", "<br>").replace("|", "\\|")
            flag = "、".join(r["flags"]) or "—"
            L.append(f"| {i} | {r['q']} | {a} | {flag} |")
        L.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))


def main():
    ap = argparse.ArgumentParser(description="人设微调效果评测")
    ap.add_argument("--adapter", default="", help="LoRA adapter 目录")
    ap.add_argument("--base-model", default=os.path.join(HERE, "models/Qwen3.5-4B"),
                    help="底座模型目录")
    ap.add_argument("--eval-set", default=os.path.join(HERE, "personas/haibara_eval.json"))
    ap.add_argument("--system", default="", help="system 提示；--system-file 可读文件")
    ap.add_argument("--system-file", default="", help="把某个文件（如人设卡）当 system 提示")
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--compare", action="store_true", help="同时跑底座，做微调前后对比")
    ap.add_argument("--base-only", action="store_true", help="只跑底座")
    ap.add_argument("--tag", default="", help="报告文件名后缀，默认 adapter 目录名")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 题（调试用）")
    args = ap.parse_args()

    system = args.system
    if args.system_file:
        system = open(args.system_file, encoding="utf-8").read().strip()

    questions = json.load(open(args.eval_set, encoding="utf-8"))["questions"]
    if args.limit:
        questions = questions[:args.limit]
    print(f"[题集] {len(questions)} 题")

    results = {}

    if args.compare or args.base_only:
        print(f"[加载底座] {args.base_model}")
        m, t, e, p = load(args.base_model)
        results["微调前（底座）"] = run(m, t, e, p, questions, system, args.max_new, "base")
        del m
        torch.cuda.empty_cache()

    if not args.base_only:
        if not args.adapter:
            sys.exit("[错误] 需要 --adapter，或用 --base-only 只跑底座")
        print(f"[加载 adapter] {args.adapter}")
        m, t, e, p = load(args.adapter)
        results["微调后（+LoRA）"] = run(m, t, e, p, questions, system, args.max_new, "lora")
        del m
        torch.cuda.empty_cache()

    tag = args.tag or (os.path.basename(os.path.dirname(args.adapter.rstrip("/")))
                       if args.adapter else "base")
    out = os.path.join(HERE, f"eval_result_{tag}.md")
    write_md(out, tag, results, system)

    print("\n" + "=" * 56)
    for label, rows in results.items():
        s = summarize(rows)
        print(f"{label}: {s['n']} 题 | 有问题 {s['bad']} 条 "
              f"({s['bad_rate']*100:.0f}%) | 平均 {s['avg_len']:.0f} 字")
    print(f"报告: {out}")
    print("=" * 56)
    print("重点看两件事：① 违规率有没有下降 ② 平均长度是不是更短了")


if __name__ == "__main__":
    main()
