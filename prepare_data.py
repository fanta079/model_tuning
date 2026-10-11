# -*- coding: utf-8 -*-
# prepare_data.py — 微调数据 校验 / 清洗 / 去重 / 划分
#
# 一个文件里可混用两种格式:
#   单轮 Alpaca : {"instruction": "...", "input": "...", "output": "..."}
#   多轮 Chat   : {"conversations": [{"role": "...", "content": "..."}, ...]}
#
# 用法:
#   python prepare_data.py customer_service/all.json
#   python prepare_data.py my.json --out-dir my_data --val-ratio 0.05 --seed 42
#
# 产出:
#   <out-dir>/train.json     训练集
#   <out-dir>/val.json       验证集（holdout，绝不可混入训练）
#   <out-dir>/dropped.json   被剔除的样本 + 原因（便于回头修数据）
#
# 只做数据，不碰模型；纯标准库，无需装依赖。

import argparse
import hashlib
import json
import os
import random
import re
import sys
from collections import Counter

MIN_REPLY, MAX_REPLY = 4, 4000          # assistant 回复的字符数合理区间
VALID_ROLES = ("system", "user", "assistant")


def norm(s):
    """归一化：去掉所有空白并小写，用于判重。"""
    return re.sub(r"\s+", "", str(s or "")).lower()


def is_chat(ex):
    return isinstance(ex.get("conversations"), list) and len(ex["conversations"]) > 0


def fingerprint(ex):
    """判重指纹：chat 用全部非 system 内容，alpaca 用 instruction+input。"""
    if is_chat(ex):
        parts = [m.get("content", "") for m in ex["conversations"] if m.get("role") != "system"]
        return hashlib.md5(norm("\n".join(parts)).encode()).hexdigest()
    return hashlib.md5((norm(ex.get("instruction")) + "|" + norm(ex.get("input"))).encode()).hexdigest()


def check(ex):
    """校验单条样本，返回 (是否通过, 原因)。"""
    if is_chat(ex):
        convs = ex["conversations"]
        for m in convs:
            if not isinstance(m, dict) or m.get("role") not in VALID_ROLES:
                return False, f"非法 role: {m.get('role') if isinstance(m, dict) else m}"
            if not str(m.get("content", "")).strip():
                return False, "存在空的 content"
        roles = [m["role"] for m in convs if m["role"] != "system"]
        if not roles:
            return False, "只有 system，没有对话"
        if roles.count("assistant") < 1 or roles.count("user") < 1:
            return False, "缺少 user 或 assistant 轮次"
        if roles[0] != "user":
            return False, "对话应以 user 开头"
        if roles[-1] != "assistant":
            return False, "对话应以 assistant 结尾"
        for m in convs:
            if m["role"] == "assistant":
                L = len(m["content"])
                if L < MIN_REPLY:
                    return False, f"assistant 回复过短({L}字)"
                if L > MAX_REPLY:
                    return False, f"assistant 回复过长({L}字)"
        return True, ""

    # 单轮 Alpaca
    if "instruction" not in ex or "output" not in ex:
        return False, "缺少 instruction 或 output 字段"
    if not str(ex["instruction"]).strip():
        return False, "instruction 为空"
    if not str(ex["output"]).strip():
        return False, "output 为空"
    L = len(str(ex["output"]))
    if L < MIN_REPLY:
        return False, f"output 过短({L}字)"
    if L > MAX_REPLY:
        return False, f"output 过长({L}字)"
    return True, ""


def describe(ex):
    """给样本一个便于人工阅读的摘要。"""
    if is_chat(ex):
        n_user = sum(1 for m in ex["conversations"] if m["role"] == "user")
        return f"[多轮 {n_user} 问] " + str(ex["conversations"][0].get("content", ""))[:40]
    return "[单轮] " + str(ex.get("instruction", ""))[:40]


def main():
    ap = argparse.ArgumentParser(description="微调数据 校验 / 清洗 / 去重 / 划分")
    ap.add_argument("src", help="原始 json 文件路径")
    ap.add_argument("--out-dir", default=None, help="输出目录（默认与源文件同目录）")
    ap.add_argument("--val-ratio", type=float, default=0.05, help="验证集比例，默认 0.05")
    ap.add_argument("--seed", type=int, default=42, help="随机种子，保证可复现")
    ap.add_argument("--keep-dropped", action="store_true", help="同时写出 dropped.json 便于排查")
    args = ap.parse_args()

    if not os.path.isfile(args.src):
        sys.exit(f"[错误] 找不到文件: {args.src}")
    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.src))
    os.makedirs(out_dir, exist_ok=True)

    with open(args.src, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        sys.exit("[错误] 顶层必须是 JSON 数组")

    total = len(data)
    passed, dropped, reasons, seen = [], [], Counter(), set()
    n_chat = 0

    for i, ex in enumerate(data):
        if not isinstance(ex, dict):
            dropped.append({"index": i, "reason": "不是对象", "raw": ex})
            reasons["不是对象"] += 1
            continue
        ok, why = check(ex)
        if not ok:
            dropped.append({"index": i, "reason": why, "sample": describe(ex)})
            reasons[why] += 1
            continue
        fp = fingerprint(ex)
        if fp in seen:
            dropped.append({"index": i, "reason": "重复", "sample": describe(ex)})
            reasons["重复"] += 1
            continue
        seen.add(fp)
        if is_chat(ex):
            n_chat += 1
        passed.append(ex)

    if not passed:
        sys.exit("[错误] 没有一条通过校验，请检查数据格式")

    random.seed(args.seed)
    random.shuffle(passed)
    n_val = max(1, int(len(passed) * args.val_ratio)) if args.val_ratio > 0 else 0
    val, train = passed[:n_val], passed[n_val:]

    def dump(name, obj):
        p = os.path.join(out_dir, name)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        return p

    p_train = dump("train.json", train)
    p_val = dump("val.json", val) if val else None
    if args.keep_dropped and dropped:
        dump("dropped.json", dropped)

    # ---------------- 报告 ----------------
    print("=" * 56)
    print(f"源文件     : {args.src}")
    print(f"输出目录   : {out_dir}")
    print("-" * 56)
    print(f"原始条数   : {total}")
    print(f"通过       : {len(passed)}  (多轮 {n_chat} / 单轮 {len(passed) - n_chat})")
    print(f"剔除       : {sum(reasons.values())}")
    for r, c in reasons.most_common():
        print(f"    - {r}: {c}")
    print("-" * 56)
    print(f"训练集     : {os.path.basename(p_train)}  {len(train)} 条")
    if p_val:
        print(f"验证集     : {os.path.basename(p_val)}  {len(val)} 条 (holdout，勿混入训练)")
    print("=" * 56)
    print("下一步:")
    print(f"  DATA_PATH={p_train} DATA_FORMAT={'chat' if n_chat else 'alpaca'} \\")
    print("  python train_lora.py")


if __name__ == "__main__":
    main()
