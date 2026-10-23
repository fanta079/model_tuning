#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rebuild_dataset.py —— 训练集重建器（风格清洗 + 来源配比 + 去重 + 划分）

为什么需要它
------------
SFT 学的是「token 分布」，不是「道理」。两个真实教训：

1. 「……」开头泛滥
   ev3_all 里有 36.4% 的回复以「……」开头，最频繁的一条回复 literally 是「……嗯。」
   （出现 9 次）。模型于是学会：assistant 的第一个 token = 「……」。表现就是
   不管你问什么，她都用「……」起头。
   —— 这不是提示词能救的，只能改数据。本脚本把开头省略号的占比压到 --keep-ellipsis。

2. 能力占比失衡
   v4 里「助手/通用」样本只占 17.7%（按回复轮次），但 help_pack 自己写的建议是
   40%~50%。闲聊短句（均长 16.9 字）在数量上碾压了长回答，长度分布被拖回短句。
   —— 本脚本按「回复轮次占比」重新采样各来源，把配比变成显式参数。

用法
----
    # 只清洗，不改配比（看效果）
    python rebuild_dataset.py --src chat=my_data/ev3_all.json --out my_data/ev3_clean.json

    # 重建 v5：助手 45% / 闲聊 45% / 打招呼 10%
    python rebuild_dataset.py \
        --src help=my_data/help_api.json:0.45 \
        --src chat=my_data/ev3_all.json:0.45 \
        --src intro=my_data/intro_api.json:0.10 \
        --out my_data/ev5_all.json --keep-ellipsis 0.08

参数里 `:0.45` 是该来源的目标占比（按 assistant 轮次计），不给则只参与不配比。
总占比会自动归一化；来源样本不够时默认报错，加 --allow-repeat 才循环复用。
"""

import argparse
import collections
import hashlib
import json
import random
import re
import sys
from pathlib import Path

# 开头省略号的多种写法
ELLIPSIS_RE = re.compile(r"^(?:…+|\.{2,}|。{2,})")


# ---------------------------------------------------------------- 基础工具

def load_json(path):
    p = Path(path)
    if not p.exists():
        sys.exit(f"[x] 文件不存在: {path}")
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        sys.exit(f"[x] {path} 顶层不是数组")
    return data


def assistant_turns(ex):
    return [m for m in ex["conversations"] if m.get("role") == "assistant"]


def fingerprint(ex):
    joined = "\n".join(m["content"] for m in assistant_turns(ex))
    return hashlib.md5(re.sub(r"\s+", "", joined).encode()).hexdigest()


def stats(exs):
    """一组样本的统计：轮次 / 唯一回复 / 均长 / ……开头占比 / 长度分箱"""
    turns = [m["content"] for ex in exs for m in assistant_turns(ex)]
    if not turns:
        return dict(turns=0, uniq=0, avg=0.0, ell=0.0, bins={})
    lens = [len(t) for t in turns]
    ell = sum(1 for t in turns if ELLIPSIS_RE.match(t.strip()))
    bins = {
        "<10": sum(1 for x in lens if x < 10),
        "10-20": sum(1 for x in lens if 10 <= x < 20),
        "20-50": sum(1 for x in lens if 20 <= x < 50),
        ">=50": sum(1 for x in lens if x >= 50),
    }
    return dict(
        turns=len(turns),
        uniq=len(set(turns)),
        avg=round(sum(lens) / len(lens), 1),
        ell=round(ell * 100 / len(turns), 1),
        bins=bins,
    )


# ---------------------------------------------------------------- 自述修复
# 为什么单独修这一块：自述是**最高频**的问题之一，而 ev3 里 30 条自述样本
# 大多是敷衍腔（「E.V.。……就这样。」「你问这个做什么。」），
# 后者甚至违反人设卡（三·五② 明令禁止用反问挡回）。
# v5 忠实学会了这个毛病，真实对话里问「介绍一下你自己」就得到「E.V.。……就这样。」。
# 处理方式：**整条丢掉**，由 v6 包生成够多的好样本补回来。
# 不在原句上改词 —— 改出来的句子风格是伪的，模型学得更歪。
SELF_INTRO_RE = re.compile(
    r"^\s*(?:请问)?(?:你|您)?\s*(?:是谁|叫什么|叫啥|什么名字|名字是什么|"
    r"介绍一下自己|介绍下自己|介绍你自己|做个?自我介绍|自我介绍一下|"
    r"能做什么|能帮我做什么|会做什么|能干什么|能帮上什么忙)")
BRUSHOFF_RE = re.compile(r"问这个做什么|问那么多|不重要|没什么好说|没什么可说|别问|套什么话")


def repair_selfintro(exs):
    """丢掉「自述题 + 敷衍答」的整条样本。返回 (保留, 丢弃数)。"""
    keep, drop = [], 0
    for ex in exs:
        convs = ex.get("conversations") or []
        if len(convs) >= 2:
            u = str(convs[0].get("content", "")).strip()
            a = str(convs[1].get("content", "")).strip()
            # 只丢"短得没法接话"的：<15 字一律丢；带挡回语的再放宽到 <26 字。
            # 不能见 BRUSHOFF 就丢 —— "离这儿不远。平时不太出门，看书的时候多。
            # 你呢，忽然问这个做什么。" 这种先答后回问是好的，丢了可惜。
            if SELF_INTRO_RE.match(u) and (len(a) <= 14 or
                                           (BRUSHOFF_RE.search(a) and len(a) < 26)):
                drop += 1
                continue
        keep.append(ex)
    return keep, drop


# ---------------------------------------------------------------- 风格清洗

def strip_ellipsis(reply, rng, keep_rate):
    """去掉开头省略号；按 keep_rate 概率保留，让它回归「偶尔」而不是「默认」。"""
    s = reply.lstrip()
    if not ELLIPSIS_RE.match(s):
        return reply
    if rng.random() < keep_rate:
        return s                      # 保留（统一成「……」三个点，别混着来）
    return ELLIPSIS_RE.sub("", s).lstrip()


def clean_dataset(exs, keep_ellipsis, seed=0):
    """返回 (新样本列表, 统计信息)。只动 assistant 开头，不碰用户话和内容。"""
    rng = random.Random(seed)
    out = []
    changed = 0
    dropped = 0
    for ex in exs:
        convs = []
        bad = False
        for m in ex["conversations"]:
            c = str(m.get("content", ""))
            if m.get("role") == "assistant":
                new = strip_ellipsis(c, rng, keep_ellipsis)
                if new != c:
                    changed += 1
                c = new
                if not c.strip():
                    bad = True
                    break
            convs.append({"role": m["role"], "content": c})
        if bad:
            dropped += 1
            continue
        out.append({"conversations": convs})
    return out, dict(changed=changed, dropped=dropped)


def dedupe(exs):
    seen = set()
    out = []
    dup = 0
    for ex in exs:
        fp = fingerprint(ex)
        if fp in seen:
            dup += 1
            continue
        seen.add(fp)
        out.append(ex)
    return out, dup


# ---------------------------------------------------------------- 配比采样

def parse_sources(specs):
    """'name=path[:0.45]' -> dict(name, paths, weight)；多个 path 用 + 连接"""
    out = []
    for spec in specs:
        if "=" not in spec:
            sys.exit(f"[x] --src 格式应为 name=path[:weight]，收到: {spec}")
        name, rest = spec.split("=", 1)
        weight = None
        if ":" in rest and not re.match(r"^[A-Za-z]:[\\/]", rest):   # 避开 E:\ 这种盘符
            rest, w = rest.rsplit(":", 1)
            try:
                weight = float(w)
            except ValueError:
                sys.exit(f"[x] 占比不是数字: {w}")
        paths = [p.strip() for p in rest.split("+") if p.strip()]
        out.append(dict(name=name.strip(), paths=paths, weight=weight))
    return out


def rebalance(pools, allow_repeat, seed=0):
    """
    pools: [(name, [samples], weight)]
    按 assistant 轮次占比采样。权重全为空时直接拼接。

    思路：设每条样本平均贡献 ratio_i 轮回复，要 n_i*ratio_i = T*w_i。
    不重复时 T 的上界是 min_i(可用轮次_i / w_i)，取这个上界即可
    （会恰好把最少的那路用满）；允许重复时 T 取「现有轮次总和」。
    """
    if all(w is None for _, _, w in pools):
        return {name: list(items) for name, items, _ in pools}, {}

    weights = {}
    for name, _, w in pools:
        weights[name] = max(0.0, w or 0.0)
    total_w = sum(weights.values())
    if total_w <= 0:
        sys.exit("[x] 占比之和为 0 — 至少给一个来源加 :0.45 这样的权重")
    weights = {k: v / total_w for k, v in weights.items()}

    info = {}
    for name, items, _ in pools:
        if not items:
            sys.exit(f"[x] 来源 {name} 是空的")
        turns = sum(len(assistant_turns(x)) for x in items)
        info[name] = dict(items=len(items), turns=turns,
                          ratio=turns / len(items) if items else 1.0)

    # T = 目标总轮次
    limits = [info[n]["turns"] / weights[n] for n in weights if weights[n] > 0]
    T = sum(info[n]["turns"] for n in weights) if allow_repeat else min(limits)

    picked = {}
    for name, items, _ in pools:
        w = weights[name]
        if w <= 0:
            picked[name] = []
            continue
        need = max(1, int(round(T * w / (info[name]["ratio"] or 1.0))))
        if need > len(items):
            if not allow_repeat:
                sys.exit(
                    f"[x] 来源 {name} 样本不足：需要 {need} 条，只有 {len(items)} 条\n"
                    f"    → 先生成更多数据，或加 --allow-repeat 允许重复采样"
                )
        rng = random.Random(seed)
        if need <= len(items):
            picked[name] = rng.sample(items, need)
        else:                                   # 循环复用 + 打散
            reps = [x for _ in range(-(-need // len(items))) for x in items]
            rng.shuffle(reps)
            picked[name] = reps[:need]
    return picked, weights


# ---------------------------------------------------------------- 主流程

def main():
    ap = argparse.ArgumentParser(
        description="训练集重建：风格清洗 + 来源配比 + 去重 + 划分",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("用法\n----")[-1],
    )
    ap.add_argument("--src", action="append", required=True,
                    metavar="NAME=PATH[:WEIGHT]",
                    help="来源，可重复；带 :WEIGHT 才参与配比")
    ap.add_argument("--out", default="my_data/rebuilt.json",
                    help="输出合并集（会同时写 <out>_train.json / <out>_val.json）")
    ap.add_argument("--keep-ellipsis", type=float, default=0.08,
                    help="保留开头「……」的比例，默认 0.08（8%%）")
    ap.add_argument("--val-ratio", type=float, default=0.05, help="验证集比例，默认 0.05")
    ap.add_argument("--allow-repeat", action="store_true",
                    help="来源样本不足时允许循环复用（会重复，慎用）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-fix-selfintro", action="store_true",
                    help="保留「自述题 + 敷衍答」的样本（默认丢弃，见 repair_selfintro）")
    ap.add_argument("--dry-run", action="store_true", help="只打印报告，不写文件")
    args = ap.parse_args()

    sources = parse_sources(args.src)

    # 1) 载入 + 清洗
    print("=" * 64)
    print("① 清洗前")
    print("=" * 64)
    pools = []
    for s in sources:
        raw = []
        for p in s["paths"]:
            raw.extend(load_json(p))
        if len(s["paths"]) > 1:
            print(f"{s['name']:8s} 合并 {len(s['paths'])} 个文件 → {len(raw)} 条")
        before = stats(raw)
        cleaned, info = clean_dataset(raw, args.keep_ellipsis, args.seed)
        si = 0
        if not args.no_fix_selfintro:
            cleaned, si = repair_selfintro(cleaned)
        cleaned, dup = dedupe(cleaned)
        after = stats(cleaned)
        print(f"{s['name']:8s} 样本 {len(raw):4d} → {len(cleaned):4d}"
              f"（去重 {dup}，改开头 {info['changed']}，丢弃 {info['dropped']}"
              + (f"，敷衍自述 {si}" if si else "") + "）")
        print(f"         均长 {before['avg']:5.1f} → {after['avg']:5.1f}"
              f"　……开头 {before['ell']:5.1f}% → {after['ell']:5.1f}%")
        pools.append((s["name"], cleaned, s["weight"]))

    # 跨来源去重：同一条样本只归属一个来源（否则采样后再去重会把配比打乱）
    seen = set()
    deduped = []
    cross = 0
    for name, items, w in pools:
        keep = []
        for x in items:
            fp = fingerprint(x)
            if fp in seen:
                cross += 1
                continue
            seen.add(fp)
            keep.append(x)
        deduped.append((name, keep, w))
    pools = deduped
    if cross:
        print(f"（跨来源重复 {cross} 条，按 --src 顺序归属靠前的来源）")

    # 2) 配比
    picked, weights = rebalance(pools, args.allow_repeat, args.seed)
    merged = [x for name, _, _ in pools for x in picked[name]]
    merged_final, dup2 = dedupe(merged)
    random.Random(args.seed).shuffle(merged_final)

    print()
    print("=" * 64)
    print("② 配比后（按 assistant 轮次）")
    print("=" * 64)
    total_turns = sum(len(assistant_turns(x)) for x in merged_final) or 1
    for name, _, _ in pools:
        t = sum(len(assistant_turns(x)) for x in picked[name])
        print(f"{name:8s} 样本 {len(picked[name]):4d}　轮次 {t:5d}"
              f"　实际占比 {t * 100 / total_turns:5.1f}%"
              f"　目标 {(weights.get(name, 0) * 100):.0f}%")
    if dup2:
        print(f"（跨来源重复剔除 {dup2} 条）")
    print()
    final = stats(merged_final)
    print(f"合计 {len(merged_final)} 条样本 / {final['turns']} 轮回复　"
          f"唯一回复 {final['uniq']}　均长 {final['avg']} 字")
    print(f"开头省略号 {final['ell']}%　长度分布 {final['bins']}")

    # 3) 划分 + 写出
    n = len(merged_final)
    n_val = max(20, int(n * args.val_ratio))
    val = merged_final[:n_val]
    train = merged_final[n_val:]
    print(f"\n③ 划分：train {len(train)} / val {len(val)}")

    if args.dry_run:
        print("\n[dry-run] 不写文件")
        return

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(merged_final, ensure_ascii=False, indent=1), encoding="utf-8")
    out.with_name(out.stem + "_train.json").write_text(
        json.dumps(train, ensure_ascii=False, indent=1), encoding="utf-8")
    out.with_name(out.stem + "_val.json").write_text(
        json.dumps(val, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n[ok] {out}（+ _train.json / _val.json）")


if __name__ == "__main__":
    main()
