#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
harvest_chatlog.py —— 把日常对话日志收割成微调数据

为什么不能「存下来就直接训」
--------------------------
真实对话是最好的数据源（分布就是你真实的说话方式），但原样灌进训练集有四个坑：

1. **自我强化**：模型答得不好 → 你原样存 → 再训 → 更差。必须有人工信号（rating）
   或至少过滤掉短到没内容的回复。
2. **STT 污染**（仅语音来源）：user 侧是 whisper 转的，错字/断句错会让「问题」和
   「回答」对不上。默认行为：语音来源照样收，但把可疑的标出来，可以用
   --only-src text 只收打字来源。
3. **分布稀释**：日常多是「你好 / 在吗 / 随便聊」，全量灌进去会把好不容易调出来的
   助手能力又冲淡（踩坑记录第 21 条的复现）。所以本脚本只管产出干净数据，
   配比交给 rebuild_dataset.py 的 --src 权重控制。
4. **上下文截断**：多轮里「那它呢」这种指代，单独拿出来训会让模型答非所问。
   本脚本只收**用户话说完整**的轮次（长度阈值 + 指代词黑名单）。

用法
----
    # 1) 先看能收多少（不写文件）
    python harvest_chatlog.py --dry-run

    # 2) 收割到 my_data/
    python harvest_chatlog.py --out my_data/chatlog_clean.json

    # 3) 当作一个新来源参与配比（建议 20%~30%，别喧宾夺主）
    python rebuild_dataset.py \
        --src help=my_data/help_api_v2.json:0.35 \
        --src chat=my_data/ev3_all.json:0.40 \
        --src real=my_data/chatlog_clean.json:0.25 \
        --out my_data/ev6_all.json

日志格式（server.py 的 _log_turn 写的，每行一条）
    {"ts": "2026-10-18T15:30:00", "src": "voice|text", "user": "...",
     "reply": "...", "ms": 3.1, "rating": null}
"""

import argparse
import collections
import hashlib
import json
import re
import sys
from pathlib import Path

# 指代词开头的话，单独拿出来没意义
PRONOUN_HEAD = re.compile(
    r"^(它|他|她|这个|那个|这|那|上面|刚才|刚刚|还有|然后|所以|但是|可是|那如果|接着)")
# 明显的 STT 噪声特征：连续重复单字 / 一堆标点 / 纯符号
NOISE = re.compile(r"(.)\1{3,}|^[，。？！、…\s]+$")


def load_jsonl(paths):
    rows, bad = [], 0
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    bad += 1
    return rows, bad


def fingerprint(u, r):
    return hashlib.md5(re.sub(r"\s+", "", u + "\x00" + r).encode()).hexdigest()


def keep(row, args):
    """返回 (是否保留, 丢弃原因)"""
    u = (row.get("user") or "").strip()
    r = (row.get("reply") or "").strip()

    if not u or not r:
        return False, "空"
    if r.startswith("<") or "请求失败" in r:
        return False, "报错"
    if row.get("rating") == 0:
        return False, "人工差评"
    if row.get("rating") is None and not args.keep_unrated:
        return False, "未打标"
    if len(r) < args.min_reply:
        return False, f"回复太短<{args.min_reply}"
    if len(u) < args.min_user:
        return False, f"提问太短<{args.min_user}"
    if args.only_src and row.get("src") != args.only_src:
        return False, f"来源={row.get('src')}"
    if args.drop_pronoun and PRONOUN_HEAD.match(u):
        return False, "指代开头（脱离上下文）"
    if NOISE.search(u) or NOISE.search(r):
        return False, "疑似转写噪声"
    return True, ""


def main():
    ap = argparse.ArgumentParser(
        description="收割对话日志 → 微调数据", epilog=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="my_data/chatlog", help="日志目录（默认 my_data/chatlog）")
    ap.add_argument("--out", default="my_data/chatlog_clean.json")
    ap.add_argument("--min-reply", type=int, default=6, help="回复至少几个字，默认 6")
    ap.add_argument("--min-user", type=int, default=2, help="提问至少几个字，默认 2")
    ap.add_argument("--only-src", default="", choices=["", "text", "voice"],
                    help="只收某种来源；语音有 STT 误差，求稳就用 text")
    ap.add_argument("--keep-unrated", action="store_true", default=True,
                    help="保留没打标的（默认保留）；--no-keep-unrated 则只收打过分的")
    ap.add_argument("--no-keep-unrated", dest="keep_unrated", action="store_false")
    ap.add_argument("--drop-pronoun", action="store_true", default=True,
                    help="丢掉「它/那个/然后」开头的（默认丢）")
    ap.add_argument("--no-drop-pronoun", dest="drop_pronoun", action="store_false")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    d = Path(args.dir)
    if not d.exists():
        sys.exit(f"[x] 没有日志目录 {d}\n    先用助手聊几句（server.py 会自动落盘）")
    files = sorted(d.glob("*.jsonl"))
    if not files:
        sys.exit(f"[x] {d} 里没有 .jsonl")

    rows, badjson = load_jsonl(files)
    print(f"[i] 读了 {len(files)} 个文件 / {len(rows)} 条记录"
          + (f"（跳过损坏行 {badjson}）" if badjson else ""))

    kept, reasons = [], collections.Counter()
    seen = set()
    for row in rows:
        ok, why = keep(row, args)
        if not ok:
            reasons[why] += 1
            continue
        fp = fingerprint(row["user"], row["reply"])
        if fp in seen:
            reasons["重复"] += 1
            continue
        seen.add(fp)
        kept.append(row)

    print(f"[i] 保留 {len(kept)} 条 / 丢弃 {sum(reasons.values())} 条")
    for why, n in reasons.most_common():
        print(f"      - {why}: {n}")
    if not kept:
        sys.exit("[x] 一条都没留下 —— 放宽条件（--min-reply / --keep-unrated），或先多聊几句")

    by_src = collections.Counter(r.get("src") for r in kept)
    lens = [len(r["reply"]) for r in kept]
    print(f"[i] 来源分布 {dict(by_src)}　均长 {sum(lens)/len(lens):.1f} 字"
          f"　最长 {max(lens)} 字")

    if args.dry_run:
        print("\n[dry-run] 不写文件。样例：")
        for r in kept[:3]:
            print(f"   你: {r['user'][:40]}")
            print(f"  E.V.: {r['reply'][:60]}")
        return

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    data = [{"conversations": [
        {"role": "user", "content": r["user"]},
        {"role": "assistant", "content": r["reply"]}]} for r in kept]
    out.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n[ok] {out}（{len(data)} 条）")
    print("     当新来源参与配比，建议占 20%~30%：")
    print(f"     --src real={args.out}:0.25")


if __name__ == "__main__":
    main()
