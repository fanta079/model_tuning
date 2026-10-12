# -*- coding: utf-8 -*-
"""wechat_paste_to_json.py — 把微信 PC 端「复制」出来的文本转成训练用 JSON

为什么需要它
------------
批量导出微信记录的工具（PyWxDump 等）原项目已删库、现存同名仓库多为仿冒钓鱼，
所以更安全的做法是：在微信 PC 端手动复制对话 -> 粘贴到文本文件 -> 用本脚本转格式。

你不需要"手写"这些 txt —— 直接从微信复制粘贴即可，格式由微信生成。
常见粘贴格式（本脚本都能识别）：
    Fanta 2026-10-11 17:20        <- 昵称 + 完整日期时间
    Fanta 昨天 17:20              <- 昵称 + 相对日期
    Fanta 下午5:20                <- 昵称 + 上下午式时间
    Fanta 17:20                   <- 昵称 + 仅时间
    Fanta：                        <- 昵称 + 冒号（无时间）

用法
----
    # 1) 先探测：核对昵称/内容有没有切错（不产出文件）
    python wechat_paste_to_json.py chat1.txt --me "你的昵称" --probe

    # 2) 单个文件转换
    python wechat_paste_to_json.py chat1.txt --me "你的昵称" --out my_data/all.json

    # 3) 批量转换：整个目录下的 txt 合并成一个 all.json
    python wechat_paste_to_json.py my_data/ --me "你的昵称"

    # 4) 自测（看内置样例的转换效果）
    python wechat_paste_to_json.py --selftest

产出格式（可直接喂给 prepare_data.py / train_lora.py）:
    [{"conversations": [{"role": "user", "content": "..."},
                        {"role": "assistant", "content": "..."}]}]

角色映射：**别人说的话 = user（输入），你说的话 = assistant（要学的目标）**
"""

import argparse
import datetime as dt
import glob
import json
import os
import re
import sys
from collections import Counter

# 行头模式，按「信息量从多到少」排序，先匹配到的先用
SPEAKER_PATTERNS = [
    # 昵称 2026-10-11 17:20[:38]   /  2026/10/11 17:20
    re.compile(r"^\s*(?P<name>.{1,30}?)\s+(?P<time>\d{4}[-/]\d{1,2}[-/]\d{1,2}\s+\d{1,2}:\d{2}(?::\d{2})?)\s*$"),
    # 昵称 昨天 17:20 / 今天 17:20 / 星期X 17:20
    re.compile(r"^\s*(?P<name>.{1,30}?)\s+(?P<time>(?:昨天|今天|前天|星期[一二三四五六日天])\s+\d{1,2}:\d{2})\s*$"),
    # 昵称 下午5:20 / 晚上 21:30
    re.compile(r"^\s*(?P<name>.{1,30}?)\s+(?P<time>(?:上午|下午|晚上|凌晨|中午)\s*\d{1,2}:\d{2})\s*$"),
    # 昵称 17:20
    re.compile(r"^\s*(?P<name>.{1,30}?)\s+(?P<time>\d{1,2}:\d{2}(?::\d{2})?)\s*$"),
    # 昵称：  （无时间）
    re.compile(r"^\s*(?P<name>.{1,30}?)\s*[:：]\s*$"),
]

# 纯占位内容（非文本消息），这类整条丢掉
PLACEHOLDER = re.compile(
    r"^\s*[\[【(（]?\s*(图片|照片|表情|动画表情|语音|视频|文件|位置|链接|"
    r"转账|红包|名片|音乐|小程序|视频号|聊天记录|拍一拍|系统消息)\s*[\]】)）]?\s*$"
)


def match_header(line):
    """返回 (name, time_str) 或 None"""
    for pat in SPEAKER_PATTERNS:
        m = pat.match(line)
        if m:
            return m.group("name").strip(), m.group("time").strip()
    return None


def parse_blocks(text):
    """把文本切成 [{name, time, lines[]}, ...] —— 两遍法，靠出现次数过滤误判。"""
    raw = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")

    # ---- 第一遍：统计候选昵称 ----
    cand = Counter()
    for line in raw:
        h = match_header(line)
        if h:
            cand[h[0]] += 1

    # 出现 >=2 次才算真昵称（防止正文里偶然匹配被当成行头）
    speakers = {n for n, c in cand.items() if c >= 2} or set(cand)

    # ---- 第二遍：切块 ----
    blocks, cur = [], None
    for line in raw:
        h = match_header(line)
        if h and h[0] in speakers:
            if cur and cur["lines"]:
                blocks.append(cur)
            cur = {"name": h[0], "time": h[1], "lines": []}
        elif cur is not None and line.strip():
            cur["lines"].append(line.rstrip())
    if cur and cur["lines"]:
        blocks.append(cur)

    return blocks, speakers


def block_messages(blocks):
    """把块整理成 [{name, time, content}]，并丢掉纯占位/空内容。"""
    msgs = []
    for b in blocks:
        content = "\n".join(b["lines"]).strip()
        if not content or PLACEHOLDER.match(content):
            continue
        msgs.append({"name": b["name"], "time": b["time"], "content": content})
    return msgs


def time_key(t):
    """尽力把时间转成可比较的 datetime；转不了就返回 None。"""
    if not t:
        return None
    s = t.replace("年", "-").replace("月", "-").replace("日", " ").strip()
    m = re.match(r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})\s+(\d{1,2}):(\d{2})(?::(\d{2}))?", s)
    if m:
        y, mo, d, h, mi, se = (int(x) if x else 0 for x in m.groups())
        return dt.datetime(y, mo, d, h, mi, se)
    m = re.match(r"(\d{1,2}):(\d{2})(?::(\d{2}))?$", s)
    if m:                                    # 只有时间：用今天补
        today = dt.date.today()
        return dt.datetime(today.year, today.month, today.day,
                           int(m.group(1)), int(m.group(2)))
    return None


def to_conversations(msgs, me, gap_minutes=30, min_turns=1):
    """合并同人连续发言 -> 按时间间隔切分 -> 映射角色。"""
    # 1) 合并同一个人连续说的
    merged = []
    for m in msgs:
        if merged and merged[-1]["name"] == m["name"]:
            merged[-1]["content"] += "\n" + m["content"]
        else:
            merged.append(dict(m))

    # 2) 按时间间隔切成会话
    sessions, cur = [], []
    for m in merged:
        if cur:
            t0, t1 = time_key(cur[-1]["time"]), time_key(m["time"])
            if t0 and t1 and (t1 - t0) > dt.timedelta(minutes=gap_minutes):
                sessions.append(cur)
                cur = []
        cur.append(m)
    if cur:
        sessions.append(cur)

    # 3) 映射角色 + 校验
    out, n_short = [], 0
    for s in sessions:
        conv = []
        for m in s:
            role = "assistant" if m["name"] == me else "user"
            if conv and conv[-1]["role"] == role:           # 理论上已合并，保险
                conv[-1]["content"] += "\n" + m["content"]
            else:
                conv.append({"role": role, "content": m["content"]})
        # 掐掉首尾不完整的部分（而不是整段丢弃）：
        #   开头若是 assistant -> 前面没有提问，删掉
        #   结尾若是 user      -> 这句没被回复，没有学习目标，删掉
        while conv and conv[0]["role"] == "assistant":
            conv.pop(0)
        while conv and conv[-1]["role"] == "user":
            conv.pop()
        if len(conv) < min_turns * 2 or not conv:
            n_short += 1
            continue
        out.append({"conversations": conv})
    return out, n_short


def analyze(text, me, gap_minutes, min_turns):
    """解析一段文本，返回 (conversations, stats)"""
    blocks, speakers = parse_blocks(text)
    msgs = block_messages(blocks)
    convs, n_short = to_conversations(msgs, me, gap_minutes, min_turns)
    total_turns = sum(len(c["conversations"]) // 2 for c in convs)
    stats = {
        "speakers": sorted(speakers),
        "msgs": len(msgs),
        "mine": sum(1 for m in msgs if m["name"] == me),
        "convs": len(convs),
        "turns": total_turns,
        "short": n_short,
        "sample": msgs[:5],
    }
    return convs, stats


def collect_files(paths):
    """把「文件 + 目录」的混合输入展开成待处理的 txt 列表。"""
    files = []
    for p in paths:
        if os.path.isdir(p):
            files += sorted(glob.glob(os.path.join(p, "*.txt")))
        elif os.path.isfile(p):
            files.append(p)
        else:
            sys.exit(f"[错误] 找不到: {p}")
    if not files:
        sys.exit("[错误] 没有可处理的 .txt 文件")
    return files


def main():
    ap = argparse.ArgumentParser(description="微信 PC 复制文本 -> 训练用 JSON（支持批量）")
    ap.add_argument("input", nargs="*", help="一个或多个 .txt，或一个目录（自动取里面的 *.txt）")
    ap.add_argument("--me", default="", help="你自己的昵称（必须和文本里完全一致）")
    ap.add_argument("--out", default="", help="输出 JSON 路径；目录输入时默认 <目录>/all.json")
    ap.add_argument("--gap-minutes", type=int, default=30, help="超过多少分钟算新会话，默认 30")
    ap.add_argument("--min-turns", type=int, default=1, help="每段最少几轮问答，默认 1")
    ap.add_argument("--probe", action="store_true", help="只探测识别结果，不写文件")
    ap.add_argument("--selftest", action="store_true", help="用内置样例自测")
    args = ap.parse_args()

    if args.selftest:
        print("--- 自测：内置样例（me = Fanta，即「要克隆的那个人」）---\n")
        convs, stats = analyze(SELFTEST, "Fanta", 30, 1)
        report("(内置样例)", stats)
        print(json.dumps(convs, ensure_ascii=False, indent=2))
        return

    if not args.input:
        ap.error("请提供 txt 文件或目录，或用 --selftest")
    if not args.me:
        ap.error('必须指定 --me "你的昵称"（否则无法判断哪句是你说的）')

    files = collect_files(args.input)

    # ---------------- 探测模式 ----------------
    if args.probe:
        for f in files:
            with open(f, encoding="utf-8-sig") as fh:
                _, stats = analyze(fh.read(), args.me, args.gap_minutes, args.min_turns)
            report(f, stats, show_sample=True)
        return

    # ---------------- 转换模式 ----------------
    all_convs, grand = [], {"msgs": 0, "mine": 0, "convs": 0, "turns": 0, "short": 0}
    for f in files:
        with open(f, encoding="utf-8-sig") as fh:
            convs, stats = analyze(fh.read(), args.me, args.gap_minutes, args.min_turns)
        report(f, stats)
        all_convs += convs
        for k in ("msgs", "mine", "convs", "turns", "short"):
            grand[k] += stats[k]

    if not all_convs:
        print("\n[警告] 一段都没切出来 —— 检查 --me 是否写对了昵称（要和文本里完全一致）")
        sys.exit(1)

    # 输出路径：目录输入默认 <目录>/all.json
    if args.out:
        out = args.out
    elif os.path.isdir(args.input[0]):
        out = os.path.join(args.input[0], "all.json")
    else:
        out = re.sub(r"\.[^.]+$", "", args.input[0]) + ".json"

    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(all_convs, fh, ensure_ascii=False, indent=2)

    print("=" * 60)
    print(f"合计: {len(files)} 个文件 / {grand['msgs']} 条消息"
          f"（你说的 {grand['mine']}）/ {grand['convs']} 段会话 / {grand['turns']} 轮问答")
    print(f"输出: {out}")
    print("=" * 60)
    print(f"下一步: python prepare_data.py {out} --out-dir {os.path.dirname(out) or '.'}")


def report(tag, s, show_sample=False):
    print("=" * 60)
    print(f"[{tag}]")
    print(f"  识别昵称 : {s['speakers']}")
    print(f"  消息条数 : {s['msgs']}  (你说的 {s['mine']})")
    print(f"  会话/问答: {s['convs']} 段 / {s['turns']} 轮   丢弃过短: {s['short']}")
    if s["msgs"] and s["mine"] == 0:
        print("  [警告] --me 指定的昵称在文本里没出现过，请核对拼写")
    if show_sample and s["sample"]:
        print("  前几条（核对有没有切错）:")
        for m in s["sample"]:
            print(f"    [{m['name']}] ({m['time']}) "
                  f"{m['content'].replace(chr(10), ' / ')[:50]}")
    print("=" * 60)


SELFTEST = """Fanta 2026-10-11 17:20
在吗？今天有空不

E.V. 2026-10-11 17:21
在的，怎么啦

Fanta 2026-10-11 17:22
晚上想去看电影，你去不去

E.V. 2026-10-11 17:23
去啊，看哪部

Fanta 2026-10-11 17:24
还没定，你有什么想看的

E.V. 2026-10-11 17:25
最近那个科幻的不错，评分挺高

Fanta 2026-10-11 18:30
那就它了，几点场

E.V. 2026-10-11 18:31
七点半的吧，我提前买票

E.V. 2026-10-11 18:31
顺便把爆米花也订了

Fanta 2026-10-11 18:32
哈哈哈好，你懂我
"""


if __name__ == "__main__":
    main()
