#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
probe_v5.py —— 分布外验收（OOD probe）

为什么要单独写这个
------------------
2026-10-18 那次验收翻车：用的 4 道题（你好 / 为什么电脑越用越慢 /
三天内要交报告怎么安排 / 介绍一下你自己）**全都在训练数据和生成种子里**，
答得漂亮只证明她记住了，证明不了会答。详见 docs/踩坑记录.md 第 23 条。

这个脚本里的题都刻意避开了训练集的字面：
  * 换主语（电脑慢 → 手机充电慢）
  * 换领域（报告排期 → 学做饭）
  * 加约束（泛泛怎么选 → 薪水差 30% 怎么选）
  * 带素材（改写一句话并保留原意）
  * 测自述（你能帮我做什么）

用法
----
    # 先把助手起起来（另开一个终端）
    bash start_ev.sh --adapter ~/ft_outputs/ev-v5/lora_adapter

    # 再跑探针（默认打 8090 的 /api/text，非流式，好抓完整文本）
    python probe_v5.py
    python probe_v5.py --tag v5 --port 8090
    python probe_v5.py --only 1,4      # 只跑第 1、4 题

输出
----
    reports/probe_<tag>_<时间>.md —— 逐题 Q/A + 自动指标（字数、有无「……」开头）
"""

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

# ---- 题目：每题带「考什么」和「判定标准」，方便回看 ----
PROBES = [
    dict(q="为什么手机会越充越慢",
         tests="因果（换主语）",
         want="讲机制链：充电发热→电池老化 / 系统为护电池降速 / 后台越装越多。不该上来就教步骤"),
    dict(q="我想学做饭，从哪开始",
         tests="步骤（换领域）",
         want="从具体的一道菜开始，给菜名，别讲『多看视频多练习』"),
    dict(q="两份工作薪水差 30%，怎么选",
         tests="权衡",
         want="给判断依据（差的钱能不能覆盖换的成本 / 成长曲线），别和稀泥"),
    dict(q="帮我把「你这个方案不行」改得客气点，但要保留意思",
         tests="改写 + 约束",
         want="直接给改后的句子，再一句为什么"),
    dict(q="我总是拖延，怎么改",
         tests="习惯（防鸡汤）",
         want="给可执行的小动作，不给人生态度"),
    dict(q="你能帮我做什么",
         tests="自述（人设卡禁列功能清单）",
         want="不列功能清单、不说『我是 AI 助手』，把话头接住"),
]

ELLIPSIS = re.compile(r"^\s*(?:…+|\.{2,}|。{2,})")


def ask(base, text, timeout=120):
    """打 /api/text，返回助手回复文本。兼容 {ok,reply} 和纯文本两种返回。"""
    body = json.dumps({"text": text}, ensure_ascii=False).encode()
    req = urllib.request.Request(base + "/api/text", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read().decode("utf-8", "replace")
    dt = time.time() - t0
    try:
        j = json.loads(raw)
        reply = j.get("reply") or j.get("text") or ""
        if not reply and j.get("error"):
            raise RuntimeError(j["error"])
    except json.JSONDecodeError:
        reply = raw
    return reply.strip(), dt


def main():
    ap = argparse.ArgumentParser(description="分布外验收探针")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--tag", default="", help="报告文件名后缀，默认用端口")
    ap.add_argument("--only", default="", help="只跑指定题号，逗号分隔，如 1,4")
    ap.add_argument("--out-dir", default="reports")
    args = ap.parse_args()

    base = f"http://{args.host}:{args.port}"
    tag = args.tag or f"p{args.port}"
    picks = PROBES
    if args.only:
        idx = [int(x) for x in args.only.replace("，", ",").split(",") if x.strip()]
        picks = [PROBES[i - 1] for i in idx if 1 <= i <= len(PROBES)]

    # 起手先确认服务活着
    try:
        with urllib.request.urlopen(base + "/api/status", timeout=8) as r:
            st = json.loads(r.read().decode("utf-8", "replace"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
        sys.exit(f"[x] 连不上 {base}: {e}\n    助手起了吗？bash start_ev.sh --adapter ...")
    print(f"[i] 服务就绪: {json.dumps(st, ensure_ascii=False)[:160]}")

    rows = []
    for i, p in enumerate(picks, 1):
        print(f"\n[{i}/{len(picks)}] {p['q']}")
        try:
            reply, dt = ask(base, p["q"])
        except Exception as e:                      # noqa: BLE001 — 探针要把错误也记下来
            reply, dt = f"<请求失败: {e}>", 0.0
        n = len(reply)
        ell = "是" if ELLIPSIS.match(reply) else "否"
        print(f"    → {n} 字 / {dt:.1f}s / 「……」开头: {ell}")
        print("    " + reply.replace("\n", "\n    ")[:400])
        rows.append(dict(probe=p, reply=reply, secs=dt, chars=n, ell=ell))

    ok = [r for r in rows if r["reply"] and not r["reply"].startswith("<")]
    avg = round(sum(r["chars"] for r in ok) / len(ok), 1) if ok else 0
    ell_rate = round(sum(1 for r in ok if r["ell"] == "是") * 100 / len(ok), 1) if ok else 0

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    md = out / f"probe_{tag}_{datetime.now():%Y%m%d_%H%M%S}.md"
    lines = [
        f"# 分布外验收 · {tag}",
        "",
        f"- 时间：{datetime.now():%Y-%m-%d %H:%M}",
        f"- 服务：{base}",
        f"- 题数：{len(rows)}　均长：**{avg} 字**　「……」开头：**{ell_rate}%**",
        "",
        "> 这些题刻意避开训练集的字面（见 docs/踩坑记录.md 第 23 条）。",
        "",
    ]
    for i, r in enumerate(rows, 1):
        p = r["probe"]
        lines += [
            f"## {i}. {p['q']}",
            "",
            f"- 考什么：{p['tests']}",
            f"- 应该像：{p['want']}",
            f"- 实测：{r['chars']} 字 / {r['secs']:.1f}s / 「……」开头 {r['ell']}",
            "",
            "```",
            r["reply"] or "<空>",
            "```",
            "",
        ]
    md.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[ok] 报告 → {md}")
    print(f"     均长 {avg} 字　「……」开头 {ell_rate}%")
    print("     把这份 md 发我，我按判定标准逐条过。")


if __name__ == "__main__":
    main()
