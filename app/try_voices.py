# -*- coding: utf-8 -*-
"""try_voices.py — E.V. 音色试听台

把同一句 E.V. 风格的话，用多组「音色 + 语速 + 音调」合成出来，
一次性生成一个可以点击播放的网页，你听完选一个填进 app/voice.json。

为什么音色要单独调
------------------
默认的 TTS 音色是"通用客服女声"——温柔、饱满、有起伏。
但 E.V. 的人设是**冷淡、克制、话少**。两者直接撞车。

调法就三个旋钮（edge-tts 能控制的全部）：
    voice   选哪个底子
    rate    语速，负值更慢 → 显得不在乎
    pitch   音调，负值更低 → 显得冷

用法
----
    python app/try_voices.py                      # 生成 8 组候选 + 试听网页
    python app/try_voices.py --text "……无聊。"     # 换一句话试
    python app/try_voices.py --only 3             # 只重生成第 3 组

产物
----
    app/voice_samples/01_xxx.mp3 ...
    app/voice_samples/index.html   <- 浏览器打开这个，逐个点着听
"""

import argparse
import asyncio
import json
import os

import edge_tts

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "voice_samples")

# 默认用一句最能体现人设的话来试音：
# 够短、够冷、而且带省略号 —— 一听就知道音色对不对
DEFAULT_TEXT = "……无聊。你自己不是已经有答案了吗。"

# 候选组合：同一个底子配不同参数 + 不同底子对照
CANDIDATES = [
    # (编号, 显示名, 说明, voice, rate, pitch)
    (1, "Xiaoxiao 原样",     "基准对照 —— 通用客服女声是什么样", "zh-CN-XiaoxiaoNeural",      "+0%",  "+0Hz"),
    (2, "Xiaoxiao 微冷",     "略降调放慢，日常自然",             "zh-CN-XiaoxiaoNeural",      "-10%", "-8Hz"),
    (3, "Xiaoxiao 冷",       "明显更冷更平 —— 推荐起点",         "zh-CN-XiaoxiaoNeural",      "-15%", "-15Hz"),
    (4, "Xiaoxiao 很冷",     "最冷淡的一组，注意别听成没感情",   "zh-CN-XiaoxiaoNeural",      "-22%", "-22Hz"),
    (5, "Xiaoyi 原样",       "另一个女声底子，对照用",           "zh-CN-XiaoyiNeural",        "+0%",  "+0Hz"),
    (6, "Xiaoyi 冷",         "活泼底子降下来会怎样",             "zh-CN-XiaoyiNeural",        "-15%", "-15Hz"),
    (7, "HsiaoChen 原样",    "台湾女声，底气更软",               "zh-TW-HsiaoChenNeural",     "+0%",  "+0Hz"),
    (8, "HsiaoChen 冷",      "软底子 + 降调",                    "zh-TW-HsiaoChenNeural",     "-15%", "-15Hz"),
]


async def synth(text, voice, rate, pitch, path):
    comm = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
    with open(path, "wb") as f:
        async for chunk in comm.stream():
            if chunk["type"] == "audio":
                f.write(chunk["data"])
    return os.path.getsize(path)


async def main_async(args):
    os.makedirs(OUT, exist_ok=True)
    items = CANDIDATES if not args.only else [c for c in CANDIDATES if c[0] == args.only]
    if not items:
        raise SystemExit(f"[错误] --only {args.only} 没有对应的候选组")

    print(f"试音文本：{args.text}\n")
    made = []
    for num, name, note, voice, rate, pitch in items:
        fn = f"{num:02d}_{voice.replace('zh-', '').replace('Neural', '')}_{rate}_{pitch}.mp3"
        fn = fn.replace("%", "").replace("+", "p").replace("-", "m")
        path = os.path.join(OUT, fn)
        try:
            size = await synth(args.text, voice, rate, pitch, path)
            print(f"  [{num}] {name:<16} {voice:<24} rate={rate:<5} pitch={pitch:<6} "
                  f"-> {fn}  ({size/1024:.0f} KB)")
            made.append((num, name, note, voice, rate, pitch, fn))
        except Exception as e:                  # noqa: BLE001
            print(f"  [{num}] {name:<16} 失败: {type(e).__name__}: {e}")

    # 生成试听网页
    rows = "\n".join(
        f"""  <div class="card">
    <div class="head"><b>[{n}] {name}</b><span class="note">{note}</span></div>
    <div class="meta">{voice}　rate={rate}　pitch={pitch}</div>
    <audio controls preload="none" src="{fn}"></audio>
  </div>""" for n, name, note, voice, rate, pitch, fn in made)

    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>E.V. 音色试听</title>
<style>
  body{{background:#f7f6f2;color:#2c2c2a;font:15px/1.6 system-ui,"Microsoft YaHei",sans-serif;
       max-width:760px;margin:0 auto;padding:32px 20px}}
  h1{{font-size:20px;margin:0 0 6px}}
  .tip{{color:#5f5e5a;font-size:13px;margin-bottom:22px}}
  .card{{background:#fff;border:1px solid #e2e0d8;border-radius:10px;
        padding:14px 16px;margin-bottom:12px}}
  .head{{display:flex;justify-content:space-between;align-items:baseline;gap:12px}}
  .note{{color:#888780;font-size:12.5px;font-weight:400}}
  .meta{{color:#888780;font-size:12px;margin:4px 0 10px;font-family:ui-monospace,monospace}}
  audio{{width:100%;height:34px}}
  code{{background:#f1efe8;padding:1px 5px;border-radius:4px;font-size:13px}}
</style></head><body>
<h1>E.V. 音色试听</h1>
<div class="tip">试音文本：<code>{args.text}</code><br>
  听完把选中那组的 <code>voice / rate / pitch</code> 填进 <code>app/voice.json</code>，重启服务生效。</div>
{rows}
</body></html>"""
    with open(os.path.join(OUT, "index.html"), "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\n试听网页：{os.path.join(OUT, 'index.html')}")
    print("提示：在 WSL 里可以直接把整个目录拷到 Windows 上用浏览器打开，")
    print("      或者把 app/voice_samples 用 VSCode 打开、右键 index.html -> Open with Live Server。")


def main():
    ap = argparse.ArgumentParser(description="E.V. 音色试听")
    ap.add_argument("--text", default=DEFAULT_TEXT, help="用来试音的句子")
    ap.add_argument("--only", type=int, default=0, help="只生成某一组（编号见脚本内 CANDIDATES）")
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
