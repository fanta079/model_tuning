# -*- coding: utf-8 -*-
"""bench_tts.py — 量一遍"从按下发送到听见声音"的每一段耗时

为什么单独写一个
----------------
"语音慢"是个很模糊的描述。实际链路有四个可能变慢的地方：

    文字首字  <- LLM 首个 token
    首段音频  <- 切句 + 第一次 TTS 调用（本地 0.6B 的大头在这）
    段间隔    <- 后续每句的 TTS 是否跟得上播放
    总时长    <- 整体是不是被 LLM 拖住

只有把四个数分开量，才知道该动哪一段。凭感觉优化必然白干。

用法
----
    # 只测 TTS 引擎本身（直连 8092，不走 LLM）
    python app/bench_tts.py --tts-only

    # 测一整轮文字对话（默认）
    python app/bench_tts.py

    # 换句子 / 多跑几次取中位数
    python app/bench_tts.py --text "今天有点困，但是还有一堆事没做完。" --repeat 3

在 Windows 上也能跑（只用标准库 + HTTP），不用进 WSL。
"""

import argparse
import json
import statistics
import sys
import time
import urllib.request

APP = "http://127.0.0.1:8090"
TTS = "http://127.0.0.1:8092"


def _post_json(url, obj, timeout=180):
    req = urllib.request.Request(url, data=json.dumps(obj).encode(),
                                 headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


def bench_tts(sentences, repeat):
    """直连 TTS 服务：每句调一次，记录墙钟时间与音频长度，算 RTF。"""
    print("\n== TTS 引擎（直连 %s）==" % TTS)
    try:
        h = json.loads(urllib.request.urlopen(TTS + "/health", timeout=5).read())
    except Exception as e:
        print("  连不上 TTS 服务：%s" % e)
        print("  先起：bash start_ev.sh --tts")
        return
    print("  引擎 %s  参考音频 %ss  模型 %s" % (h.get("engine"), h.get("ref_seconds"),
                                              str(h.get("model"))[-46:]))
    print("  %-24s %6s %8s %8s %7s" % ("文本", "字数", "音频", "用时", "RTF"))
    rows = []
    for _ in range(repeat):
        for s in sentences:
            t0 = time.time()
            try:
                r = _post_json(TTS + "/tts", {"text": s})
                data = r.read()
            except Exception as e:
                print("  合成失败：%s" % e)
                continue
            dt = time.time() - t0
            # wav: 24kHz 16bit 单声道 -> 字节数 / 48000 = 秒
            sec = max(0.0, (len(data) - 44) / 48000.0)
            rows.append((len(s), sec, dt))
            print("  %-24s %6d %7.2fs %7.2fs %7.2f" % (s[:12], len(s), sec, dt, dt / max(sec, 1e-6)))
    if len(rows) >= 2:
        # 用最小二乘估一下固定开销：dt ≈ a + b*字数
        n = len(rows)
        xs = [r[0] for r in rows]; ys = [r[2] for r in rows]
        mx, my = sum(xs) / n, sum(ys) / n
        den = sum((x - mx) ** 2 for x in xs) or 1e-9
        b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den
        a = my - b * mx
        print("  —— 拟合：每次调用固定 %.2fs + 每字 %.2fs（这就是短句也慢的原因）" % (a, b))


def bench_turn(text, repeat):
    """走完整链路：/api/stream/text，记录首字 / 首音 / 每段音频到达时刻 / 结束。"""
    print("\n== 一整轮对话（%s）==" % APP)
    agg = []
    for i in range(repeat):
        t0 = time.time()
        marks = {}
        audio_at = []
        try:
            r = _post_json(APP + "/api/stream/text", {"text": text})
            for raw in r:
                line = raw.decode("utf-8", "ignore").strip()
                if not line.startswith("data:"):
                    continue
                try:
                    ev = json.loads(line[5:].strip())
                except Exception:
                    continue
                t = ev.get("t")
                dt = time.time() - t0
                if t == "delta":
                    marks.setdefault("首字", dt)
                elif t == "audio":
                    audio_at.append(dt)
                elif t == "done":
                    marks["结束"] = dt
                    marks["timing"] = ev.get("timing")
        except Exception as e:
            print("  请求失败：%s" % e)
            return
        marks["首音"] = audio_at[0] if audio_at else float("nan")
        print("  第 %d 轮：首字 %.2fs  首音 %.2fs  音频段 %d  结束 %.2fs"
              % (i + 1, marks["首字"], marks["首音"], len(audio_at), marks["结束"]))
        if len(audio_at) > 1:
            gaps = [round(audio_at[k + 1] - audio_at[k], 2) for k in range(len(audio_at) - 1)]
            print("          段间隔 %s" % gaps)
        if marks.get("timing"):
            print("          服务端 timing %s" % marks["timing"])
        agg.append(marks)
    if len(agg) > 1:
        print("  —— 中位数：首字 %.2fs  首音 %.2fs"
              % (statistics.median([m["首字"] for m in agg]),
                 statistics.median([m["首音"] for m in agg if m["首音"] == m["首音"]] or [0])))



def bench_voice(wav_path, repeat):
    """完整语音回合：把 wav 当成麦克风录音 POST 给 /api/stream/turn。

    没有麦克风也能测 —— 这条路径和真说话走的是同一个端点、同一套流水线，
    唯一的差别是省掉了浏览器端的 VAD 判停时间（那部分要真说话才量得到）。
    """
    import uuid
    print("\n== 语音回合（%s）==" % APP)
    print("   音频 %s" % wav_path)
    boundary = "----ev" + uuid.uuid4().hex
    raw = open(wav_path, "rb").read()
    head = ("--%s\r\nContent-Disposition: form-data; name=\"audio\"; filename=\"voice.wav\"\r\n"
            "Content-Type: audio/wav\r\n\r\n" % boundary).encode()
    tail = ("\r\n--%s--\r\n" % boundary).encode()
    body = head + raw + tail

    for i in range(repeat):
        t0 = time.time()
        req = urllib.request.Request(APP + "/api/stream/turn", data=body,
                                     headers={"Content-Type": "multipart/form-data; boundary=" + boundary})
        marks, audio_at = {}, []
        try:
            r = urllib.request.urlopen(req, timeout=180)
            for line in r:
                line = line.decode("utf-8", "ignore").strip()
                if not line.startswith("data:"):
                    continue
                try:
                    ev = json.loads(line[5:].strip())
                except Exception:
                    continue
                dt = time.time() - t0
                if ev.get("t") == "meta":
                    marks["听成"] = ev.get("user_text", "")
                elif ev.get("t") == "delta":
                    marks.setdefault("首字", dt)
                elif ev.get("t") == "audio":
                    audio_at.append(dt)
                elif ev.get("t") == "done":
                    marks["结束"] = dt
                    marks["timing"] = ev.get("timing")

        except Exception as e:
            print("  请求失败：%s" % e)
            return
        print("  第 %d 轮：首字 %.2fs  首音 %.2fs  音频段 %d  结束 %.2fs"
              % (i + 1, marks.get("首字", 0), audio_at[0] if audio_at else float("nan"),
                 len(audio_at), marks.get("结束", 0)))
        if marks.get("听成"):
            print("          识别为：%s" % marks["听成"])
        if marks.get("timing"):
            print("          服务端 timing %s" % marks["timing"])

def main():
    ap = argparse.ArgumentParser(description="E.V. 语音链路耗时基准")
    ap.add_argument("--text", default="今天有点困，但是还有一堆事没做完。")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--tts-only", action="store_true", help="只测 TTS 引擎，不走 LLM")
    ap.add_argument("--turn-only", action="store_true", help="只测完整回合")
    ap.add_argument("--voice", nargs="?", const="app/voice_lock/check.wav", default=None,
                    metavar="WAV", help="把一段 wav 当作你说的话，测完整语音回合（默认用内置样例）")
    args = ap.parse_args()

    if args.voice:
        bench_voice(args.voice, args.repeat)
    else:
        sentences = ["嗯，在。", args.text, "这个嘛，我觉得可以先把手头那件事收个尾，剩下的明天再说。"]
        if not args.turn_only:
            bench_tts(sentences, args.repeat)
        if not args.tts_only:
            bench_turn(args.text, args.repeat)
    print("\n判读指南")
    print("  语音回合 = VAD 判停(约 0.85s，浏览器端) + 识别 + 首字 + 首段 TTS")
    print("  首音 < 1.5s  流畅；1.5~3s 能接受；> 4s 就是「文字读完声音才来」")
    print("  固定开销 > 2s：短句也慢，问题在模型/加载路径，不在切句")
    print("  RTF > 1：合成比播放还慢，句子一多必然越拖越后")


if __name__ == "__main__":
    sys.exit(main())
