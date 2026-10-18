# -*- coding: utf-8 -*-
"""tts_server.py — 本地 Qwen3-TTS 语音合成服务（E.V. 的嗓子）

它和 design_voice.py 是两件事：
  design_voice.py   1.7B VoiceDesign，**一次性**，用文字造出音色 → ref.wav
  tts_server.py     0.6B Base + ref.wav 克隆，**常驻**，把任意文字念出来

为什么要单独起一个进程
----------------------
* 显存独立核算：它占 ~2GB，和 serve_hf(3.4GB) 并存；挂了不影响对话主链路
* 模型只加载一次：每次请求重新加载要几十秒，没法用
* 换音色只换 ref.wav + 重启它，server.py 一行都不用改

接口（故意做得极简，够 server.py 按句调用就行）
------------------------------------------------
    GET  /health          -> {"ok":true,"engine":"qwen3tts","ref_seconds":12.4,...}
    POST /tts  {"text":"…","language":"Chinese"}
         -> audio/wav 二进制（24kHz 16bit 单声道）

用法
----
    # 1) 下 0.6B Base（约 1.5GB，跑一次就够）
    bash setup_tts.sh --no-design          # 只补 Base 权重

    # 2) 起服务（独占 2GB 显存，可以和 serve_hf 同时跑）
    source ~/venvs/tts/bin/activate
    python app/tts_server.py                 # 默认 :8092

    # 3) 自检：合成一句存成 wav，不用开前端
    python app/tts_server.py --check

    # 4) 让语音助手用它
    TTS_BASE=http://127.0.0.1:8092 bash start_ev.sh

自检为什么重要
--------------
克隆模型最容易出问题的不是"不像"，而是**第一句哑**（prompt 没建好）或
**长句跑飞**。--check 会把参考音频、参考文本、生成时长都打出来。
"""

import argparse
import io
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
VOICE_JSON = os.path.join(HERE, "voice.json")

DEFAULT_MODEL = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
LOCAL_NAME = "Qwen3-TTS-12Hz-0.6B-Base"

_state = {"model": None, "prompt": None, "sr": 24000, "ref_seconds": 0.0,
          "ref_text": "", "model_id": ""}
_lock = threading.Lock()          # torch 推理不保证并发安全，串行化请求
TRIM = os.environ.get("TTS_TRIM", "1") != "0"      # 剪首尾静音（默认开）
GAP  = os.environ.get("TTS_GAP", "0") != "0"       # 压句中长停顿（默认关，见 _compress_gaps）


def _resolve_model(arg: str) -> str:
    if arg:
        return arg
    env = os.environ.get("QWEN_TTS_CLONE_MODEL")
    if env:
        return env
    local = os.path.join(ROOT, "models", LOCAL_NAME)
    if os.path.isdir(local) and os.listdir(local):
        return local
    return DEFAULT_MODEL


def _read_voice_cfg():
    cfg = {}
    if os.path.isfile(VOICE_JSON):
        try:
            cfg = json.load(open(VOICE_JSON, encoding="utf-8"))
        except Exception as e:                  # noqa: BLE001
            print(f"[警告] voice.json 解析失败: {e}", flush=True)
    ref = cfg.get("qwen_ref_audio") or "app/voice_lock/ref.wav"
    if not os.path.isabs(ref):
        ref = os.path.join(ROOT, ref)
    ref_text = (cfg.get("qwen_ref_text") or "").strip()
    return ref, ref_text, cfg


def load(model_arg, device, dtype_name, ref_audio, ref_text):
    import torch
    import soundfile as sf
    from qwen_tts import Qwen3TTSModel

    if not os.path.isfile(ref_audio):
        raise SystemExit(
            f"[错误] 参考音频不存在: {ref_audio}\n"
            "  先用 app/design_voice.py 造音色并 --lock 一个：\n"
            "    python app/design_voice.py --lock 3")
    if not ref_text:
        raise SystemExit("[错误] voice.json 里没有 qwen_ref_text —— 克隆需要参考文本")

    wav, sr = sf.read(ref_audio)
    dur = len(wav) / float(sr)
    print(f"[参考] {ref_audio}  {dur:.1f}s @ {sr}Hz", flush=True)
    print(f"[参考] 文本: {ref_text[:40]}…", flush=True)
    if dur < 3:
        print("[警告] 参考音频不足 3 秒，音色可能不稳", flush=True)
    if dur > 20:
        print("[提示] 参考音频超过 20 秒，建议截到 8~12 秒（越干净越稳）", flush=True)

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[dtype_name]

    if device.startswith("cuda") and torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        print(f"[GPU] 空闲 {free/2**30:.1f}GB / 共 {total/2**30:.1f}GB", flush=True)

    t0 = time.time()
    model_id = _resolve_model(model_arg)
    print(f"[加载] {model_id}", flush=True)
    model = Qwen3TTSModel.from_pretrained(model_id, device_map=device, dtype=dtype)
    print(f"[加载] 完成 {time.time() - t0:.1f}s", flush=True)

    # 建一次 clone prompt，之后每次合成复用 —— 这正是不用"每句重新提特征"的关键
    t1 = time.time()
    prompt = model.create_voice_clone_prompt(ref_audio=(wav, sr), ref_text=ref_text)
    print(f"[音色] clone prompt 就绪 {time.time() - t1:.1f}s", flush=True)

    _state.update({"model": model, "prompt": prompt, "sr": sr,
                   "ref_seconds": round(dur, 1), "ref_text": ref_text,
                   "model_id": model_id})
    return _state["sr"]


def _trim_silence(audio, sr, head_ms=40, tail_ms=60, thr_ratio=0.08):
    """剪掉首尾静音，保留一点点余量。

    为什么必须剪：实测 8 个字的短句，模型会在前面吐 1190ms 空白（ref.wav 只有 30ms）。
    server.py 是按句切分逐句推给前端的，一句 1.2 秒空白 → 三句话就是 3.6 秒死气，
    听起来像卡死。剪首尾（不动句中停顿）是确定性的，不依赖模型行为。
    """
    import numpy as np

    a = np.asarray(audio, dtype=np.float32).reshape(-1)
    if a.size == 0:
        return a
    fl = max(1, int(sr * 0.01))                 # 10ms 一帧
    nf = a.size // fl
    if nf < 3:
        return a
    env = np.sqrt((a[:nf * fl].reshape(nf, fl) ** 2).mean(axis=1))
    thr = float(env.max()) * thr_ratio
    idx = np.where(env > thr)[0]
    if idx.size == 0:
        return a                                 # 整段都静音？原样返回，让调用方去报错
    start = max(0, int(idx[0]) * fl - int(sr * head_ms / 1000))
    end = min(a.size, (int(idx[-1]) + 1) * fl + int(sr * tail_ms / 1000))
    return a[start:end]


def _compress_gaps(audio, sr, min_ms=600, target_ms=320, thr_ratio=0.08):
    """压缩句中长停顿：>=min_ms 的静音压到 target_ms，短的保持原样。

    为什么需要：克隆会继承参考音频的停顿长度。实测「嗯，在。有事就说。」里有两个
    840ms 停顿，8 个字占了 2.66 秒 —— 单句像"冷淡地顿一下"，但一段回答有三四句时，
    这些停顿会累积成好几秒死气，听起来像卡住。压到 320ms 仍然保留语气，只是不拖。
    """
    import numpy as np

    a = np.asarray(audio, dtype=np.float32).reshape(-1)
    if a.size == 0:
        return a
    fl = max(1, int(sr * 0.01))
    nf = a.size // fl
    if nf < 3:
        return a
    env = np.sqrt((a[:nf * fl].reshape(nf, fl) ** 2).mean(axis=1))
    quiet = env <= float(env.max()) * thr_ratio
    tgt = int(sr * target_ms / 1000)
    hard = int(sr * min_ms / 1000)

    out, i = [], 0
    while i < nf:
        j = i
        if quiet[i]:
            while j < nf and quiet[j]:
                j += 1
            run = (j - i) * fl
            out.append(a[i * fl: i * fl + (tgt if run >= hard else run)])
        else:
            while j < nf and not quiet[j]:
                j += 1
            out.append(a[i * fl: j * fl])
        i = j
    out.append(a[nf * fl:])                      # 不足一帧的尾巴
    return np.concatenate(out)


def synthesize(text: str, language: str = "Chinese"):
    """返回 (wav_bytes, sample_rate)。失败抛异常。"""
    import soundfile as sf

    text = (text or "").strip()
    if not text:
        raise ValueError("空文本")
    with _lock:
        t0 = time.time()
        wavs, sr = _state["model"].generate_voice_clone(
            text=text, language=language, voice_clone_prompt=_state["prompt"])
        dt = time.time() - t0
    raw_audio = wavs[0]
    audio = _trim_silence(raw_audio, sr) if TRIM else raw_audio
    if GAP:
        audio = _compress_gaps(audio, sr)
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
    data = buf.getvalue()
    print(f"[合成] {len(text):>3}字 {len(audio)/sr:>5.2f}s 音频 "
          f"用时 {dt:>4.2f}s  RTF {dt / max(len(audio) / sr, 1e-6):.2f}  "
          f"| 剪掉 {len(raw_audio)/sr - len(audio)/sr:>4.2f}s 首尾静音 | {text[:24]}",
          flush=True)
    return data, sr


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):          # 静音默认访问日志
        pass

    def _json(self, code, obj):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.rstrip("/") in ("/health", ""):
            self._json(200, {
                "ok": _state["model"] is not None,
                "engine": "qwen3tts",
                "model": _state["model_id"],
                "sample_rate": _state["sr"],
                "ref_seconds": _state["ref_seconds"],
            })
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        if self.path.rstrip("/") not in ("/tts", "/tts/"):
            self._json(404, {"error": "not found"})
            return
        try:
            n = int(self.headers.get("Content-Length", "0"))
            req = json.loads(self.rfile.read(n) or b"{}")
        except Exception as e:                  # noqa: BLE001
            self._json(400, {"error": f"bad request: {e}"})
            return

        if _state["model"] is None:
            self._json(503, {"error": "模型还没加载完"})
            return

        try:
            data, _sr = synthesize(req.get("text", ""),
                                   req.get("language", "Chinese"))
        except Exception as e:                  # noqa: BLE001
            self._json(500, {"error": f"{type(e).__name__}: {e}"})
            return

        self.send_response(200)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main():
    ap = argparse.ArgumentParser(description="E.V. 本地 TTS 服务（Qwen3-TTS 克隆）")
    ap.add_argument("--model", default="", help="0.6B Base 的 id 或本地目录")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=int(os.environ.get("TTS_PORT", "8092")))
    ap.add_argument("--ref", default="", help="覆盖 voice.json 里的参考音频")
    ap.add_argument("--ref-text", default="",
                    help="覆盖参考文本（换参考音频时必须同步改，否则音色会漂）")
    ap.add_argument("--check", action="store_true", help="合成一句自检后退出")
    ap.add_argument("--no-trim", action="store_true",
                    help="不剪首尾静音（默认剪；模型短句会在前面吐 ~1.2s 空白）")
    ap.add_argument("--compress-gaps", action="store_true",
                    help="压句中 >=600ms 的长停顿到 320ms（默认关）")
    args = ap.parse_args()

    global TRIM, GAP
    if args.no_trim:
        TRIM = False
    if args.compress_gaps:
        GAP = True

    ref_audio, ref_text, _cfg = _read_voice_cfg()
    if args.ref:
        ref_audio = args.ref
    if args.ref_text:
        ref_text = args.ref_text
    load(args.model, args.device, args.dtype, ref_audio, ref_text)

    if args.check:
        sample = "嗯，在。有事就说。"
        t0 = time.time()
        data, sr = synthesize(sample)
        out = os.path.join(HERE, "voice_lock", "check.wav")
        with open(out, "wb") as f:
            f.write(data)
        print(f"\n[自检] {len(data)/1024:.0f}KB / {sr}Hz / 总耗时 {time.time() - t0:.2f}s")
        print(f"[自检] 已保存 {out} —— 打开听一下，这就是以后 E.V. 的声音")
        return

    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"\nTTS 服务: http://{args.host}:{args.port}", flush=True)
    print(f"  让语音助手用它: TTS_BASE=http://{args.host}:{args.port} bash start_ev.sh\n",
          flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")


if __name__ == "__main__":
    main()
