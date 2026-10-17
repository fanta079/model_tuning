# -*- coding: utf-8 -*-
"""E.V. 语音对话后端（P1）

链路: 浏览器录音 -> faster-whisper 转文字 -> llama-server(E.V.) 生成回复 -> edge-tts 合成语音 -> 浏览器播放

特点
----
* 半双工: 一轮说完才允许下一轮（由前端按钮锁保证，见 app/index.html）
* 无 agent 能力: 纯对话，不接工具调用
* 组件可缺省: STT / TTS / LLM 各自独立，哪个没起会在 /api/status 里标出来

启动
----
    bash run_app.sh            # 推荐（会一起检查依赖）

或用环境变量覆盖默认值:
    LLM_BASE=http://127.0.0.1:8080/v1 WHISPER_MODEL=medium python app/server.py
"""

import asyncio
import base64
import json
import os
import tempfile
import threading
import time

import httpx
import edge_tts
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

# ---- 兼容补丁：PyAV >= 14 删掉了 av.open(metadata_errors=...) ----
# faster-whisper 1.2.x 解码时仍在传这个参数（它只声明 av>=11，没有上界），
# 于是第一次识别就报：open() got an unexpected keyword argument 'metadata_errors'
# 这里把这个参数吞掉；根治办法是把 av 固定在 12~13（setup_app.sh 已加约束）。
try:
    import inspect
    import av as _av

    if "metadata_errors" not in inspect.signature(_av.open).parameters:
        _av_open_orig = _av.open

        def _av_open_compat(*args, **kwargs):
            kwargs.pop("metadata_errors", None)
            return _av_open_orig(*args, **kwargs)

        _av.open = _av_open_compat
        print(f"[STT] 已启用 PyAV 兼容补丁（av {_av.__version__} 无 metadata_errors）", flush=True)
except Exception:                                # noqa: BLE001
    pass

# ---------------- 配置（全部可用环境变量覆盖） ----------------
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

LLM_BASE = os.environ.get("LLM_BASE", "http://127.0.0.1:8080/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "local")
LLM_TIMEOUT = float(os.environ.get("LLM_TIMEOUT", "120"))

WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "large-v3")
WHISPER_DEVICE = os.environ.get("WHISPER_DEVICE", "cuda")
WHISPER_COMPUTE = os.environ.get("WHISPER_COMPUTE", "int8_float16")
# 这句是让 Whisper 输出简体中文的关键，别删
WHISPER_PROMPT = os.environ.get(
    "WHISPER_PROMPT", "以下是普通话的句子，请用简体中文输出。")

# ---- 音色配置 ----
# 优先级: 环境变量 > app/voice.json > 下面的默认值
# 先用 `python app/try_voices.py` 试听，选定后把结果填进 app/voice.json
VOICE_FILE = os.path.join(HERE, "voice.json")
_vcfg = {}
if os.path.isfile(VOICE_FILE):
    try:
        _vcfg = json.load(open(VOICE_FILE, encoding="utf-8"))
    except Exception as e:                      # noqa: BLE001
        print(f"[TTS] voice.json 解析失败，用默认音色: {e}", flush=True)

TTS_VOICE = os.environ.get("TTS_VOICE") or _vcfg.get("voice", "zh-CN-XiaoxiaoNeural")
TTS_RATE = os.environ.get("TTS_RATE") or _vcfg.get("rate", "-12%")
TTS_PITCH = os.environ.get("TTS_PITCH") or _vcfg.get("pitch", "-10Hz")
TTS_VOLUME = os.environ.get("TTS_VOLUME") or _vcfg.get("volume", "+0%")

# system 提示词放在 app/system.txt，随时可改，改完重启生效
SYSTEM_FILE = os.path.join(HERE, "system.txt")
SYSTEM_PROMPT = os.environ.get("EV_SYSTEM", "")
if not SYSTEM_PROMPT and os.path.isfile(SYSTEM_FILE):
    SYSTEM_PROMPT = open(SYSTEM_FILE, encoding="utf-8").read().strip()

MAX_HISTORY = int(os.environ.get("MAX_HISTORY", "12"))   # 保留最近几轮上下文

app = FastAPI(title="E.V. Voice")


@app.on_event("startup")
def _warmup():
    """后台加载 whisper —— 不占住 /api/status，否则前端会一直显示"准备中"。"""
    threading.Thread(target=get_whisper, daemon=True).start()

# 模型懒加载：第一次请求时才载入，避免启动就吃显存
_whisper = None
_whisper_err = ""
_history: list[dict] = []


def get_whisper():
    """返回 (model, err)。载入失败不抛异常，便于前端显示原因。

    GPU 起不来（典型：libcublas.so.12 缺失、cuDNN 版本不对）时**自动退回 CPU**，
    不要因为一个可选加速把小助手整体搞挂。
    """
    global _whisper, _whisper_err, WHISPER_DEVICE, WHISPER_COMPUTE
    if _whisper is not None or _whisper_err:
        return _whisper, _whisper_err

    from faster_whisper import WhisperModel

    tries = [(WHISPER_DEVICE, WHISPER_COMPUTE)]
    if WHISPER_DEVICE != "cpu":
        tries.append(("cpu", "int8"))          # 兜底：CPU 永远能用

    last_err = ""
    for dev, comp in tries:
        try:
            print(f"[STT] 载入 whisper {WHISPER_MODEL} ({dev}/{comp}) ...", flush=True)
            _whisper = WhisperModel(WHISPER_MODEL, device=dev, compute_type=comp)
            WHISPER_DEVICE, WHISPER_COMPUTE = dev, comp   # 记录实际生效的设备
            print(f"[STT] 就绪（{dev}）", flush=True)
            return _whisper, ""
        except Exception as e:                 # noqa: BLE001
            last_err = f"{type(e).__name__}: {e}"
            print(f"[STT] {dev} 载入失败: {last_err}", flush=True)
            if dev != "cpu":
                print("[STT] 自动退回 CPU（GPU 缺 CUDA 12 运行库？见 run_app.sh 提示）",
                      flush=True)

    _whisper_err = last_err
    return None, _whisper_err


# ---------------- 三个组件 ----------------

def stt(audio_path: str) -> str:
    global _whisper, _whisper_err, WHISPER_DEVICE, WHISPER_COMPUTE

    def _run(model):
        segs, _ = model.transcribe(
            audio_path, language="zh", beam_size=5,
            vad_filter=True, initial_prompt=WHISPER_PROMPT,
        )
        return "".join(s.text for s in segs).strip()

    model, err = get_whisper()
    if model is None:
        raise RuntimeError(f"语音识别不可用：{err}")

    try:
        return _run(model)
    except Exception as e:                  # noqa: BLE001
        # 有些机器「模型加载」能过，真正推理时才炸（缺 libcublas / 驱动不匹配）
        # 这里再兜一次：丢掉 GPU 模型，换 CPU 重建，重跑同一段音频
        if WHISPER_DEVICE == "cpu":
            raise
        print(f"[STT] 推理失败：{type(e).__name__}: {e}", flush=True)
        print("[STT] 切换 CPU 重建模型后重试……", flush=True)
        _whisper, _whisper_err = None, ""
        WHISPER_DEVICE, WHISPER_COMPUTE = "cpu", "int8"
        model, err = get_whisper()
        if model is None:
            raise RuntimeError(f"语音识别不可用：{err}")
        return _run(model)


async def llm(user_text: str) -> str:
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}] if SYSTEM_PROMPT else []
    msgs += _history[-MAX_HISTORY:]
    msgs.append({"role": "user", "content": user_text})
    async with httpx.AsyncClient(timeout=LLM_TIMEOUT) as c:
        r = await c.post(f"{LLM_BASE}/chat/completions", json={
            "model": LLM_MODEL, "messages": msgs,
            "temperature": 0.75, "top_p": 0.9, "max_tokens": 200,
        })
        r.raise_for_status()
        reply = r.json()["choices"][0]["message"]["content"].strip()
    # 清掉可能残留的模板标记
    for t in ("<|im_end|>", "<|im_start|>"):
        reply = reply.replace(t, "")
    return reply.strip()


async def tts(text: str) -> bytes:
    comm = edge_tts.Communicate(text, TTS_VOICE, rate=TTS_RATE,
                                pitch=TTS_PITCH, volume=TTS_VOLUME)
    chunks = [c["data"] async for c in comm.stream() if c["type"] == "audio"]
    if not chunks:
        raise RuntimeError("TTS 没有返回音频")
    return b"".join(chunks)


# ---------------- 路由 ----------------

async def _respond(user_text: str, t0: float, t_stt: float):
    """从"用户文本"往后走完：LLM -> TTS -> 记上下文 -> 返回。语音和打字两条入口共用。"""
    try:
        reply = await llm(user_text)
    except Exception as e:                      # noqa: BLE001
        return JSONResponse(
            {"ok": False, "user_text": user_text,
             "error": f"模型连接失败（serve_hf.py 起了吗？）：{e}"}, status_code=502)
    t_llm = time.time() - t0

    audio_b64, tts_err = "", ""
    try:
        audio_b64 = base64.b64encode(await tts(reply)).decode()
    except Exception as e:                      # noqa: BLE001
        tts_err = f"{type(e).__name__}: {e}"    # 静默降级：只回文字

    _history.append({"role": "user", "content": user_text})
    _history.append({"role": "assistant", "content": reply})

    print(f"[轮次] STT {t_stt:.1f}s / LLM {t_llm - t_stt:.1f}s / "
          f"总 {time.time() - t0:.1f}s | 你: {user_text[:30]} | E.V.: {reply[:30]}",
          flush=True)
    return {"ok": True, "user_text": user_text, "reply_text": reply,
            "audio_b64": audio_b64, "tts_error": tts_err,
            "timing": {"stt": round(t_stt, 1), "llm": round(t_llm - t_stt, 1),
                       "total": round(time.time() - t0, 1)}}


@app.get("/", response_class=HTMLResponse)
async def index():
    p = os.path.join(HERE, "index.html")
    html = open(p, encoding="utf-8").read()
    # 页面里带构建时间戳，改完一眼能确认浏览器拿到的不是缓存
    stamp = time.strftime("%m-%d %H:%M:%S", time.localtime(os.path.getmtime(p)))
    # 禁掉缓存：否则改了前端、刷新还是旧逻辑，排查时会被误导
    return HTMLResponse(html.replace("__BUILD__", stamp),
                        headers={"Cache-Control": "no-store, max-age=0"})


@app.get("/api/status")
async def status():
    # 不在这里阻塞地加载模型 —— 前端要秒回状态，否则按钮会一直"准备中"
    model, err = _whisper, _whisper_err
    loading = model is None and not err
    llm_ok, llm_msg = False, ""
    try:
        async with httpx.AsyncClient(timeout=5) as c:
            r = await c.get(f"{LLM_BASE}/models")
            llm_ok = r.status_code == 200
            llm_msg = "就绪" if llm_ok else f"HTTP {r.status_code}"
    except Exception as e:                      # noqa: BLE001
        llm_msg = f"{type(e).__name__}"
    return {
        "stt": {"ok": model is not None, "msg": err or f"{WHISPER_MODEL}",
                "loading": loading},
        "llm": {"ok": llm_ok, "msg": llm_msg, "base": LLM_BASE},
        "tts": {"ok": True, "msg": f"{TTS_VOICE} rate={TTS_RATE} pitch={TTS_PITCH}"},
        "system_prompt": SYSTEM_PROMPT[:120],
        "turns": len(_history) // 2,
    }


@app.post("/api/reset")
async def reset():
    _history.clear()
    return {"ok": True}


@app.post("/api/turn")
async def turn(audio: UploadFile = File(...)):
    """一轮对话：音频进 -> 音频出。半双工，调用方需自行保证串行。"""
    t0 = time.time()
    raw = await audio.read()
    if len(raw) < 1200:
        return JSONResponse({"ok": False, "error": "录音太短，再说一次？"}, status_code=400)

    # 1) 语音 -> 文字（存临时文件，让 PyAV 按扩展名解码）
    ext = os.path.splitext(audio.filename or "")[1] or ".webm"
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
        f.write(raw)
        tmp = f.name
    try:
        user_text = stt(tmp)
    except Exception as e:                      # noqa: BLE001
        return JSONResponse({"ok": False, "error": f"语音识别失败：{e}"}, status_code=500)
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass

    if not user_text:
        return JSONResponse({"ok": False, "error": "没听清，再说一次？"}, status_code=400)
    t_stt = time.time() - t0

    return await _respond(user_text, t0, t_stt)


@app.post("/api/text")
async def text_turn(payload: dict):
    """打字入口：跳过 STT，直接走 LLM -> TTS。麦克风不可用时用它验证整条链路。"""
    text = (payload or {}).get("text", "").strip()
    if not text:
        return JSONResponse({"ok": False, "error": "空消息"}, status_code=400)
    if len(text) > 2000:
        return JSONResponse({"ok": False, "error": "太长了，说重点"}, status_code=400)
    return await _respond(text, time.time(), 0.0)


if __name__ == "__main__":
    import uvicorn
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8090"))
    print(f"E.V. 语音服务: http://{host}:{port}  （浏览器打开这个地址）")
    print(f"  STT  : faster-whisper {WHISPER_MODEL} @ {WHISPER_DEVICE}")
    print(f"  LLM  : {LLM_BASE}")
    print(f"  TTS  : {TTS_VOICE}")
    uvicorn.run(app, host=host, port=port, log_level="warning")
