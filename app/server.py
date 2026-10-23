# -*- coding: utf-8 -*-
"""E.V. 语音对话后端（P1）

链路: 浏览器录音 -> faster-whisper 转文字 -> llama-server(E.V.) 流式生成 -> edge-tts 分段合成 -> 浏览器边收边播

特点
----
* **真流式**: LLM 每吐几个 token 就切句 -> 立刻 TTS -> 立刻推给浏览器播，
  不用等整段回答生成完（首字到出声约 0.4~1.2s）
* **可打断**: 前端 abort 请求后后端 generator 被取消，模型生成也停下（不会白烧显存）
* 半双工按钮 / 电话模式都复用同一条流式管道
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
import re
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path

import httpx
import edge_tts
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse

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

# ---- 换引擎开关：设了 TTS_BASE 就不走 edge-tts，改调本地 Qwen3-TTS 克隆服务 ----
#   TTS_BASE=http://127.0.0.1:8092 bash start_ev.sh
# 为什么是"服务"而不是在本进程里加载模型：
#   1) 显存独立核算，TTS 挂了不影响对话；2) 模型只加载一次；3) 换音色不用重启前端
TTS_BASE = os.environ.get("TTS_BASE", "").rstrip("/")
TTS_TIMEOUT = float(os.environ.get("TTS_TIMEOUT", "30"))
TTS_LANG = os.environ.get("TTS_LANG", "Chinese")

# ---- 运行时语音模式（可以在界面里改，不用重启）----
#   local : 本地 Qwen3-TTS 克隆（她要的声音，但 4060Ti 上实测 RTF≈4）
#   cloud : edge-tts（首包 0.3~0.5s，但不是这个音色）
#   off   : 不朗读，只出文字
# 为什么要有 off：打字的时候人本来就在看字，声音晚 8 秒反而是干扰。
# 默认值尊重原来的启动方式：起了本地服务就是 local，没起就是 cloud。
# 默认 cloud（edge-tts）。改这个默认值是有依据的：本地 Qwen3-TTS 0.6B 在
# 4060Ti 上实测 RTF≈4（19 个字要 9.2 秒），voice 回合的首段音频要等到 15 秒 ——
# 那不叫实时语音。edge-tts 首包 0.3~0.5s，先保证"能对话"。
# 想要她的克隆音色：TTS_MODE=local（start_ev.sh --tts 会自动这么设）。
_TTS_MODE_ENV = os.environ.get("TTS_MODE", "").strip().lower()
TTS_MODE = _TTS_MODE_ENV if _TTS_MODE_ENV in ("local", "cloud", "off") else "cloud"
if TTS_MODE == "local" and not TTS_BASE:
    TTS_MODE = "cloud"          # 没起本地服务还要 local，只能退回云端，别静默失败

# system 提示词放在 app/system.txt，随时可改，改完重启生效
SYSTEM_FILE = os.path.join(HERE, "system.txt")
SYSTEM_PROMPT = os.environ.get("EV_SYSTEM", "")
if not SYSTEM_PROMPT and os.path.isfile(SYSTEM_FILE):
    SYSTEM_PROMPT = open(SYSTEM_FILE, encoding="utf-8").read().strip()

MAX_HISTORY = int(os.environ.get("MAX_HISTORY", "12"))   # 保留最近几轮上下文

# ---- 流式切句参数（想调"多快开口"改这三个）----
MIN_SEG = int(os.environ.get("MIN_SEG", "4"))            # 少于这么多字不单独切（"嗯。"单念很怪）
MAX_SEG = int(os.environ.get("MAX_SEG", "42"))           # 一直没标点，最多攒这么长就硬切
EAGER_DELAY = float(os.environ.get("EAGER_DELAY", "0.38"))  # 首句等这么久还没标点 -> 先切给 TTS

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
    """把一句文字变成音频字节。引擎由运行时 TTS_MODE 决定。

    返回的是**完整一段**（mp3 或 wav）—— 前端用 decodeAudioData 解，
    格式无所谓，所以本地服务直接给 wav（少一层编码，还能少几十毫秒）。
    """
    if TTS_MODE == "local":
        if not TTS_BASE:
            raise RuntimeError("语音模式是 local，但没配 TTS_BASE（本地服务没起）")
        return await _tts_remote(text)
    if TTS_MODE == "off":
        raise RuntimeError("语音已关闭")
    return await _tts_cloud(text)


async def _tts_cloud(text: str) -> bytes:
    comm = edge_tts.Communicate(text, TTS_VOICE, rate=TTS_RATE,
                                pitch=TTS_PITCH, volume=TTS_VOLUME)
    chunks = [c["data"] async for c in comm.stream() if c["type"] == "audio"]
    if not chunks:
        raise RuntimeError("TTS 没有返回音频")
    return b"".join(chunks)


_tts_http: "httpx.AsyncClient | None" = None


def _tts_client() -> httpx.AsyncClient:
    """复用连接池：一轮回答要按句调 3~6 次，每次新建 client 会白加几十毫秒。"""
    global _tts_http
    if _tts_http is None:
        _tts_http = httpx.AsyncClient(timeout=TTS_TIMEOUT)
    return _tts_http


async def _tts_remote(text: str) -> bytes:
    r = await _tts_client().post(f"{TTS_BASE}/tts",
                                 json={"text": text, "language": TTS_LANG})
    if r.status_code != 200:
        raise RuntimeError(f"TTS 服务 {r.status_code}: {r.text[:120]}")
    if not r.content:
        raise RuntimeError("TTS 服务返回空音频")
    return r.content


# ---------------- 流式：切句 / LLM / SSE ----------------

# 硬标点：碰到就切（一句说完）
HARD_END = "。！？!?…\n"
# 软标点：句子太长时优先在这些位置下刀
SOFT_END = "，,、；;：:"
# 模板残留：模型偶尔把特殊 token / 角色标签吐出来
CUT_MARKS = ("<|im_end|>", "<|im_start|>", "\nuser", "\nassistant",
             "\nUser", "\nAssistant")


def clean_for_tts(t: str) -> str:
    """把「给人看的文本」变成「能念的文本」。

    括号里的舞台动作（（轻轻叹气））不该念出来 —— 这是人设数据里最容易
    带进来的东西，直接送进 TTS 会变成"括号轻轻叹气括号"。
    """
    t = re.sub(r"[（(【\[][^）)】\]]*[）)】\]]", "", t)   # 括号旁白
    t = re.sub(r"[*_`#>~|]+", "", t)                     # markdown 记号
    t = t.replace("——", "，").replace("—", "，")
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def take_sentence(buf: str):
    """从缓冲区切出"一句"。返回 (句子, 剩余)；切不出来返回 ('', buf)。

    三个下刀条件（按优先级）：
      1. 遇到。！？… 且已经攒够 MIN_SEG 字  -> 正常断句
      2. 超过 MAX_SEG 字还没句号            -> 退到最后一个逗号切，没有就硬切
      3. 其他                               -> 继续攒
    """
    for i, ch in enumerate(buf):
        if ch in HARD_END and i + 1 >= MIN_SEG:
            return buf[:i + 1].strip(), buf[i + 1:].lstrip()
    if len(buf) >= MAX_SEG:
        soft = max((buf.rfind(c) for c in SOFT_END), default=-1) + 1
        cut = soft if soft >= MIN_SEG else MAX_SEG
        return buf[:cut].strip(), buf[cut:].lstrip()
    return "", buf


def _cut_marks(s: str):
    """遇到模板残留就截断，返回 (截断后的文本, 是否该停)。"""
    for m in CUT_MARKS:
        if m in s:
            return s.split(m)[0], True
    return s, False


async def llm_stream(user_text: str):
    """异步产出 LLM 的增量文本。

    后端要是**不支持** stream（比如老版 llama-server、或某个只做非流式的代理），
    这里会自动降级成整包返回 —— 前端照样能用，只是没有逐字效果。
    """
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}] if SYSTEM_PROMPT else []
    msgs += _history[-MAX_HISTORY:]
    msgs.append({"role": "user", "content": user_text})
    payload = {"model": LLM_MODEL, "messages": msgs, "stream": True,
               "temperature": 0.75, "top_p": 0.9, "max_tokens": 200}

    async with httpx.AsyncClient(timeout=LLM_TIMEOUT) as c:
        async with c.stream("POST", f"{LLM_BASE}/chat/completions",
                            json=payload) as r:
            if r.status_code != 200:
                body = (await r.aread()).decode("utf-8", "ignore")[:300]
                raise RuntimeError(f"HTTP {r.status_code} {body}")

            ctype = r.headers.get("content-type", "")
            if "event-stream" not in ctype:
                raw = await r.aread()
                try:
                    txt = json.loads(raw)["choices"][0]["message"]["content"]
                except Exception as e:                      # noqa: BLE001
                    raise RuntimeError(f"后端返回无法解析（不是流式也不是标准 JSON）：{e}")
                print("[LLM] 后端不支持流式，已降级为整包返回", flush=True)
                if txt:
                    yield txt
                return

            async for line in r.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if not data:
                    continue
                if data == "[DONE]":
                    break
                try:
                    d = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if d.get("error"):
                    raise RuntimeError(str(d["error"])[:300])
                ch = (d.get("choices") or [{}])[0].get("delta", {}).get("content") or ""
                if ch:
                    yield ch


def sse(obj) -> str:
    return "data: " + json.dumps(obj, ensure_ascii=False) + "\n\n"


# 8GB 卡只够跑一份 4B。打断时旧请求可能还没完全退出，
# 用信号量保证同一时刻只有一路在生成 —— 否则两个 KV cache 直接把显存顶爆。
_LLM_LOCK = asyncio.Semaphore(1)


def _sse_response(gen) -> StreamingResponse:
    return StreamingResponse(gen, media_type="text/event-stream",
                             headers={"Cache-Control": "no-store",
                                      "X-Accel-Buffering": "no"})


async def _sse_error(msg: str) -> StreamingResponse:
    async def g():
        yield sse({"t": "error", "msg": msg})
    return _sse_response(g())


async def turn_stream(user_text: str, t0: float, t_stt: float, src: str = "voice"):
    """一轮完整流水线：LLM 流 -> 切句 -> TTS -> SSE 推给浏览器。

    关键设计：**TTS 和 LLM 并行**。LLM 还在生成第 3 句时，第 1 句的音频
    已经在浏览器里播了 —— 这就是"不像对讲机"的全部秘密。
    """
    out: asyncio.Queue = asyncio.Queue()

    async def emit(obj):
        await out.put(sse(obj))

    async def worker():
        seg_q: asyncio.Queue = asyncio.Queue(maxsize=4)   # 有界队列 -> 背压
        parts: list[str] = []
        stat = {"first_tok": 0.0, "first_audio": 0.0, "tts_err": "", "seg": 0}

        async def tts_worker():
            while True:
                seg = await seg_q.get()
                if seg is None:
                    return
                txt = clean_for_tts(seg)
                if not txt:
                    continue
                # 语音关掉时也要把队列吃掉：seg_q 有界（4），不消费会反压住 LLM 生成
                if TTS_MODE == "off":
                    continue
                try:
                    audio = await tts(txt)
                except Exception as e:                    # noqa: BLE001
                    stat["tts_err"] = f"{type(e).__name__}: {e}"
                    continue
                if not stat["first_audio"]:
                    stat["first_audio"] = time.time() - t0
                await emit({"t": "audio", "i": stat["seg"], "text": txt,
                            "b64": base64.b64encode(audio).decode()})
                stat["seg"] += 1

        tw = asyncio.create_task(tts_worker())
        reply = ""
        try:
            await emit({"t": "meta", "user_text": user_text,
                        "stt": round(t_stt, 2)})
            buf = ""
            t_seg = time.time()
            stop = False
            # 抢许可：8GB 卡只够一份 4B，打断后旧请求还没退干净就再来一轮会 OOM
            async with _LLM_LOCK:
                async for d in llm_stream(user_text):
                    if not stat["first_tok"]:
                        stat["first_tok"] = time.time() - t0
                    d, stop = _cut_marks(d)
                    if d:
                        parts.append(d)
                        await emit({"t": "delta", "d": d})
                        buf += d
                    while True:
                        seg, buf = take_sentence(buf)
                        if not seg:
                            break
                        await seg_q.put(seg)
                        t_seg = time.time()
                    # 首句兜底：模型爱铺垫（"这个嘛，其实……"），等不到句号就先出声。
                    # 优先切在逗号上（停顿自然），实在没有才硬切 —— 硬切会听着断气。
                    if not stat["first_audio"] and len(buf) >= MIN_SEG:
                        waited = time.time() - t_seg
                        if waited > EAGER_DELAY:
                            soft = max((buf.rfind(c) for c in SOFT_END), default=-1) + 1
                            cut = soft if soft >= MIN_SEG else 0
                            if not cut and waited > EAGER_DELAY * 2.4:
                                cut = min(len(buf), MAX_SEG)
                            if cut:
                                await seg_q.put(buf[:cut].strip())
                                buf, t_seg = buf[cut:].lstrip(), time.time()
                    if stop:
                        break
            if buf.strip():
                await seg_q.put(buf.strip())
            await seg_q.put(None)
            await tw

            reply = "".join(parts).strip()
            for m in CUT_MARKS:
                reply = reply.split(m)[0]
            reply = reply.strip()

            if reply:
                _history.append({"role": "user", "content": user_text})
                _history.append({"role": "assistant", "content": reply})

            _log_turn(user_text, reply, (time.time() - t0) * 1000, src)
            await emit({"t": "done", "reply": reply, "tts_error": stat["tts_err"],
                        "timing": {"stt": round(t_stt, 2),
                                   "first_token": round(stat["first_tok"], 2),
                                   "first_audio": round(stat["first_audio"], 2),
                                   "total": round(time.time() - t0, 2)}})
            print(f"[流式] 首字 {stat['first_tok']:.1f}s / 出声 {stat['first_audio']:.1f}s / "
                  f"总 {time.time() - t0:.1f}s / {stat['seg']} 段 | "
                  f"你: {user_text[:20]} | E.V.: {reply[:24]}", flush=True)
        except asyncio.CancelledError:
            # 用户打断 / 关页面：把已经说出去的部分记进上下文，别让历史断层
            part = "".join(parts).strip()
            if part:
                _history.append({"role": "user", "content": user_text})
                _history.append({"role": "assistant", "content": part})
            print(f"[流式] 已打断（已播 {stat['seg']} 段，保留 {len(part)} 字）",
                  flush=True)
            raise
        except Exception as e:                            # noqa: BLE001
            await emit({"t": "error",
                        "msg": f"模型连接失败（serve_hf.py 起了吗？）：{e}"})
        finally:
            if not tw.done():
                tw.cancel()
            out.put_nowait(None)      # 用 put_nowait：取消中的任务里 await 可能再次被打断

    async def run():
        task = asyncio.create_task(worker())
        try:
            while True:
                item = await out.get()
                if item is None:
                    break
                yield item
        finally:
            if not task.done():
                task.cancel()

    async for chunk in run():
        yield chunk


# ---------------- 对话落盘（为下一轮微调攒真实数据） ----------------
# 为什么必须落盘：前端日志只活在浏览器内存里，刷新即失。而「真实使用分布」是
# 比 API 生成好得多的微调数据 —— 但只有存下来才有得筛。见 docs/踩坑记录.md。
CHATLOG_ON = os.environ.get("CHATLOG", "1") != "0"
CHATLOG_DIR = Path(os.environ.get("CHATLOG_DIR", os.path.join(ROOT, "my_data", "chatlog")))


def _log_turn(user_text: str, reply: str, ms: float, src: str) -> None:
    """把一轮问答追加进当日 jsonl。绝不因为写日志而搞坏一轮对话。

    注意 ms 是**毫秒**。早期版本把秒直接塞进了这个字段（日志里 ms=4.2），
    读老日志时别按毫秒解释；harvest_chatlog.py 不用这个字段，所以不影响训练数据。
    """
    if not CHATLOG_ON or not reply:
        return
    try:
        CHATLOG_DIR.mkdir(parents=True, exist_ok=True)
        rec = {"ts": datetime.now().isoformat(timespec="seconds"),
               "src": src,              # voice | text
               "user": user_text,
               "reply": reply,
               "ms": round(ms, 2),
               "rating": None}          # 留给人打标：1=好 0=不好；不标也不影响筛选
        p = CHATLOG_DIR / f"{datetime.now():%Y-%m-%d}.jsonl"
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception as e:                  # noqa: BLE001
        print(f"[chatlog] 写日志失败（不影响对话）: {e}", flush=True)


# ---------------- 路由 ----------------

async def _respond(user_text: str, t0: float, t_stt: float, src: str = "voice"):
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

    _log_turn(user_text, reply, (time.time() - t0) * 1000, src)
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


# ---------------- PWA：手机"添加到主屏幕"要的两样东西 ----------------
# manifest 决定名字/图标/是否全屏（standalone = 没有地址栏，看着就是 App）
_ICONS = {"192", "512"}


@app.get("/manifest.webmanifest")
async def manifest():
    return FileResponse(os.path.join(HERE, "manifest.webmanifest"),
                        media_type="application/manifest+json")


@app.get("/icon-{size}.png")
async def icon(size: str):
    if size not in _ICONS:
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(os.path.join(HERE, f"icon-{size}.png"),
                        media_type="image/png")


@app.post("/api/tts_mode")
async def set_tts_mode(body: dict):
    """运行时切语音引擎：local / cloud / off。改完立刻生效，不用重启、不用重新加载模型。"""
    global TTS_MODE
    mode = str(body.get("mode", "")).strip().lower()
    if mode not in ("local", "cloud", "off"):
        return JSONResponse({"ok": False, "error": "mode 只能是 local/cloud/off"}, 400)
    if mode == "local" and not TTS_BASE:
        return JSONResponse({"ok": False, "error": "本地 TTS 没启动（没配 TTS_BASE）"}, 400)
    TTS_MODE = mode
    print(f"[TTS] 模式切换 -> {mode}", flush=True)
    return {"ok": True, "mode": TTS_MODE}


@app.get("/api/status")
async def status():
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

    # TTS：走本地服务时真的探一下活，不然它挂了界面还显示绿的
    tts_ok, tts_msg = True, f"edge {TTS_VOICE} rate={TTS_RATE} pitch={TTS_PITCH}"
    if TTS_MODE == "off":
        tts_msg = "不朗读（只出文字）"
    elif TTS_MODE == "local" and TTS_BASE:
        try:
            async with httpx.AsyncClient(timeout=2) as c:
                r = await c.get(f"{TTS_BASE}/health")
                d = r.json() if r.status_code == 200 else {}
            tts_ok = bool(d.get("ok"))
            if tts_ok:
                name = os.path.basename(str(d.get("model", "")).rstrip("/"))
                tts_msg = (f"qwen3tts {name} "
                           f"ref {d.get('ref_seconds', '?')}s")
            else:
                tts_msg = "模型还没加载完" if r.status_code == 200 else f"HTTP {r.status_code}"
        except Exception as e:                  # noqa: BLE001
            tts_ok, tts_msg = False, f"连不上 {TTS_BASE}（{type(e).__name__}）"

    return {
        "stt": {"ok": model is not None, "msg": err or f"{WHISPER_MODEL}",
                "loading": loading},
        "llm": {"ok": llm_ok, "msg": llm_msg, "base": LLM_BASE},
        "tts": {"ok": tts_ok, "msg": tts_msg, "base": TTS_BASE or "edge-tts",
                "mode": TTS_MODE, "has_local": bool(TTS_BASE)},
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
    """打字入口（非流式，留给老调用方/curl 调试）。新前端请用 /api/stream/text。"""
    text = (payload or {}).get("text", "").strip()
    if not text:
        return JSONResponse({"ok": False, "error": "空消息"}, status_code=400)
    if len(text) > 2000:
        return JSONResponse({"ok": False, "error": "太长了，说重点"}, status_code=400)
    return await _respond(text, time.time(), 0.0, src="text")


# ---------------- 流式入口（前端用的就是这两个） ----------------

@app.post("/api/stream/turn")
async def stream_turn(audio: UploadFile = File(...)):
    """一轮对话的流式版：音频进 -> SSE 事件流（meta / delta / audio / done）。

    注意：STT 这一段仍然是阻塞的（whisper 是同步库），用 to_thread 丢到线程池，
    不挡事件循环 —— 否则 TTS / 其他请求会被一起卡住。
    """
    t0 = time.time()
    raw = await audio.read()
    if len(raw) < 1200:
        return await _sse_error("录音太短，再说一次？")

    ext = os.path.splitext(audio.filename or "")[1] or ".webm"
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
        f.write(raw)
        tmp = f.name
    try:
        user_text = await asyncio.to_thread(stt, tmp)
    except Exception as e:                            # noqa: BLE001
        return await _sse_error(f"语音识别失败：{e}")
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass

    if not user_text:
        return await _sse_error("没听清，再说一次？")
    return _sse_response(turn_stream(user_text, t0, time.time() - t0))


@app.post("/api/stream/text")
async def stream_text(payload: dict):
    """打字入口的流式版：跳过 STT，直接 LLM 流 -> TTS 流。"""
    text = (payload or {}).get("text", "").strip()
    if not text:
        return await _sse_error("空消息")
    if len(text) > 2000:
        return await _sse_error("太长了，说重点")
    return _sse_response(turn_stream(text, time.time(), 0.0, src="text"))


if __name__ == "__main__":
    import uvicorn
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8090"))
    # HTTPS 开关：麦克风只在安全上下文可用（localhost 或 https）。
    # 手机通过局域网 IP 访问时，必须走 https，否则 getUserMedia 直接不可用。
    tls_cert = os.environ.get("TLS_CERT", "")
    tls_key = os.environ.get("TLS_KEY", "")
    scheme = "https" if (tls_cert and tls_key) else "http"
    print(f"E.V. 语音服务: {scheme}://{host}:{port}  （浏览器打开这个地址）")
    if scheme == "https":
        print(f"  TLS  : {tls_cert}")
        print(f"         手机首次访问要点「继续前往」，并把证书装成受信任 CA")
    else:
        like_lan = os.environ.get("HOST", "") not in ("127.0.0.1", "localhost")
        if like_lan:
            print("  提示 : 用局域网 IP 打开时麦克风会被浏览器禁用（非安全上下文）")
            print("         要语音就用 https（TLS_CERT/TLS_KEY），或改用 localhost")
    print(f"  STT  : faster-whisper {WHISPER_MODEL} @ {WHISPER_DEVICE}")
    print(f"  LLM  : {LLM_BASE}")
    print(f"  TTS  : {TTS_VOICE}")
    if scheme == "https":
        uvicorn.run(app, host=host, port=port, log_level="warning",
                    ssl_certfile=tls_cert, ssl_keyfile=tls_key)
    else:
        uvicorn.run(app, host=host, port=port, log_level="warning")
