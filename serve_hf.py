# -*- coding: utf-8 -*-
"""serve_hf.py — 用 Unsloth(PyTorch + GPU) 提供 OpenAI 兼容接口

为什么需要它
------------
`llama.cpp` 你编的是 **CPU 版**，4B 模型只有 3~8 token/s，语音对话太慢。
重编 CUDA 版要装 3GB 的 CUDA Toolkit —— 但你不需要：

    ~/venvs/ft 里已经有 torch + CUDA + Unsloth（训练时验证过能用）

所以直接用 PyTorch 在 GPU 上跑，对外暴露和 llama-server 一模一样的接口。
P1 的 app/server.py 只要把 LLM_BASE 指过来就行，**一行代码都不用改**。

跟 llama.cpp 的取舍
------------------
| | llama.cpp (CUDA版) | 本脚本 (PyTorch) |
|---|---|---|
| 显存 | ~2.6GB (Q4) | ~4GB (4bit) |
| 速度 | 最快 | 够用（20~40 tok/s）|
| 启动 | 秒开 | 30~60 秒（加载模型）|
| 需要编译 | ✅ 要 CUDA Toolkit | ❌ 不需要 |

**先用这个把 P1 跑通**，哪天有空再编 CUDA 版替换 —— 接口一样，换过去零改动。

用法（在训练环境里跑）
----------------------
    source ~/venvs/ft/bin/activate
    python serve_hf.py --adapter /home/fanta/ft_outputs/ev-v2/lora_adapter

    # 换端口 / 换成跑底座（不加 LoRA）
    python serve_hf.py --adapter ... --port 8081
    python serve_hf.py --base

验证
----
    curl http://127.0.0.1:8080/v1/models
    curl http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' \
      -d '{"model":"ev","messages":[{"role":"user","content":"我今天很累"}]}'
"""

import argparse
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer
from unsloth import FastLanguageModel

HERE = os.path.dirname(os.path.abspath(__file__))

SYSTEM = (
    "你是 E.V.，一个能帮上忙的聊天助手；性格冷淡、话少、毒舌，但心里在意对方。"
    # 定位：人设是语气，不是拒答的理由（用户明确：核心是助手，人设是附加）
    "先解决问题，再保持人设。对方问事情就把事情讲清楚，不要为了冷淡只回半句。"
    "长度按事情大小：打招呼/闲聊/情绪一到两句；解释原因、给方案、教步骤三到五句。"
    "默认只说「你」，只在郑重时刻才叫「Fanta」。"
    "不用 emoji、不用感叹号、不寒暄、不说客套话。"
    # ↓ 打招呼 / 自我介绍的约束（v2 实测：这两个场景最容易滑回助手腔）
    "别人只打招呼就自然应一声（「嗯，在。怎么了。」「在。」），不要展开自我介绍，"
    "也不要写「坐吧 / 坐下 / 喝水」这类舞台动作（这是对话不是剧本）。"
    "被问你是谁、介绍一下自己，只说「E.V.。……就这样。」，"
    "绝不说自己是 AI / 助手 / 语言模型，绝不列功能清单。"
    "被问你能干什么，不列功能，回「看情况。」"
    # ↓ 这两句是 v2 实测补的：v2 把"回避"泛化成了万能挡箭牌，
    #   连「你是谁」都回「你问这个做什么」。用它做软约束兜一层。
    "被问到自己是谁、叫什么，直接回答（我是 E.V.），不许反问。"
    "只有对方真的问到你的过去时才回避，一场对话里最多反问一次。"
)

_model = None
_tok = None
_eos = []
_pad = 0
_max_new = 256


def load(adapter: str, base: str, max_seq: int):
    global _model, _tok, _eos, _pad
    path = adapter or base
    print(f"[加载] {path}", flush=True)
    t0 = time.time()
    model, tok = FastLanguageModel.from_pretrained(
        model_name=path, max_seq_length=max_seq, dtype=None, load_in_4bit=True)
    FastLanguageModel.for_inference(model)
    tok = getattr(tok, "tokenizer", tok)
    eos = [tok.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>")]
    _eos = [i for i in eos if isinstance(i, int) and i >= 0]
    _pad = tok.pad_token_id if tok.pad_token_id is not None else (_eos[0] if _eos else 0)
    _model, _tok = model, tok
    print(f"[加载] 完成，用时 {time.time() - t0:.1f}s", flush=True)


def render(messages):
    try:
        return _tok.apply_chat_template(messages, add_generation_prompt=True,
                                        tokenize=False, enable_thinking=False)
    except Exception:                                   # noqa: BLE001
        return _tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)


def generate(messages, max_new, temperature, top_p):
    prompt = render(messages)
    ids = _tok(prompt, return_tensors="pt").to(_model.device)
    t0 = time.time()
    with torch.inference_mode():
        out = _model.generate(
            **ids, max_new_tokens=max_new, do_sample=temperature > 0,
            temperature=max(temperature, 0.01), top_p=top_p,
            eos_token_id=_eos or None, pad_token_id=_pad)
    new = out[0][ids["input_ids"].shape[1]:]
    txt = _tok.decode(new, skip_special_tokens=True)
    for m in ("<|im_end|>", "\nuser", "\nassistant", "\nUser", "\nAssistant",
              "<|im_start|>"):
        if m in txt:
            txt = txt.split(m)[0]
    dt = time.time() - t0
    n = len(new)
    print(f"[生成] {n} tokens / {dt:.1f}s = {n / dt:.1f} tok/s", flush=True)
    return txt.strip()


class _StopOnEvent(StoppingCriteria):
    """客户端断开了（打断/关页面）就让 generate 尽快停，别继续烧 GPU。"""

    def __init__(self, ev: threading.Event):
        self.ev = ev

    def __call__(self, input_ids, scores, **kwargs):    # noqa: ARG002
        return self.ev.is_set()


def generate_stream(messages, max_new, temperature, top_p, stop_ev=None):
    """边生成边吐文本。用 TextIteratorStreamer + 后台线程实现。

    为什么要自己攒 acc：Qwen 的分词是字节级 BPE，单个 token 解码出来
    可能是半个汉字（`\\ufffd`），直接发出去前端就会闪一下乱码。
    """
    prompt = render(messages)
    ids = _tok(prompt, return_tensors="pt").to(_model.device)
    streamer = TextIteratorStreamer(_tok, skip_prompt=True, skip_special_tokens=True)
    kw = dict(
        max_new_tokens=max_new, do_sample=temperature > 0,
        temperature=max(temperature, 0.01), top_p=top_p,
        eos_token_id=_eos or None, pad_token_id=_pad, streamer=streamer)
    if stop_ev is not None:
        kw["stopping_criteria"] = StoppingCriteriaList([_StopOnEvent(stop_ev)])

    err = {}

    def work():
        try:
            with torch.inference_mode():
                _model.generate(**ids, **kw)
        except Exception as e:                          # noqa: BLE001
            err["e"] = e
        finally:
            try:
                streamer.end()
            except Exception:                           # noqa: BLE001
                pass

    th = threading.Thread(target=work, daemon=True)
    th.start()

    acc = ""
    for chunk in streamer:
        acc += chunk
        if "\ufffd" in chunk:            # 半个汉字，等下一块补齐
            continue
        if acc:
            yield acc
            acc = ""
    if acc:
        yield acc
    th.join(timeout=1.0)
    if "e" in err:
        raise err["e"]


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):                  # 静音默认日志
        pass

    def _json(self, code, obj):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # ---------------- SSE / chunked ----------------
    def _chunk(self, data: bytes):
        """手写 chunked 分块。BaseHTTPRequestHandler 不会自动加，必须自己来。"""
        self.wfile.write(b"%X\r\n" % len(data) + data + b"\r\n")
        self.wfile.flush()

    def _sse_headers(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")     # 有反代时别缓冲
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

    def _sse(self, obj):
        payload = json.dumps(obj, ensure_ascii=False)
        self._chunk(("data: " + payload + "\n\n").encode())

    def _stream(self, msgs, max_new, temperature, top_p):
        stop_ev = threading.Event()
        t0 = time.time()
        n_char = 0
        self._sse_headers()
        try:
            for text in generate_stream(msgs, max_new, temperature, top_p, stop_ev):
                if not text:
                    continue
                n_char += len(text)
                self._sse({"choices": [{"index": 0, "delta": {"content": text}}]})
        except Exception as e:                          # noqa: BLE001
            stop_ev.set()
            try:
                self._sse({"error": {"message": f"{type(e).__name__}: {e}"}})
            except OSError:
                pass
        finally:
            stop_ev.set()
            try:
                self._chunk(b"data: [DONE]\n\n")
                self._chunk(b"")
            except OSError:                             # 客户端已断开
                pass
            dt = time.time() - t0
            if dt > 0:
                print(f"[生成-流式] {n_char} 字 / {dt:.1f}s = {n_char / dt:.1f} 字/s",
                      flush=True)

    def do_GET(self):
        if self.path.rstrip("/").endswith("/models"):
            self._json(200, {"object": "list", "data": [
                {"id": "ev", "object": "model", "owned_by": "local"}]})
        else:
            self._json(404, {"error": {"message": "not found"}})

    def do_POST(self):
        if not self.path.rstrip("/").endswith("/chat/completions"):
            self._json(404, {"error": {"message": "not found"}})
            return
        try:
            n = int(self.headers.get("Content-Length", "0"))
            req = json.loads(self.rfile.read(n) or b"{}")
        except Exception as e:                          # noqa: BLE001
            self._json(400, {"error": {"message": f"bad request: {e}"}})
            return

        msgs = req.get("messages") or []
        if not any(m.get("role") == "system" for m in msgs):
            msgs = [{"role": "system", "content": SYSTEM}] + msgs
        max_new = min(int(req.get("max_tokens") or _max_new), 512)
        temperature = float(req.get("temperature", 0.7))
        top_p = float(req.get("top_p", 0.9))

        # 流式：OpenAI 的 stream=true，返回 SSE，边生成边吐 token
        if req.get("stream"):
            self._stream(msgs, max_new, temperature, top_p)
            return

        try:
            text = generate(msgs, max_new, temperature, top_p)
        except Exception as e:                          # noqa: BLE001
            self._json(500, {"error": {"message": f"{type(e).__name__}: {e}"}})
            return

        self._json(200, {
            "id": f"chatcmpl-{int(time.time() * 1000)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req.get("model", "ev"),
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": text}}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        })


def main():
    global _max_new
    ap = argparse.ArgumentParser(description="Unsloth + OpenAI 兼容接口")
    ap.add_argument("--adapter", default="/home/fanta/ft_outputs/ev-v2/lora_adapter")
    ap.add_argument("--base", action="store_true", help="跑底座（不加载 LoRA）")
    ap.add_argument("--base-model", default=os.path.join(HERE, "models/Qwen3.5-4B"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8080")))
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--max-seq", type=int, default=2048)
    ap.add_argument("--max-new", type=int, default=256)
    args = ap.parse_args()

    _max_new = args.max_new
    load("" if args.base else args.adapter, args.base_model, args.max_seq)

    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"\nOpenAI 兼容接口: http://{args.host}:{args.port}/v1")
    print(f"  P1 这样用:  LLM_BASE=http://127.0.0.1:{args.port}/v1 bash run_app.sh")
    print("  Ctrl+C 退出\n", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")


if __name__ == "__main__":
    main()
