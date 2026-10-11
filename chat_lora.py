# -*- coding: utf-8 -*-
# chat_lora.py — 多轮对话 + 流式输出（终端 REPL）
#
# 运行环境: WSL（Unsloth 不支持原生 Windows）
#
# 用法:
#   source ~/venvs/ft/bin/activate
#   python /mnt/e/VMOS_workspace/model_tuning/chat_lora.py
#
# 常用环境变量:
#   ADAPTER=/path/to/lora_adapter  微调权重目录；改成底座目录就退化为原始模型
#                                  默认: /home/fanta/ft_outputs/qwen35-4b-lora/lora_adapter
#   MAX_NEW=512                    单轮最多生成多少 token
#   TEMPERATURE=0.7   TOP_P=0.9
#   SYSTEM="你是一个..."            可选系统提示
#   KEEP_TURNS=8                   最多携带最近几轮历史（防止上下文过长）
#
# 会话内命令:  /exit 退出   /reset 清空上下文   /help 帮助
#              /remember 内容   记住一条长期记忆
#              /memory          查看全部长期记忆
#              /forget 关键词   删除一条
#
# 外挂长期记忆 (重要):
#   存在同目录的 memory.md —— Markdown 文件，可直接用编辑器改，改完下一轮即生效。
#   记忆靠文件，风格靠微调：加记忆**不需要重新训练**。
#   路径可用环境变量 MEMORY_MD 覆盖。
#
# 已内置处理的三个坑:
#   1. Qwen3.5 是多模态模型, from_pretrained 返回的 "tokenizer" 实为 Processor,
#      直接调用会把文本当成图片 -> 用底层纯文本分词器 tok 做编码/解码
#   2. 模型不会在 <|im_end|> 处停止而续写下一轮对话 -> 显式传 eos_token_id
#   3. 默认会输出思考链 <think> -> 渲染模板时 enable_thinking=False

import os
import sys
import threading

import torch
from unsloth import FastLanguageModel
from transformers import TextIteratorStreamer

from memory import Memory

ADAPTER     = os.environ.get("ADAPTER", "/home/fanta/ft_outputs/qwen35-4b-lora/lora_adapter")
MAX_NEW     = int(os.environ.get("MAX_NEW", "512"))
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.7"))
TOP_P       = float(os.environ.get("TOP_P", "0.9"))
SYSTEM      = os.environ.get("SYSTEM", "").strip()
KEEP_TURNS  = int(os.environ.get("KEEP_TURNS", "8"))

print("=" * 60)
print(f"[加载] {ADAPTER}")
print("=" * 60, flush=True)

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name     = ADAPTER,
    max_seq_length = 2048,
    dtype          = None,
    load_in_4bit   = True,
)
FastLanguageModel.for_inference(model)

# 坑 1: 取底层纯文本分词器（Processor 有 .tokenizer 属性，普通分词器则返回自身）
tok = getattr(tokenizer, "tokenizer", tokenizer)

# 坑 2: 显式终止符，否则模型会在答完后继续编造下一轮 user/assistant
EOS_IDS = [tok.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>")]
EOS_IDS = [i for i in EOS_IDS if isinstance(i, int) and i >= 0]
PAD_ID  = tok.pad_token_id if tok.pad_token_id is not None else (EOS_IDS[0] if EOS_IDS else 0)

# 兜底：一旦出现这些标记，说明模型开始编下一轮，立即截断
STOP_MARKS = ("<|im_end|>", "\nuser", "\nassistant", "\nUser", "\nAssistant")

# 外挂长期记忆：Markdown 文件，人可读可手改；加记忆不需要重新训练
MEM_PATH = os.environ.get(
    "MEMORY_MD",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "memory.md"))
mem = Memory(MEM_PATH)


def render(messages):
    """渲染 chat 模板；坑 3: 优先关闭思考链，模板不支持时回退。"""
    try:
        return tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False,
            enable_thinking=False)
    except Exception:
        return tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False)


def _run_generate(**kwargs):
    with torch.inference_mode():
        model.generate(**kwargs)


def build_system():
    """system 提示 = 环境变量人设 + 外挂记忆（每轮现读，改了立刻生效）"""
    parts = []
    if SYSTEM:
        parts.append(SYSTEM)
    block = mem.as_prompt_block()
    if block:
        parts.append(block)
    return "\n\n".join(parts)


def stream_reply(turns):
    """流式生成并实时打印，返回完整回答文本。turns 只含 user/assistant。"""
    sys_text = build_system()
    messages = ([{"role": "system", "content": sys_text}] if sys_text else []) + turns
    inputs = tok(render(messages), return_tensors="pt").to(model.device)
    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)

    kwargs = dict(
        **inputs,
        max_new_tokens = MAX_NEW,
        do_sample      = True,
        temperature    = TEMPERATURE,
        top_p          = TOP_P,
        eos_token_id   = EOS_IDS or None,
        pad_token_id   = PAD_ID,
        streamer       = streamer,
    )
    thread = threading.Thread(target=_run_generate, kwargs=kwargs)
    thread.start()

    shown, printed = "", 0
    for piece in streamer:
        shown += piece
        cut = -1
        for m in STOP_MARKS:
            i = shown.find(m)
            if i != -1 and (cut == -1 or i < cut):
                cut = i
        visible = shown if cut == -1 else shown[:cut]
        if len(visible) > printed:
            sys.stdout.write(visible[printed:])
            sys.stdout.flush()
            printed = len(visible)
        if cut != -1:          # 命中停止标记，收工
            shown = visible
            break
    thread.join()
    return shown.strip()


def trim(history):
    """只保留最近 KEEP_TURNS 轮对话（system + 记忆由 build_system() 每轮现拼）。"""
    return history[-KEEP_TURNS * 2:]


def main():
    history = []
    print(f"已就绪 | 温度 {TEMPERATURE} | 单轮上限 {MAX_NEW} token | 上下文 {KEEP_TURNS} 轮")
    print(f"长期记忆: {MEM_PATH}  (当前 {len(mem.list())} 条)")
    print("命令: /exit 退出   /reset 清空上下文   /remember 内容   /memory 查看   /forget 关键词")
    print("-" * 60)

    while True:
        try:
            user = input("\n你 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见。")
            break
        if not user:
            continue
        if user in ("/exit", "/quit", "/q"):
            print("再见。")
            break
        if user == "/reset":
            history = []
            print("(上下文已清空；长期记忆不受影响)")
            continue
        if user.startswith("/remember"):
            text = user[len("/remember"):].strip()
            if not text:
                print("用法: /remember 我住在杭州")
            elif mem.remember(text):
                print(f"(已记住: {text})")
            else:
                print("(这条已经在记忆里了)")
            continue
        if user == "/memory":
            items = mem.list()
            if not items:
                print("(还没有记忆，用 /remember 添加，或直接编辑 memory.md)")
            else:
                for i, t in enumerate(items, 1):
                    print(f"  {i}. {t}")
            continue
        if user.startswith("/forget"):
            kw = user[len("/forget"):].strip()
            if not kw:
                print("用法: /forget 关键词")
            else:
                gone = mem.forget(kw)
                print(f"(已忘记: {gone})" if gone else f"(没找到包含「{kw}」的记忆)")
            continue
        if user == "/help":
            print("直接输入问题回车即可。")
            print("  /reset        清空当前对话上下文")
            print("  /remember 内容 记住一条长期记忆（写入 memory.md）")
            print("  /memory       查看全部长期记忆")
            print("  /forget 关键词 删除一条")
            print("  /exit         退出")
            continue

        history.append({"role": "user", "content": user})
        history = trim(history)
        print("\n模型 > ", end="", flush=True)
        reply = stream_reply(history)
        history.append({"role": "assistant", "content": reply})
        print()


if __name__ == "__main__":
    main()
