# -*- coding: utf-8 -*-
"""export_gguf.py — 把 LoRA adapter 合并进底座并导出 GGUF

为什么长期使用要导出 GGUF
--------------------------
直接跑 `chat_lora.py`（加载 4B 底座 + adapter）有三个不便：
  · 每次启动都要加载整个底座，慢
  · 4bit 底座常驻显存，占得多
  · 一次只能一个会话

导出成 GGUF 后可以用 llama.cpp 的 llama-server 跑：
  · 启动快、显存占用低
  · 自带 OpenAI 兼容 API + Web UI
  · 可以接任意前端（和你现在跑 9B 的 run_chat.sh 一个姿势）

用法（WSL 内）
--------------
    source ~/venvs/ft/bin/activate

    python export_gguf.py \
        --adapter /home/fanta/ft_outputs/qwen35-4b-haibara-v1/lora_adapter \
        --out ~/gguf/qwen35-4b-haibara

    # 可选：同时导出多个量化等级
    python export_gguf.py --adapter ... --out ... --quants q4_k_m,q8_0

注意
----
* 第一次导出时 Unsloth 会自动下载并编译 llama.cpp，耗时较长（十几分钟），之后就有缓存了
* 导出的是「合并后的完整模型」，之后不再需要 adapter
* 量化等级：q4_k_m 体积/质量最平衡（4B 约 2.5GB）；q8_0 更准但更大
"""

import argparse
import os
import sys

from unsloth import FastLanguageModel

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser(description="导出 GGUF 供 llama.cpp 使用")
    ap.add_argument("--adapter", required=True, help="LoRA adapter 目录")
    ap.add_argument("--out", required=True, help="GGUF 输出目录")
    ap.add_argument("--quants", default="q4_k_m", help="量化方式，逗号分隔，如 q4_k_m,q8_0")
    ap.add_argument("--merged-only", action="store_true",
                    help="只把 LoRA 合并成 HF 格式，不转 GGUF —— 不需要 llama.cpp、不需要联网。"
                         "网络/编译有问题时先用这个，之后再单独转 GGUF。")
    ap.add_argument("--max-seq", type=int, default=2048)
    args = ap.parse_args()

    if not os.path.isdir(args.adapter):
        sys.exit(f"[错误] 找不到 adapter: {args.adapter}")

    out = os.path.abspath(os.path.expanduser(args.out))
    os.makedirs(out, exist_ok=True)

    print("=" * 60)
    print(f"[加载] {args.adapter}")
    print("=" * 60, flush=True)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.adapter, max_seq_length=args.max_seq, dtype=None,
        load_in_4bit=True,
    )

    quants = [q.strip() for q in args.quants.split(",") if q.strip()]

    # ---- 只合并：完全不碰 llama.cpp，也不需要联网 ----
    if args.merged_only:
        print(f"\n[合并] LoRA -> HF 完整模型  ->  {out}", flush=True)
        model.save_pretrained_merged(out, tokenizer, save_method="merged_16bit")
        print("\n" + "=" * 60)
        print("合并完成（HF 格式，不是 GGUF）:", out)
        print("下一步（需要 llama.cpp）:")
        print(f"  python ~/llama.cpp/convert_hf_to_gguf.py {out} "
              f"--outfile {out}/ev-f16.gguf --outtype f16")
        print(f"  ~/llama.cpp/build/bin/llama-quantize {out}/ev-f16.gguf "
              f"{out}/ev-q4_k_m.gguf Q4_K_M")
        print("=" * 60)
        return

    for q in quants:
        print(f"\n[导出] 量化方式 = {q}  ->  {out}", flush=True)
        model.save_pretrained_gguf(out, tokenizer, quantization_method=q)

    print("\n" + "=" * 60)
    print("导出完成。目录内容：")
    for f in sorted(os.listdir(out)):
        p = os.path.join(out, f)
        if os.path.isfile(p):
            print(f"  {f}   {os.path.getsize(p)/1024**2:.0f} MB")
    print("=" * 60)
    print("""
下一步 —— 用 llama-server 跑起来（改一下你现有的 run_chat.sh 即可）：

  llama-server \\
    -m <上面那个 .gguf 文件> \\
    -ngl 99 -c 8192 --port 8080 \\
    --jinja \\
    -sys "现在请完全以灰原哀的身份回答，不要跳戏。"

然后浏览器打开 http://localhost:8080 就能聊了。

提示：日常使用建议**保留 system 提示词**（微调学的是说话倾向，
     system 负责明确身份，两者配合最稳）。
""")


if __name__ == "__main__":
    main()
