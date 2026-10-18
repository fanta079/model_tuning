# -*- coding: utf-8 -*-
"""design_voice.py — 用文字"造"一个 E.V. 的音色（Qwen3-TTS VoiceDesign）

为什么需要它
------------
edge-tts 只能在别人做好的音色上拧 rate/pitch，天花板就在那儿。
Qwen3-TTS 的 VoiceDesign 反过来：**你描述，它造** —— 不需要任何参考音频，
也就不存在"克隆真人声音"的版权问题。

一次跑完做三件事：
  1. 按内置的 8 条「气质描述」批量生成候选音频
  2. 生成一个可点击试听的网页（顺带把描述词原文列出来，方便你改）
  3. `--lock N` 把选中的那条锁定成参考音频，写进 app/voice.json

为什么是 1.7B（而不是 0.6B）
----------------------------
VoiceDesign 只有 1.7B 版本。它**只在"造音色"这一下用**，跑完就该退出：
    bash start_ev.sh --stop        # 先腾显存，1.7B bf16 约 4.5GB
之后真正说话用的是 0.6B Base + 这段参考音频做 voice clone（约 2GB，能和 LLM 共存）。

用法
----
    python app/design_voice.py --dry-run          # 先看 8 条描述词，不加载模型、不花钱
    python app/design_voice.py                    # 全跑（首次会下载 ~4.5GB 权重）
    python app/design_voice.py --only 1,3,4       # 只跑几条
    python app/design_voice.py --device cpu       # 没显卡（1.7B 在 CPU 上很慢，仅应急）
    python app/design_voice.py --lock 3           # 把第 3 条锁定成 E.V. 的音色

产物
----
    app/voice_samples/design/01_xxx.wav ...
    app/voice_samples/design/candidates.json      # 每条用了什么描述词
    app/voice_samples/design/index.html           # 浏览器打开这个听
    app/voice_lock/ref.wav + ref.txt              # --lock 之后才有

安装（WSL 内，建议独立 venv，别装进 ~/venvs/ft）
----
    python -m venv ~/venvs/tts && source ~/venvs/tts/bin/activate
    pip install -U qwen-tts soundfile -i https://mirrors.aliyun.com/pypi/simple/
    # 国内下载权重走魔搭（比 HF 快很多）：
    modelscope download --model Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign \
        --local_dir ~/models/Qwen3-TTS-12Hz-1.7B-VoiceDesign
    # 然后用 --model 指到本地目录
"""

import argparse
import json
import os
import shutil
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, "voice_samples", "design")
LOCK_DIR = os.path.join(HERE, "voice_lock")
VOICE_JSON = os.path.join(HERE, "voice.json")

DEFAULT_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
LOCAL_NAME = "Qwen3-TTS-12Hz-1.7B-VoiceDesign"

# 试音台词：短句 + 停顿 + 长句 + 中间有省略号，一句话就能听出"冷"和"稳"
TEXT_CN = "嗯，在。有事就说。这个嘛……你想清楚了再问。别把简单的事弄复杂，数据不会骗人。"
TEXT_EN = "Good evening. I have already run the diagnostics. Nothing unusual, so far."

# (编号, 显示名, 一句话说明, 语言, 描述词)
# 描述词写法要点：性别/年龄 → 音高 → 语速 → 句尾走向 → 情绪态度。
# 「句尾上扬」是最要命的：上扬=亲切，冷感全靠把它压掉。
CANDIDATES = [
    (1, "冷淡女管家", "基准组：中低音、平稳、句尾不上扬",
     "Chinese",
     "成年女性，三十岁上下，中低音，语速平稳偏慢，句尾持平不上扬，语气克制疏离，"
     "像在陈述事实，没有笑意。"),

    (2, "深冷电台", "更低压更平，几乎没有起伏",
     "Chinese",
     "成年女性，低沉略沙哑的嗓音，几乎没有情绪起伏，语速缓慢均匀，字字清楚，"
     "像深夜电台的独白。"),

    (3, "清冷少女", "年龄感更轻，贴近灰原哀那种冷淡",
     "Chinese",
     "十七八岁少女，嗓音偏低，语速不快，语气冷静疏远，句尾略下沉，偶尔透出一点不耐烦。"),

    (4, "中性系统音", "最像 AI，几乎没有重音起伏",
     "Chinese",
     "成年女性，中性冷淡，语调平直，几乎不带重音起伏，语速中等，"
     "像高级设备的语音提示，但比机械音自然。"),

    (5, "温柔但疏离", "软一点，冷感靠不热情而不是靠压低",
     "Chinese",
     "二十五岁左右女性，声音干净偏软，语速慢，语气平静但不热情，句尾轻轻落下，不拖腔。"),

    (6, "低哑克制", "成熟、见惯场面那种笃定",
     "Chinese",
     "三十多岁女性，偏低沉的胸腔共鸣，语速平缓，语气笃定克制，不拖尾音。"),

    (7, "EN 英式男管家", "对照用：Jarvis 那一路的英音男声",
     "English",
     "Adult British male, mid-low register, received pronunciation, steady measured pace, "
     "restrained and courteous, sentences falling at the end, no warmth."),

    (8, "EN 英式女助理", "对照用：Friday 那一路的英音女声",
     "English",
     "Adult British female, mid-low register, crisp received pronunciation, slightly quicker pace, "
     "cool and precise, flat sentence endings."),
]


def text_for(lang: str, override: str) -> str:
    if override:
        return override
    return TEXT_EN if lang == "English" else TEXT_CN


def resolve_model(arg: str) -> str:
    """优先用本地目录 —— setup_tts.sh 会把权重下到 models/ 下，
    传目录名加载可以完全避免运行时又回 HF/魔搭拉一遍。"""
    if arg:
        return arg
    env = os.environ.get("QWEN_TTS_MODEL")
    if env:
        return env
    local = os.path.join(ROOT, "models", LOCAL_NAME)
    if os.path.isdir(local) and os.listdir(local):
        return local
    return DEFAULT_MODEL


def slug(s: str) -> str:
    keep = []
    for ch in s:
        if ch.isalnum() or ch in "-_":
            keep.append(ch)
    return "".join(keep) or "voice"


def print_candidates(items):
    print(f"{'#':<3}{'名称':<16}{'语言':<9}说明")
    print("-" * 78)
    for n, name, note, lang, instruct in items:
        print(f"{n:<3}{name:<16}{lang:<9}{note}")
    print("\n描述词原文（想改就改 CANDIDATES）：")
    for n, name, _note, lang, instruct in items:
        print(f"\n[{n}] {name}  ({lang})")
        print(f"    {instruct}")


def load_model(model_id, device, dtype_name, flash):
    try:
        import torch
        from qwen_tts import Qwen3TTSModel
    except ImportError as e:
        raise SystemExit(
            "[错误] 缺依赖：%s\n"
            "  python -m venv ~/venvs/tts && source ~/venvs/tts/bin/activate\n"
            "  pip install -U qwen-tts soundfile -i https://mirrors.aliyun.com/pypi/simple/\n"
            "  只想先看描述词、不装模型：加 --dry-run" % e)

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[dtype_name]

    if device.startswith("cuda") and torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        print(f"[GPU] {torch.cuda.get_device_name(0)}  "
              f"空闲 {free/2**30:.1f}GB / 共 {total/2**30:.1f}GB")
        if free / 2**30 < 5.0:
            print("[警告] 空闲显存 < 5GB，1.7B VoiceDesign 可能 OOM。"
                  "先 `bash start_ev.sh --stop` 把模型服务停掉。")

    kw = {"device_map": device, "dtype": dtype}
    if flash:
        kw["attn_implementation"] = "flash_attention_2"
    print(f"[加载] {model_id}（首次会下载约 4.5GB）……")
    t0 = time.time()
    model = Qwen3TTSModel.from_pretrained(model_id, **kw)
    print(f"[加载] 完成，{time.time() - t0:.1f}s")
    return model


def generate(model, items, text_override):
    """先试批量（一条推理出多条），失败就退回逐条 —— 批量能省一大半时间。"""
    try:
        import soundfile as sf  # noqa: F401
    except ImportError:
        raise SystemExit("[错误] 缺 soundfile：pip install soundfile")

    texts = [text_for(c[3], text_override) for c in items]
    langs = [c[3] for c in items]
    instructs = [c[4] for c in items]

    print(f"\n[生成] {len(items)} 条候选 ……")
    t0 = time.time()
    try:
        wavs, sr = model.generate_voice_design(
            text=texts, language=langs, instruct=instructs)
        print(f"[生成] 批量完成，{time.time() - t0:.1f}s，采样率 {sr}Hz")
        return list(wavs), sr
    except Exception as e:                      # noqa: BLE001
        print(f"[生成] 批量失败（{type(e).__name__}: {e}），改为逐条生成")

    wavs, sr = [], 24000
    for i, c in enumerate(items):
        t1 = time.time()
        w, sr = model.generate_voice_design(
            text=texts[i], language=langs[i], instruct=instructs[i])
        wavs.append(w[0])
        print(f"  [{c[0]}] {c[1]}  {time.time() - t1:.1f}s")
    return wavs, sr


def write_outputs(items, wavs, sr, text_override, meta):
    import soundfile as sf

    os.makedirs(OUT, exist_ok=True)
    made = []
    for (n, name, note, lang, instruct), wav in zip(items, wavs):
        fn = f"{n:02d}_{slug(name)}.wav"
        path = os.path.join(OUT, fn)
        sf.write(path, wav, sr)
        size = os.path.getsize(path)
        dur = len(wav) / float(sr)
        print(f"  [{n}] {name:<16} -> {fn}  {dur:.1f}s  {size/1024:.0f} KB")
        made.append({
            "id": n, "name": name, "note": note, "lang": lang,
            "instruct": instruct, "text": text_for(lang, text_override),
            "file": fn, "seconds": round(dur, 2),
        })

    payload = {"model": meta["model"], "sample_rate": sr,
               "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
               "candidates": made}
    with open(os.path.join(OUT, "candidates.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    html_path = os.path.join(OUT, "index.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(build_html(payload))
    print(f"\n试听网页：{html_path}")
    print("在 WSL 里可以直接 `explorer.exe` 或者在 Windows 浏览器打开这个路径。")
    return made


def build_html(payload):
    cards = []
    for c in payload["candidates"]:
        cards.append(f"""
  <div class="card">
    <div class="head"><b>[{c['id']}] {c['name']}</b><span class="note">{c['note']}</span></div>
    <div class="meta">{c['lang']}　{c['seconds']}s　{c['file']}</div>
    <audio controls preload="none" src="{c['file']}"></audio>
    <details><summary>描述词 / 台词</summary>
      <p class="q">{c['text']}</p>
      <p class="q">{c['instruct']}</p>
    </details>
  </div>""")

    return f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>E.V. 音色设计 · Qwen3-TTS VoiceDesign</title>
<style>
  body{{background:#f7f6f2;color:#2c2c2a;font:15px/1.6 system-ui,"Microsoft YaHei",sans-serif;
       max-width:800px;margin:0 auto;padding:32px 20px}}
  h1{{font-size:20px;margin:0 0 6px;font-weight:500}}
  .tip{{color:#5f5e5a;font-size:13px;margin-bottom:22px}}
  .card{{background:#fff;border:1px solid #e2e0d8;border-radius:10px;padding:14px 16px;margin-bottom:12px}}
  .head{{display:flex;justify-content:space-between;align-items:baseline;gap:12px}}
  .note{{color:#888780;font-size:12.5px}}
  .meta{{color:#888780;font-size:12px;margin:4px 0 10px;font-family:ui-monospace,monospace}}
  audio{{width:100%;height:34px}}
  details{{margin-top:10px}} summary{{color:#888780;font-size:12.5px;cursor:pointer}}
  .q{{color:#5f5e5a;font-size:13px;background:#f7f6f2;border-radius:6px;padding:8px 10px;margin:8px 0 0}}
  code{{background:#f1efe8;padding:1px 5px;border-radius:4px;font-size:13px}}
</style></head><body>
<h1>E.V. 音色设计（VoiceDesign）</h1>
<div class="tip">
  模型：<code>{payload['model']}</code>　生成于 {payload['created_at']}<br>
  听完记住编号，然后 <code>python app/design_voice.py --lock N</code> 把它锁定成参考音频。<br>
  音色会随描述词变，同一编号重跑结果也不会完全一致 —— 锁定就是为了"定下来"。
</div>
{''.join(cards)}
</body></html>"""


def lock(candidate_id):
    path = os.path.join(OUT, "candidates.json")
    if not os.path.isfile(path):
        raise SystemExit(f"[错误] 没有找到 {path} —— 先生成候选")
    data = json.load(open(path, encoding="utf-8"))
    hit = next((c for c in data["candidates"] if c["id"] == candidate_id), None)
    if not hit:
        raise SystemExit(f"[错误] 没有编号 {candidate_id} 的候选")

    src = os.path.join(OUT, hit["file"])
    if not os.path.isfile(src):
        raise SystemExit(f"[错误] 音频不存在：{src}")

    os.makedirs(LOCK_DIR, exist_ok=True)
    ref_wav = os.path.join(LOCK_DIR, "ref.wav")
    shutil.copyfile(src, ref_wav)
    with open(os.path.join(LOCK_DIR, "ref.txt"), "w", encoding="utf-8") as f:
        f.write(hit["text"])
    with open(os.path.join(LOCK_DIR, "ref.json"), "w", encoding="utf-8") as f:
        json.dump(hit, f, ensure_ascii=False, indent=2)

    cfg = {}
    if os.path.isfile(VOICE_JSON):
        try:
            cfg = json.load(open(VOICE_JSON, encoding="utf-8"))
        except Exception:                       # noqa: BLE001
            cfg = {}
    cfg["_当前选择"] = (f"Qwen3-TTS VoiceDesign 候选 [{hit['id']}] {hit['name']}，"
                       f"已锁定为 app/voice_lock/ref.wav")
    cfg["engine"] = "qwen3tts"
    cfg["qwen_ref_audio"] = "app/voice_lock/ref.wav"
    cfg["qwen_ref_text"] = hit["text"]
    cfg["qwen_design"] = {"id": hit["id"], "name": hit["name"],
                          "instruct": hit["instruct"], "lang": hit["lang"]}
    cfg["locked_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(VOICE_JSON, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    print(f"[锁定] [{hit['id']}] {hit['name']} -> {ref_wav}")
    print(f"[锁定] 已写入 {VOICE_JSON}（engine=qwen3tts）")
    print("\n下一步（还没做，需要单独接）：")
    print("  1. 用 0.6B Base 建可复用的 clone prompt：")
    print("     model.create_voice_clone_prompt(ref_audio=(wav, sr), ref_text=...)")
    print("  2. server.py 的 tts() 改成调本地 TTS 服务（TTS_BASE=http://127.0.0.1:8092）")
    print("  3. 试听不满意就 --lock 别的编号，随时可换，不动代码")


def main():
    ap = argparse.ArgumentParser(description="用文字造 E.V. 的音色（Qwen3-TTS VoiceDesign）")
    ap.add_argument("--dry-run", action="store_true", help="只打印描述词，不加载模型")
    ap.add_argument("--only", default="", help="只跑指定编号，逗号分隔，如 1,3,4")
    ap.add_argument("--text", default="", help="覆盖试音台词（默认中文那句）")
    ap.add_argument("--model", default="",
                    help="模型 id 或本地目录（默认自动找 models/Qwen3-TTS-12Hz-1.7B-VoiceDesign）")
    ap.add_argument("--device", default="cuda:0", help="cuda:0 / cpu")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--flash", action="store_true", help="用 flash-attention-2（省显存，需已安装）")
    ap.add_argument("--lock", type=int, default=0, help="把某条候选锁定成参考音频")
    args = ap.parse_args()

    if args.lock:
        lock(args.lock)
        return

    items = CANDIDATES
    if args.only:
        want = {int(x) for x in args.only.replace(" ", "").split(",") if x}
        items = [c for c in CANDIDATES if c[0] in want]
        if not items:
            raise SystemExit(f"[错误] --only {args.only} 没有匹配的候选")

    if args.dry_run:
        print_candidates(items)
        print(f"\n共 {len(items)} 条。去掉 --dry-run 就会真的生成（首跑要下 ~4.5GB 权重）。")
        print(f"模型：{resolve_model(args.model)}")
        return

    model_id = resolve_model(args.model)
    model = load_model(model_id, args.device, args.dtype, args.flash)
    wavs, sr = generate(model, items, args.text)
    write_outputs(items, wavs, sr, args.text, {"model": model_id})

    try:
        import torch
        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
    except Exception:                           # noqa: BLE001
        pass
    print("\n听完：python app/design_voice.py --lock <编号>")


if __name__ == "__main__":
    main()
