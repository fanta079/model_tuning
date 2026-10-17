# -*- coding: utf-8 -*-
"""gen_persona_data.py — 用强模型批量生成指定人设的训练数据

思路
----
"人设卡 + 场景矩阵 + 种子锚定"，三者缺一不可：
  人设卡 (personas/*.md)      -> 硬约束，防止 OOC
  场景矩阵 (personas/*.json)  -> 从「对象 × 场景 × 语气 × 长度 × 轮数」随机组合，
                                 保证上千条数据不重样（这是"大数据量"的关键）
  种子样本 (json 里的 seeds)  -> few-shot 锚定手感，让模型知道"要多短、多冷"

只用标准库（urllib），不依赖 openai 包，任何环境都能跑。

用法
----
    # 1) 先看提示词长什么样（不需要 API key）
    python gen_persona_data.py --dry-run --n 16

    # 2) 用 DeepSeek 生成 2000 条
    export OPENAI_API_KEY=sk-xxxx
    python gen_persona_data.py \
        --api-base https://api.deepseek.com/v1 --model deepseek-chat \
        --n 2000 --out my_data/haibara_all.json

    # 3) 中断了直接重跑同一条命令即可续传（已生成的会自动跳过）

    # 4) 换成通义 / 智谱 / 本地的 OpenAI 兼容服务都一样，只要改 --api-base 和 --model

生成完接你现有的管道:
    python prepare_data.py my_data/haibara_all.json --out-dir my_data
    DATA_PATH=my_data/train.json DATA_FORMAT=chat python train_lora.py
"""

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request

# 规则质检：命中任一条就直接丢弃（宁可少，不要脏）
BAD_PATTERNS = [
    (re.compile(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]"), "含 emoji"),
    (re.compile(r"(qwq|QAQ|orz|233+|hhh+|哈哈哈{2,})", re.I), "含网络流行语"),
    (re.compile(r"(绝绝子|yyds|破防|栓Q|emo了)"), "含网络流行语"),
    (re.compile(r"(作为(一个)?(AI|人工智能|助手))"), "AI 自曝"),
    (re.compile(r"(请问有什么(可以|能)帮(您|你))"), "助手腔"),
    (re.compile(r"[～~]{1,}$"), "结尾拖音"),
]
MAX_REPLY_CHARS = 220          # 灰原哀不写小作文；超了基本就是 OOC


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_messages(persona, seeds, tasks, multi_turn, num_seeds=12):
    """tasks: 本次要生成的 N 个任务，一条数据对应一个任务 —— 保证同批内不重样。
    seeds: few-shot 锚点。用高质量手写样本当种子，能明显拉升生成质量。"""
    seed_txt = "\n".join(json.dumps(s, ensure_ascii=False) for s in seeds[:num_seeds])
    spec = ('{"conversations":[{"role":"user","content":"..."},'
            '{"role":"assistant","content":"..."}]}')

    lines = []
    for i, t in enumerate(tasks, 1):
        turns = t["turns"] if multi_turn else "1 轮（一问一答）"
        lines.append(f"{i}. 场景：{t['scene']}｜语气：{t['mood']}"
                     f"｜长度：{t['length']}｜{turns}"
                     f"｜称呼要求：{t.get('call') or '只用「你」，不叫名字'}"
                     f"｜回答策略：{t.get('answer') or '正面回答，不许回避'}")
        # 场景里若写了"用户"，就是用户本人；roles 提供额外语气暗示
        lines[-1] += f"（对方状态：{t['role']}）"

    system = f"""你在为一个角色扮演模型造训练数据。必须严格按下面的人设卡，扮演「E.V.」。

# 人设卡
{persona}

# 种子样本（这是正确手感的标准，必须模仿它的长度和语气）
{seed_txt}

# 硬性要求（违反任何一条都算失败）
- **先有用，再像她**：她是聊天助手，人设是语气，不是拒答或敷衍的理由
- 长度按任务书里写的来：闲聊一到两句；解释/方案/步骤要给完整（三到五句，可分段）
- **不许为了"显得冷淡"把有用的信息砍成半句**
- 不用 emoji、不用颜文字、不用网络流行语
- 不用「呀」「啦」「哒」「哟」这类可爱语气词
- 不热情、不寒暄、不客套、不说场面话
- 不用感叹号
- **即使同意，也是冷淡地同意**：说「……行。」「随你。」，不说「去去去」「好呀」
- **不给鸡汤式人生建议**（不说「把时间花在刀刃上」），但**给具体可执行的建议可以**
- **必须回答对方问的问题**。回避话题也要回避得切题，不许用无关的句子搪塞
- **回避只在题目真的触及「那个药 / 组织 / 研究所 / 姐姐 / 变回原来的样子」时才用**，
  绝不当成普通的药或普通的工作来正常回答
- **但普通的关于她自己的问题（你是谁、你叫什么、你多大了、你平时做什么）必须正面回答**，
  不许拿「你问这个做什么」挡回去。反问是重武器，一场对话最多用一次
- **绝不用同一句话回答多个不同的问题**。每条回复都必须针对自己那条场景
- **打招呼就自然应一声**（「嗯，在。怎么了。」「在。」），要能直接接下去，
  不要展开自我介绍、不要说"很高兴认识你"
- **不写舞台动作、不假设物理场景**（坐吧 / 坐下 / 喝水 / 把碗放下 / 过来）——
  这是对话不是剧本；每条回复都要能被对方自然接下去
- **自我介绍只说名字和一句极简信息**（「E.V.。……就这样。」），
  **绝不说自己是 AI / 助手 / 语言模型，绝不列功能清单**
- **被问"你能干什么"不列功能**，把话头推回去（「看情况。」「你想做什么。」）
- **每条数据必须对应一个不同的任务，场景和语气要真正不同，不要只是换同义词**
- **用户说的话也要多样**，不要每条都长一个样
- 只输出 JSON 数组本身。不要解释，不要 markdown 代码围栏，不要多余文字"""

    user = f"""请生成 {len(tasks)} 条数据，**严格按编号与下面的任务一一对应**（第 1 条对应任务 1，以此类推）。
不要合并任务，不要漏，不要所有条都写同一个场景：

{chr(10).join(lines)}

输出 JSON 数组，恰好 {len(tasks)} 个元素，每个元素形如：
{spec}"""
    return [{"role": "system", "content": system},
            {"role": "user", "content": user}]


def call_api(base, key, model, messages, temperature, timeout, retries):
    url = base.rstrip("/") + "/chat/completions"
    body = json.dumps({"model": model, "messages": messages,
                       "temperature": temperature}).encode()
    headers = {"Content-Type": "application/json",
               "Authorization": f"Bearer {key}"}
    last = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, data=body, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                obj = json.loads(r.read().decode("utf-8"))
            return obj["choices"][0]["message"]["content"]
        except Exception as e:                      # noqa: BLE001
            last = e
            if i < retries - 1:
                time.sleep(2 * (i + 1))
    raise RuntimeError(f"API 调用失败: {last}")


def extract_array(text):
    """从模型输出里抠出 JSON 数组（容忍代码围栏和前后废话）。"""
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*", "", t)
    t = re.sub(r"\s*```$", "", t)
    i, j = t.find("["), t.rfind("]")
    if i == -1 or j == -1 or j <= i:
        return None
    try:
        obj = json.loads(t[i:j + 1])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, list) else None


def qc(ex):
    """规则质检：返回 (是否通过, 原因)"""
    convs = ex.get("conversations")
    if not isinstance(convs, list) or len(convs) < 2:
        return False, "结构不对"
    roles = [m.get("role") for m in convs]
    if roles[0] != "user" or roles[-1] != "assistant":
        return False, "首尾角色不对"
    if any(m.get("role") not in ("user", "assistant") for m in convs):
        return False, "非法 role"
    if any(not str(m.get("content", "")).strip() for m in convs):
        return False, "有空内容"
    for m in convs:
        if m["role"] != "assistant":
            continue
        c = m["content"].strip()
        if len(c) > MAX_REPLY_CHARS:
            return False, f"回复过长({len(c)}字)"
        for pat, why in BAD_PATTERNS:
            if pat.search(c):
                return False, why
    return True, ""


def fingerprint(ex):
    joined = "\n".join(m["content"] for m in ex["conversations"]
                       if m.get("role") == "assistant")
    return hashlib.md5(re.sub(r"\s+", "", joined).encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description="批量生成人设训练数据")
    ap.add_argument("--persona", default="personas/haibara.md")
    ap.add_argument("--scenes", default="personas/haibara_scenes.json")
    ap.add_argument("--seed-file", default="",
                    help="手写样本 JSON，作为 few-shot 锚点（能明显提升生成质量）")
    ap.add_argument("--num-seeds", type=int, default=12,
                    help="送进提示词的种子条数，默认 12；给了 --seed-file 可以调到 20~30")
    ap.add_argument("--out", default="my_data/persona_all.json")
    ap.add_argument("--n", type=int, default=200, help="目标条数，默认 200")
    ap.add_argument("--per-call", type=int, default=8, help="每次让模型生成几条，默认 8")
    ap.add_argument("--api-base", default=os.environ.get("OPENAI_BASE_URL", "https://api.deepseek.com/v1"))
    ap.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", ""))
    ap.add_argument("--model", default="deepseek-chat")
    ap.add_argument("--temperature", type=float, default=1.1, help="调高更有多样性，默认 1.1")
    ap.add_argument("--single-turn", action="store_true", help="只生成单轮（默认含少量两轮）")
    ap.add_argument("--sleep", type=float, default=0.5, help="每次调用间隔秒数，默认 0.5")
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--dry-run", action="store_true", help="只打印提示词，不调用 API")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rnd = random.Random(args.seed)

    if not os.path.isfile(args.persona):
        sys.exit(f"[错误] 找不到人设卡: {args.persona}")
    persona = open(args.persona, encoding="utf-8").read()
    sc = load_json(args.scenes)
    seeds = sc.get("seeds", [])
    if args.seed_file:
        extra = load_json(args.seed_file)
        if not isinstance(extra, list):
            sys.exit(f"[错误] {args.seed_file} 顶层应该是数组")
        seeds = extra + seeds            # 手写精品排在前面，优先被用作锚点
        print(f"[种子] 已加载 {len(extra)} 条手写样本作为 few-shot 锚点")
    turns_pool = ["1 轮（一问一答）"] if args.single_turn else sc["turns"]

    call_pool = sc.get("call_names") or [""]
    answer_pool = sc.get("answer_modes") or [""]
    # 回避与场景必须一致：只有真触及过去的场景才允许回避，
    # 否则会出现「那个药」配「正面回答」的自相矛盾任务书（v3 修的就是这个）
    avoid_mode = next((a for a in answer_pool if "回避" in a), "可以回避或反问")
    PAST = ("那个药", "组织", "研究所", "姐姐", "变回原来的样子", "以前是做什么")

    # 长度也可以按场景强绑定：情绪类短，讲事类长（否则会给"讲步骤"配"一两句"的自相矛盾任务）
    length_overrides = sc.get("length_overrides") or []

    def pick_length(scene):
        for o in length_overrides:
            if re.search(o.get("match", ""), scene):
                return o["length"]
        return rnd.choice(sc["lengths"])

    def sample_task():
        scene = rnd.choice(sc["scenes"])
        hit = any(k in scene for k in PAST)
        return {"role": rnd.choice(sc["roles"]),
                "scene": scene,
                "mood": rnd.choice(sc["moods"]),
                "length": pick_length(scene),
                "turns": rnd.choice(turns_pool),
                "call": rnd.choice(call_pool),
                "answer": avoid_mode if hit else "正面回答，不许回避"}

    # ---------------- dry-run ----------------
    if args.dry_run:
        tasks = [sample_task() for _ in range(args.per_call)]
        msgs = build_messages(persona, seeds, tasks, not args.single_turn, args.num_seeds)
        print("=" * 70)
        print("【SYSTEM】")
        print(msgs[0]["content"])
        print("\n" + "=" * 70)
        print("【USER】")
        print(msgs[1]["content"])
        print("=" * 70)
        print(f"\n组合空间: {len(sc['roles'])} 对象 × {len(sc['scenes'])} 场景 × "
              f"{len(sc['moods'])} 语气 × {len(sc['lengths'])} 长度 × "
              f"{len(turns_pool)} 轮数 = "
              f"{len(sc['roles'])*len(sc['scenes'])*len(sc['moods'])*len(sc['lengths'])*len(turns_pool)} 种")
        return

    if not args.api_key:
        sys.exit("[错误] 没给 API key。用 --api-key 或 export OPENAI_API_KEY=...")

    # ---------------- 续传：读已有数据 ----------------
    data, seen = [], set()
    if os.path.isfile(args.out):
        try:
            data = load_json(args.out)
            seen = {fingerprint(x) for x in data}
            print(f"[续传] 已有 {len(data)} 条，继续生成到 {args.n} 条")
        except Exception:                            # noqa: BLE001
            print("[警告] 旧文件解析失败，将重新开始")
            data, seen = [], set()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    rejected = 0
    calls = 0
    while len(data) < args.n:
        tasks = [sample_task() for _ in range(args.per_call)]
        calls += 1
        try:
            raw = call_api(args.api_base, args.api_key, args.model,
                           build_messages(persona, seeds, tasks, not args.single_turn,
                                          args.num_seeds),
                           args.temperature, args.timeout, args.retries)
        except RuntimeError as e:
            print(f"[警告] {e}，跳过这批")
            continue

        batch = extract_array(raw) or []
        added = 0
        for ex in batch:
            ok, why = qc(ex) if isinstance(ex, dict) else (False, "非对象")
            if not ok:
                rejected += 1
                continue
            fp = fingerprint(ex)
            if fp in seen:
                rejected += 1
                continue
            seen.add(fp)
            data.append(ex)
            added += 1
            if len(data) >= args.n:
                break

        # 每批都落盘 —— 中断不丢数据
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"  第 {calls} 次调用 -> 新增 {added} 条，"
              f"累计 {len(data)}/{args.n}（累计剔除 {rejected}）")
        time.sleep(args.sleep)

    print("=" * 60)
    print(f"完成: {len(data)} 条 -> {args.out}")
    print(f"调用次数 {calls}，规则质检剔除 {rejected} 条")
    print("=" * 60)
    out_dir = os.path.dirname(args.out) or "."
    print(f"下一步: python prepare_data.py {args.out} --out-dir {out_dir}")
    print("       DATA_PATH=" + os.path.join(out_dir, "train.json") +
          " DATA_FORMAT=chat python train_lora.py")


if __name__ == "__main__":
    main()
