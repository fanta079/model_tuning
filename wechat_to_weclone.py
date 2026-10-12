# -*- coding: utf-8 -*-
"""wechat_to_weclone.py — 把微信导出的聊天记录转成 WeClone 期望的 CSV

WeClone 的输入要求（从仓库自带样例 tests/tests_data/test_person/test_0_730.csv 反推）:
    放在 dataset/wechat/ 目录下，表头固定 9 列：
    id,MsgSvrID,type_name,is_sender,talker,room_name,msg,src,CreateTime

    其中:
      type_name : 中文类型名(文本/图片/动画表情/系统通知/引用回复...)
      is_sender : 1 = 自己, 0 = 对方
      talker    : 发送者 ID; is_sender=1 时是自己的 ID
      room_name : 会话名(一般是对面的人)
      src       : 媒体文件路径, 文本消息留空
      CreateTime: 形如 2024/10/4 11:43

PyWxDump 的原始 CSV 列名是 localId/Type/IsSender/StrContent/StrTalker 这类,
且 Type 是数字 —— 所以要转换。

用法
----
    # 最简：自动识别列名, 自动推断"我"
    python wechat_to_weclone.py 导出.csv --out-dir ./dataset/wechat

    # 指定"我"的标识（推荐, 避免猜错）
    python wechat_to_weclone.py 导出.csv --me "我的昵称" --out-dir ./dataset/wechat

    # 列名太怪？手动映射
    python wechat_to_weclone.py 导出.csv --map "内容=StrContent" --map "时间=CreateTime"

支持输入: .csv / .tsv / .json / .jsonl（JSON 需是对象数组）

自测:
    python wechat_to_weclone.py --selftest
"""

import argparse
import csv
import datetime as dt
import json
import os
import re
import sys

# WeClone 要求的固定表头
HEADER = ["id", "MsgSvrID", "type_name", "is_sender", "talker",
          "room_name", "msg", "src", "CreateTime"]

# 目标字段 -> 常见别名（不区分大小写；数字类型会另做映射）
ALIASES = {
    "id":         ["id", "localid", "local_id", "序号", "msgid"],
    "MsgSvrID":   ["msgsvrid", "msg_svr_id", "svrid", "serverid", "消息id"],
    "type_name":  ["type_name", "typename", "type", "类型", "消息类型", "msgtype"],
    "is_sender":  ["is_sender", "issender", "issend", "是否发送", "发送方向"],
    "talker":     ["talker", "strtalker", "sender", "发送者", "昵称", "nickname", "from"],
    "room_name":  ["room_name", "roomname", "chatroom", "会话", "群名", "聊天对象"],
    "msg":        ["msg", "strcontent", "content", "内容", "消息", "text", "body"],
    "src":        ["src", "path", "媒体", "文件", "filepath", "url"],
    "CreateTime": ["createtime", "strtime", "time", "时间", "日期", "datetime", "timestamp"],
}

# PyWxDump / 微信数字 Type -> 中文类型名（WeClone 的 skip/cut 名单用的就是这些词）
TYPE_MAP = {
    "1": "文本",
    "3": "图片",
    "34": "语音",
    "42": "名片",
    "43": "视频",
    "47": "动画表情",
    "48": "位置",
    "49": "分享卡片",
    "50": "语音通话",
    "10000": "系统通知",
    "10002": "消息撤回",
}

# 只有这些类型会进入训练（WeClone 的 include_type 默认 ["text"]）
TEXT_LIKE = {"文本", "text", ""}


def norm_key(k):
    return str(k).strip().lower().replace(" ", "").replace("-", "")


def detect_columns(rows, overrides):
    """返回 {目标字段: 源列名}。overrides 优先。"""
    src_keys = list(rows[0].keys())
    normalized = {norm_key(k): k for k in src_keys}
    mapping = {}

    for field, aliases in ALIASES.items():
        if field in overrides:
            mapping[field] = overrides[field]
            continue
        for a in aliases:
            if a in normalized:
                mapping[field] = normalized[a]
                break
        else:
            mapping[field] = None
    return mapping


def to_type_name(raw):
    """把 Type 转成中文类型名；已经是中文的原样返回。"""
    v = str(raw or "").strip()
    if not v:
        return "文本"
    if v in TYPE_MAP:
        return TYPE_MAP[v]
    if re.fullmatch(r"-?\d+", v):        # 未知的数字类型
        return "未知"
    return v                              # 已经是中文


def parse_time(raw):
    """尽量把各种时间写法归一成 2024/10/4 11:43。"""
    s = str(raw or "").strip()
    if not s:
        return ""
    # 纯数字时间戳（秒 / 毫秒）—— 手工格式化，避免 %-m / %#m 的平台差异
    if re.fullmatch(r"\d{10,13}", s):
        ts = int(s) / (1000 if len(s) == 13 else 1)
        d = dt.datetime.fromtimestamp(ts)
        return f"{d.year}/{d.month}/{d.day} {d.hour}:{d.minute:02d}"
    # 统一分隔符
    s2 = s.replace("年", "/").replace("月", "/").replace("日", " ").replace("-", "/")
    m = re.match(r"(\d{4})/(\d{1,2})/(\d{1,2})[ T]?(\d{1,2})?:?(\d{1,2})?", s2)
    if m:
        y, mo, d, h, mi = m.groups()
        return f"{int(y)}/{int(mo)}/{int(d)} {int(h or 0)}:{int(mi or 0):02d}"
    return s


def load_rows(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".json", ".jsonl"):
        with open(path, encoding="utf-8") as f:
            txt = f.read().strip()
        if ext == ".jsonl":
            return [json.loads(l) for l in txt.splitlines() if l.strip()]
        data = json.loads(txt)
        if isinstance(data, dict):
            for key in ("messages", "data", "items", "list"):
                if isinstance(data.get(key), list):
                    return data[key]
            raise SystemExit("[错误] JSON 顶层是对象但不含 messages/data/items/list")
        return data
    # CSV / TSV
    with open(path, encoding="utf-8-sig", newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",\t;")
        except csv.Error:
            dialect = csv.excel
        return list(csv.DictReader(f, dialect=dialect))


def convert(rows, mapping, me, room_name, verbose=True):
    out, skipped = [], 0
    if verbose:
        print("[列映射]")
        for f, src in mapping.items():
            print(f"   {f:12s} <- {src or '(缺失)'}")

    for i, r in enumerate(rows, 1):
        def get(field):
            col = mapping.get(field)
            return (r.get(col) if col else None)

        msg = str(get("msg") or "").strip()
        tname = to_type_name(get("type_name"))
        src = str(get("src") or "").strip()

        # 非文本消息的正文常常是占位符（"图片"/"表情"），统一清空
        if tname not in TEXT_LIKE:
            if msg in ("图片", "表情", "语音", "视频", "文件", "位置", "动画表情"):
                msg = ""

        # 判定发送方
        raw_is = str(get("is_sender") or "").strip()
        if raw_is in ("1", "1.0", "true", "True"):
            is_sender = 1
        elif raw_is in ("0", "0.0", "false", "False"):
            is_sender = 0
        elif me:
            is_sender = 1 if str(get("talker") or "").strip() == str(me).strip() else 0
        else:
            is_sender = 0

        if not msg and not src:
            skipped += 1
            continue

        out.append({
            "id":                 str(get("id") or i).strip(),
            "MsgSvrID":           str(get("MsgSvrID") or f"local{i}").strip(),
            "type_name":          tname,
            "is_sender":          is_sender,
            "talker":             str(get("talker") or (me if is_sender else room_name) or "").strip(),
            "room_name":          str(get("room_name") or room_name or "").strip(),
            "msg":                msg,
            "src":                src,
            "CreateTime":         parse_time(get("CreateTime")),
        })
    return out, skipped


def main():
    ap = argparse.ArgumentParser(description="微信导出 -> WeClone CSV")
    ap.add_argument("input", nargs="?", help="源文件 (.csv/.tsv/.json/.jsonl)")
    ap.add_argument("--out-dir", default="./dataset/wechat", help="输出目录，默认 ./dataset/wechat")
    ap.add_argument("--out-name", default="", help="输出文件名，默认沿用源文件名")
    ap.add_argument("--me", default="", help="你自己的标识（昵称或 ID），用来判定 is_sender")
    ap.add_argument("--room", default="", help="会话名（room_name），默认取第一条的 talker")
    ap.add_argument("--map", action="append", default=[],
                    metavar="目标=源列名", help="手动指定列映射，可多次")
    ap.add_argument("--selftest", action="store_true", help="用内置样例自测")
    args = ap.parse_args()

    if args.selftest:
        rows = [
            {"localId": 1, "MsgSvrID": "111", "Type": 1, "IsSender": 1,
             "StrTalker": "me_id", "StrContent": "在吗", "StrTime": "2024-10-04 11:43"},
            {"localId": 2, "MsgSvrID": "222", "Type": 3, "IsSender": 0,
             "StrTalker": "friend", "StrContent": "图片", "StrTime": "2024-10-04 11:44"},
            {"localId": 3, "MsgSvrID": "333", "Type": 1, "IsSender": 0,
             "StrTalker": "friend", "StrContent": "在的", "StrTime": "2024-10-04 11:45"},
        ]
        mp = {f: None for f in HEADER}
        mp.update({"id": "localId", "MsgSvrID": "MsgSvrID", "type_name": "Type",
                   "is_sender": "IsSender", "talker": "StrTalker",
                   "msg": "StrContent", "CreateTime": "StrTime"})
        out, _ = convert(rows, mp, "me_id", "friend")
        print("\n[自测输出]")
        w = csv.DictWriter(sys.stdout, fieldnames=HEADER)
        w.writeheader()
        w.writerows(out)
        return

    if not args.input or not os.path.isfile(args.input):
        ap.error("请提供存在的输入文件，或用 --selftest")

    overrides = {}
    for kv in args.map:
        if "=" not in kv:
            ap.error(f"--map 需要 目标=源列名 的形式，收到: {kv}")
        k, v = kv.split("=", 1)
        overrides[k.strip()] = v.strip()

    rows = load_rows(args.input)
    if not rows:
        sys.exit("[错误] 没读到任何数据行")
    print(f"[读取] {len(rows)} 行，源列名: {list(rows[0].keys())}")

    mapping = detect_columns(rows, overrides)

    room = args.room
    if not room and mapping.get("talker"):
        # 启发式：把第一条的 talker 当作"对方"，其相反的就是"我"
        room = str(rows[0].get(mapping["talker"]) or "").strip()
    me = args.me
    if not me and mapping.get("talker"):
        # 从数据里找出现过的所有 talker，取"少数派"当自己（1v1 场景通常只有两个）
        vals = [str(r.get(mapping["talker"]) or "").strip() for r in rows]
        uniq = [v for v in dict.fromkeys(vals) if v]
        if len(uniq) == 2:
            me = uniq[1] if uniq[0] == room else uniq[0]
            print(f"[推断] room_name={room!r}, me={me!r}（可用 --me/--room 覆盖）")

    out, skipped = convert(rows, mapping, me, room)

    os.makedirs(args.out_dir, exist_ok=True)
    name = args.out_name or (os.path.splitext(os.path.basename(args.input))[0] + ".csv")
    path = os.path.join(args.out_dir, name)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=HEADER)
        w.writeheader()
        w.writerows(out)

    n_text = sum(1 for r in out if r["type_name"] in TEXT_LIKE)
    n_mine = sum(1 for r in out if r["is_sender"] == 1)
    print()
    print("=" * 56)
    print(f"输出       : {path}")
    print(f"有效消息   : {len(out)}  (文本类 {n_text})")
    print(f"其中你说的 : {n_mine}  /  对方说的 {len(out) - n_mine}")
    print(f"跳过空行   : {skipped}")
    print("=" * 56)
    if n_mine == 0 or n_mine == len(out):
        print("[警告] 全是同一个人说的 —— 检查 --me 是否正确，否则 is_sender 会错")
    print("下一步: cd WeClone && source .venv/bin/activate && weclone-cli make-dataset")


if __name__ == "__main__":
    main()
