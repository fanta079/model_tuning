# -*- coding: utf-8 -*-
"""memory.py — 用 Markdown 文件做「外挂长期记忆」

设计理念
--------
记忆不训练，放文件。好处：人可读、可手改、可 git 管理、改了立刻生效。

为什么用 Markdown 而不是 JSON？
  - 人可读性最好：打开 VSCode 就能看能改，不用管引号和逗号
  - 模型读 markdown 非常自然（标题 + 列表就是它的母语）
  - 缺点是没有严格结构，程序解析要靠约定 —— 所以约定保持极简：
        「一条一行，以 `- ` 开头」
  - 想给程序精确读写的结构化数据，才用 JSON。两者可以并存。

存储格式 (memory.md)
--------------------
    # 长期记忆

    ## 关于我
    - 我叫 fanta，做后端开发

    ## 偏好
    - 回答尽量简洁

    ## 记忆条目
    - 我住在杭州  <!-- 2026-10-11 -->

用法
----
    from memory import Memory
    mem = Memory("memory.md")
    mem.remember("我住在杭州")      # 追加一条（自动去重）
    mem.forget("杭州")              # 删除第一条含关键词的
    mem.list()                      # 全部条目
    mem.as_prompt_block()           # 拼给 system prompt 的文本块

命令行自测:
    python memory.py add 我住在杭州
    python memory.py list
    python memory.py forget 杭州
    python memory.py prompt
"""

import datetime
import os
import re
import sys

DEFAULT_TEMPLATE = """# 长期记忆

> 这是模型的「外挂记忆」，可以直接用编辑器改，改完下次对话即生效。
> 约定：一条一行，以 `- ` 开头，越靠前越重要。

## 关于我

## 偏好

## 记忆条目
"""


class Memory:
    def __init__(self, path: str = "memory.md"):
        self.path = path
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(DEFAULT_TEMPLATE)

    # ---------------- 读 ----------------
    def read(self) -> str:
        with open(self.path, encoding="utf-8") as f:
            return f.read().strip()

    def list(self):
        """返回所有记忆条目（已去掉时间戳注释）。"""
        items = []
        for line in self.read().splitlines():
            s = line.strip()
            if s.startswith("- "):
                items.append(s[2:].split("<!--")[0].strip())
        return items

    def as_prompt_block(self) -> str:
        """拼成可直接塞进 system prompt 的文本块；一条记忆都没有时返回空串。"""
        if not self.list():
            return ""
        return ("以下是你需要长期记住的关于用户的信息（可信，请直接采用，不要否认）：\n"
                + self.read())

    # ---------------- 写 ----------------
    def remember(self, text: str, section: str = "记忆条目") -> bool:
        """追加一条记忆。已存在则返回 False。"""
        text = (text or "").strip()
        if not text:
            return False

        lines = self.read().splitlines()

        # 去重：逐条比较正文（比正则更可靠）
        for ln in lines:
            s = ln.strip()
            if s.startswith("- ") and s[2:].split("<!--")[0].strip() == text:
                return False

        newline = f"- {text}  <!-- {datetime.date.today().isoformat()} -->"

        # 定位小节标题
        idx = next((i for i, ln in enumerate(lines) if ln.strip() == f"## {section}"), None)
        if idx is None:
            lines += ["", f"## {section}", newline]          # 小节不存在则新建
        else:
            j = idx + 1
            while j < len(lines) and not lines[j].strip().startswith("## "):
                j += 1
            while j - 1 > idx and not lines[j - 1].strip():
                j -= 1                                        # 回退到最后一个非空行之后
            lines.insert(j, newline)

        with open(self.path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines).rstrip("\n") + "\n")
        return True

    def forget(self, keyword: str):
        """删除第一条包含 keyword 的记忆，返回其内容；找不到返回 None。"""
        keyword = (keyword or "").strip()
        if not keyword:
            return None
        lines = self.read().splitlines()
        for i, ln in enumerate(lines):
            s = ln.strip()
            if s.startswith("- ") and keyword in s:
                removed = s[2:].split("<!--")[0].strip()
                del lines[i]
                with open(self.path, "w", encoding="utf-8") as f:
                    f.write("\n".join(lines) + "\n")
                return removed
        return None


# ---------------- 命令行自测 ----------------
if __name__ == "__main__":
    mem = Memory(sys.argv[1] if len(sys.argv) > 2 and sys.argv[1].endswith(".md") else "memory.md")
    args = [a for a in sys.argv[1:] if not a.endswith(".md")]
    cmd = args[0] if args else "list"
    rest = args[1:]

    if cmd == "add":
        ok = mem.remember(" ".join(rest))
        print("已记住" if ok else "已存在，跳过")
    elif cmd == "forget":
        gone = mem.forget(" ".join(rest))
        print(f"已忘记: {gone}" if gone else "没找到")
    elif cmd == "prompt":
        print(mem.as_prompt_block() or "(暂无记忆)")
    else:
        items = mem.list()
        print(f"共 {len(items)} 条记忆:")
        for i, t in enumerate(items, 1):
            print(f"  {i}. {t}")
