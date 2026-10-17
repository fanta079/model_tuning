#!/usr/bin/env python
"""upload_models.py — 把 adapter 传到 ModelScope（自动建仓 + 上传 + 校验）

用法（在 ~/venvs/ft 里跑）:
    python upload_models.py --dir /mnt/e/VMOS_workspace/ev-models/ev-v4/lora_adapter --name ev-v4
    python upload_models.py --dir ... --name ev-v4 --token ms-xxxx     # 不想登录时直接给 token
    python upload_models.py --whoami                                   # 只查用户名

为什么需要这个脚本:
    `modelscope upload fanta079/ev-v4 ...` 会失败，因为
    ① 仓库必须先存在（CLI 不会自动建）
    ② ModelScope 的用户名 ≠ GitHub 用户名（fanta079 是 GitHub 的）
"""
from __future__ import annotations

import argparse
import os
import sys


def get_api(token: str = ""):
    from modelscope.hub.api import HubApi
    api = HubApi()
    if token:
        api.login(token, force=True)
    return api


def username_of(api) -> str:
    """尽量从 API 里挖出当前账号的用户名 / namespace。"""
    info = api.get_user_info()
    if not isinstance(info, dict):
        raise SystemExit(f"[错误] get_user_info 返回了意外内容: {info!r}")
    for key in ("Username", "Name", "username", "name", "Account", "account"):
        v = info.get(key)
        if v:
            return str(v)
    raise SystemExit(
        "[错误] 没能从账号信息里找到用户名，原始返回如下，请手动填 --owner:\n"
        f"  {info}"
    )


def ensure_repo(api, model_id: str, private: bool = True) -> None:
    """建仓；已存在就跳过。"""
    try:
        api.create_model(model_id, visibility=1 if private else 5,
                         license="Apache License 2.0")
        print(f"[建仓] 已创建 {model_id}（{'私有' if private else '公开'}）")
    except Exception as e:                        # noqa: BLE001
        msg = str(e)
        if any(k in msg for k in ("already exist", "已存在", "existed", "Conflict")):
            print(f"[建仓] {model_id} 已存在，直接上传")
        else:
            # 建不成就直接尝试上传 —— 有些账号建仓接口权限不同，但上传是通的
            print(f"[建仓] 跳过（{type(e).__name__}: {msg[:120]}）")


def main() -> int:
    ap = argparse.ArgumentParser(description="上传 LoRA adapter 到 ModelScope")
    ap.add_argument("--dir", help="本地 adapter 目录")
    ap.add_argument("--name", default="ev-v4", help="模型仓库名（默认 ev-v4）")
    ap.add_argument("--owner", default="", help="ModelScope 用户名（不填则自动探测）")
    ap.add_argument("--token", default="", help="访问令牌（不填则用已登录的凭证）")
    ap.add_argument("--public", action="store_true", help="建成公开仓库（默认私有）")
    ap.add_argument("--whoami", action="store_true", help="只打印用户名")
    args = ap.parse_args()

    api = get_api(args.token)

    if args.whoami or not args.dir:
        name = username_of(api)
        print(f"[账号] ModelScope 用户名: {name}")
        print(f"[提示] 上传命令里的 owner 应该是「{name}」，不是 GitHub 的名字")
        return 0

    local = os.path.abspath(args.dir)
    if not os.path.isdir(local):
        print(f"[错误] 目录不存在: {local}")
        return 1
    weights = os.path.join(local, "adapter_model.safetensors")
    if not os.path.exists(weights):
        print(f"[错误] 没有 adapter_model.safetensors，这不是一个 adapter 目录: {local}")
        return 1

    owner = args.owner or username_of(api)
    model_id = f"{owner}/{args.name}"
    print(f"[目标] {model_id}  <-  {local}")
    print(f"[大小] {os.path.getsize(weights) / 2**20:.0f} MB (adapter_model.safetensors)")

    ensure_repo(api, model_id, private=not args.public)

    print("[上传] 开始（断了可以重跑，已传的会跳过）……")
    api.upload_folder(
        model_id=model_id,
        folder_path=local,
        repo_type="model",
        commit_message=f"upload {args.name}",
    )

    print()
    print("[完成] 仓库地址:")
    print(f"  https://www.modelscope.cn/models/{model_id}")
    print()
    print("[校验] 本地 sha256:")
    import hashlib
    h = hashlib.sha256()
    with open(weights, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    print(f"  {h.hexdigest()}  adapter_model.safetensors")
    print("  与 ev-models 仓里的 MANIFEST.md 对照，一致就说明传对了")
    return 0


if __name__ == "__main__":
    sys.exit(main())
