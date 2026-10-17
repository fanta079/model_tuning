#!/usr/bin/env python
"""upload_models.py — 把 adapter 传到 ModelScope（自动建仓 + 上传 + 校验）

用法（在 ~/venvs/ft 里跑）:
    python upload_models.py --whoami                                   # 只查用户名
    python upload_models.py --dir /mnt/e/VMOS_workspace/ev-models/ev-v4/lora_adapter --name ev-v4
    python upload_models.py --dir ... --name ev-v4 --owner Fanta79      # 手动指定用户名

踩过的两个坑（写在这里免得再犯）:
    ① ModelScope 用户名 ≠ GitHub 用户名。
       本机是 ModelScope=`Fanta79` / GitHub=`fanta079`，用错就 404：
       "The project you were looking for could not be found: models/fanta079/ev-v4"
    ② modelscope 1.x 的 `HubApi.get_user_info()` 是 **staticmethod**，
       读本地凭证文件 ~/.modelscope/credentials/user，不是请求接口。
       当实例方法用会 AttributeError: 'HubApi' object has no attribute 'get_user_info'
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys


# ---------------------------------------------------------------- 账号

def read_username() -> str:
    """从本地凭证拿 ModelScope 用户名。"""
    try:
        from modelscope.hub.api import HubApi
        info = HubApi.get_user_info()            # staticmethod -> (username, email)
        if isinstance(info, (tuple, list)) and info and info[0]:
            return str(info[0])
        if isinstance(info, dict):
            for k in ("Username", "Name", "username", "name"):
                if info.get(k):
                    return str(info[k])
    except Exception:                            # noqa: BLE001
        pass

    for cand in (
        os.path.expanduser("~/.modelscope/credentials/user"),
        os.path.expanduser("~/.cache/modelscope/credentials/user"),
    ):
        if os.path.isfile(cand):
            name = open(cand, encoding="utf-8").read().split(":", 1)[0].strip()
            if name:
                return name

    raise SystemExit(
        "[错误] 拿不到 ModelScope 用户名。\n"
        "  先跑一次: modelscope login --token ms-xxx\n"
        "  或指定:   --owner <你的ModelScope用户名>"
    )


def get_api(token: str = ""):
    from modelscope.hub.api import HubApi
    api = HubApi()
    if token:
        api.login(token, force=True)
    return api


# ---------------------------------------------------------------- 建仓

def ensure_repo(api, model_id: str, private: bool = True) -> None:
    """建仓；已存在就跳过。优先新接口 create_repo，失败退回 create_model。"""
    vis = "private" if private else "public"
    exists_kw = ("already exist", "已存在", "existed", "Conflict", "conflict")
    first_err = ""

    try:
        api.create_repo(model_id, "model", visibility=vis,
                        license="Apache License 2.0")
        print(f"[建仓] 已创建 {model_id}（{vis}）")
        return
    except Exception as e:                       # noqa: BLE001
        if any(k in str(e) for k in exists_kw):
            print(f"[建仓] {model_id} 已存在，直接上传")
            return
        first_err = f"{type(e).__name__}: {e}"

    try:
        api.create_model(model_id, visibility=1 if private else 5)
        print(f"[建仓] 已创建 {model_id}（旧接口）")
        return
    except Exception as e:                       # noqa: BLE001
        if any(k in str(e) for k in exists_kw):
            print(f"[建仓] {model_id} 已存在，直接上传")
            return
        print(f"[建仓] 建仓失败（{type(e).__name__}: {str(e)[:160]}）")
        print("       若随后上传仍失败，去网页手动建：")
        print(f"       modelscope.cn → 我的 → 模型 → 创建模型 → 名称 {model_id.split('/')[-1]} → 私有")
        if first_err:
            print(f"       （新接口错误：{first_err[:160]}）")


# ---------------------------------------------------------------- 上传

def upload(api, model_id: str, local: str, name: str) -> None:
    try:
        api.upload_folder(repo_id=model_id, folder_path=local,
                          repo_type="model", commit_message=f"upload {name}")
    except TypeError:
        # 老版本签名是 model_id=
        api.upload_folder(model_id=model_id, folder_path=local,
                          repo_type="model", commit_message=f"upload {name}")


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser(description="上传 LoRA adapter 到 ModelScope")
    ap.add_argument("--dir", help="本地 adapter 目录")
    ap.add_argument("--name", default="ev-v4", help="模型仓库名（默认 ev-v4）")
    ap.add_argument("--owner", default="", help="ModelScope 用户名（默认自动读本地凭证）")
    ap.add_argument("--token", default="", help="访问令牌（默认用已登录的凭证）")
    ap.add_argument("--public", action="store_true", help="建成公开仓库（默认私有）")
    ap.add_argument("--whoami", action="store_true", help="只打印用户名")
    args = ap.parse_args()

    if args.whoami or not args.dir:
        print(f"[账号] ModelScope 用户名: {args.owner or read_username()}")
        print("[提示] upload 的 owner 必须是这个，不是 GitHub 的 fanta079")
        return 0

    local = os.path.abspath(args.dir)
    weights = os.path.join(local, "adapter_model.safetensors")
    if not os.path.isfile(weights):
        print(f"[错误] 不是 adapter 目录（缺 adapter_model.safetensors）: {local}")
        return 1

    owner = args.owner or read_username()
    model_id = f"{owner}/{args.name}"
    size_mb = os.path.getsize(weights) / 2**20
    print(f"[目标] {model_id}  <-  {local}")
    print(f"[大小] {size_mb:.0f} MB (adapter_model.safetensors)")

    api = get_api(args.token)
    ensure_repo(api, model_id, private=not args.public)

    print("[上传] 开始（断了可重跑，已传的会跳过）……")
    upload(api, model_id, local, args.name)

    print()
    print("[完成] https://www.modelscope.cn/models/" + model_id)
    print("[校验] 本地 sha256:")
    print(f"  {sha256_of(weights)}  adapter_model.safetensors")
    print("  与 ev-models 仓的 MANIFEST.md 对照，一致即传对")
    return 0


if __name__ == "__main__":
    sys.exit(main())
