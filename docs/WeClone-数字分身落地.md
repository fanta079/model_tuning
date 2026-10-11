# WeClone 数字分身 — 落地方案（RTX 4060 Ti 8GB / WSL2）

> 目标：用你自己的聊天记录微调一个「说话像你」的模型。
> 上游项目：<https://github.com/xming521/WeClone>（本文对应 v0.4.01）
> 官方文档：<https://docs.weclone.love>

---

## 0. 先明确三件事（不看会踩坑）

### ① 这是与现有 Unsloth 管线**完全独立**的第二条管线

WeClone 的依赖是固定的：

| 组件 | WeClone 要求 | 你 `~/venvs/ft` 现状 |
|---|---|---|
| torch | `2.7.1+cu126` | `2.14.1+cu130` |
| transformers | `4.53.2` | 更高版本 |
| 框架 | `llamafactory==0.9.4` + `vllm==0.10.0` | `unsloth` |

> ⚠️ **绝对不要把 WeClone 装进 `~/venvs/ft`** —— 会把现有能跑的环境搞坏。
> 正确做法：用它**项目内的 `.venv`**（第 1 步），两套环境互不干扰。

### ② 显存：官方默认配置你跑不动，必须改

- 官方默认 `Qwen2.5-7B-Instruct` + **LoRA(fp16) ≈ 16GB** → 你的 8GB 不够
- 官方显存表里 **QLoRA 4bit / 7B ≈ 6GB** → ✅ 改配置即可（第 3 步）

### ③ 合规红线

微信官方有专门的
《[针对违规获取及利用微信终端用户数据行为的打击公告](https://mp.weixin.qq.com/s/A6h4ZLTE2EPrY7kJ5fHE2g)》。

- 仅限**个人学习自用**，不对外提供服务、不商用
- 需获得**对话相关方的授权**
- 详情见第 8 节

---

## 1. 环境搭建（WSL 内执行）

```bash
cd ~                      # 放 Linux 家目录，不要放 /mnt/e（IO 慢）
git clone https://github.com/xming521/WeClone.git
cd WeClone

uv venv .venv --python=3.12          # 必须 3.12（pyproject 限定 >=3.12,<3.13）
source .venv/bin/activate

# 主依赖（国内走清华镜像）
uv pip install --group main -e . -i https://pypi.tuna.tsinghua.edu.cn/simple/

# 中文脱敏要用的 spaCy 模型（Presidio 依赖）
uv pip install https://github.com/explosion/spacy-models/releases/download/zh_core_web_sm-3.8.0/zh_core_web_sm-3.8.0-py3-none-any.whl

# 复制配置模板 → 之后所有配置都改这个文件
cp settings.template.jsonc settings.jsonc

# 验证 CUDA（宿主驱动要求 ≥ 12.6，你是 13.1 ✅）
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

用完只需 `deactivate` 就回到原环境 —— **`.venv` 与 `~/venvs/ft` 完全隔离**。

---

## 2. 数据准备

### 2.1 先下模型（国内走魔搭）

```bash
pip install modelscope
modelscope download --model Qwen/Qwen2.5-7B-Instruct --local_dir ./models/Qwen2.5-7B-Instruct
```

（也可以换更小的模型，见第 3 节小模型选项。）

### 2.2 聊天记录

| 平台 | 官方支持 | 做法 |
|---|---|---|
| **Telegram** | ✅ 内置 | Telegram Desktop → 右上角「导出聊天记录」→ 选 **JSON** → 把每个联系人的 `ChatExport_*` 文件夹一起放进 `./dataset/telegram/`（**不建议群聊**，会学成"混合声线"）|
| **微信** | ✅ **源码里是一等公民** | 用 `platform: "chat"` |

> 🔍 **源码证据**（`weclone/data/models.py`）：里面内置了微信专属的消息类型名，中英双语——
> **丢弃**：图片 / 视频 / 语音 / 合并转发的聊天记录 / 分享卡片 / 粘贴的文本；
> **跳过**：添加好友 / 动画表情 / 位置 / 文件 / 引用回复 / 转账 / 语音通话 / **系统通知 / 消息撤回 / 拍一拍**……
> 全是微信术语 —— 说明 `chat` 平台本来就是给微信设计的。
>
> ⚠️ 但 v0.4.01 的 **README 没写微信导出步骤**（只写了 Telegram），需要你自行导出后整理格式。
>
> 消息结构（源码里的 `ChatMessage`）：
> `id, MsgSvrID, type_name, is_sender, talker, msg, src, CreateTime, room_name`
> 其中 `is_sender`：**0 = 对方，1 = 自己**
>
> → 把一份**脱敏样例**（几行，真实内容替换掉）发我，我按这个结构给你写转换脚本。

---

## 3. 适配 8GB 显存（改 `settings.jsonc`）

改这几处就够了：

| 配置项 | 官方默认 | 改成 | 原因 |
|---|---|---|---|
| `common_args.finetuning_type` | `lora` | **`qlora`** | 4bit 量化，16GB → 约 6GB |
| `train_sft_args.per_device_train_batch_size` | `2` | **`1`** | 8GB 只放得下 1 条 |
| `train_sft_args.gradient_accumulation_steps` | `16` | **`32`** | 保持等效 batch 不缩水 |
| `train_sft_args.cutoff_len` | `2048` | `1024` | 先降保稳，跑通再往上加 |
| `train_sft_args.flash_attn` | `"fa2"` | **先删掉这行** | 免装 flash-attn，跑通再说 |

**可选：换更小/更顺手的基础模型**

```jsonc
"model_name_or_path": "./models/Qwen2.5-7B-Instruct",   // 官方默认
"template": "qwen",
```
- 想更省显存 → 换 `Qwen2.5-3B-Instruct`
- 想复用你已有的 `Qwen3.5-4B` → 可以，但 `template` 要按它的 chat 模板改

---

## 4. 数据处理

```bash
weclone-cli make-dataset                     # 清洗 / 脱敏 / 按时间窗配对成 QA
weclone-cli build-profile --profile-only     # (可选) 生成用户画像
```

关键机制（`settings.jsonc` 的 `make_dataset_args`）：

| 参数 | 默认 | 含义 |
|---|---|---|
| `single_combine_time_window` | 2（分钟）| 同一人连续消息在 2 分钟内**合并成一句** |
| `qa_match_time_window` | 5（分钟）| 5 分钟内的一来一回**配成一个 QA** |
| `include_type` | `["text"]` | 只保留文本消息 |
| `blocked_words` | 占位 | **自己补上**要过滤的词（含该词的整句会被删）|
| `clean_dataset.enable_clean` | `false` | 开启后用 LLM 给数据打分过滤（需 API）|

脱敏：默认用 Microsoft Presidio 去掉电话/邮箱/信用卡/IP/地理位置/身份证等，
但**不能保证 100%**，所以 `blocked_words` 一定要自己补。

**加密**：`security_args.storage_mode` 默认 `encrypted`，首次运行会让你设密码，产物以 `.enc` 结尾。

> ⚠️ **`.weclone/` 目录必须保留** —— 里面的密钥是读取加密数据的唯一凭据；
> 加密模式下**忘记密码无法恢复**，只能从原始聊天重新生成。

---

## 5. 训练

```bash
weclone-cli train-sft
```

产物路径 = `common_args.adapter_name_or_path`（默认 `./model_output`）。

官方默认训练参数（供参考，别乱改）：

```
lora_rank: 8        lora_target: "q_proj,v_proj"     lora_dropout: 0.25
learning_rate: 1e-4 num_train_epochs: 2              lr_scheduler: cosine
```

> 注意 `lora_target` 只挂 2 个投影层、`rank` 只有 8 —— 参数量极小，
> **好处**：不容易过拟合、训得快；**代价**："像"的程度有限。
> 想更像就把 `lora_target` 扩到 `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`、`lora_rank` 提到 16~32。

---

## 6. 验证与使用

```bash
weclone-cli webchat-demo                        # 浏览器 demo，调 temperature / top_p
weclone-cli server --inference --port 8005      # OpenAI 兼容 API
weclone-cli test-model                          # 内置闲聊题测试 → test_result-my.txt
```

接机器人（AstrBot / LangBot）三个要点：

1. 模型名填 `gpt-3.5-turbo`，API Key **随便填**
2. **必须关掉工具调用**（发 `/tool off_all`），否则微调效果会被工具调用覆盖
3. 系统提示词要与 `settings.jsonc` 的 `default_system`（默认「请你扮演一名人类，不要说自己是人工智能」）**保持一致**

---

## 7. 坑清单

| 坑 | 说明 |
|---|---|
| **装错环境** | WeClone 必须独立 venv；装进 `~/venvs/ft` 会毁掉 Unsloth |
| Python 版本 | 必须 **3.12**（3.13 会被 pyproject 拒绝）|
| CUDA 版本 | 要求 ≥ 12.6（宿主 13.1 ✅ 没问题）|
| 项目放 `/mnt/e` | IO 慢、权限怪；建议放 Linux 家目录 |
| 忘记密码 | 加密模式下**不可恢复** |
| flash-attn | 先别装，8GB 上收益有限还折腾 |
| 7B 效果一般 | 官方自己写「7B 效果一般，14B+ 更好」——你 8GB 只能 7B/3B，**预期要放低** |
| 群聊数据 | 会学成"混合声线"，建议只用 1v1 |

---

## 8. 合规（务必读）

- WeClone 的免责声明明确：**仅供学习研究，生产环境风险自担**，违法后果自负
- 微信官方公告已明确打击"违规获取及利用微信终端用户数据"
- 自用建议：**只用自己的记录、不对外、不商用、界面标注"AI 生成"**

---

## 附：与现有管线的关系

| | 现有管线 | WeClone |
|---|---|---|
| 用途 | 学通用指令 / 话术 | 学"某人的说话风格" |
| 框架 | Unsloth + TRL | LLaMA-Factory |
| 环境 | `~/venvs/ft` | `WeClone/.venv` |
| 数据格式 | `conversations:[{role,content}]` | LLaMA-Factory sharegpt |
| 脚本 | `train_lora.py` / `chat_lora.py` | `weclone-cli` |

**两者并存、互不干扰。** 你现有那套继续用来做话术/指令微调，
WeClone 用来做数字分身 —— 目标不同，工具不同，环境隔离。
