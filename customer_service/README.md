# customer_service — 企业客服微调数据

## 核心原则：知识 ≠ 话术

| 内容 | 交给谁 | 为什么 |
|---|---|---|
| **知识**（产品参数、价格、政策、流程）| **RAG 检索** | 会变、要准、要能溯源；灌进权重后就"焊死"，改价得重训 |
| **话术**（语气、称呼、致歉、转人工）| **微调（LoRA）** | 正是微调擅长的：学"怎么说"，不是"说什么" |

所以本目录的数据**刻意不写产品参数**——比如"防水多少米"这类问题，示例回答是
「方便告诉我型号吗，我帮您查」而不是直接报数字。事实由 RAG 提供。

## 文件

| 文件 | 说明 |
|---|---|
| `all.json` | 原始数据（人工编写 / 生成后待清洗）|
| `train.json` / `val.json` | 由 `prepare_data.py` 生成，喂给训练脚本 |

## 数据格式（多轮）

```json
[
  {"conversations": [
    {"role": "system",    "content": "你是「XX」的客服小林，语气亲切专业，不确定的不编造。"},
    {"role": "user",      "content": "用户说的话"},
    {"role": "assistant", "content": "客服的回复"}
  ]}
]
```

- `role` 只允许 `system` / `user` / `assistant`；`user` 与 `assistant` 必须成对
- `system` 可省略（训练时用 `SYSTEM=` 环境变量统一注入）
- 兼容单轮 Alpaca：`{"instruction": "...", "input": "...", "output": "..."}`

## 使用流程

```bash
source ~/venvs/ft/bin/activate
cd /mnt/e/VMOS_workspace/model_tuning

# 1) 清洗 + 去重 + 划分
python prepare_data.py customer_service/all.json

# 2) 训练（冒烟）
DATA_PATH=customer_service/train.json DATA_FORMAT=chat RESPONSES_ONLY=1 \
SYSTEM="你是「星辰科技」的客服小林，语气亲切专业，不确定的信息绝不编造。" \
python train_lora.py

# 3) 全量训练（后台）
DATA_PATH=customer_service/train.json DATA_FORMAT=chat RESPONSES_ONLY=1 NUM_EPOCHS=2 \
SMOKE_TEST=0 OUTDIR_KEY=cs \
nohup python train_lora.py > ~/train_cs.log 2>&1 &
```

## 数据来源建议（按真实性排序）

1. **真实客服会话记录** —— 最值钱，脱敏后直接用
2. **工单 + 人工回复** —— 天然一问一答
3. **产品文档 / FAQ** —— 用强模型批量生成问答对，再人工挑一遍

> 量不必大：**500~2000 条高质量数据**通常就能明显改变话术风格。
