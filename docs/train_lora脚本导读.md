# train_lora.py 逐段导读（零基础版）

> **写给谁**：会一点 Python，但没接触过大模型训练的人。
> **读完能得到什么**：知道这份脚本每一段在干什么、每个专业名词是什么意思、参数为什么是这个值、改哪里会发生什么。
> **不包含**：数据怎么造（见导航里的《人设数据生产指南》）。

---

## 0. 阅读地图

这份文档只讲**训练**这一段。项目里其他环节各有专门文档，按需跳转：

| 我想…… | 去看 |
|---|---|
| **搞懂 `train_lora.py` 每一行、每个名词** | **本文** |
| 知道一共有哪些脚本、各自怎么跑 | [训练与推理脚本](./训练与推理脚本.md) |
| 造人设数据（人设卡 / 场景矩阵 / 种子） | [人设数据生产指南](./人设数据生产指南.md) |
| 环境搭不起来、报错查不到原因 | [踩坑记录](./踩坑记录.md) |
| 把模型接成语音助手 | [P1-语音助手](./P1-语音助手.md) |
| 用微信聊天记录做数字分身 | [WeClone-数字分身落地](./WeClone-数字分身落地.md) |

**建议阅读顺序**：本文第 1 节（建立直觉）→ 第 2 节（看清全貌）→ 第 4 节（逐段走读）→ 遇到不懂的名词回第 3 节或第 9 节查。

---

## 1. 先建立直觉：训练到底在改什么

不先建立直觉，后面每个名词都会悬空。这一节只有四个概念：**权重、loss、梯度、学习率**。

### 1.1 大模型的三段成长

```
① 预训练 Pretraining
   读几万亿字网页/书籍，只学一件事："下一个字最可能是什么"
   ↓ 产物：底座模型 (base model) —— 知识很足，但不会好好聊天
        │
② 监督微调 SFT (Supervised Fine-Tuning)
   用「问题 → 标准答案」的配对数据，教它"按人类期望的方式回答"
   ↓ 产物：对话模型 / LoRA adapter        ◀━━ train_lora.py 在这一步
        │
③ 偏好对齐 RLHF / DPO
   用人类的"喜欢/不喜欢"排序，让它更有用、更无害
   ↓ 产物：最终产品
```

本项目**只做 ②**。你现在训的"人设模型"，本质是让底座学会一种**说话风格**（冷淡、极短、毒舌），这是典型的 SFT 任务。

### 1.2 训练在数学上只有一件事

```
        ┌──────────────────────────────────────────┐
        │                                          │
        ↓                                          │
  ① 给模型看一段文本                                 │
        ↓                                          │
  ② 模型预测"下一个 token 是什么"                     │
        ↓                                          │
  ③ 和真实的下一个 token 对比 → 算出 loss（差多少）      │
        ↓                                          │
  ④ 反向传播：算出每个参数"往哪边挪能降低 loss"（梯度）  │
        ↓                                          │
  ⑤ 按学习率挪一小步（优化器更新参数）────────────────┘
```

重复几万次，loss 从 3.x 降到 1.x，"说话方式"就变了。

**四个核心词**：

| 词 | 一句话 | 类比 |
|---|---|---|
| **权重（参数）** | 模型内部几亿～几十亿个数字，决定了它的一切行为 | 大脑里的突触强度 |
| **loss** | 模型预测和标准答案的差距，越小越好 | 考试扣的分数 |
| **梯度** | loss 对每个参数的导数，指明"往哪调" | 蒙眼下山时脚底感觉到的坡度 |
| **学习率** | 每次沿梯度挪多大一步 | 步子大小（太大滚下山，太小学得慢） |

### 1.3 LoRA 的比喻（这份脚本最关键的概念）

- **底座模型** = 一位读过万卷书的老教授，知识渊博，但说话一板一眼像论文。
- **全参微调** = 把这颗大脑熔了重铸。4B 模型要存 4B 个参数的梯度 + 优化器状态，8GB 显卡直接爆。
- **LoRA** = **不碰他的大脑**，只在他手边放一本 85MB 的小便签，每次开口前翻一眼便签。

技术上：

```
原始计算：  y = W · x            （W 是几百万维的巨型矩阵，冻结不动）
LoRA 之后：  y = W · x + (α/r) · B · A · x
                          └──────┬──────┘
                          只训练 A 和 B 两个小矩阵
                          A: r × d_in     （很扁）
                          B: d_out × r    （很瘦）
                          r = 16，所以 BA 的参数量只有 W 的千分之几
```

**一个必须知道的细节**：`B` 的初始值全是 **0**，所以训练第 0 步时 `B·A = 0`，也就是——

> 刚挂上 adapter、还没训练时，模型输出和底座**一模一样**。

这就是为什么"我加载了 adapter 但看不出区别"往往是正常的：要么没训，要么 `ADAPTER` 路径指错了（指到了 `checkpoint-*` 而不是 `lora_adapter`）。

### 1.4 三种"8GB 才能跑"的技术叠在一起 = QLoRA

| 技术 | 省的是什么 | 在本脚本哪里 |
|---|---|---|
| **量化（4bit）** | 底座权重显存：8.8GB → 2.6GB | `load_in_4bit=True` |
| **LoRA** | 需要训练/存梯度的参数：4B → 4000 万 | `get_peft_model` |
| **梯度检查点** | 中间激活值显存 | `use_gradient_checkpointing="unsloth"` |

三者合一就是 **QLoRA**（Quantized LoRA）。名字里的 Q 就是"量化过的底座"。

---

## 2. 五步全景

脚本 172 行，实际只做五件事：

```
┌─────────────────────────────────────────────────────────────┐
│  第 1 步  加载底座（4bit 量化）                                │
│           FastLanguageModel.from_pretrained(load_in_4bit=True)│
│           显存 +2.6GB                                        │
├─────────────────────────────────────────────────────────────┤
│  第 2 步  挂 LoRA（在关键层旁边插小矩阵）                       │
│           FastLanguageModel.get_peft_model(r=16, alpha=32)   │
│           可训练参数 4000 万（约 85MB）                        │
├─────────────────────────────────────────────────────────────┤
│  第 3 步  数据 → 文本（渲染成模型认识的对话格式）                │
│           load_dataset().map(to_text)                        │
│           每条样本变成一整段带特殊 token 的字符串                │
├─────────────────────────────────────────────────────────────┤
│  第 4 步  训练（SFTTrainer 跑训练循环）                        │
│           trainer.train()                                    │
│           loss 下降，LoRA 矩阵被更新                           │
├─────────────────────────────────────────────────────────────┤
│  第 5 步  保存 adapter（85MB，不含底座）                       │
│           model.save_pretrained(OUTPUT_DIR/lora_adapter)     │
└─────────────────────────────────────────────────────────────┘
```

| 步骤 | 核心代码对象 | 一句话 | 显存影响 |
|---|---|---|---|
| 1 | `FastLanguageModel` | 把大模型装进显卡 | +2.6GB（4bit） |
| 2 | `get_peft_model` | 给它挂一本小便签 | +0.1GB |
| 3 | `apply_chat_template` | 把问答对翻译成模型的语言 | 0（在内存） |
| 4 | `SFTTrainer` | 反复"预测→算 loss→改参数" | 峰值 ~7GB |
| 5 | `save_pretrained` | 只存便签，不存大脑 | 0 |

---

## 3. 名词速查（先扫一眼，看不懂就跳过）

后面第 4 节会逐个展开。这里先建立"这个词属于哪一类"的地图：

**模型与权重类**

| 名词 | 全称 | 一句话 |
|---|---|---|
| 底座 / base model | base model | 预训练完、还没做过指令微调的原始模型 |
| 权重 / 参数 | weights / parameters | 模型里那些数字，训练就是在改它们 |
| adapter | — | LoRA 训练出的那 85MB 小文件，必须配底座用 |
| safetensors | — | 一种权重文件格式，比 pickle 安全、加载快 |
| BF16 / FP16 | BFloat16 / Float16 | 数字精度格式，bf16 范围大不易溢出，训练首选 |
| 4bit / NF4 | NormalFloat4 | 把每个权重从 16 位压到 4 位，体积降到 1/4 |

**训练方法类**

| 名词 | 全称 | 一句话 |
|---|---|---|
| SFT | Supervised Fine-Tuning | 监督微调：用"问题→答案"配对数据教它回答 |
| LoRA | Low-Rank Adaptation | 低秩适配：不改编原权重，只训两个小矩阵 |
| QLoRA | Quantized LoRA | 4bit 底座 + LoRA，8GB 显存训 4B 的关键 |
| PEFT | Parameter-Efficient Fine-Tuning | LoRA 所属的大类：只训极少量参数 |
| RLHF / DPO | — | 用人类偏好做对齐，本项目不做 |

**训练过程类**

| 名词 | 一句话 |
|---|---|
| token | 模型眼里的"字块"，中文一个词常是 1 个 token |
| chat template | 模型训练时见过的对话拼接格式，用错就等于"说外语" |
| epoch | 把整个数据集完整过一遍叫 1 个 epoch |
| batch | 一次喂几条样本；显存只跟"一次喂几条"有关 |
| 梯度累积 | 攒够 N 次梯度再更新一次，用小显存模拟大 batch |
| learning rate | 每次更新挪多大一步 |
| warmup | 开头几十步让学习率从 0 慢慢升上来，防止一上来就把权重带崩 |
| loss | 预测与答案的差距 |
| 梯度检查点 | 用"反向时重算"换显存，慢 30% 但省一半以上 |
| packing | 把多条短样本拼成一条长样本，减少浪费 |
| label mask / -100 | 把某些 token 的 loss 关掉（PyTorch 约定 -100 = 忽略） |
| checkpoint | 训练中途的存档，含优化器状态，可续训 |

**代码对象类**

| 名词 | 是谁 | 干什么 |
|---|---|---|
| `FastLanguageModel` | Unsloth 库 | 加速版的模型加载器 + LoRA 挂载器 |
| `SFTTrainer` | TRL 库（HuggingFace） | **训练器**，封装了整个训练循环 |
| `SFTConfig` | TRL 库 | **训练配置**，一个装超参的盒子 |
| `trainer.train()` | — | 真正开始跑循环的那一行 |
| `load_dataset` | datasets 库 | 读 json/csv/parquet 成数据集对象 |
| `.map(fn)` | datasets 库 | 对每条样本执行一个函数，加新字段 |
| `apply_chat_template` | tokenizer | 把 `[{role,content}]` 渲染成模型认识的字符串 |

---

## 4. 逐段走读

下面按脚本实际顺序，一段代码配一段"人话翻译"。

### 4.1 第 36–40 行：三个 import 分别是谁

```python
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
```

| 库 | 定位 | 在这份脚本里负责 |
|---|---|---|
| **Unsloth** | 训练加速库（重写了注意力/反向传播内核） | 加载模型、挂 LoRA、省显存。**只在 Linux/WSL 可用** |
| **datasets** | HuggingFace 的数据集库 | 读 json、做 map 变换 |
| **TRL** | Transformer Reinforcement Learning，HuggingFace 官方的训练框架 | `SFTTrainer` 提供训练循环，`SFTConfig` 装超参 |

> **为什么不直接用 PyTorch 写训练循环？**
> `SFTTrainer` 已经帮你处理了：自动算 loss、混合精度、梯度累积、梯度裁剪、学习率调度、
> 日志、存档、断点续训、多卡分布式……自己写要几百行，而且处处是坑。**新手一律先用 Trainer。**

三者的关系：`datasets` 供数据 → `SFTTrainer` 拿数据训练 → `Unsloth` 让"模型"这一侧跑得更快更省。

### 4.2 第 42–60 行：路径与开关

```python
BASE       = os.environ.get("BASE", "/mnt/e/VMOS_workspace/model_tuning")
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(BASE, "models/Qwen3.5-4B"))
DATA_PATH  = os.environ.get("DATA_PATH", os.path.join(BASE, "alpaca_zh/alpaca_data_zh_51k.json"))
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/home/fanta/ft_outputs/qwen35-4b-lora")
```

全部用 `os.environ.get("变量名", 默认值)` 的写法，好处是：**换数据、换输出目录不用改代码**，命令行前面加一句就行。

```bash
DATA_PATH=/path/to/train.json OUTPUT_DIR=/home/fanta/ft_outputs/ev-v3 python train_lora.py
```

> ⚠️ `OUTPUT_DIR` 默认在 `/home/fanta`（WSL 原生 ext4），**不要改成 `/mnt/e/...`**。
> 9P 文件系统上 `torch.save` 写 zip 会损坏，报错信息还很不直观（见《踩坑记录》）。

接下来的四个开关，决定"训什么、怎么训"：

```python
MAX_SEQ_LEN    = 2048                                   # 硬编码常量，要改得动文件
SMOKE_TEST     = os.environ.get("SMOKE_TEST", "1") == "1"   # 默认只跑 500 条
DATA_FORMAT    = os.environ.get("DATA_FORMAT", "alpaca").lower()
SYSTEM         = os.environ.get("SYSTEM", "").strip()
RESPONSES_ONLY = os.environ.get("RESPONSES_ONLY", "0") == "1"
NUM_EPOCHS     = int(os.environ.get("NUM_EPOCHS", "1"))
```

| 开关 | 意思 | 新手最容易踩的 |
|---|---|---|
| `MAX_SEQ_LEN` | 一条样本最多 2048 个 token（约 1000~1500 汉字） | 超了会被**截断**，答案尾部直接消失 |
| `SMOKE_TEST` | **默认 1 = 只跑 500 条**，全量必须写 `SMOKE_TEST=0` | **忘了写，以为在训全量，其实只训了 500 条** |
| `DATA_FORMAT` | `alpaca` 单轮 / `chat` 多轮 | 多轮人设必须用 `chat` |
| `RESPONSES_ONLY` | 只对 assistant 回答算 loss | 人设任务不开，会学到用户的语气 |
| `NUM_EPOCHS` | 数据过几遍 | 数据只有几百条时 1 轮远远不够，人设要 3~5 |

### 4.3 第 62–68 行：第 1 步，加载底座

```python
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name      = MODEL_PATH,
    max_seq_length  = MAX_SEQ_LEN,
    dtype           = None,          # 自动选择 bf16
    load_in_4bit    = True,
)
```

**这一行做四件事**：① 从磁盘读 safetensors 权重；② 把每个 16 位数字压成 4 位（量化）；③ 搬进显存；④ 准备好 tokenizer。

| 参数 | 值 | 为什么 |
|---|---|---|
| `model_name` | 底座目录 | 传本地路径就不联网。传 HF 名字则会去下载 |
| `max_seq_length` | 2048 | 决定了 KV cache 和位置编码的预算，越大学得越多也越费显存 |
| `dtype` | `None` | `None` = 自动挑。有 bf16 支持就用 bf16，没有才退 fp16 |
| `load_in_4bit` | `True` | **8GB 显存的命根子**：8.8GB → 2.6GB |

**为什么 4bit 量化不毁模型？**
神经网络权重对精度不敏感，尤其是"说话风格"这种高层特征。4bit 量化（NF4）会带来一点点损失，
但配合 LoRA 训练，最终效果和全精度微调的差距通常小到肉眼看不出。**这是 QLoRA 论文的核心发现。**

> 顺便解释 `dtype`：bf16 和 fp16 都是 16 位浮点，区别在"能表示多大的数"。
> bf16 指数位多、范围大，训练时不容易梯度溢出，所以现代训练首选 bf16；fp16 精度高但容易 nan。

### 4.4 第 70–81 行：第 2 步，挂 LoRA

这是整份脚本**最值得逐字读懂**的一段。

```python
model = FastLanguageModel.get_peft_model(
    model,
    r                       = 16,
    lora_alpha              = 32,
    lora_dropout            = 0,
    target_modules          = ["q_proj", "k_proj", "v_proj", "o_proj",
                               "gate_proj", "up_proj", "down_proj"],
    bias                    = "none",
    use_gradient_checkpointing = "unsloth",
    random_state            = 42,
)
```

#### `r`（秩）= 16

低秩矩阵的"宽度"。`A` 是 `r × d_in`，`B` 是 `d_out × r`，`r` 越小参数越少。

| r | adapter 大小 | 效果 | 适合 |
|---|---|---|---|
| 4~8 | ~40MB | 只能学很浅的风格 | 数据少、只调语气 |
| **16** | **85MB** | **通用起点** | **本项目在用** |
| 32~64 | 170~340MB | 学得更细，也更容易过拟合 | 数据上千条、任务复杂 |

#### `lora_alpha` = 32

缩放系数，实际生效的是 `α/r = 2`。它的作用是：**让 r 变化时，有效学习率不跟着漂**。
经验规则：`alpha = 2 × r` 起步。

#### `lora_dropout` = 0

Dropout 是防过拟合的随机丢弃。LoRA 训练数据量小、训练步数少，开 dropout 反而拖慢收敛，业界普遍用 0。

#### `target_modules`（挂在哪）= 7 个矩阵

```
Transformer 每一层里有两块：
┌──────────────────────────────────────────────────┐
│  ① 注意力（Attention）                            │
│     q_proj  k_proj  v_proj  o_proj               │
│     "这句话里每个字该关注哪些字"                    │
├──────────────────────────────────────────────────┤
│  ② 前馈网络（FFN / MLP）                          │
│     gate_proj  up_proj  down_proj                │
│     "把关注到的信息做一轮加工"                      │
└──────────────────────────────────────────────────┘
```

七个全挂 = 效果最好、参数最多（85MB）。只挂 `q_proj, v_proj` 是老论文的省参数做法，对人设这种
"说话风格"任务明显不够 —— 风格不只存在注意力里，FFN 里也存了大量"表达习惯"。

#### `use_gradient_checkpointing` = "unsloth"

**梯度检查点**：正常训练时，前向传播要把每一层的中间结果（激活值）都存下来，反向传播要用。
但存这些值最占显存。梯度检查点的做法是：**前向只存层边界，反向时把中间值重新算一遍**。

> 用 30% 左右的计算时间，换掉一半以上的激活显存。**这是 8GB 能开 2048 上下文的前提。**

值写 `"unsloth"` 是用他们优化过的版本，比 HuggingFace 原生实现再省一点、快一点。

#### `bias` = "none"

只训 LoRA 矩阵，偏置项也不动。保持最小改动。

#### 挂完之后发生了什么

```
每一层的 q_proj 变成：
    原权重 W  (冻结，标了 requires_grad=False)
    +  LoRA 的 A、B（可训练，requires_grad=True）

优化器只会看到那 4000 万个可训练参数，
4B 个原始参数一动不动。
```

**这就是 LoRA 的省显存原理**：梯度、优化器状态都只对应那 4000 万参数，而不是 4B。

### 4.5 第 83–114 行：第 3 步，把数据变成模型能吃的文本

模型不认 `{"instruction": ..., "output": ...}` 这种 JSON。它只认**一段字符串**。
这一步就是把结构化数据"渲染"成字符串。

```python
def to_text_alpaca(ex):
    """单轮 Alpaca: instruction(+input) -> user, output -> assistant"""
    inst = ex["instruction"].strip()
    inp  = (ex.get("input") or "").strip()
    user = inst + ("\n" + inp if inp else "")
    msgs = [
        {"role": "user",      "content": user},
        {"role": "assistant", "content": ex["output"].strip()},
    ]
    return {"text": tokenizer.apply_chat_template(msgs, tokenize=False,
                                                  add_generation_prompt=False)}
```

拆开看：

1. 把 `instruction` 和 `input` 拼成一条 user 消息（`input` 为空就不拼）
2. `output` 作为 assistant 消息
3. `apply_chat_template` 把 `[{role, content}]` **渲染成模型训练时见过的格式**

#### chat template 到底渲染出了什么

大模型不认识 "role: user" 这种东西，它认识的是训练时见过的特殊 token。Qwen 系列大致长这样：

```
<|im_start|>system
你是 E.V.，回答极短。<|im_end|>
<|im_start|>user
我今天很累。<|im_end|>
<|im_start|>assistant
……坐下。别撑着。<|im_end|>
```

`<|im_start|>` / `<|im_end|>` 是**特殊 token**，它们会被当成"格式控制符"学进去。
**用错模板 = 模型听不懂**，这是新手第一大坑：拿 Llama 的数据格式喂 Qwen，训出来的东西会胡言乱语。

> 所以永远用 `tokenizer.apply_chat_template()`，**不要自己手拼字符串**。
> 模板来自底座自带的 `chat_template.jinja`，是模型作者定义的唯一正确答案。

#### `add_generation_prompt` 为什么是 False

| 场景 | 值 | 效果 |
|---|---|---|
| **训练**（本脚本） | `False` | 最后就是 `<|im_start|>assistant\n回答<|im_end|>`，答案完整 |
| **推理**（chat/serve 脚本） | `True` | 末尾多一个 `<|im_start|>assistant\n`，等于对模型说"该你了，说吧" |

训练时答案已经在里面了，不需要你"提示它开口"；推理时才需要。

#### 多轮版本（`DATA_FORMAT=chat`）

```python
def to_text_chat(ex):
    msgs = [{"role": m["role"], "content": str(m["content"]).strip()}
            for m in ex["conversations"]]
    if SYSTEM and not any(m["role"] == "system" for m in msgs):
        msgs = [{"role": "system", "content": SYSTEM}] + msgs
    return {"text": tokenizer.apply_chat_template(msgs, tokenize=False,
                                                  add_generation_prompt=False)}
```

区别只有两点：整段 `conversations` 一起渲染（多轮共享上下文）、可选注入 `SYSTEM`。

> **为什么 Alpaca 不注入 system？** 那份数据本身没有 system，硬加会造出"训练时见过、推理时没见过"
> 的分布偏移。**数据长什么样，推理时就得怎么喂** —— 这条原则比参数调优重要得多。

#### 读数据与变换

```python
ds = load_dataset("json", data_files=DATA_PATH, split="train").map(to_text)
if SMOKE_TEST:
    ds = ds.select(range(min(500, len(ds))))
```

- `load_dataset("json", ...)`：把 json 数组读成数据集对象（惰性加载，不占内存）
- `.map(to_text)`：对每条样本执行 `to_text`，把返回的 `{"text": ...}` **合并成新字段**
- `.select(...)`：冒烟测试只取前 500 条

变换后每条样本大致是：

```json
{"instruction": "...", "output": "...", "text": "<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n...<|im_end|>"}
```

**训练真正用的只有 `text` 这一个字段**。

### 4.6 第 116–140 行：第 4 步，SFTTrainer

```python
trainer = SFTTrainer(
    model           = model,
    tokenizer       = tokenizer,
    train_dataset   = ds,
    args = SFTConfig(
        dataset_text_field          = "text",
        max_seq_length              = MAX_SEQ_LEN,
        packing                     = True,
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 8,
        warmup_steps                = 20,
        num_train_epochs            = NUM_EPOCHS,
        learning_rate               = 2e-4,
        bf16                        = True,
        logging_steps               = 10,
        optim                       = "adamw_8bit",
        weight_decay                = 0.01,
        lr_scheduler_type           = "linear",
        save_steps                  = 500,
        output_dir                  = OUTPUT_DIR,
        report_to                   = "none",
        seed                        = 42,
    ),
)
```

#### `SFTTrainer` 是什么

**训练器**。你把「模型 + 数据 + 配置」三样交给它，它负责跑完整个训练循环：

```
for 每个 batch:
    ① 把 text 切成 token id
    ② 前向传播：模型预测下一个 token
    ③ 算 loss（和真实的下一个 token 比交叉熵）
    ④ 反向传播：算梯度
    ⑤ 累积够 8 个 batch → 优化器更新参数 → 清零梯度
    ⑥ 按调度器调学习率
    ⑦ 定期打日志 / 存档
```

第 ②~⑤ 步就是第 1.2 节那张图，只是被封装起来了。

#### `SFTConfig` 是什么

**训练配置**，一个纯装超参的盒子。名字里的 SFT 表示"这套配置是为监督微调准备的"（比如它知道
`dataset_text_field`、`packing`、`max_seq_length` 这些 SFT 专属概念）。

下面逐个说，**按"最影响结果"排序**：

---

**① `learning_rate = 2e-4`（最影响结果）**

| 量级 | 用在哪 |
|---|---|
| 1e-5 ~ 5e-5 | 全参微调（所有 4B 参数都在动，必须很小心） |
| **2e-4** | **LoRA 微调的标准值**（只有少量参数在动，可以迈大步） |
| 1e-3 以上 | 几乎必然训崩 |

太大 → loss 震荡甚至 nan；太小 → 训几千步也不动。**新手不要动这个值。**

---

**② `per_device_train_batch_size = 1` × `gradient_accumulation_steps = 8`**

- **batch size**：一次喂几条样本。显存占用基本和它成正比，所以 8GB 只能填 1。
- **梯度累积**：算完 1 条先不更新，攒够 8 条把梯度加起来，再更新一次。

```
显存占用  ←── 只看 batch size = 1
梯度质量  ←── 等效于 batch size = 8
```

这叫"**用小显存模拟大 batch**"。等效 batch = 1 × 8 = 8，是 LoRA 微调的常用配置。

---

**③ `packing = True`**

把多条短样本**首尾相接**拼进一个 2048 的窗口里，而不是每条都 padding 到 2048。

```
不开 packing： [样本A 80token][padding 1968 个无效token]   ← 96% 算力浪费
开 packing：   [样本A 80][样本B 120][样本C 200]...[样本N]  ← 基本填满
```

人设数据每条只有几十到两三百 token，**收益极大**（吞吐能翻好几倍）。

> ⚠️ 副作用：理论上相邻样本会拼在同一条序列里。TRL 会通过重置 position ids 让它们互不"看见"，
> 但如果发现模型开始**串味、复读上一轮**，先把 `packing=False` 试一次，排除这个因素。

---

**④ `num_train_epochs`**

数据过几遍。判断标准很简单：

| 数据量 | 建议 |
|---|---|
| 几十条 | 10 轮以上（但基本学不会，先加数据） |
| 几百条（本项目） | **3~5 轮** |
| 几万条以上 | 1 轮 |

轮数太多的典型症状：**复读**。模型开始不管问什么都回同一句，或者逐字重复你的问题。

---

**⑤ `max_seq_length` / `dataset_text_field`**

前者是窗口大小（这里再传一次是为了兼容旧版 API）；后者告诉 trainer "用哪个字段当训练文本"。
因为第 3 步产出的是 `text` 字段，所以这里写 `"text"`。

---

**⑥ `warmup_steps = 20` + `lr_scheduler_type = "linear"`**

```
学习率
2e-4 ┤        ╭────────────────────╮
     │      ╱                      ╲
     │    ╱                         ╲
   0 ┤──╱                           ╲────
     └──┬──────┬──────────────────┬──────→ 步数
        0     20                结束
        │warmup│      线性衰减      │
```

- **warmup**：前 20 步从 0 爬到 2e-4。因为训练刚开始时梯度又大又乱，直接满速容易把权重带崩
- **linear**：之后一路线性降到 0，后期步长越来越小，便于收敛到细处

---

**⑦ `optim = "adamw_8bit"` + `weight_decay = 0.01` + `bf16 = True`**

- **AdamW**：最常用的优化器。它给每个参数维护两个状态（一阶动量、二阶动量），所以很吃显存
- **adamw_8bit**：把这些状态用 8 位存，省显存。LoRA 参数量小，这里省的其实不多，但没坏处
- **weight_decay**：L2 正则，轻微抑制过拟合，0.01 是标准值
- **bf16**：混合精度训练，前向/反向用 16 位算，主权重保持高精度，速度和显存都受益

---

**⑧ `logging_steps` / `save_steps` / `report_to` / `seed`**

| 参数 | 值 | 含义 |
|---|---|---|
| `logging_steps` | 10 | 每 10 步往屏幕打一行 loss |
| `save_steps` | 500 | 每 500 步存一次 `checkpoint-*` |
| `report_to` | `"none"` | 不上报 wandb/tensorboard（国内网络也传不上去） |
| `seed` | 42 | 随机种子，保证同配置可复现 |

> ⚠️ **`save_steps=500` 的隐藏坑**：人设训练通常只有一两百步，**中途根本不会存 checkpoint**，
> 万一跑到一半崩了就得从头再来。想保险就把它改成 50 或 100（代价是多占磁盘）。

### 4.7 第 141–149 行：只对回答算 loss（人设必开）

```python
if RESPONSES_ONLY:
    from unsloth.chat_templates import train_on_responses_only
    trainer = train_on_responses_only(
        trainer,
        instruction_part = "<|im_start|>user\n",
        response_part    = "<|im_start|>assistant\n",
    )
```

#### 它到底改了什么

一条训练样本里有两类 token：

```
<|im_start|>user\n  我今天很累。  <|im_end|>\n <|im_start|>assistant\n ……坐下。别撑着。 <|im_end|>
└─────────────── 用户的输入 ───────────────┘   └────────── 模型的回答 ──────────┘
       这部分不该让模型学                                这部分才是要学的
```

`train_on_responses_only` 会遍历每条样本的 **label 数组**，把"用户输入"那一段全部改成 **`-100`**。

#### `-100` 是什么

PyTorch 的 `CrossEntropyLoss` 有个约定：**label = -100 的位置直接忽略，不参与 loss 计算**。
这是官方的 `ignore_index` 默认值。

```
改之前：loss = 用户段误差 + 回答段误差     ← 模型被迫学"用户会怎么说话"
改之后：loss =             回答段误差     ← 信号 100% 来自"该怎么回"
```

#### 为什么人设任务必须开

1. **不污染风格**：数据里用户的语气（活泼、刷表情、说长句）会被一起学进去，人设直接崩
2. **信号更纯**：用户段可能占样本长度的一半，这些 token 的 loss 在稀释真正要学的部分
3. **收敛更快**：同样的步数，有效学习信号翻倍

什么时候可以不开：纯指令跟随（alpaca 那种"问一句答一句"的通用数据），用户段本来就短且中性。

### 4.8 第 151 行：开始训练

```python
trainer.train()
```

一行，但会在终端滚出几十行。每 `logging_steps=10` 步打一次：

```
{'loss': 2.3141, 'grad_norm': 1.203, 'learning_rate': 0.000192, 'epoch': 0.42}
```

| 字段 | 怎么看 |
|---|---|
| `loss` | **核心指标**。起步 2~5 都正常；稳定在 0.8~1.5 说明学进去了；掉到 0.1 且不再动 → 过拟合/复读 |
| `grad_norm` | 梯度大小。突然飙到几百 → 学习率太大或有脏样本；一直是 0 → 梯度断了 |
| `learning_rate` | 当前学习率。能看到它按 warmup→linear 衰减，说明调度器在工作 |
| `epoch` | 当前是第几轮 |

**进度条怎么读**

除了日志字典，tqdm 还会打一行实时进度条：

```
 1%|▏         | 5/372 [00:25<24:04,  3.93s/it]
```

| 片段 | 含义 |
|---|---|
| `1%` | 完成百分比 |
| `5/372` | 当前第 **5 步** / 总共 **372 步** |
| `00:25` | 已经跑了 25 秒 |
| `<24:04` | 预计还要 24 分 04 秒 |
| `3.93s/it` | **每一步耗 3.93 秒** |

- **`it` = iteration = 一次参数更新**，不是一条样本。因为 `gradient_accumulation_steps = 8`，
  **1 it = 8 条样本**
- **总步数** ≈ `样本数 × epochs ÷ 8`（packing 不改变步数，它只改变每个 batch 里塞多满）
- `3.93s/it` 和日志里的 `it/s` 是倒数关系（3.93 s/it ≈ 0.25 it/s）

**速度对照**（4B / 4bit / seq 2048 / batch 1）：

| s/it | 判断 |
|---|---|
| 1~2 | 很快，说明样本短、packing 填得满 |
| **3~5** | **8GB 上的正常区间** |
| 8~15 | 偏慢：序列太长、显存吃紧在换页、或有别的进程抢卡 |
| 20+ | 大概率跑在 CPU 上，`nvidia-smi` 确认 |

> 总时间 ≈ `总步数 × s/it`。上例 372 × 3.93 ≈ 1462 秒 ≈ 24 分钟。
> 想知道"还要不要等"，先看总步数，再看 `s/it`，乘一下就知道 —— 比盯着进度条有用。

**判断训练是否健康的三个信号**：loss 总体向下、没有 nan、grad_norm 在个位数到几十之间波动。

#### loss 到底是什么（算法 + 数值直觉）

**一句话**：loss 是模型"答错了多少"的分数，训练的唯一目标就是让它变小。

具体算法是**交叉熵**，落到代码里只有一步：

```
模型对每个词都给出一个概率 → 只取「正确答案」那个词的概率 p → loss = −log(p)
```

以「我今天很累」为例，模型在看到「很」之后要预测下一个字，它给出的概率可能是：

| 候选 | 概率 p | −log(p) |
|---|---|---|
| **累**（正确答案） | **0.02** | **3.9** ← 这一步的 loss |
| 困 | 0.31 | 1.17 |
| 忙 | 0.12 | 2.12 |
| 其他几千个词 | 0.55 | — |

只在正确答案上算 loss，其他候选的概率只是"分母"。**p 越低，罚分越狠**：

| 模型给正确答案的概率 | loss | 人话 |
|---|---|---|
| 0.9 | 0.11 | 基本确定 |
| 0.5 | 0.69 | 一半一半 |
| 0.1 | 2.30 | 猜的 |
| 0.02 | 3.91 | 从 50 个候选里瞎蒙 |
| 0.01 | 4.61 | 完全没概念 |

**为什么用 −log 而不是直接 `1 − p`？**
因为 `1 − p` 封顶只有 1，学不到"差得有多离谱"；`−log(p)` 在 p→0 时会趋向无穷大，
等于告诉模型"你这次错得非常严重"，梯度信号更强。它还有个漂亮的副产品：

> `e^loss` = **困惑度（perplexity）** = 模型平均在多少个候选之间犹豫。
> loss 2.0 → e²≈7.4（在 7 个词里挑）；loss 1.0 → 2.7；loss 0.1 → 1.1。

**日志里的数字怎么读**：

| 现象 | 含义 |
|---|---|
| 起步 3~5，几十步内开始下降 | 正常（4B 模型常见起步值） |
| 稳定在 0.8~1.5 | 学进去了（困惑度 2~4，说话基本确定） |
| 掉到 0.1 以下且不再动 | ⚠️ 多半是**背下来了**，不是学会了 |
| 一直在 2.x 上下抖、不降 | 学习率太大、数据太乱，或 label 掩码把该学的也 mask 了 |
| 突然 nan | 精度溢出或数据里有空样本 |

**最重要的一条**：loss 只衡量"**像不像训练数据**"，不衡量"**人设对不对**"。

本项目就撞过这个坑：v2 的 loss 很低，但模型见人就回"你问这个做什么" —— 因为数据里"回避"
样本占比高，**它把回避学得很好**。loss 低得漂亮，行为却是错的。

> 所以：loss 用来看"训练有没有跑通"，**人设好不好必须用 `eval_persona.py` 的固定题集判断**。

### 4.9 第 153–157 行：第 5 步，保存

```python
ADAPTER = os.path.join(OUTPUT_DIR, "lora_adapter")
model.save_pretrained(ADAPTER)
tokenizer.save_pretrained(ADAPTER)
```

`save_pretrained` 保存的是 **adapter 而不是完整模型** —— 因为底座是冻结的，只有 LoRA 矩阵变了。

```
lora_adapter/
├── adapter_model.safetensors   ← 85MB，真正的训练成果（A、B 两个小矩阵）
├── adapter_config.json         ← 记录了底座路径 + r/alpha/挂载层
├── tokenizer.json              ← 分词表（必须跟训练时一致）
├── chat_template.jinja         ← 对话模板（必须跟训练时一致）
└── processor_config.json       ← 多模态 Processor 配置
```

**为什么必须带 tokenizer 和模板？** 推理时如果分词方式或对话格式和训练不一致，等于让模型
"用另一种语言回答"，前面所有努力归零。所以打包保存，一次拷走。

#### adapter vs checkpoint

| | adapter | checkpoint-* |
|---|---|---|
| 是什么 | 最终成果 | 训练途中的存档 |
| 含什么 | 只有 LoRA 矩阵 | LoRA 矩阵 + **优化器状态** + 调度器状态 |
| 大小 | 85MB | 几百 MB ~ 1GB+ |
| 能干嘛 | 直接加载推理 | 可以**断点续训** |
| 需要归档吗 | ✅ 必须 | ❌ 可以删 |

> 加载时 `model_name` 要指向 **`lora_adapter/` 目录**，不是 `checkpoint-138/`。
> 指错目录的表现是"加载成功但行为完全没变"，很难查。

### 4.10 第 159–172 行：注释里的两件事

```python
# model.save_pretrained_gguf(...)   ← 导出 GGUF，已注释掉
```

GGUF 是 llama.cpp 用的格式，好处是启动快、显存低。但 Qwen3.5 是**混合注意力架构**，
llama.cpp 支持不完整，导出后输出会变成"有语法没语义"的字词汤。

> **现状**：这条路本项目已放弃。日常用 `chat_lora.py` 调试，对外服务用 `serve_hf.py`。
> 详见《踩坑记录》第六节。

```python
# 旧版 TRL 兼容写法 ...
```

如果 `SFTConfig` 报"没有这个参数"，说明 TRL 版本旧，把注释里那段 `TrainingArguments` 的写法
替换掉 `args=SFTConfig(...)` 即可。功能一样，只是参数放在另一个类里。

---

## 5. 显存账：8GB 到底怎么塞下 4B

| 占用项 | 大小 | 说明 |
|---|---|---|
| 底座权重（4bit NF4） | **~2.6GB** | 8.8GB BF16 → 压到 30% |
| LoRA 参数（约 4000 万，bf16） | ~0.08GB | 只有这些带梯度 |
| 梯度 + 8bit 优化器状态 | ~0.2GB | 只对应 LoRA 参数，很小 |
| **激活值**（seq 2048, batch 1） | **~3~4GB** | 最大头，被梯度检查点压过一轮 |
| CUDA 上下文 + 显存碎片 | ~0.5GB | 固定开销 |
| **合计** | **~6.5~7.5GB** | 8GB 刚好，余量不到 1GB |

**结论**：训练和推理**不能同时跑**（`serve_hf` 约 3.4GB，加起来直接 OOM）。训练前先
`pkill -f serve_hf.py`，训完再启动。

**爆显存时的调整顺序**（从代价最小开始）：

1. `MAX_SEQ_LEN` 2048 → 1024（最有效，代价是长样本被截断）
2. 确认没有别的进程占卡：`nvidia-smi`
3. `gradient_accumulation_steps` 8 → 16（等效 batch 不变，峰值略降）
4. 还不行 → 换更小的底座，或者上云

---

## 6. 改参数指南

| 我想要 | 改什么 | 副作用 |
|---|---|---|
| 训练快一点 | `NUM_EPOCHS=1`、数据少一点、`packing=True` | 学不进去 |
| 省显存 | `MAX_SEQ_LEN=1024` | 长对话被截断 |
| 人设更稳 | 提升数据质量、`RESPONSES_ONLY=1`、`NUM_EPOCHS=3~5` | 轮数过头会复读 |
| 别把通用能力训坏 | `r=8`、`lr=1e-4` | 风格迁移变弱 |
| adapter 更小 | `r=8`、去掉 FFN 三个模块 | 效果下降明显 |
| 学得更细 | `r=32`、数据上千条 | 过拟合、adapter 变大 |
| 中途崩溃能从存档恢复 | `save_steps=50` | 多占磁盘 |

**改动优先级建议**：数据质量 > `RESPONSES_ONLY` > `NUM_EPOCHS` > `lr` > `r` > 其他。
新手期**别碰 `lr` 和 `r`**，先把手感建立起来。

---

## 7. 常见困惑 Q&A

**Q1：为什么训练数据只要一段 `text`，不需要标签？**
因为语言模型是自回归的：把 `text` 喂进去，它自己把每个位置"右移一位"当作答案。
`"我今天很累"` → 输入 `"我今天很"` 预测 `"累"`。**答案就藏在原文里**，这就是"自监督"。

**Q2：loss 降到 0 是不是最好？**
不是。loss=0 意味着模型把训练数据逐字背下来了，遇到新问题会崩。**目标是有下降且不抖，不是越低越好。**

**Q3：85MB 的 adapter 真能改变 4B 模型？**
能。两个原因：① 它挂在 **7 个关键投影**上，每层都改，不是只改一层；② `α/r=2` 的缩放让
小矩阵的改动被放大到足够影响输出。学术界把这叫"低秩假设"——微调所需的改动本来就是低秩的。

**Q4：训练时要不要带 system 提示？**
看数据。人设/客服这类**多轮**任务要带（`SYSTEM="..."`），而且推理时必须带**同样的** system。
Alpaca 单轮通用数据不要带。

**Q5：能不能用 A 模型的模板训 B 模型？**
绝对不能。模板里是特殊 token，不同模型完全不同。**永远用 `tokenizer.apply_chat_template()`。**

**Q6：训练时能同时聊天吗？**
不能。8GB 显存放不下"训练 + 推理"，必然 OOM。用另一台设备，或者训完再聊。

**Q7：为什么默认只跑 500 条（`SMOKE_TEST=1`）？**
防止你写错数据路径、写错格式，白等两小时。**先冒烟 5 分钟，确认 loss 在动，再上全量。**

**Q8：训一版要多久？**
看日志里的 `it/s`：`总步数 ÷ it/s` 就是秒数。总步数 = `样本数 × epochs ÷ 8`。
例：797 条 × 5 轮 ÷ 8 ≈ 498 步。

**Q9：训完效果不好，先怀疑什么？**
顺序是：① 数据里有没有"崩人设"的样本（最常见）；② `RESPONSES_ONLY` 开没开；
③ 推理时 system 和数据里是否一致；④ 轮数够不够；⑤ 最后才怀疑 lr 和 r。

**Q10：为什么脚本里到处都在写 `<|im_end|>`？**
因为模型不一定会自己停。不显式给终止符，它会在答完后继续编造下一轮"用户提问 + 自己回答"。
这是 Qwen 系微调最经典的现象，脚本里用 `eos_token_id` + `STOP_MARKS` 双保险解决。

---

## 8. 术语速查表（打印版）

| 中文 | 英文 | 一句话 |
|---|---|---|
| 底座模型 | base model | 预训练完、未做指令微调的原始模型 |
| 权重 / 参数 | weights / parameters | 模型里的数字，训练就是改它们 |
| 微调 | fine-tuning | 在已有权重上继续训练 |
| 监督微调 | SFT | 用「问题→答案」配对数据训练 |
| 低秩适配 | LoRA | 冻结原权重，只训两个小矩阵 |
| 量化低秩适配 | QLoRA | 4bit 底座 + LoRA |
| 参数高效微调 | PEFT | 只训极少量参数的方法总称 |
| 秩 | rank (r) | LoRA 小矩阵的宽度 |
| 适配器 | adapter | LoRA 训练出的 85MB 文件 |
| 分词 | tokenize | 把文本切成 token id |
| 词元 | token | 模型处理的最小单位 |
| 对话模板 | chat template | 把角色消息拼成模型认识的格式 |
| 轮次 | epoch | 数据完整过一遍 |
| 批大小 | batch size | 一次喂几条样本 |
| 梯度累积 | gradient accumulation | 攒 N 次梯度再更新 |
| 学习率 | learning rate | 每次更新挪多大步 |
| 热身 | warmup | 开头让学习率从 0 爬升 |
| 损失 | loss | 预测与答案的差距 |
| 梯度 | gradient | loss 对参数的导数 |
| 反向传播 | backpropagation | 从 loss 倒推每个参数怎么改 |
| 梯度检查点 | gradient checkpointing | 反向时重算中间值以省显存 |
| 打包 | packing | 多条短样本拼成一条长样本 |
| 标签掩码 | label mask (-100) | 让某些 token 不参与 loss |
| 检查点 | checkpoint | 训练中途存档，可续训 |
| 轮数 | num_train_epochs | 数据过几遍 |
| 权重衰减 | weight decay | L2 正则，抑制过拟合 |
| 混合精度 | mixed precision (bf16) | 用 16 位算，省显存提速 |

---

## 9. 下一步

读完本文，建议按这个顺序动手：

1. **先跑冒烟**，亲眼看到 loss 在动
   ```bash
   source ~/venvs/ft/bin/activate
   python train_lora.py          # 默认 500 条
   ```
2. **再看一篇好数据长什么样** → [人设数据生产指南](./人设数据生产指南.md)
3. **训一版完整人设** → [训练与推理脚本](./训练与推理脚本.md) 第十一节「工作流 A」
4. **用固定题集评测**，别靠"感觉还行"
5. 训崩了就查 → [踩坑记录](./踩坑记录.md)

> 一句话总结这份脚本：**加载一个压扁的底座，在 7 个关键矩阵旁边挂两个小矩阵，
> 把对话渲染成模型认识的格式，让 trainer 反复"预测→算差距→改小矩阵"，
> 最后只把小矩阵存下来。**
