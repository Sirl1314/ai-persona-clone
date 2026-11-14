# WeChat Persona LoRA

A personal chat model that talks like me. Fine-tuned from Qwen2.5-7B-Instruct with QLoRA on 9,815 real conversations. Runs locally on a 4GB laptop GPU as a 4.4GB Q4_K_M GGUF, and on a 10GB lab GPU with full offload.

一个"像本人说话"的私人对话模型。基于 Qwen2.5-7B-Instruct 做 QLoRA 微调，训练集是 9,815 条真实聊天。量化到 Q4_K_M（4.4GB）后，能在 4G 显存的笔记本显卡上跑，也能在 10G 显存的实验室机器上全量放显存。

| Metric | Value |
|---|---|
| Base model | Qwen2.5-7B-Instruct |
| Training samples | 9,815 multi-turn conversations |
| Method | QLoRA, rank 64 / alpha 128, all linear layers |
| Train time | 1,228 steps, 1,927s on a single RTX 4090 |
| Final train loss | 2.6136 |
| Quantization | Q4_K_M, 4,683,073,632 bytes |
| Human eval (strict) | 78% "accurate and human-like" |
| Human eval (no-error) | 91% "no derailment, no role break" |
| Laptop throughput | ~16 tok/s on RTX 3050 Ti Laptop (4GB) |

---

## 1. 这个项目在做什么

我想要一个能替我回微信的东西。不是那种"您好，我理解您的感受"的客服腔，而是短句、口语、会怼人、会玩梗、被骂了会顶回去的那种。

三个约束决定了技术路线：

1. **数据只有自己的一份**：9,815 条真实对话，没有公开语料可对齐，所以只能小数据微调。
2. **推理必须在自己电脑上**：4G 显存的 3050Ti，7B 全精度放不下，所以必须量化。
3. **目标不是"答对题"**：是"像本人说话"，所以评测口径要围绕对话行为，不是知识问答。

最终交付：一个 4.7GB 的 ollama 模型 `wechat-v38`，接在飞书机器人里实际跑。

## 2. 数据

原始数据是四个会话的聊天记录（私聊 + 群聊），清洗后得到 9,815 条 ShareGPT 格式样本。

处理链条：

```
原始聊天记录
  → 会话切分、去广告/去系统消息
  → 多轮上下文重建（保留 ctx_turns）
  → 真名替换、敏感内容过滤
  → 维度打标（日常/互怼/玩梗/活泼/询问/关心/情感正负）
  → ShareGPT 格式（messages: human → gpt 交替）
```

维度分布（多标签，合计超过 100%）：

| 维度 | 占比 | 主样本数 |
|---|---|---|
| 日常 | 41.6% | 4,078 |
| 互怼 | 17.9% | 1,753 |
| 玩梗 | 14.3% | 1,402 |
| 询问 | 13.3% | 1,304 |
| 活泼 | 12.9% | 1,262 |
| 情感-正面 | 9.4% | 918 |
| 关心 | 8.0% | 785 |
| 情感-负面 | 3.9% | 383 |

**关键统计（用来判断"像不像人"）**：本人每条回复的长度中位数是 **11 字**，p90 是 31 字，超过 80 字的只占 0.7%。这个数字后来成了所有长度类判定的基准线。

## 3. 训练配置

```yaml
base_model: Qwen/Qwen2.5-7B-Instruct
stage: sft
finetuning_type: lora
quantization_bit: 4          # nf4 + double quant
lora_rank: 64
lora_alpha: 128
lora_dropout: 0.1
lora_target: all
template: qwen
cutoff_len: 2048
num_train_epochs: 2
per_device_train_batch_size: 4
gradient_accumulation_steps: 4
learning_rate: 1e-4
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
gradient_checkpointing: true
```

单卡 4090，1,228 步，1,927 秒（32 分钟），最终 train loss 2.6136。

## 4. 评测：三次自我纠偏

这部分是我认为整个项目里最值得看的地方。**一开始的三个结论，后来全被我自己推翻了。**

### 4.1 第一次：贪心解码让我误判"训练崩了"

用 `do_sample=False` 跑 312 条测试，出现 17 条重复退化——某条回答把"算了"重复了 47 次。

第一反应是权重崩了。但改成采样（`temperature 0.7 / top_p 0.9 / repetition_penalty 1.15 / no_repeat_ngram_size 3`）重跑同一批：

| 指标 | 贪心 | 采样 |
|---|---|---|
| 重复退化条数 | 17 | **0** |
| 超过 80 字的回复 | 19 | 5 |
| 长度中位数 | 16 字 | **13 字** |

结论：**重复 collapse 是贪心解码在长句里进了 n-gram 循环，不是权重坏。** 拿贪心当评测口径，会把解码缺陷误判成训练失败。

### 4.2 第二次："相同输出 31 组"不是过拟合

贪心版里 31 组、72 条输出完全相同，一度判为背题。查下去发现是测试集里「好！」「好呀」「我去」这类 prompt 本身重复出现——相同输入给相同输出是必然的。采样版降到 6 组 16 条。

真正的背题判据应该是**照搬率**：拿生成结果去比对训练集 25,583 条回复，8-gram 精确命中 12 条，占 3.8%，而且全是"是的""当然""好好好"这种 ≤7 字习惯用语，**没有任何整段照搬**。

### 4.3 第三次："精准像人"和"不出错"是两个口径

- 按**不出错**算（不答非所问、不串味、不退化、不角色错位）：90 条里真错 8 条，**91% 过线**
- 按**精准像人**算（要求回答切题且风格到位）：**78%**

我原本的目标是"能回答 90% 的问题不出错"，对应的是前者。

### 4.4 最终维度表现

720 条盲测，9 个维度各 80 条，每维度抽 10 条人工看：

| 维度 | 通过 | 维度 | 通过 |
|---|---|---|---|
| 日常 | 10/10 | 活泼 | 8/10 |
| 情感-负面 | 10/10 | 冷淡 | 7/10 |
| 情感-正面 | 9/10 | 互怼 | 5/10 |
| 询问 | 9/10 | 玩梗 | **4/10** |
| 图片 | 8/10 | | |

**短板明确：玩梗 4/10、互怼 5/10。** 这两个维度的样本量本来就少（各 1,400-1,750 条），而且玩梗高度依赖具体语境，模型学不到那个"点"。

## 5. 候选重排：best-of-4

推理时对同一输入采样 4 条候选，用规则打分选最高分：

```python
score = 0
# 长度：5-40 字 +2；<5 或 >60 字 -1
# 开头是 666/我的发/我去/稳了/哈哈/？/。。 -2
# 含 url 或单引号 -6
# 分条 1-4 行 +1，>6 行 -1
# 与 prompt 有中文 2-gram 重叠 +3
# prompt ≥10 字且零重叠 -4
# 长度 ≤3 且 prompt ≥12 字 -2
# 含 砍死/我恨/去死/杀/打死 -3
```

720 条上，best-of-4 比单次采样提升了约 3 个点。这套规则是从数据分布反推的，不是拍脑袋定的。

## 6. 量化与部署

### 6.1 一条走过的错路

第一次量化：bf16 → **q8_0** → Q4_K_M（二次压缩）。

结果模型完全失效：问"1+1 等于几"答"大于 2 / 可能等于 2 / 看那两个人"，问"中国的首都是哪里"答"哈哈 / 你猜我在干嘛"。

排查链条：

1. 排除 ollama 环境 → 同机 `qwen2.5:3b-instruct` 答"1 + 1 equals 2."完全正常
2. 排除模板覆盖 → 删掉手写 TEMPLATE，改用 GGUF 自带 Qwen 模板
3. 排除 system 不一致 → 补上训练时的 system 描述
4. 排除显存/性能 → 清空显存、强制纯 CPU，一样错
5. 直测云端 bf16 合并模型 → **5 句全对**，证明合并没问题

**根因：`llama-quantize` 从 q8_0 二次压到 Q4_K_M 会显著破坏权重**（工具本身对 requantize 有警告，需要 `--allow-requantize` 才放行，这本身就是个信号）。

正确路径：

```
bf16 merged (15.5GB)
  → convert_hf_to_gguf.py --outtype f16   (15.5GB)
  → llama-quantize Q4_K_M                 (4.68GB)
```

直转后 MD5 校验一致，云端和本地同一文件，5 句话全对。

### 6.2 本地部署

`Modelfile`：

```
FROM ./model-q4km.gguf

TEMPLATE "{{ if .System }}<|im_start|>system\n{{ .System }}<|im_end|>\n{{ end }}<|im_start|>user\n{{ .Prompt }}<|im_end|>\n<|im_start|>assistant\n"

SYSTEM 你是微信里的我，用短句、口语回复，想说的话可以分成多条，用换行分隔。

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER repeat_penalty 1.15
PARAMETER num_predict 64
PARAMETER num_ctx 2048
PARAMETER stop <|im_end|>
```

两个部署环境：

| 环境 | 显存 | 策略 | 实际速度 |
|---|---|---|---|
| RTX 3050 Ti Laptop | 4GB | 部分层卸载到 CPU | ~16 tok/s |
| RTX 3080 | 10GB | 全量放显存 | 明显更快，长回复稳定 |

## 7. 目录结构

```
wechat-persona-lora/
├── README.md
├── configs/
│   └── train_qlora.yaml          # LLaMA-Factory 训练配置
├── src/
│   ├── build_dataset.py          # ShareGPT 构造（脱敏示例）
│   ├── eval_sampled.py           # 采样评测
│   └── best_of_n.py              # 候选重排打分器
├── scripts/
│   ├── merge_lora.sh             # 合并 adapter 到基座
│   └── convert_quantize.sh       # bf16 → f16 → Q4_K_M
├── samples/
│   └── demo_dialog.md            # 脱敏对话样例
├── docs/
│   ├── evaluation.md             # 评测方法论完整记录
│   └── troubleshooting.md        # 量化/模板/特殊token 踩坑
└── Modelfile
```

## 8. 复现

```bash
# 1. 训练（需要一张 24GB 以上显卡）
llamafactory-cli train configs/train_qlora.yaml

# 2. 合并
bash scripts/merge_lora.sh

# 3. 量化（注意必须是 f16 直转，不要经过 q8）
bash scripts/convert_quantize.sh

# 4. 本地运行
ollama create mypersona -f Modelfile
ollama run mypersona
```

## 9. 数据与隐私

**仓库不含任何真实聊天记录。** `samples/demo_dialog.md` 里的所有对话都是虚构的，只用来说明输出风格。

原始数据涉及他人隐私，不公开、不分享、不上传。模型权重也不在仓库里。

## 10. 后续可以做的

- 补"被骂 → 直接回怼"的样本。当前模型的先验偏忍让，提示词压不住，只能靠数据。
- 玩梗维度需要更多同语境样本，当前的 4/10 是数据问题不是训练问题。
- 上下文窗口 2048 偏小。放到 4096 需要更多显存，或者上 14B 时就该重新权衡。

## License

MIT
