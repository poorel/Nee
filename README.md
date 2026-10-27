# Nee — 中文评论负面片段抽取

> Negative span Extraction。给定一条中文评论，**字符级圈出所有负面评价片段**。

```json
输入: "房间有点小，但服务很好，就是空调特别吵"
输出: {"spans": ["房间有点小", "空调特别吵"]}
```

基于 [hfl/chinese-macbert](https://huggingface.co/hfl/chinese-macbert) 的 token 分类（BIO 序列标注），本仓库包含**训练数据、模型权重、推理服务**三件套，开箱即用。

## 快速开始

```bash
pip install -r requirements.txt          # torch + transformers
MODEL_PATH=./model PORT=8910 python server.py
```

调用（HTTP POST，JSON）：

```bash
curl -X POST localhost:8910 -d '{"text":"商场停车特别难，导购态度也很冷淡"}'
# {"spans": ["商场停车特别难", "导购态度也很冷淡"],
#  "highlighted": "<span class=neg>商场停车特别难</span>，<span class=neg>导购态度也很冷淡</span>"}
```

`highlighted` 是可直接用于前端展示的高亮 HTML。

## 仓库结构

```
Nee/
├── model/               # 模型权重 + tokenizer（model.safetensors 走 Git LFS）
├── data/
│   ├── train.jsonl      # 训练集 6664 条
│   └── val.jsonl        # 验证集 350 条
├── server.py            # 推理服务（含解码护栏，见下）
├── MODEL_CARD.md        # 模型卡片（底座/指标/局限）
├── requirements.txt
└── LICENSE              # Apache-2.0
```

## 模型

| 项 | 说明 |
|------|------|
| 任务 | 字符级 BIO 序列标注：`O` / `B-NEG`（负面片段首字）/ `I-NEG`（片段后续字） |
| 底座 | chinese-macbert（Apache-2.0）→ 情感分类中间微调 → 3 类 token 分类头 |
| 规模 | BERT-base 规格，约 1 亿参数，权重 389MB（fp32） |
| 上下文 | 训练位置编码上限 **256 token**，超长文本需分窗（见下） |

**指标**：验证集字符级 P=0.873 / R=0.820 / **F1=0.846**；真实评论验收 11/11 全对；300+ 字长混排验收 1.000。

### 解码护栏（工程重点）

模型本体是逐字独立三分类，不建模"片段完整性"，因此服务端带两条实测有效的解码规则（`server.py`）：

1. **I-run 无 B 起点修复**：连续 ≥2 个 `I-NEG` 且前面无 `B` 时，首字提升为 `B`。治"模型知道负面但首字 B 置信 0.47 不点火、整段被解码惯例丢弃"——长文验收 0.896→0.955 的关键。
2. **单字 span 护栏**：单字 span 且起始字 `B-NEG` 置信 <0.7 时丢弃。治方面词头字临界误点火（如「配」B=0.52 被拼成单字假 span）。

## 训练数据

**来源构成**（按条数混合）：

- [Meituan-Dianping/ASAP](https://github.com/Meituan-Dianping/ASAP) 公开数据集（Apache-2.0，美团点评 46730 条真实评论）——经 LLM 重标为字符级 BIO
- 模板合成增强专项：好评包裹稀释、转折句式、长混排、口语化好评、方面词误报等
- 少量人工精标真实评论样例

**格式**（JSONL，每行一条）：

```json
{
  "text":   "房间有点小，但服务很好",
  "chars":  ["房", "间", "点", ...],      // 逐字列表，与 text 对齐
  "labels": ["B-NEG", "I-NEG", "I-NEG", "O", ...],  // 字符级 BIO，与 chars 等长
  "spans":  ["房间有点小"]                // 片段列表（由 BIO 还原，含边界信息）
}
```

训练/验证均为负例（全 O）与正例混合，好评无负面时 `spans` 为空。标注口径：只圈**评价性负面语义**，方面词前缀不整体圈入；转折结构（"X 了点，其他还不错"）中的弱负面照圈。

## 长文使用注意

位置编码训练上限 256 token。超过 ~140 字的评论建议**按句子边界打包窗口**推理（不切断片段、负面段尽量靠窗前），`server.py` 内置 `extract_long` 可直接参考——等长滑窗会切断片段导致漏报。

## 已知局限

- **领域不均**：酒店/餐饮类较强，商场/卖场类偏弱（训练分布所致）
- **缓和表达**：长好评包裹中的弱负面（"略不方便""性价比只能算中等"）易漏
- **弱遗憾**：反讽、隐含不满的召回有限

## License

Apache-2.0（代码、权重、数据同）。底座 chinese-macbert 与 ASAP 数据集均为 Apache-2.0。

## 致谢

- [HIT-SCIR chinese-macbert](https://huggingface.co/hfl/chinese-macbert)
- [Meituan-Dianping/ASAP](https://github.com/Meituan-Dianping/ASAP)
