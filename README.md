# MalayCN: 马来语 -> 中文翻译

这是一个马来语到中文翻译小项目。默认推理链路直接使用 Hugging Face 预训练模型 [`facebook/m2m100_418M`](https://huggingface.co/facebook/m2m100_418M)，语言方向是 `ms -> zh`。

仓库里也保留了一个 TensorFlow Transformer 训练版本，方便学习机器翻译完整训练流程。训练数据格式使用最简单的 TSV：

```text
ms	zh
selamat pagi	早上好
terima kasih	谢谢你
```

## 1. 安装

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

默认依赖只包含预训练翻译所需的 `torch`、`transformers` 和 `sentencepiece`。如果你还想跑 TensorFlow 自训练版本，建议使用 Python 3.10 或 3.11；Apple Silicon 上可以优先使用 Miniforge/Conda 环境，然后额外安装：

```bash
pip install -r requirements-train.txt
```

## 2. 直接使用预训练模型

完整使用说明见 [USAGE.md](USAGE.md)。

命令行翻译：

```bash
python -m src.malaycn.translate --text "terima kasih"
```

第一次运行会下载 `facebook/m2m100_418M` 模型权重，后续会复用本地缓存。

也可以指定模型和语言代码：

```bash
python -m src.malaycn.translate \
  --model-name facebook/m2m100_418M \
  --source-lang ms \
  --target-lang zh \
  --text "terima kasih"
```

也可以启动 Tkinter 桌面界面：

```bash
python -m src.malaycn.gui
```

界面启动后点击“加载模型”，再输入马来语文本进行翻译。默认模型是 `facebook/m2m100_418M`，源语言代码是 `ms`，目标语言代码是 `zh`。

如果启动时报 `_tkinter` 缺失，说明当前 Python 没有图形界面支持。换用 python.org 安装包、Conda/Miniforge，或安装当前 Python 对应版本的 `python-tk` 即可。

## 3. 可选：训练自己的 TensorFlow 模型

仓库内带了一个很小的样例数据，只用于验证流程，不适合训练可用模型：

```bash
python -m src.malaycn.train \
  --data-path data/sample_ms_zh.tsv \
  --output-dir artifacts/malaycn-demo \
  --epochs 20 \
  --batch-size 8
```

训练完成后，如果你需要使用旧的 TensorFlow checkpoint，可以在代码里调用 `src.malaycn.translate.load_tensorflow_checkpoint` 和 `src.malaycn.translate.translate_tensorflow_checkpoint`。

## 4. 真实训练数据格式

准备一个 UTF-8 TSV 文件，至少包含两列：

```text
ms	zh
Saya mahu pergi ke sekolah.	我想去学校。
Cuaca hari ini sangat baik.	今天的天气很好。
```

训练命令：

```bash
python -m src.malaycn.train \
  --data-path data/your_ms_zh.tsv \
  --output-dir artifacts/malaycn \
  --epochs 30 \
  --batch-size 64 \
  --max-source-len 80 \
  --max-target-len 96 \
  --max-source-tokens 30000 \
  --max-target-tokens 30000 \
  --embed-dim 256 \
  --latent-dim 512 \
  --num-heads 8
```

## 5. 推荐可用数据集

优先建议使用 OPUS，因为它聚合了大量公开平行语料，且支持按语言对下载：

- OPUS：开放平行语料集合，包含 OpenSubtitles、NLLB、CCMatrix、WikiMatrix、Tatoeba 等多个语料源。入口：https://opus.nlpl.eu/
- OPUS API：可用 `source=ms&target=zh` 查询马来语-中文语言对可下载语料。入口：https://opus.nlpl.eu/opusapi/
- Tatoeba：句子级翻译语料，量通常比 OPUS 大型 web 语料少，但质量适合做小规模验证和评测。入口：https://tatoeba.org/

推荐组合：

1. **先用 Tatoeba / WikiMatrix / TED2020 小规模语料跑通**：噪声较少，适合调通预处理、训练和推理。
2. **再加入 OpenSubtitles / CCMatrix / NLLB 扩量**：量更大，但要做去重、长度过滤和质量过滤。
3. **最终按领域清洗**：如果你要翻译旅游、教育、客服或政务内容，最好额外收集对应领域的 `ms-zh` 平行句对微调。

## 6. 数据清洗建议

最低限度建议：

- 删除空行、重复句对。
- 删除源句或目标句过长的样本。
- 删除源句和目标句长度比例极端的样本。
- 保留一份固定验证集，避免只看训练集 loss。

可以用内置脚本把两个对齐文本文件合并成 TSV：

```bash
python scripts/parallel_to_tsv.py \
  --source-file data/train.ms \
  --target-file data/train.zh \
  --output-file data/train_ms_zh.tsv
```

## 7. 当前实现边界

默认可用版本走 M2M100 预训练模型，适合直接做原型。TensorFlow 自训练版本是教学和实验实现，重点是让你拥有完整训练链路；中文默认按字切分，马来语默认按词切分。真实生产效果通常建议继续做领域数据微调、质量评估和异常输入处理。
