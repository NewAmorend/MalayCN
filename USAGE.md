# MalayCN 使用说明

本文档说明如何直接使用预训练模型做马来语到中文翻译，以及如何启动桌面界面。

## 1. 准备环境

建议使用 Python 3.10 或 3.11。创建虚拟环境：

```bash
python -m venv .venv
source .venv/bin/activate
```

安装默认推理依赖：

```bash
pip install -r requirements.txt
```

默认依赖会安装：

- `torch`
- `transformers`
- `sentencepiece`

## 2. 命令行翻译

最简单的用法：

```bash
python -m src.malaycn.translate --text "terima kasih"
```

默认模型是 `facebook/m2m100_418M`，默认语言方向是：

- 源语言：`ms`，马来语
- 目标语言：`zh`，中文

完整参数示例：

```bash
python -m src.malaycn.translate \
  --model-name facebook/m2m100_418M \
  --source-lang ms \
  --target-lang zh \
  --text "saya mahu pergi ke sekolah" \
  --max-length 128 \
  --num-beams 4
```

第一次运行会从 Hugging Face 下载模型权重。模型较大，下载和首次加载会比较慢；后续会复用本地缓存。

## 3. 桌面界面

启动 Tkinter 界面：

```bash
python -m src.malaycn.gui
```

使用步骤：

1. 保持默认模型 `facebook/m2m100_418M`。
2. 保持源语言 `ms`、目标语言 `zh`。
3. 点击“加载模型”。
4. 在左侧输入马来语。
5. 点击“翻译”，右侧会显示中文结果。

如果启动时报 `_tkinter` 缺失，说明当前 Python 没有 Tkinter 图形界面支持。可以换用 python.org 安装包、Conda/Miniforge，或安装当前 Python 对应版本的 `python-tk`。

## 4. 常见问题

### 缺少 torch / transformers / sentencepiece

执行：

```bash
pip install -r requirements.txt
```

### 第一次运行很慢

这是正常现象。`facebook/m2m100_418M` 需要下载模型权重并加载到内存。下载完成后再次运行会快很多。

### 不想联网下载

如果模型已经缓存到本机，可以使用：

```bash
python -m src.malaycn.translate --text "terima kasih" --local-files-only
```

如果本机没有缓存，这个命令会失败，需要先联网下载一次。

### 翻译质量不稳定

M2M100 是通用多语言模型，适合快速原型。若要用于旅游、客服、教育、政务等具体场景，建议准备领域语料做微调或后处理。

## 5. 可选：训练 TensorFlow 教学模型

仓库仍保留 TensorFlow Transformer 训练链路。安装训练依赖：

```bash
pip install -r requirements-train.txt
```

用样例数据跑通训练：

```bash
python -m src.malaycn.train \
  --data-path data/sample_ms_zh.tsv \
  --output-dir artifacts/malaycn-demo \
  --epochs 20 \
  --batch-size 8
```

样例数据只用于验证流程，不适合训练真实可用模型。真实训练请准备 `ms` / `zh` 两列的 TSV 平行语料。

