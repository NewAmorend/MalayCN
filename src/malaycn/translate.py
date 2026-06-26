from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pretrained import DEFAULT_MODEL_NAME, M2M100Translator
from .text import detokenize_zh, normalize_text


def parse_args():
    parser = argparse.ArgumentParser(description="使用 M2M100 预训练模型进行马来语到中文翻译")
    parser.add_argument("--text", required=True, help="待翻译的马来语文本")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help="Hugging Face 模型名")
    parser.add_argument("--source-lang", default="ms", help="源语言代码，马来语为 ms")
    parser.add_argument("--target-lang", default="zh", help="目标语言代码，中文为 zh")
    parser.add_argument("--max-length", type=int, default=128, help="最大生成长度")
    parser.add_argument("--num-beams", type=int, default=4, help="beam search 宽度")
    parser.add_argument("--device", default=None, help="运行设备，例如 cpu、cuda")
    parser.add_argument("--local-files-only", action="store_true", help="只使用本地缓存模型")
    return parser.parse_args()


def build_vectorizer(vocabulary: list[str], sequence_length: int):
    from tensorflow.keras.layers import TextVectorization

    vectorizer = TextVectorization(
        output_mode="int",
        output_sequence_length=sequence_length,
        standardize=None,
        split="whitespace",
    )
    vectorizer.set_vocabulary(vocabulary)
    return vectorizer


def main():
    args = parse_args()
    try:
        translator = M2M100Translator(
            model_name=args.model_name,
            source_lang=args.source_lang,
            target_lang=args.target_lang,
            device=args.device,
            local_files_only=args.local_files_only,
        )
        translated = translator.translate(args.text, max_length=args.max_length, num_beams=args.num_beams)
    except ModuleNotFoundError as exc:
        raise SystemExit(f"缺少依赖: {exc.name}。请先执行 pip install -r requirements.txt") from exc
    print(translated)


def load_tensorflow_checkpoint(checkpoint_dir: str | Path):
    import tensorflow as tf

    from .model import MalayChineseTransformer

    checkpoint_dir = Path(checkpoint_dir)
    config = json.loads((checkpoint_dir / "config.json").read_text(encoding="utf-8"))
    vocab = json.loads((checkpoint_dir / "vocab.json").read_text(encoding="utf-8"))

    source_vectorizer = build_vectorizer(vocab["source"], config["max_source_len"])
    target_vectorizer = build_vectorizer(vocab["target"], config["target_vectorize_len"])
    target_index_lookup = dict(enumerate(vocab["target"]))

    model = MalayChineseTransformer(**config)
    dummy_source = tf.zeros((1, config["max_source_len"]), dtype=tf.int64)
    dummy_target = tf.zeros((1, config["max_target_len"]), dtype=tf.int64)
    model([dummy_source, dummy_target], training=False)
    model.load_weights(str(checkpoint_dir / "model.weights.h5"))
    return (
        model,
        source_vectorizer,
        target_vectorizer,
        target_index_lookup,
        config["max_target_len"],
    )


def translate_tensorflow_checkpoint(
    text: str,
    model,
    source_vectorizer,
    target_vectorizer,
    target_index_lookup: dict[int, str],
    max_decoded_len: int,
) -> str:
    import tensorflow as tf

    source = normalize_text(text, "ms")
    encoder_input = source_vectorizer([source])
    decoded = "[start]"
    output_tokens: list[str] = []

    for step in range(max_decoded_len):
        decoder_tokens = target_vectorizer([decoded])[:, :-1]
        predictions = model([encoder_input, decoder_tokens], training=False)
        sampled_token_index = int(tf.argmax(predictions[0, step, :]).numpy())
        sampled_token = target_index_lookup.get(sampled_token_index, "[UNK]")

        if sampled_token == "[end]":
            break
        output_tokens.append(sampled_token)
        decoded += f" {sampled_token}"

    return detokenize_zh(output_tokens)


if __name__ == "__main__":
    main()
