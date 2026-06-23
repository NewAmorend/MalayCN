from __future__ import annotations

import argparse
import json
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.layers import TextVectorization

from .model import MalayChineseTransformer
from .text import detokenize_zh, normalize_text


def parse_args():
    parser = argparse.ArgumentParser(description="使用训练好的模型进行马来语到中文翻译")
    parser.add_argument("--checkpoint-dir", required=True, help="训练输出目录")
    parser.add_argument("--text", required=True, help="待翻译的马来语文本")
    parser.add_argument("--max-decoded-len", type=int, default=None, help="最大生成长度")
    return parser.parse_args()


def build_vectorizer(vocabulary: list[str], sequence_length: int) -> TextVectorization:
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
    checkpoint_dir = Path(args.checkpoint_dir)
    config = json.loads((checkpoint_dir / "config.json").read_text(encoding="utf-8"))
    vocab = json.loads((checkpoint_dir / "vocab.json").read_text(encoding="utf-8"))

    source_vectorizer = build_vectorizer(vocab["source"], config["max_source_len"])
    target_vectorizer = build_vectorizer(vocab["target"], config["target_vectorize_len"])
    target_index_lookup = dict(enumerate(vocab["target"]))

    model = MalayChineseTransformer(**config)
    dummy_source = tf.zeros((1, config["max_source_len"]), dtype=tf.int64)
    dummy_target = tf.zeros((1, config["max_target_len"]), dtype=tf.int64)
    model([dummy_source, dummy_target], training=False)
    model.load_weights(checkpoint_dir / "model.weights.h5")

    translated = translate(
        args.text,
        model,
        source_vectorizer,
        target_vectorizer,
        target_index_lookup,
        max_decoded_len=args.max_decoded_len or config["max_target_len"],
    )
    print(translated)


def translate(
    text: str,
    model: MalayChineseTransformer,
    source_vectorizer: TextVectorization,
    target_vectorizer: TextVectorization,
    target_index_lookup: dict[int, str],
    max_decoded_len: int,
) -> str:
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

