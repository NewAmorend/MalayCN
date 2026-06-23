from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization

from .text import add_target_tokens, normalize_text


@dataclass
class ParallelData:
    source: list[str]
    target: list[str]


def load_parallel_tsv(
    data_path: str | Path,
    source_col: str = "ms",
    target_col: str = "zh",
    max_examples: int | None = None,
) -> ParallelData:
    source_texts: list[str] = []
    target_texts: list[str] = []

    with Path(data_path).open("r", encoding="utf-8", newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        has_header = csv.Sniffer().has_header(sample)

        if has_header:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                source = row.get(source_col, "")
                target = row.get(target_col, "")
                _append_pair(source_texts, target_texts, source, target)
                if max_examples and len(source_texts) >= max_examples:
                    break
        else:
            reader = csv.reader(f, delimiter="\t")
            for row in reader:
                if len(row) < 2:
                    continue
                _append_pair(source_texts, target_texts, row[0], row[1])
                if max_examples and len(source_texts) >= max_examples:
                    break

    if not source_texts:
        raise ValueError(f"没有从 {data_path} 读取到有效句对")
    return ParallelData(source_texts, target_texts)


def _append_pair(source_texts: list[str], target_texts: list[str], source: str, target: str) -> None:
    source = normalize_text(source, "ms")
    target = add_target_tokens(normalize_text(target, "zh"))
    if source and target:
        source_texts.append(source)
        target_texts.append(target)


def split_data(data: ParallelData, validation_fraction: float, seed: int) -> tuple[ParallelData, ParallelData]:
    rng = np.random.default_rng(seed)
    indices = np.arange(len(data.source))
    rng.shuffle(indices)
    val_size = max(1, int(len(indices) * validation_fraction))
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    if len(train_indices) == 0:
        train_indices = val_indices

    return _take(data, train_indices), _take(data, val_indices)


def _take(data: ParallelData, indices: np.ndarray) -> ParallelData:
    return ParallelData(
        [data.source[i] for i in indices],
        [data.target[i] for i in indices],
    )


def build_vectorizers(
    train_data: ParallelData,
    max_source_tokens: int,
    max_target_tokens: int,
    max_source_len: int,
    target_vectorize_len: int,
) -> tuple[TextVectorization, TextVectorization]:
    source_vectorizer = TextVectorization(
        max_tokens=max_source_tokens,
        output_mode="int",
        output_sequence_length=max_source_len,
        standardize=None,
        split="whitespace",
    )
    target_vectorizer = TextVectorization(
        max_tokens=max_target_tokens,
        output_mode="int",
        output_sequence_length=target_vectorize_len,
        standardize=None,
        split="whitespace",
    )

    source_vectorizer.adapt(tf.data.Dataset.from_tensor_slices(train_data.source).batch(1024))
    target_vectorizer.adapt(tf.data.Dataset.from_tensor_slices(train_data.target).batch(1024))
    return source_vectorizer, target_vectorizer


def make_dataset(
    data: ParallelData,
    source_vectorizer: TextVectorization,
    target_vectorizer: TextVectorization,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> tf.data.Dataset:
    dataset = tf.data.Dataset.from_tensor_slices((data.source, data.target))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(data.source), seed=seed, reshuffle_each_iteration=True)

    def vectorize(source, target):
        source_tokens = source_vectorizer(source)
        target_tokens = target_vectorizer(target)
        decoder_input = target_tokens[:, :-1]
        decoder_target = target_tokens[:, 1:]
        return (source_tokens, decoder_input), decoder_target

    return (
        dataset.batch(batch_size)
        .map(vectorize, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )

