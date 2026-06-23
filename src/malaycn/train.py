from __future__ import annotations

import argparse
import json
from pathlib import Path

import tensorflow as tf
from tensorflow import keras

from .data import build_vectorizers, load_parallel_tsv, make_dataset, split_data
from .model import MalayChineseTransformer


def masked_loss(y_true, y_pred):
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="none")
    loss = loss_fn(y_true, y_pred)
    mask = tf.cast(tf.not_equal(y_true, 0), loss.dtype)
    loss *= mask
    return tf.reduce_sum(loss) / tf.maximum(tf.reduce_sum(mask), 1.0)


def masked_accuracy(y_true, y_pred):
    predictions = tf.argmax(y_pred, axis=-1, output_type=y_true.dtype)
    matches = tf.cast(tf.equal(y_true, predictions), tf.float32)
    mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
    return tf.reduce_sum(matches * mask) / tf.maximum(tf.reduce_sum(mask), 1.0)


def parse_args():
    parser = argparse.ArgumentParser(description="训练马来语到中文的 TensorFlow Transformer 翻译模型")
    parser.add_argument("--data-path", required=True, help="TSV 数据路径，默认列名为 ms 和 zh")
    parser.add_argument("--output-dir", default="artifacts/malaycn", help="模型输出目录")
    parser.add_argument("--source-col", default="ms", help="源语言列名")
    parser.add_argument("--target-col", default="zh", help="目标语言列名")
    parser.add_argument("--max-examples", type=int, default=None, help="最多读取多少条样本")
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--max-source-tokens", type=int, default=30000)
    parser.add_argument("--max-target-tokens", type=int, default=30000)
    parser.add_argument("--max-source-len", type=int, default=80)
    parser.add_argument("--max-target-len", type=int, default=96)

    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--latent-dim", type=int, default=512)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_parallel_tsv(args.data_path, args.source_col, args.target_col, args.max_examples)
    train_data, val_data = split_data(data, args.validation_fraction, args.seed)

    target_vectorize_len = args.max_target_len + 1
    source_vectorizer, target_vectorizer = build_vectorizers(
        train_data,
        args.max_source_tokens,
        args.max_target_tokens,
        args.max_source_len,
        target_vectorize_len,
    )

    train_ds = make_dataset(
        train_data,
        source_vectorizer,
        target_vectorizer,
        args.batch_size,
        shuffle=True,
        seed=args.seed,
    )
    val_ds = make_dataset(
        val_data,
        source_vectorizer,
        target_vectorizer,
        args.batch_size,
        shuffle=False,
        seed=args.seed,
    )

    config = {
        "source_vocab_size": len(source_vectorizer.get_vocabulary()),
        "target_vocab_size": len(target_vectorizer.get_vocabulary()),
        "max_source_len": args.max_source_len,
        "max_target_len": args.max_target_len,
        "target_vectorize_len": target_vectorize_len,
        "embed_dim": args.embed_dim,
        "latent_dim": args.latent_dim,
        "num_heads": args.num_heads,
        "dropout": args.dropout,
    }

    model = MalayChineseTransformer(**config)
    model.compile(
        optimizer=keras.optimizers.Adam(args.learning_rate),
        loss=masked_loss,
        metrics=[masked_accuracy],
    )

    checkpoint_path = output_dir / "model.weights.h5"
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            str(checkpoint_path),
            save_weights_only=True,
            save_best_only=True,
            monitor="val_masked_accuracy",
            mode="max",
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_masked_accuracy",
            mode="max",
            patience=5,
            restore_best_weights=True,
        ),
    ]

    history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks)
    model.save_weights(str(checkpoint_path))

    (output_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "vocab.json").write_text(
        json.dumps(
            {
                "source": source_vectorizer.get_vocabulary(),
                "target": target_vectorizer.get_vocabulary(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (output_dir / "history.json").write_text(
        json.dumps(history.history, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"训练完成，模型已保存到: {output_dir}")


if __name__ == "__main__":
    main()
