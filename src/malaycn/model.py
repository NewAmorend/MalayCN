from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen: int, vocab_size: int, embed_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.maxlen = maxlen
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.token_emb = layers.Embedding(vocab_size, embed_dim, mask_zero=True)
        self.pos_emb = layers.Embedding(maxlen, embed_dim)

    def call(self, x):
        length = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_tokens = self.token_emb(x)
        embedded_positions = self.pos_emb(positions)
        return embedded_tokens + embedded_positions

    def compute_mask(self, inputs, mask=None):
        return self.token_emb.compute_mask(inputs)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "maxlen": self.maxlen,
                "vocab_size": self.vocab_size,
                "embed_dim": self.embed_dim,
            }
        )
        return config


class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim: int, latent_dim: int, num_heads: int, dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.dense_proj = keras.Sequential(
            [layers.Dense(latent_dim, activation="relu"), layers.Dense(embed_dim)]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.dropout_1 = layers.Dropout(dropout)
        self.dropout_2 = layers.Dropout(dropout)

    def call(self, inputs, mask=None, training=False):
        attention_mask = None
        if mask is not None:
            attention_mask = mask[:, tf.newaxis, :]

        attention_output = self.attention(
            query=inputs,
            value=inputs,
            key=inputs,
            attention_mask=attention_mask,
            training=training,
        )
        attention_output = self.dropout_1(attention_output, training=training)
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        proj_output = self.dropout_2(proj_output, training=training)
        return self.layernorm_2(proj_input + proj_output)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "latent_dim": self.latent_dim,
                "num_heads": self.num_heads,
                "dropout": self.dropout,
            }
        )
        return config


class TransformerDecoder(layers.Layer):
    def __init__(self, embed_dim: int, latent_dim: int, num_heads: int, dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.self_attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.cross_attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.dense_proj = keras.Sequential(
            [layers.Dense(latent_dim, activation="relu"), layers.Dense(embed_dim)]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.layernorm_3 = layers.LayerNormalization()
        self.dropout_1 = layers.Dropout(dropout)
        self.dropout_2 = layers.Dropout(dropout)
        self.dropout_3 = layers.Dropout(dropout)

    def call(self, inputs, encoder_outputs, encoder_mask=None, decoder_mask=None, training=False):
        causal_mask = self._causal_attention_mask(inputs)

        self_attention_mask = causal_mask
        if decoder_mask is not None:
            decoder_padding_mask = decoder_mask[:, tf.newaxis, :]
            self_attention_mask = tf.logical_and(causal_mask, decoder_padding_mask)

        cross_attention_mask = None
        if encoder_mask is not None:
            cross_attention_mask = encoder_mask[:, tf.newaxis, :]

        self_attention_output = self.self_attention(
            query=inputs,
            value=inputs,
            key=inputs,
            attention_mask=self_attention_mask,
            training=training,
        )
        self_attention_output = self.dropout_1(self_attention_output, training=training)
        out_1 = self.layernorm_1(inputs + self_attention_output)

        cross_attention_output = self.cross_attention(
            query=out_1,
            value=encoder_outputs,
            key=encoder_outputs,
            attention_mask=cross_attention_mask,
            training=training,
        )
        cross_attention_output = self.dropout_2(cross_attention_output, training=training)
        out_2 = self.layernorm_2(out_1 + cross_attention_output)

        proj_output = self.dense_proj(out_2)
        proj_output = self.dropout_3(proj_output, training=training)
        return self.layernorm_3(out_2 + proj_output)

    def _causal_attention_mask(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size = input_shape[0]
        sequence_length = input_shape[1]
        i = tf.range(sequence_length)[:, tf.newaxis]
        j = tf.range(sequence_length)
        mask = i >= j
        mask = tf.reshape(mask, (1, sequence_length, sequence_length))
        return tf.tile(mask, [batch_size, 1, 1])

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "latent_dim": self.latent_dim,
                "num_heads": self.num_heads,
                "dropout": self.dropout,
            }
        )
        return config


class MalayChineseTransformer(keras.Model):
    def __init__(
        self,
        source_vocab_size: int,
        target_vocab_size: int,
        max_source_len: int,
        max_target_len: int,
        target_vectorize_len: int | None = None,
        embed_dim: int = 256,
        latent_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.source_vocab_size = source_vocab_size
        self.target_vocab_size = target_vocab_size
        self.max_source_len = max_source_len
        self.max_target_len = max_target_len
        self.target_vectorize_len = target_vectorize_len or (max_target_len + 1)
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.dropout = dropout

        self.encoder_embedding = TokenAndPositionEmbedding(max_source_len, source_vocab_size, embed_dim)
        self.decoder_embedding = TokenAndPositionEmbedding(max_target_len, target_vocab_size, embed_dim)
        self.encoder = TransformerEncoder(embed_dim, latent_dim, num_heads, dropout)
        self.decoder = TransformerDecoder(embed_dim, latent_dim, num_heads, dropout)
        self.dropout_layer = layers.Dropout(dropout)
        self.output_dense = layers.Dense(target_vocab_size)

    def call(self, inputs, training=False):
        encoder_inputs, decoder_inputs = inputs
        encoder_mask = tf.not_equal(encoder_inputs, 0)
        decoder_mask = tf.not_equal(decoder_inputs, 0)

        encoder_embeddings = self.encoder_embedding(encoder_inputs)
        encoder_outputs = self.encoder(encoder_embeddings, mask=encoder_mask, training=training)

        decoder_embeddings = self.decoder_embedding(decoder_inputs)
        decoder_outputs = self.decoder(
            decoder_embeddings,
            encoder_outputs,
            encoder_mask=encoder_mask,
            decoder_mask=decoder_mask,
            training=training,
        )
        decoder_outputs = self.dropout_layer(decoder_outputs, training=training)
        return self.output_dense(decoder_outputs)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "source_vocab_size": self.source_vocab_size,
                "target_vocab_size": self.target_vocab_size,
                "max_source_len": self.max_source_len,
                "max_target_len": self.max_target_len,
                "target_vectorize_len": self.target_vectorize_len,
                "embed_dim": self.embed_dim,
                "latent_dim": self.latent_dim,
                "num_heads": self.num_heads,
                "dropout": self.dropout,
            }
        )
        return config
