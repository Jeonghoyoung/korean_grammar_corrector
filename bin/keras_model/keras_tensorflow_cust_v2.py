import tensorflow as tf
from tensorflow import keras
import numpy as np

# 인코더, 디코더 layer 구현 필요. , 일부 에러가 발생됨.
class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self, dff, d_model, num_heads, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        # Hyper Parameters
        # self.vocab_size = vocab_size
        self.num_heads = num_heads
        # self.num_layers = num_layers
        self.dff = dff
        self.d_model = d_model

        # Layers
        self.attention = tf.keras.layers.MultiHeadAttention(
            num_heads=self.num_heads, key_dim=self.d_model
        )
        self.ffnn = tf.keras.Sequential(
            [tf.keras.layers.Dense(units=self.dff, activation='relu'), tf.keras.layers.Dense(units=self.d_model)]
        )
        self.layernorm_1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm_2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        # self.supports_masking = True

    def call(self, inputs, mask=None):
        if mask is not None:
            padding_mask = tf.cast(mask[:, tf.newaxis, tf.newaxis, :], dtype="int32")
        attention_output = self.attention(
            query=inputs, value=inputs, key=inputs, attention_mask=padding_mask
        )
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.ffnn(proj_input)
        return self.layernorm_2(proj_input + proj_output)

class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, max_length, vocab_size, d_model, **kwargs):
        super(PositionalEmbedding, self).__init__(**kwargs)
        self.token_embeddings = tf.keras.layers.Embedding(
            input_dim=vocab_size, output_dim=d_model
        )
        self.position_embeddings = tf.keras.layers.Embedding(
            input_dim=max_length, output_dim=d_model
        )
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.d_model = d_model

    def call(self, inputs):
        length = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions

    def compute_mask(self, inputs, mask=None):
        return tf.math.not_equal(inputs, 0)


class TransformerDecoder(tf.keras.layers.Layer):
    def __init__(self, dff, d_model, num_heads, **kwargs):
        super(TransformerDecoder, self).__init__(**kwargs)
        # Hyper parameters
        # self.vocab_size = vocab_size
        # self.num_layers = num_layers
        self.dff = dff
        self.d_model = d_model
        self.num_heads = num_heads

        # Layers
        self.attention_1 = tf.keras.layers.MultiHeadAttention(
            num_heads = num_heads, key_dim = d_model
        )
        self.attention_2 = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model
        )
        self.ffnn = tf.keras.Sequential(
            [tf.keras.layers.Dense(dff, activation='relu'), tf.keras.layers.Dense(d_model)]
        )

        self.layernorm_1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm_2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm_3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        # self.supports_masking = True
        # self.dropout_layer = tf.keras.layers.Dropout(rate=dropout)

    def get_causal_attention_mask(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size, sequence_length = input_shape[0], input_shape[1]
        i = tf.range(sequence_length)[:, tf.newaxis]
        j = tf.range(sequence_length)
        mask = tf.cast(i >= j, dtype="int32")
        mask = tf.reshape(mask, (1, input_shape[1], input_shape[1]))
        mult = tf.concat(
            [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)],
            axis=0,
        )
        return tf.tile(mask, mult)

    def call(self, inputs, encoder_outputs, mask=None):
        causal_mask = self.get_causal_attention_mask(inputs)
        if mask is not None:
            padding_mask = tf.cast(mask[:, tf.newaxis, :], dtype="int32")
            padding_mask = tf.minimum(padding_mask, causal_mask)

        attention_output_1 = self.attention_1(
            query=inputs, value=inputs, key=inputs, attention_mask=causal_mask
        )
        out_1 = self.layernorm_1(inputs + attention_output_1)

        attention_output_2 = self.attention_2(
            query=out_1,
            value=encoder_outputs,
            key=encoder_outputs,
            attention_mask=padding_mask,
        )
        out_2 = self.layernorm_2(out_1 + attention_output_2)

        proj_output = self.ffnn(out_2)
        return self.layernorm_3(out_2 + proj_output)


def KerasTransformer(vocab_size, d_model, dff, num_heads, max_length):
    # encoder input
    encoder_inputs = keras.Input(shape=(None,), dtype="int64", name="inputs")
    encoder_x = PositionalEmbedding(max_length, vocab_size, d_model)(encoder_inputs)
    encoder_outputs = TransformerEncoder(dff, d_model, num_heads)(encoder_x)
    # Encoder Model
    encoder = keras.Model(encoder_inputs, encoder_outputs, name='transformer_encoder')

    # Decoder input
    decoder_inputs = keras.Input(shape=(None,), dtype="int64", name="dec_inputs")
    # encoded_seq_inputs = keras.Input(shape=(None, d_model), name="encoder_outputs")
    encoded_seq_inputs = encoder([encoder_inputs])

    decoder_x = PositionalEmbedding(max_length, vocab_size, d_model)(decoder_inputs)
    decoder_x = TransformerDecoder(dff, d_model, num_heads)(decoder_x, encoded_seq_inputs)

    decoder_x = tf.keras.layers.Dropout(0.5)(decoder_x)
    outputs = tf.keras.layers.Dense(vocab_size, activation="softmax")(decoder_x)

    # Decoder Model
    decoder = keras.Model([decoder_inputs, encoded_seq_inputs], outputs, name='transformer_decoder')

    decoder_outputs = decoder([decoder_inputs, encoder_outputs])

    # TransFormer Model
    transformer = keras.Model(
        [encoder_inputs, decoder_inputs], decoder_outputs, name="keras_transformer")
    return transformer


def Model(vocab_size, dff, d_model, num_heads, max_length):
    tf.keras.backend.clear_session()
    model = KerasTransformer(
        vocab_size=vocab_size,
        dff=dff,
        d_model=d_model,
        num_heads=num_heads,
        max_length=max_length
    )
    return model


def model_plot(model, save_path):
    tf.keras.utils.plot_model(model, to_file=save_path, show_shapes=False)
    return None


if __name__ == '__main__':
    epochs=1
    test_transformer = Model(vocab_size=2000, d_model=256, num_heads=4, dff=512, max_length=70)
    print(test_transformer.summary())
    model_plot(test_transformer, '../test_transformer2.png')

    # test_transformer.compile(
    #     'rmsprop', loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    # )

    # test_transformer.fit()

