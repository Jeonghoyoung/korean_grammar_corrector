import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.dense_proj = keras.Sequential(
            [layers.Dense(dense_dim, activation="relu"), layers.Dense(embed_dim),]
        ) # Feed Forward Neural Network
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.supports_masking = True

    def call(self, inputs, mask=None):
        if mask is not None:
            padding_mask = tf.cast(mask[:, tf.newaxis, tf.newaxis, :], dtype="int32")
            attention_output = self.attention(
                query=inputs, value=inputs, key=inputs, attention_mask=padding_mask
            )
        else:
            attention_output = self.attention(
                query=inputs, value=inputs, key=inputs)
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)


class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, vocab_size, embed_dim, **kwargs):
        super(PositionalEmbedding, self).__init__(**kwargs)
        self.token_embeddings = layers.Embedding(
            input_dim=vocab_size, output_dim=embed_dim
        )
        self.position_embeddings = layers.Embedding(
            input_dim=sequence_length, output_dim=embed_dim
        )
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

    def call(self, inputs):
        length = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions

    def compute_mask(self, inputs, mask=None):
        return tf.math.not_equal(inputs, 0)


class TransformerDecoder(layers.Layer):
    def __init__(self, embed_dim, latent_dim, num_heads, **kwargs):
        super(TransformerDecoder, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.attention_1 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.attention_2 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.dense_proj = keras.Sequential(
            [layers.Dense(latent_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.layernorm_3 = layers.LayerNormalization()
        self.supports_masking = True

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

        proj_output = self.dense_proj(out_2)
        return self.layernorm_3(out_2 + proj_output)

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


def KerasTransformer(vocab_size, d_model, dff, num_heads):
    encoder_inputs = tf.keras.Input(shape=(None, ), name='encoder_inputs')
    encoder_x = PositionalEmbedding(sequence_length=30,vocab_size=vocab_size, embed_dim=d_model)(encoder_inputs)
    encoder_x = tf.keras.layers.Dropout(0.1)(encoder_x)

    encoder_outputs = TransformerEncoder(embed_dim=d_model, dense_dim=dff, num_heads=num_heads)(encoder_x)
    encoder = tf.keras.Model(encoder_inputs, encoder_outputs)

    decoder_inputs = tf.keras.Input(shape=(None,), name="decoder_inputs")
    # encoded_seq_inputs = tf.keras.Input(shape=(None, d_model), name="decoder_state_inputs")
    encoded_seq_inputs = encoder([encoder_inputs])

    decoder_x = PositionalEmbedding(sequence_length=30, vocab_size=vocab_size, embed_dim=d_model)(decoder_inputs)
    decoder_x = tf.keras.layers.Dropout(0.1)(decoder_x)
    decoder_x = TransformerDecoder(embed_dim=d_model, latent_dim=dff, num_heads=num_heads)(decoder_x, encoded_seq_inputs)
    decoder_outputs = tf.keras.layers.Dense(vocab_size, activation='softmax', name='outputs')(decoder_x)
    decoder = tf.keras.Model([decoder_inputs, encoded_seq_inputs], decoder_outputs)

    decoder_outputs = decoder([decoder_inputs, encoder_outputs])
    keras_transformer = tf.keras.Model(
        [encoder_inputs, decoder_inputs], decoder_outputs, name='KerasTransformer'
    )
    return keras_transformer


def model_plot(model, save_path):
    tf.keras.utils.plot_model(model, to_file=save_path, show_shapes=False)
    return None


if __name__ == '__main__':
    epochs=1
    test_transformer = KerasTransformer(2000, 256, 512, 4)
    print(test_transformer.summary())
    # model_plot(test_transformer, '../test_transformer.png')

    # test_transformer.compile(
    #     'rmsprop', loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    # )

    # test_transformer.fit()

