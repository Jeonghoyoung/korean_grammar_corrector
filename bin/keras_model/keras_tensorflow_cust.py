import tensorflow as tf


class Transformer_Encoder(tf.keras.layers.Layer):
    def __init__(self, vocab_size, dff, d_model, num_heads, dropout,**kwargs):
        super(Transformer_Encoder, self).__init__(**kwargs)
        # Hyper Parameters
        self.vocab_size = vocab_size
        # self.num_layers = num_layers
        self.dff = dff
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = dropout

        # Layers
        self.attention = tf.keras.layers.MultiHeadAttention(
            num_heads=self.num_heads, key_dim=self.d_model
        )
        self.ffnn = tf.keras.Sequential(
            [tf.keras.layers.Dense(self.dff, activation='relu'), tf.keras.layers.Dense(self.d_model)]
        )
        self.layernorm_1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm_2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.supports_masking = True

    def call(self, inputs, mask=None):
        if mask is not None:
            padding_mask = tf.cast(mask[:, tf.newaxis, tf.newaxis, :], dtype="int32")
        attention_output = self.attention(
            query=inputs, value=inputs, key=inputs, attention_mask=padding_mask
        )
        attention_output = tf.keras.layers.Dropout(rate=self.dropout)(attention_output)
        attention_output = self.layernorm_1(inputs + attention_output)

        outputs = self.ffnn(attention_output)
        outputs = tf.keras.layers.Dropout(rate=self.dropout)(outputs)
        outputs = self.layernorm_2(attention_output + outputs)
        return outputs

class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, max_length, vocab_size, d_model,**kwargs):
        super(PositionalEmbedding, self).__init__(**kwargs)
        self.token_embeddings = tf.keras.layers.Embedding(
            input_dim=vocab_size, output_dim= d_model
        )
        self.position_embedding = tf.keras.layers.Embedding(
            input_dim = max_length, output_dim=d_model
        )
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.d_model = d_model


    def call(self, inputs):
        length = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embedding(positions)
        return embedded_tokens + embedded_positions

    def compute_mask(self, inputs, mask=None):
        return tf.math.not_equal(inputs, 0)


class Transformer_Decoder(tf.keras.layers.Layer):
    def __init__(self, dff, d_model, num_heads, dropout, **kwargs):
        super(Transformer_Decoder, self).__init__(**kwargs)
        # Hyper parameters
        self.dff = dff
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = dropout

        # Layers
        self.attention_1 = tf.keras.layers.MultiHeadAttention(
            num_heads = num_heads, key_dim = d_model
        )
        self.attention_2 = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model
        )
        self.ffnn = tf.keras.Sequential(
            [tf.keras.layers.Dense(self.dff, activation='relu'), tf.keras.layers.Dense(self.d_model)]
        )

        self.layernorm_1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm_2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm_3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.supports_masking = True
        # self.dropout_layer = tf.keras.layers.Dropout(rate=dropout)

    def call(self, inputs, encoder_outputs, mask=None):
        causal_mask = self.get_causal_attention_mask(inputs)
        if mask is not None:
            padding_mask = tf.cast(mask[:, tf.newaxis, :], dtype="int32")
            padding_mask = tf.minimum(padding_mask, causal_mask)

        attention_output_1 = self.attention_1(
            query=inputs, value=inputs, key=inputs, attention_mask = causal_mask
        )
        output_1 = self.layernorm_1(inputs + attention_output_1)

        attention_output_2 = self.attention_2(
            query=output_1, value=encoder_outputs, key=encoder_outputs, attention_mask=padding_mask,
        )
        attention_output_2 = tf.keras.layers.Dropout(rate=self.dropout)(attention_output_2)

        output_2 = self.layernorm_2(output_1 + attention_output_2)
        outputs = self.ffnn(output_2)
        outputs = tf.keras.layers.Dropout(rate=self.dropout)(outputs)
        outputs = self.layernorm_3(output_2 + outputs)
        return outputs


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


def KerasTransformer(vocab_size, d_model, dff, num_heads, max_length):
    encoder_inputs = tf.keras.Input(shape=(None, ), dtype='int64', name='encoder_inputs')
    encoder_x = PositionalEmbedding(max_length=max_length, vocab_size=vocab_size, d_model=d_model)(encoder_inputs)
    encoder_x = tf.keras.layers.Dropout(0.2)(encoder_x)

    encoder_outputs = Transformer_Encoder(vocab_size=vocab_size,d_model=d_model, dff=dff, num_heads=num_heads, dropout=0.2)(encoder_x)
    encoder = tf.keras.Model(encoder_inputs, encoder_outputs)

    decoder_inputs = tf.keras.Input(shape=(None,), dtype="int64", name="decoder_inputs")
    # encoded_seq_inputs = tf.keras.Input(shape=(None, d_model), name="decoder_state_inputs")
    encoded_seq_inputs = encoder([encoder_inputs])

    decoder_x = PositionalEmbedding(max_length=max_length, vocab_size=vocab_size, d_model=d_model)(decoder_inputs)
    decoder_x = tf.keras.layers.Dropout(0.2)(decoder_x)
    decoder_x = Transformer_Decoder(d_model=d_model, dff=dff, num_heads=num_heads, dropout=0.2)(decoder_x, encoded_seq_inputs)
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
    test_transformer = KerasTransformer(2000, 256, 512, 4, 20)
    print(test_transformer.summary())
    model_plot(test_transformer, '../test_transformer.png')

    test_transformer.compile(
        'rmsprop', loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    test_transformer.fit()

