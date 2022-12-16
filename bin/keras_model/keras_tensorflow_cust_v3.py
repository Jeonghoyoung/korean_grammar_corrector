import tensorflow as tf
import numpy as np


# 기존 모델(Keras MultiAttention 사용 X)에서 Dropout만 제외한 모델
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
            [tf.keras.layers.Dense(units=self.dff, activation='relu'), tf.keras.layers.Dense(units=self.d_model)]
        )
        self.layernorm_1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm_2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        # self.supports_masking = True

    def encoder_layer(self, name='encoder_layer',  mask=None):
        inputs = tf.keras.Input(shape=(None, self.d_model), name="inputs")

        if mask is not None:
            padding_mask = tf.cast(mask[:, tf.newaxis, tf.newaxis, :], dtype="int32")
            attention_output = self.attention(
                query=inputs, value=inputs, key=inputs, attention_mask=padding_mask
            )
        else:
            attention_output = self.attention(
                query=inputs, value=inputs, key=inputs)

        attention_output = self.layernorm_1(inputs + attention_output)

        outputs = self.ffnn(attention_output)
        outputs = self.layernorm_2(attention_output + outputs)
        return tf.keras.Model(inputs=inputs, outputs=outputs, name=name)

    def encoder(self, num_layers, mask=None, name='encoder'):
        inputs = tf.keras.Input(shape=(None,), name="inputs")
        embeddings = PositionalEmbedding(vocab_size=self.vocab_size, d_model=self.d_model)(inputs)
        outputs = tf.keras.layers.Dropout(rate=self.dropout)(embeddings)

        for i in range(num_layers):
            outputs = self.encoder_layer(name="encoder_layer_{}".format(i),  mask=None)(embeddings)
        return tf.keras.Model(inputs=inputs, outputs=outputs, name=name)


class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, vocab_size, d_model, **kwargs):
        super(PositionalEmbedding, self).__init__(**kwargs)
        self.token_embeddings = tf.keras.layers.Embedding(
            input_dim=vocab_size, output_dim= d_model
        )
        self.position_embedding = tf.keras.layers.Embedding(
            input_dim = vocab_size, output_dim=d_model
        )

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.pos_encoding = self.positional_encoding(self.vocab_size, self.d_model)

    def get_angles(self, vocab_size, i, d_model):
        angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        return vocab_size * angles

    def positional_encoding(self, vocab_size, d_model):
        angle_rads = self.get_angles(
            vocab_size=tf.range(vocab_size, dtype=tf.float32)[:, tf.newaxis],
            i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
            d_model=d_model)  # shape : (position, d_model)
        # 오른쪽으로 짝수번째 인덱스는 sin 함수를 적용
        sines = tf.math.sin(angle_rads[:, 0::2])
        # 오른쪽으로 홀수번째 인덱스는 cos 함수를 적용
        cosines = tf.math.cos(angle_rads[:, 1::2])

        angle_rads = np.zeros(angle_rads.shape)
        angle_rads[:, 0::2] = sines
        angle_rads[:, 1::2] = cosines

        pos_encoding = tf.constant(angle_rads)
        pos_encoding = pos_encoding[tf.newaxis, ...]  # pos_encoding shape : (1, position, d_model)

        return tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, inputs):
        embedding_token = self.token_embeddings(inputs)
        embedding_token *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        outputs = embedding_token + self.pos_encoding[:, :tf.shape(embedding_token)[1], :]
        return outputs

class Transformer_Decoder(tf.keras.layers.Layer):
    def __init__(self, vocab_size, dff, d_model, num_heads, dropout, **kwargs):
        super(Transformer_Decoder, self).__init__(**kwargs)
        # Hyper parameters
        self.vocab_size = vocab_size
        # self.num_layers = num_layers
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

    def decoder_layer(self, name='decoder_layer', mask=None):
        inputs = tf.keras.Input(shape=(None, self.d_model), name="inputs")
        encoder_outputs = tf.keras.Input(shape=(None, self.d_model), name="encoder_outputs")

        causal_mask = self.get_causal_attention_mask(inputs)
        attention_output_1 = self.attention_1(
            query=inputs, value=inputs, key=inputs, attention_mask=causal_mask
        )
        output_1 = self.layernorm_1(inputs + attention_output_1)

        if mask is not None:
            padding_mask = tf.cast(mask[:, tf.newaxis, :], dtype=tf.int32)
            padding_mask = tf.minimum(padding_mask, causal_mask)

            attention_output_2 = self.attention_2(
                query=output_1, value=encoder_outputs, key=encoder_outputs, attention_mask=padding_mask,
            )
        else:
            attention_output_2 = self.attention_2(
                query=output_1, value=encoder_outputs, key=encoder_outputs)
        output_2 = self.layernorm_2(output_1 + attention_output_2)

        outputs = self.ffnn(output_2)
        outputs = self.layernorm_3(output_2 + outputs)
        return tf.keras.Model(inputs=[inputs, encoder_outputs], outputs=outputs,name=name)

    def decoder(self, num_layers, name='decoder'):
        inputs = tf.keras.Input(shape=(None,), name='inputs')
        encoder_outputs = tf.keras.Input(shape=(None, self.d_model), name='encoder_outputs')

        embeddings = PositionalEmbedding(vocab_size=self.vocab_size, d_model=self.d_model)(inputs)
        outputs = tf.keras.layers.Dropout(rate=self.dropout)(embeddings)

        for i in range(num_layers):
            outputs = self.decoder_layer(name='decoder_layer_{}'.format(i))([outputs, encoder_outputs])
        return tf.keras.Model(inputs=[inputs, encoder_outputs], outputs=outputs,name=name)



def KerasTransformer(vocab_size, num_layers, d_model, dff, num_heads, dropout, name):
    # encoder input
    inputs = tf.keras.Input(shape=(None,), name='inputs')

    # decoder input
    decoder_inputs = tf.keras.Input(shape=(None,), name='dec_inputs')

    encoder = Transformer_Encoder(vocab_size, dff, d_model, num_heads, dropout)
    decoder = Transformer_Decoder(vocab_size, dff, d_model, num_heads, dropout)

    encoder_outputs = encoder.encoder(num_layers=num_layers)(inputs=inputs)
    decoder_outputs = decoder.decoder(num_layers=num_layers)(inputs=[decoder_inputs, encoder_outputs])

    outputs = tf.keras.layers.Dense(units=vocab_size, name="outputs")(decoder_outputs)

    return tf.keras.Model(inputs=[inputs, decoder_inputs], outputs=outputs, name=name)


def Model(vocab_size, num_layers, dff, d_model, num_heads, dropout, name):
    tf.keras.backend.clear_session()
    model = KerasTransformer(
        vocab_size=vocab_size,
        num_layers=num_layers,
        dff=dff,
        d_model=d_model,
        num_heads=num_heads,
        dropout=dropout,
        name=name
    )
    return model


def model_plot(model, save_path):
    tf.keras.utils.plot_model(model, to_file=save_path, show_shapes=False)
    return None


if __name__ == '__main__':
    epochs=1
    test_transformer = Model(vocab_size=2000, d_model=256, num_layers=1, num_heads=4, dff=512, dropout=0.5, name='test')
    print(test_transformer.summary())
    model_plot(test_transformer, '../test_transformer.png')

    # test_transformer.compile(
    #     'rmsprop', loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    # )

    # test_transformer.fit()

