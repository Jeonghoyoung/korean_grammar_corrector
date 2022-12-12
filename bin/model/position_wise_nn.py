import tensorflow as tf


class Pointwise_FeedForward_Network(tf.keras.layers.Layer):
    # Pointwise_FeedForward_Network 에서는 인코더의 출력에서 512개의 차원이 2048차원까지 확장되고, 다시 512개의 차원으로 압축된다.
    # dff = 은닉층의 size
    def __init__(self, d_model, dff):
        super().__init__()
        self.d_model = d_model
        self.dff = dff

        self.middle = tf.keras.layers.Dense(dff, activation='relu')
        self.out = tf.keras.layers.Dense(d_model)

    def __call__(self, x):
        middle = self.middle(x) # middle shape : (batch_size, seq_len, dff)
        out = self.out(middle) # out shape : (batch_size, seq_len, d_model)
        return out