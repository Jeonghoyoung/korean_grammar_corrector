import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


class PositionalEncoding(tf.keras.layers.Layer):
    '''
    포지셔널 인코딩 방법을 사용하면 순서 정보가 보존되는데, 예를 들어 각 임베딩 벡터에 포지셔널 인코딩의 값을 더하면
    같은 단어라고 하더라도 문장 내의 위치에 따라서 트랜스포머의 입력으로 들어가는 임베딩 벡터의 값이 달라진다.
    이에 따라 트랜스포머의 입력은 순서 정보가 고려된 임베딩 벡터가 된다.

    position : 입력 문장에서의 임베딩 벡터의 위치
    i : 임베딩 벡터 내의 차원의 인덱스
    d_model : 트랜스포머의 모든 층의 출력 차원을 의미하는 트랜스포머의 하이퍼파라미터

    Positional Embedding 의 position / 10000 ** (2i / d_model) 부분 산출 함수

    PE(position, 2i) = sin(position / 10000 ** (2i / d_model))     ((pos, 2i)일때 sin함수))
    PE(position, 2i + 1) = cos(position / 10000 ** (2i / d_model)) ((pos, 2i+1일때 cos함수))

    shape = (1, d_model)
    angle_rates shape = (1, d_model)
    return shape = 행렬곱 (position, 1), (1, d_model) =  (position, d_model)
    '''

    def __init__(self, position, d_model):
        super(PositionalEncoding, self).__init__()
        self.position = position
        self.d_model = d_model
        self.pos_encoding = self.positional_encoding(self.position, self.d_model)

    def get_angles(self, position, i, d_model):
        angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        return position * angles

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            position=tf.range(position, dtype=tf.float32)[:, tf.newaxis],
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
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]

if __name__ == '__main__':
    sample_pos_encoding = PositionalEncoding(50, 128)

    plt.pcolormesh(sample_pos_encoding.pos_encoding.numpy()[0], cmap='RdBu')
    plt.xlabel('Depth')
    plt.xlim((0, 128))
    plt.ylabel('Position')
    plt.colorbar()
    plt.show()