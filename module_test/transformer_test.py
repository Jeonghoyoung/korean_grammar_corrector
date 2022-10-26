'''
Reference : https://github.com/kimwoonggon/publicservant_AI/(심화2_트랜스포머_쉽게 구현해보기)
            https://wikidocs.net/22893 (15. 어텐션 메커니즘, 16. 트랜스포머)
Transformer 주요 Hyper Parameters

d_model :transformer의 인코더와 디코더에서의 정해진 입력과 출력의 크기를 의미.
         임베딩 벡터의 차원 또한 d_model 이며, 각 인코더와 디코더가 다음 층의 인코더와 디코더로 값을 보낼떄에도 이차원을 유지.

num_layers : 트랜스포머에서 하나의 인코더와 디코더를 층으로 생각하였을 때, 트랜스포머 모델에서 인코더와 디코더가 총 몇 층으로 구성되었는지를 의미.

num_heads : 트랜스포머에서는 어텐션을 사용할 때, 한 번 하는 것 보다 여러 개로 분할해서 병렬로 어텐션을 수행하고 결과값을 다시 하나로 합치는 방식,
            이 병렬의 개수를 의미.

d_ff : 피드 포워드 신경망이 존재하며 해당 신경망의 은닉층의 크기를 의미합니다. 피드 포워드 신경망의 입력층과 출력층의 크기는 d_model.

- 트랜스포머는 단어 임베딩 + 위치 임베딩 두 가지 임베딩을 거치게 된다.
- 위치 임베딩을 하는 이유 : RNN 계열과 같이 문장 내부 단어의 순서를 나타낼 방법이 없기 때문.
- 같은 단어라도 위치 임베딩 때문에 순서에 따라서 단어가 임베딩 된 것은 다를 수 있다.

'''

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def get_angles(pos, i, d_model):
    '''
    Positional Embedding 의 pos / 10000 ** (2i / d_model) 부분 산출 함수

    PE(pos, 2i) = sin(pos / 10000 ** (2i / d_model))
    PE(pos, 2i + 1) = cos(pos / 10000 ** (2i / d_model))
    '''
    angle_rates = 1 / np.power(10000, (2 * i // 2) / np.float32(d_model))
    # shape = (1, d_model)
    # pos shape :
    # angle_rates shape = (1, d_model)
    # return shape = 핼렬곱 (pos, 1), (1, d_model) =  (pos, d_model)
    return np.matmul(pos, angle_rates)


def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)  # shape : (position, d_model)
    # 오른쪽으로 짝수번째 인덱스는 sin 함수를 적용
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    # 오른쪽으로 홀수번째 인덱스는 cos 함수를 적용
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]  # pos_encoding shape : (1, position, d_model)
    # 왜 shape에 1을 추가해주냐면, batch_size 만큼 학습하기 위함임

    return tf.cast(pos_encoding, dtype=tf.float32)


class TransformerEmbedding(tf.keras.layers.Layer):
    def __init__(self, d_model, input_vocab_size, maximum_position_encoding, dropout_rate=0.1):
        super().__init__()

        self.d_model = d_model  # 하나의 단어가 d_model의 차원으로 인코딩 됨
        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        # vocab_size는 tokenizer 내부 vocab.txt의 사이즈
        self.pos_encoding = positional_encoding(maximum_position_encoding, self.d_model)  # 포지셔널 인코딩
        self.dropout = tf.keras.layers.Dropout(dropout_rate)  # 드롭아웃 설정

    def __call__(self, x, training):
        # 최초 x의 shape = (batch_size, seq_len)
        seq_len = tf.shape(x)[1]
        out = self.embedding(x)  # shape : (batch_size, input_seq_len, d_model)
        out = out * tf.math.sqrt(
            tf.cast(self.d_model, tf.float32))  # x에 sqrt(d_model) 만큼을 곱해주냐면, 임베딩 벡터보다 포지셔널 인코딩 임베딩 벡터의 영향력을 줄이기 위해서임
        # 포지셔널 인코딩은 순서만을 의미하기 때문에 임베딩 벡터보다 영향력이 적어야 이치에 맞음
        out = out + self.pos_encoding[:, :seq_len, :]
        out = self.dropout(out, training=training)

        return out  # shape : (batch_size, input_seq_len, d_model)


def scaled_dot_product_attention(q, k, v, mask=None):
    # q shape : (batch_size, seq_len, d_model)
    # k shape : (batch_size, seq_len, d_model)
    # v shape : (batch_size, seq_len, d_model)
    matmul_qk = tf.matmul(q, k, transpose_b = True)
    #matmul_qk shape : (batch_size, seq_len, seq_len)

    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # scaled_attetion_logits shape : (batch_size, seq_len, seq_len)

    if mask is not None:
        scaled_attention_logits = scaled_attention_logits + (mask * -1e9)

    softmax = tf.nn.softmax(scaled_attention_logits, axis=-1)

    # softmax shape : (batch_size, seq_len, seq_len)

    output = tf.matmul(softmax, v)

    # output(attention_value) shape : (batch_size, seq_len, d_model)
    # 즉 처음 입력 차원인 (batch_size, seq_len, d_model) 차원을 아웃풋으로 반환
    # 인풋과 아웃풋의 사이즈가 동일하다.

    # scaled_dot_product_attention 의 결과는 단어들 간의 연관성을 학습.
    return output, softmax


class MultiHeadAttention(tf.keras.layers.Layer):
    # 멀티 헤드 어텐션은 전체 어텐션을 분리하여 병렬적으로 어텐션을 수행하는 기법.
    # 이렇게 하는 이유는, 깊은 차원을 한번에 어텐션을 수행하는 것보다, 병렬로 각각 수행하는 것이 더 심도있는 언어들간의 관계를 학습할 수 있기 때문.

    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0,2,1,3])

    def __call__(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)


        attention_weights, softmax = scaled_dot_product_attention(q, k, v, mask)

        scaled_attention = tf.transpose(attention_weights, perm=[0,2,1,3])

        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))

        output = self.dense(concat_attention)

        return output, softmax


class Pointwise_FeedForward_Network(tf.keras.layers.Layer):
    # Pointwise_FeedForward_Network 에서는 인코더의 출력에서 512개의 차원이 2048차원까지 확장되고, 다시 512개의 차원으로 압축된다.
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


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = Pointwise_FeedForward_Network(d_model, dff)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def __call__(self, x, training, mask=None):
        # x : 위치 임베딩 + 단어 임베딩 된 인코딩의 인풋
        attn_output, _ = self.mha(x, x, x, mask)
        # 멀티헤드 어텐션
        # attn_output shape : (batch_size, input_seq_len, d_model)

        attn_output = self.dropout1(attn_output, training=training)

        out1 = self.layernorm1(x + attn_output)  # Residual Network 거침, 레이어 노멀레이제이션을 통한 값 평준화

        ffn_output = self.ffn(out1)  # ffn_output_shape : (batch_size, input_seq_len, d_model), 포인트와이즈 피드포워드 네트워크
        ffn_output = self.dropout2(ffn_output, training=training)

        out2 = self.layernorm2(out1 + ffn_output)  # Residual Network 거침
        # out2 shape : (batch_size, input_seq_len, d_model)

        return out2




if __name__ == '__main__':
    pos_encoding = positional_encoding(50, 512)
    print(pos_encoding.shape)

    plt.pcolormesh(pos_encoding[0], cmap='RdBu')
    plt.xlabel("Depth")
    plt.xlim((0, 512))
    plt.ylabel("Position")
    plt.colorbar()
    plt.show()