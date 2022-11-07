import tensorflow as tf
from multi_head_attention import *
from position_wise_nn import *
from positional_encoding import *

# 인코더 층의 내부 아키텍처

# 어텐션시 패딩 토큰을 제외하도록 패딩 마스크 사용. (MultiHeadAttention 함수의 mask의 인자값으로 padding_mask가 사용되는 이유)
# 인코터는 총 두개의 서브층(멀티헤드어텐션, 포지션 와이즈 피드포워드 신경망) 각 서브층 이후에는 Add & Normalization
def encoder_layer(dff, d_model, num_heads, dropout, name="encoder_layer"):
    # 하나의 인코더 층을 구현하는 코드, 실제 트랜스포머는 num_layers 개수만큼의 인코더 층을 사용하므로 이를 여러번 쌓는 코드 구현 필요.
    inputs = tf.keras.Input(shape=(None, d_model), name="inputs")

    # 인코더는 패딩 마스크 사용
    padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")

    # 멀티-헤드 어텐션 (첫번째 서브층 / 셀프 어텐션)
    attention = MultiHeadAttention(
      d_model, num_heads, name="attention")({
          'query': inputs, 'key': inputs, 'value': inputs, # Q = K = V
          'mask': padding_mask # 패딩 마스크 사용
      })

    # 드롭아웃 + 잔차 연결과 층 정규화
    attention = tf.keras.layers.Dropout(rate=dropout)(attention)
    attention = tf.keras.layers.LayerNormalization(
      epsilon=1e-6)(inputs + attention)

    # 포지션 와이즈 피드 포워드 신경망 (두번째 서브층)
    outputs = tf.keras.layers.Dense(units=dff, activation='relu')(attention)
    outputs = tf.keras.layers.Dense(units=d_model)(outputs)

    # 드롭아웃 + 잔차 연결과 층 정규화
    outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)
    outputs = tf.keras.layers.LayerNormalization(
      epsilon=1e-6)(attention + outputs)

    return tf.keras.Model(
      inputs=[inputs, padding_mask], outputs=outputs, name=name)


# 인코더 쌓기
# 인코더 층을 쌓는 코드
# 인코더 층을 num_layers개만큼 쌓고, 마지막 인코더 층에서 얻는 (seq_len , d_model)크기의 행렬을 디코더로 보내주므로서 인코딩 연산이 끝나게됨.
def encoder(vocab_size, num_layers, dff, d_model, num_heads, dropout, name="encoder"):
    inputs = tf.keras.Input(shape=(None,), name="inputs")

    # 인코더는 패딩 마스크 사용
    padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")

    # 포지셔널 인코딩 + 드롭아웃
    embeddings = tf.keras.layers.Embedding(vocab_size, d_model)(inputs)  # Embedding
    embeddings *= tf.math.sqrt(tf.cast(d_model, tf.float32))
    embeddings = PositionalEncoding(vocab_size, d_model)(embeddings)  # PositionalEncoding
    outputs = tf.keras.layers.Dropout(rate=dropout)(embeddings)  # Dropout

    # 인코더를 num_layers개 쌓기
    for i in range(num_layers):
        outputs = encoder_layer(dff=dff, d_model=d_model, num_heads=num_heads,
                                dropout=dropout, name="encoder_layer_{}".format(i),
                                )([outputs, padding_mask])

    return tf.keras.Model(
        inputs=[inputs, padding_mask], outputs=outputs, name=name)