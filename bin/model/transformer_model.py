import sys
sys.path.append('/Users/hoyoung/Desktop/pycharm_work/korean_grammar_corrector/bin/model')
import tensorflow as tf
from multi_head_attention import *
from position_wise_nn import *
from positional_encoding import *
from transformer_encoder import *
from transformer_decoder import *


# 트랜스포머 구현

# 인코더의 출력은 디코더에서 인코더-디코더 어텐션에서 사용되기 위해 디코더로 전달하고 디코더의 끝단에는 다중 클래스 분류 문제를 풀 수 있도록, vocab_size 만큼의
# 뉴런을 가지는 출력층 추가


def transformer(vocab_size, num_layers, dff, d_model, num_heads, dropout, name='transformer'):
    # encoder input
    inputs = tf.keras.Input(shape=(None,), name='inputs')

    # decoder input
    dec_inputs = tf.keras.Input(shape=(None,), name='dec_inputs')

    # encoder padding mask
    # 인코더의 패딩 마스크
    enc_padding_mask = tf.keras.layers.Lambda(
        create_padding_mask, output_shape=(1, 1, None),
        name='enc_padding_mask')(inputs)

    # 디코더의 룩어헤드 마스크(첫번째 서브층)
    look_ahead_mask = tf.keras.layers.Lambda(
        create_look_ahead_mask, output_shape=(1, None, None),
        name='look_ahead_mask')(dec_inputs)

    # 디코더의 패딩 마스크(두번째 서브층)
    dec_padding_mask = tf.keras.layers.Lambda(
        create_padding_mask, output_shape=(1, 1, None),
        name='dec_padding_mask')(inputs)

    # 인코더의 출력은 enc_outputs. 디코더로 전달된다.
    enc_outputs = encoder(vocab_size=vocab_size, num_layers=num_layers, dff=dff,
                          d_model=d_model, num_heads=num_heads, dropout=dropout,
                          )(inputs=[inputs, enc_padding_mask])  # 인코더의 입력은 입력 문장과 패딩 마스크

    # 디코더의 출력은 dec_outputs. 출력층으로 전달된다.
    dec_outputs = decoder(vocab_size=vocab_size, num_layers=num_layers, dff=dff,
                          d_model=d_model, num_heads=num_heads, dropout=dropout,
                          )(inputs=[dec_inputs, enc_outputs, look_ahead_mask, dec_padding_mask])

    # 다음 단어 예측을 위한 출력층
    outputs = tf.keras.layers.Dense(units=vocab_size, name="outputs")(dec_outputs)

    return tf.keras.Model(inputs=[inputs, dec_inputs], outputs=outputs, name=name)


def Transformer_Model(vocab_size, num_layers, dff, d_model, num_heads, dropout, name):
    tf.keras.backend.clear_session()
    model = transformer(
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
    small_transformer = Transformer_Model(vocab_size=9000, d_model=256, num_layers=4, num_heads=4, dff=512, dropout=0.1, name='test')
