import sys
sys.path.append('/Users/hoyoung/Desktop/pycharm_work/korean_grammar_corrector/utils')
import tensorflow_preprocess as tp
import tensorflow as tf
import tensorflow_datasets as tfds
from transformer_model import *
import re


class Predictor:
    def __init__(self, max_length, model, tokenizer):
        self.max_length = max_length
        self.model = model,
        self.tokenizer = tokenizer


    def evaluate(self, sentence, model):
        sentence = tp.full_stop_filter(sentence)

        # 입력 문장에 시작 토큰과 종료 토큰을 추가
        sentence = tf.expand_dims(
            [self.tokenizer.vocab_size] + self.tokenizer.encode(sentence) + [self.tokenizer.vocab_size + 1], axis=0)

        output = tf.expand_dims([self.tokenizer.vocab_size], 0)

        # 디코더의 예측 시작
        for i in range(self.max_length):
            predictions = model(inputs=[sentence, output], training=False)

            # 현재 시점의 예측 단어를 받아온다.
            predictions = predictions[:, -1:, :]
            predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

            # 만약 현재 시점의 예측 단어가 종료 토큰이라면 예측을 중단
            if tf.equal(predicted_id, [self.tokenizer.vocab_size + 1][0]):
                break

            # 현재 시점의 예측 단어를 output(출력)에 연결한다.
            # output은 for문의 다음 루프에서 디코더의 입력이 된다.
            output = tf.concat([output, predicted_id], axis=-1)

        # 단어 예측이 모두 끝났다면 output을 리턴.
        return tf.squeeze(output, axis=0)

    def predict(self, sentence, model, debug=False):
        prediction = self.evaluate(sentence, model)

        # prediction == 디코더가 리턴한 챗봇의 대답에 해당하는 정수 시퀀스
        # tokenizer.decode()를 통해 정수 시퀀스를 문자열로 디코딩.
        predicted_sentence = self.tokenizer.decode(
            [i for i in prediction if i < self.tokenizer.vocab_size])
        
        if debug:
            print('Input: {}'.format(sentence))
            print('Output: {}'.format(predicted_sentence))

        return predicted_sentence


if __name__ == '__main__':
    train_tok = tfds.deprecated.text.SubwordTextEncoder.load_from_file('../../tokenizer.tok')
    print(train_tok)

    model_path = "../../checkpoint_20221205/cp.ckpt"
    d_model = 128
    num_layers = 4
    num_heads = 4
    dff = 256
    dropout = 0.2
    vocab_size = train_tok.vocab_size + 2

    model = transformer(
        vocab_size=vocab_size,
        num_layers=num_layers,
        dff=dff,
        d_model=d_model,
        num_heads=num_heads,
        dropout=dropout,
        name="transformer")
    model.load_weights(model_path)

    sent= '이유른 신차의 경우 야 기퍼센트에서 시자캐요.'
    predict = Predictor(24, model, train_tok)
    print(predict.predict(sent, model))