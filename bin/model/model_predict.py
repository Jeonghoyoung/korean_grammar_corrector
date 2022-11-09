import utils.tensorflow_preprocess as tp
import tensorflow as tf


def evaluate(sentence, MAX_LENGTH, tokenizer):
    # 입력 문장에 대한 전처리
    sentence = tp.full_stop_filter(sentence)

    # 입력 문장에 시작 토큰과 종료 토큰을 추가
    sentence = tf.expand_dims(
      [tokenizer.vocab_size] + tokenizer.encode(sentence) + [tokenizer.vocab_size+1], axis=0)

    output = tf.expand_dims([tokenizer.vocab_size], 0)

    # 디코더의 예측 시작
    for i in range(MAX_LENGTH):
        predictions = model(inputs=[sentence, output], training=False)

        # 현재 시점의 예측 단어를 받아온다.
        predictions = predictions[:, -1:, :]
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        # 만약 현재 시점의 예측 단어가 종료 토큰이라면 예측을 중단
        if tf.equal(predicted_id, [tokenizer.vocab_size+1][0]):
            break

        # 현재 시점의 예측 단어를 output(출력)에 연결한다.
        # output은 for문의 다음 루프에서 디코더의 입력이 된다.
        output = tf.concat([output, predicted_id], axis=-1)

    # 단어 예측이 모두 끝났다면 output을 리턴.
    return tf.squeeze(output, axis=0)


def predict(sentence, MAX_LENGTH, tokenizer):
    prediction = evaluate(sentence, MAX_LENGTH, tokenizer)

    predicted_sentence = tokenizer.decode(
      [i for i in prediction if i < tokenizer.vocab_size])

    print('Input: {}'.format(sentence))
    print('Output: {}'.format(predicted_sentence))

    return predicted_sentence