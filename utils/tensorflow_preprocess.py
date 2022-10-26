import re
import tensorflow as tf
import tensorflow_datasets as tfds


# tensorflow data preprocessing
def full_stop_filter(text):
    return re.sub(r'([?.!,])', r' \1', text).strip()


def wordpiece_tokenizer(src_list, tgt_list):
    tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(src_list + tgt_list, target_vocab_size=2**13)
    start_token, end_token = [tokenizer.vocab_size], [tokenizer.vocab_size + 1]
    vocab_size = tokenizer.vocab_size + 2

    return tokenizer, start_token, end_token, vocab_size


def tokenize_and_filter(src_list, tgt_list, max_length=20):
    tokenizer, START_TOKEN, END_TOKEN, VOCAB_SIZE = wordpiece_tokenizer(src_list, tgt_list)
    tokenized_src,tokenized_tgt = [], []

    for sent1, sent2 in zip(src_list, tgt_list):
        # encoding, 시작 토큰, 종료 토큰 추가
        sent1 = START_TOKEN + tokenizer.encode(sent1) + END_TOKEN
        sent2 = START_TOKEN + tokenizer.encode(sent2) + END_TOKEN

        tokenized_src.append(sent1)
        tokenized_tgt.append(sent2)

    # padding
    tokenized_src = tf.keras.preprocessing.sequence.pad_sequences(tokenized_src, maxlen=max_length, padding='post')
    tokenized_tgt = tf.keras.preprocessing.sequence.pad_sequences(tokenized_tgt, maxlen=max_length, padding='post')

    return tokenized_src, tokenized_tgt


def create_train_dataset(inputs, outputs, batch_size = 64, buffer_size=1024):
    '''
    :param inputs: Source data :list
    :param outputs: Target data :list
    :param batch_size: batch_size
    :param buffer_size: 고정된 버퍼 크기로 데이터를 섞는데, 데이터가 완전히 랜덤적으로 뒤섞기 위해서는 입력된 데이터 크기보다 큰 수를 입력해 주셔야 한다.
    :return:
    '''
    dataset = tf.data.Dataset.from_tensor_slices((
        {
            'inputs': inputs,
            'dec_inputs': outputs[:, :-1]
        },
        {
            'outputs': outputs[:, 1:]
        },
    ))

    dataset = dataset.cache()
    dataset = dataset.shuffle(buffer_size)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset
