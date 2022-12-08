import re
import tensorflow as tf
import tensorflow_datasets as tfds


# tensorflow data preprocessing
def full_stop_filter(text):
    return re.sub(r'([?.!,])', r' \1', str(text)).strip()


def wordpiece_tokenizer(src_list, tgt_list):
    tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(src_list + tgt_list, target_vocab_size=2**13)
    start_token, end_token = [tokenizer.vocab_size], [tokenizer.vocab_size + 1]
    vocab_size = tokenizer.vocab_size + 2

    return tokenizer, start_token, end_token, vocab_size


def tokenize_and_filter(src_list, tgt_list, max_length=20):
    tokenizer, START_TOKEN, END_TOKEN, VOCAB_SIZE = wordpiece_tokenizer(src_list, tgt_list)

    tokenized_src = [START_TOKEN + tokenizer.encode(sent) + END_TOKEN for sent in src_list]
    tokenized_tgt = [START_TOKEN + tokenizer.encode(sent) + END_TOKEN for sent in tgt_list]

    # padding
    tokenized_src = tf.keras.preprocessing.sequence.pad_sequences(tokenized_src, maxlen=max_length, padding='post')
    tokenized_tgt = tf.keras.preprocessing.sequence.pad_sequences(tokenized_tgt, maxlen=max_length, padding='post')

    return tokenized_src, tokenized_tgt, tokenizer


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

    dataset = dataset.cache() # cache를 활용해서 데이터를 로드할 때 빠른 처리를 기대해 봄
    dataset = dataset.shuffle(buffer_size) # buffer_size를 전체 데이터 수보다 크게하여 완전하게 섞음
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset


def data2tensor(src_list, tgt_list, max_length, batch_size):
    src = [full_stop_filter(x) for x in src_list]
    tgt = [full_stop_filter(x) for x in tgt_list]
    
    inputs, outputs, tokenizer = tokenize_and_filter(src, tgt, max_length)
    tensor_dataset = create_train_dataset(inputs, outputs, batch_size, buffer_size=len(src)+1)
    return tensor_dataset, tokenizer


def save_tokenizer(tokenizer, path, filename):
    return tokenizer.save_to_file(path + '/' + filename)


def load_tokenizer(path):
    load_model = tfds.deprecated.text.SubwordTextEncoder.load_from_file(path)
    return load_model


if __name__ == '__main__':
    i = '안뇽하세요 저는 뉴규'
    t = '안녕하세요 저는 누구입니다.'
    tok, s, e, v = wordpiece_tokenizer(i, t)
    encode_i = tok.encode(i)
    print(tok.decode(encode_i))

