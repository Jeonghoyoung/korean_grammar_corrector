import re
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds


def full_stop_filter(text):
    return re.sub(r'([?.!,])', r' \1', text).strip()

def wordpiece_tokenizer(src_list, tgt_list):
    tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(src_list + tgt_list, target_vocab_size=2**13)
    start_token, end_token = [tokenizer.vocab_size], [tokenizer.vocab_size + 1]
    vocab_size = tokenizer.vocab_size + 2

    return tokenizer, start_token, end_token, vocab_size


def tokenize_and_filter(src_list, tgt_list, max_length=20):
    tokenizer, START_TOKEN, END_TOKEN, VOCAB_SIZE = wordpiece_tokenizer(src_list, tgt_list)
    tokenized_src, tokenized_tgt = [], []

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


df = pd.read_csv('../data/colloquial_correct_train_data.csv')
df = df.loc[:9]
print(df)
df['n_src'] = df['src'].apply(lambda x: full_stop_filter(x))
df['n_tgt'] = df['tgt'].apply(lambda x: full_stop_filter(x))

tokenizer, START_TOKEN, END_TOKEN, VOCAB_SIZE = wordpiece_tokenizer(df['n_src'], df['n_tgt'])

print(tokenizer.vocab_size)

print(f'Strat token index : {START_TOKEN}')
print(f'End token index : {END_TOKEN}')
print(f'Size word vocab : {VOCAB_SIZE}')

# encoding, decoding test

# encode() : 텍스트 시퀀스 -> 정수 시퀀스로 변환
tokenized_string = tokenizer.encode(df["n_tgt"][0])
print(f'임의의 질문 샘플을 정수 인코딩: {tokenized_string}')

org_string = tokenizer.decode(tokenized_string)
print(f'기존 문장: {org_string}')

inputs, outputs = tokenize_and_filter(df['n_src'], df['n_tgt'], max_length=24)

print(f'Input shape: {inputs.shape}')
print(f'Output shape: {outputs.shape}')

print(f'0번 샘플: {inputs[0]}')
print(f'0번 샘플: {outputs[0]}')

# tf.data.Dataset을 사용하여 데이터를 배치  단위로 불러온다.
# tensorflow dataset을 이용하여 shuffle을 수행하되, 배치 크기로 데이터를 묶는다.
# 또한 이과정에서 teacher forcing을 사용하기 위해서 디코더의 입력과 실제값 시퀀스를 구성한다.

batch_size = 2
buffer_size = 1024

# 디코더의 실제값 시퀀스에서는 시작 토큰을 제거

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
dataset = dataset.shuffle(buffer_size) # 고정된 버퍼 크기로 데이터를 섞는데, 데이터가 완전히 랜덤적으로 뒤섞기 위해서는 입력된 데이터 크기보다 큰 수를 입력해 주셔야 한다.
dataset = dataset.batch(batch_size)
dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

print(dataset)

