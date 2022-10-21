import sys
from collections import Counter
import re
import time
import pandas as pd
from mecab import MeCab
import utils.random_sample_util as rst
import argparse
from tokenizers import SentencePieceBPETokenizer, Tokenizer
import tensorflow as tf
import tensorflow_datasets as tfds


def args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=False, default='')
    parser.add_argument('--output', required=False, default='')
    return parser.parse_args()

# tensorflow data preprocessing
def morphs_replace_text(text):
    mecab = MeCab()

    text = text.replace(' ', '▁')

    morphs_text = mecab.morphs(text)
    return morphs_text


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

# 김기현 딥러닝 책 참고한 data preprocessing
def tokenizer_text(text, tokenizer):
    output = tokenizer.encode(text)
    tok_text = ''.join(output.tokens)
    return tok_text.replace('▁▁▁', '▁▁')


def detokenizer_text(text):
    return text.replace(' ', '').replace('▁▁', ' ').replace('▁','')

def subword(df, col,tokenizer=None):
    if tokenizer is None:
        with open('../data/sample/replace_morphs.txt', 'w') as f:
            for line in df[col].values:
                try:
                    f.write(line + '\n')
                except:
                    print(line)

        time.sleep(5)

        tokenizer = SentencePieceBPETokenizer()
        min_frequency = 5
        vocab_size = 30000

        tokenizer.train(["../data/sample/replace_morphs.txt"], vocab_size=vocab_size,
                        min_frequency=min_frequency)
        tokenizer.save('../data/subword_tokenizer/tokenizer.json')
    else:
        tokenizer = Tokenizer.from_file('../data/subword_tokenizer/tokenizer.json')

    return tokenizer


def dataload(df,col, tokenizer=None):
    df[f'{col}_morphs_ko'] = df[col].apply(lambda x: morphs_replace_text(str(x)))
    df[f'{col}_morphs_length'] = df[f'{col}_morphs_ko'].apply(lambda x: len(x))

    df_idx = [i for i in range(len(df)) if 14 <= df[f'{col}_morphs_length'][i] < 25]
    df = df.loc[df_idx]
    df.reset_index(inplace=True, drop=True)

    df[f'{col}_replace_morphs_text'] = df[f'{col}_morphs_ko'].apply(lambda x: ' '.join(x))
    tokenizer = subword(df, f'{col}_replace_morphs_text',tokenizer)

    df[f'{col}_tokenizing_text'] = df[f'{col}_replace_morphs_text'].apply(lambda x: tokenizer_text(x, tokenizer))
    return df


if __name__ == '__main__':
    config = args()

    df = pd.read_csv('../data/colloquial_correct_train_data.csv')
    print(df.head())

    src = dataload(df, 'src')
    tgt = dataload(df, 'tgt')
    print(src.head())
    print(tgt.head())
    tgt = tgt.drop(['src', 'tgt'], axis=1)
    concat_df = pd.concat([src, tgt], axis=1)
    print(concat_df.columns)
    print(concat_df.head())
    concat_df.to_csv('../data/sample/colloquial_correct_train_data.csv', index=False)







