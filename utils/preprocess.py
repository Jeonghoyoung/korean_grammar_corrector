import re
import time
import pandas as pd
from mecab import MeCab
import argparse
from tokenizers import SentencePieceBPETokenizer, Tokenizer


# 김기현 딥러닝 책 참고한 data preprocessing
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







