import pandas as pd
import os, sys
import random
import argparse
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))
from utils.g2p_util import *
from utils.convert_jamo_util import *
from utils.parallel_util import *

# 속도 개선 -> pandas.DataFrame.apply 를 병렬 프로그램으로 변경

def args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=False, default='../../data/train/20221124/colloquial_correct_test_data.csv', help='절대경로 기입')
    parser.add_argument('--output', required=False, default='../../data', help='절대경로 기입')
    return parser.parse_args()


def create_errdata(input_path, output):
    df = pd.read_csv(input_path, names=['src','tgt'])
    df['tgt'] = df.apply(lambda x: x['tgt'].strip(), axis=1)

    disc_cnt = len(df) // 2
    t_idx = random.sample(range(len(df)), disc_cnt)

    gtp = df[['tgt']].loc[t_idx]
    jamo_error = df[['tgt']].drop(t_idx)

    gtp = parallel_dataframe(gtp, convert_g2p)
    gtp.to_csv(f'{output}/corpus_repair_g2p_data.csv', encoding='utf-8-sig', index=False)

    jamo_error = parallel_dataframe(jamo_error, convert_jamo)
    jamo_error.to_csv(f'{output}/corpus_repair_jamo_error_data.csv', encoding='utf-8-sig', index=False)

    result = pd.concat([gtp, jamo_error], axis=0)
    result.to_csv(f'{output}/korean_corpus_test_20221201.csv', encoding='utf-8-sig', index=False, header=False)


if __name__ == '__main__':
    config = args()
    create_errdata(config.input, config.output)