import pandas as pd
from utils.parallel_util import *
from g2pk import G2p
import time

def convert_g2p(df):
    g2p = G2p()
    df['error'] = df['tgt'].apply(lambda x: g2p(x))
    return df


if __name__ == '__main__':
    test_df = pd.read_csv('../data/train/20221124/colloquial_correct_test_data.csv')
    print(test_df)
    s_time = time.time()
    t = parallel_dataframe(test_df, convert_g2p)
    print(time.time() - s_time)
    # 5000개의 데이터로 테스트 결과 병렬로 구성했을때 단일 코어보다 약 2배 가량 빠름 (135s -> 58s)
    