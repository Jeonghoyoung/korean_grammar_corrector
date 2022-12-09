from joblib import Parallel, delayed
import os, sys
import multiprocessing as mp
import pandas as pd
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from g2pk import G2p
import time
from tqdm import tqdm


def parallel_dataframe(df, func, n_cores=4):
    df_split = np.array_split(df, n_cores)
    pool = mp.Pool(n_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df


def g2p_data(df):
    g2p = G2p()
    df['g2p'] = df['tgt'].apply(lambda x: g2p(x))
    return df


if __name__ == '__main__':
    test_df = pd.read_csv('../data/train/20221124/colloquial_correct_test_data.csv')
    # print(test_df)
    # s_time = time.time()
    # t = parallel_dataframe(test_df, g2p_data)
    # print(time.time() - s_time)
    # print()
    p = G2p()
    s2_time = time.time()
    t2 = [p(i) for i in tqdm(test_df['tgt'].tolist())]
    print(time.time() - s2_time)