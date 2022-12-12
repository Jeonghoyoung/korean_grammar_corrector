import pandas as pd
import numpy as np
import multiprocessing as mp


def parallel_dataframe(df, func, n_cores= mp.cpu_count()):
    df_split = np.array_split(df, n_cores)
    pool = mp.Pool(n_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df