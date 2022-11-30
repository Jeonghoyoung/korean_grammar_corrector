import os, sys
import pandas as pd
import re
sys.path.append('../../')
from utils.file_util import *


if __name__ == '__main__':
    tech = pd.read_csv('../../data/raw/tech/aihub_2022_tech_science_enko.csv')
    print(len(tech))
    tech = tech[['tgt']]
    tech['no_space'] = tech['tgt'].apply(lambda x: re.sub('\s', '', x))
    tech = tech.drop_duplicates(subset=['no_space'])
    print(len(tech))
    print(tech.tail(3))

    aihub = pd.read_csv('../../data/raw/aihub_2022/colloquial_ko.csv')
    aihub['no_space'] = aihub['tgt'].apply(lambda x: re.sub('\s', '', x))
    aihub = aihub.drop_duplicates(subset=['no_space'])

    concat_df = pd.concat([aihub, tech], axis=0)
    print(len(concat_df))
    print(concat_df.tail())
    concat_df = concat_df.drop_duplicates(subset=['no_space'])
    print(len(concat_df))
    concat_df.reset_index(inplace=True, drop=True)
    
    drop_idx = []
    for i in range(len(concat_df)):
        if len(re.findall('[a-zA-Z0-9]', str(concat_df['tgt'][i]))) > 0:
            drop_idx.append(i)
    print(len(drop_idx))

    result = concat_df.drop(drop_idx)
    result = result[['tgt']]

    result.to_csv('../../data/raw/korean_corpus_all_data.csv', index=False, header=False, encoding='utf-8')