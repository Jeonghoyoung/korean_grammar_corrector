import os, sys
import pandas as pd
import re
sys.path.append('../../')
from utils.file_util import *

if __name__ == '__main__':
    aihub = pd.read_csv('../../data/raw/aihub_2022/colloquial_ko.csv', names=['tgt'])
    nia = pd.read_csv('../../data/raw/nia_2021/nia_colloquial_ko.csv', names=['tgt'])

    print(len(aihub))
    print(len(nia))
    
    