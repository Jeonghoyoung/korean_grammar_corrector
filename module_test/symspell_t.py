from symspellpy import SymSpell, Verbosity
from hangul_utils import split_syllable_char, split_syllables, join_jamos
import pandas as pd
'''

'''
# 자모 분리를 하지 않고 실행

# syms = SymSpell()
# dict_path = '../ko_50k.txt'
# syms.load_dictionary(dict_path, 0, 1) # ko_50k 빈도수 데이터를 불러와서 사전 구축
#
# t = '안뇽하세요'
#
# sug = syms.lookup(t, Verbosity.ALL, max_edit_distance=2) # 유사 어절을 찾기
#
# print(len(sug))
#
# for s in sug:
#     print(s.term, s.distance, s.count)

# hangul_utils 의 자모분리를 사용하여 확인.
vocab = pd.read_csv('../ko_50k.txt', sep=' ', names=['term', 'count'])

vocab.term = vocab.term.map(split_syllables)
print(vocab.head())
# vocab.to_csv('ko_50k_decomposed.csv', header=None, index=False)