from konlpy.tag import Mecab
from joblib import Parallel, delayed
import multiprocessing as mp
from nltk.tokenize import TreebankWordTokenizer
# import jieba
# import MeCab as mecab_ja

# Prerequisite: ko nlp
# 1.
# konlp: pip3 install konlpy

# 2. yum install git, yum install python3-devel, yum install gcc-c++
# 3. install mecab: https://konlpy-ko.readthedocs.io/ko/v0.4.3/install/#id1
# bash <(curl -s https://raw.githubusercontent.com/konlpy/konlpy/master/scripts/mecab.sh)

# Prerequisite: en nlp
# 1. pip3 install nltk
# 2. run the codes to download data(optional): nltk.download()

# Prerequisite: other nlps
# janlp: pip install mecab-python3, pip install unidic-lite
# zhnlp: pip install jieba


def morphs(lang, tup_list, num_cores=mp.cpu_count(), verbose_status=0):
    if lang == 'en':
        return morph_nltk(tup_list, num_cores, verbose_status)
    elif lang == 'ko':
        return morphs_mecab(tup_list, num_cores, verbose_status)
    elif lang == 'zh' or lang == 'ja':
        return morphs_cjk(tup_list, num_cores, verbose_status)


def morphs_mecab(tup_list, num_cores=mp.cpu_count(), verbose_status=0):
    if tup_list is None:
        return None
    else:
        obj = KoNlpModule(Mecab('C:/mecab/mecab-ko-dic'))
        return Parallel(n_jobs=num_cores, verbose=verbose_status)(delayed(obj.morphs)(i) for i in tup_list)


# Use this for Chinese and Japanese
def morphs_cjk(tup_list, num_cores=1, verbose_status=0):
    if tup_list is None:
        return None
    else:
        obj = CjkNlpModule()
        return Parallel(n_jobs=num_cores, verbose=verbose_status)(delayed(obj.morphs)(i) for i in tup_list)


# Use this for Chinese only
# def morphs_jieba(tup_list, num_cores=mp.cpu_count(), verbose_status=0):
#     if tup_list is None:
#         return None
#     else:
#         return [(list(jieba.cut(t[0], cut_all=False)), list(jieba.cut(t[1], cut_all=False))) for t in tup_list]
#
#
# def morphs_mecab_ja(tup_list, num_cores=mp.cpu_count(), verbose_status=0):
#     if tup_list is None:
#         return None
#     else:
#         # todo: make this paralleled
#         wakati = mecab_ja.Tagger("-Owakati")
#         return [(wakati.parse(t[0]).split(), wakati.parse(t[1]).split()) for t in tup_list]


# Use this for English NLP
def morph_nltk(tup_list, num_cores=mp.cpu_count(), verbose_status=0):
    if tup_list is None:
        return None
    else:
        obj = TreebankWordTokenizer()
        return Parallel(n_jobs=num_cores, verbose=verbose_status)(delayed(obj.tokenize_sents)(i) for i in tup_list)


# Use this for comparing character level
def morph_atomic(tup_list, num_cores=mp.cpu_count(), verbose_status=0):
    if tup_list is None:
        return None
    else:
        obj = AtomicModule()
        return Parallel(n_jobs=num_cores, verbose=verbose_status)(delayed(obj.morphs)(i) for i in tup_list)


class KoNlpModule:
    '''
    tokenizing tuple list with various NLP.
    '''
    def __init__(self, obj):
        self.obj = obj

    def morphs(self, text_list):
        if isinstance(text_list, tuple):
            return self.obj.morphs(text_list[0]), self.obj.morphs(text_list[1])
        elif isinstance(text_list, str):
            return self.obj.morphs(text_list)
        else:
            return None


class CjkNlpModule:
    def morphs(self, tup):
        return self.split(tup[0]), self.split(tup[1])

    def split(self, slist):
        return [s for s in slist]


# class JiebaNlpModule:
#     def morphs(self, tup):
#         return list(jieba.cut(tup[0], cut_all=False)), list(jieba.cut(tup[1], cut_all=False))


class AtomicModule:
    def morphs(self, tup):
        return list(tup[0]), list(tup[1])

