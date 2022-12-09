import re
import random
from utils.parallel_util import *
import time

random.seed(123)

kor_begin = 44032
kor_end = 55203

chosung_base = 588
jungsung_base = 28

jaum_begin = 12593
jaum_end = 12622

moum_begin = 12623
moum_end = 12643

chosung_list = [ 'ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ',
        'ㅅ', 'ㅆ', 'ㅇ' , 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

jungsung_list = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ',
        'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ',
        'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ',
        'ㅡ', 'ㅢ', 'ㅣ', ' ']

jongsung_list = [
    ' ', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ',
        'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ',
        'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ',
        'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

jaum_list = ['ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄸ', 'ㄹ',
              'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ',
              'ㅃ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

moum_list = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ',
              'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']


def compose(chosung, jungsung, jongsung):
    # 자음 모음 결합
    char = chr(
        kor_begin +
        chosung_base * chosung_list.index(chosung) +
        jungsung_base * jungsung_list.index(jungsung) +
        jongsung_list.index(jongsung)
    )
    return char


def decompose(c):
    # 자음 모음 분해
    if not character_is_korean(c):
        return None
    i = ord(c)
    if jaum_begin <= i <= jaum_end:
        return c, ' ', ' '
    if moum_begin <= i <= moum_end:
        return ' ', c, ' '

    # decomposition rule
    i -= kor_begin
    cho  = i // chosung_base
    jung = ( i - cho * chosung_base ) // jungsung_base
    jong = ( i - cho * chosung_base - jung * jungsung_base )
    return chosung_list[cho], jungsung_list[jung], jongsung_list[jong]


def character_is_korean(c):
    i = ord(c)
    return ((kor_begin <= i <= kor_end) or
            (jaum_begin <= i <= jaum_end) or
            (moum_begin <= i <= moum_end))


def clean_korean(df:pd.DataFrame):
    ko_regex = r'[^ㄱ-힣|\.|\s]'
    only_ko_list = [i for i in range(len(df)) if len(re.findall(ko_regex, df['tgt'][i]))==0]
    only_ko = df.loc[only_ko_list]
    only_ko.reset_index(inplace=True, drop=True)
    return only_ko


def jamo_error_data(text:str):
    global comp
    text = str(text)
    if len(text) >= 1:
        # 띄어쓰기를 기준으로 split
        split_t = text.split()

        # Select sample number
        sample_num = random.randrange(1,4)

        # Random Choose words
        if len(split_t) <= sample_num:
            # print(f'{split_t} split count : {len(split_t)}')
            # print()
            sample_num = len(split_t) - 1
            rand_t = random.sample(range(len(split_t)), sample_num)
        else:
            rand_t = random.sample(range(len(split_t)), sample_num)

        for n in rand_t:
            # 한글 제외 문자 모두 제거
            if len(re.findall('[ㄱ-힣]', split_t[n])) > 0:
                replace_text = re.sub('[^ㄱ-힣]', '', split_t[n])
                replace_text = replace_text.replace(' ', '')
            else:
                replace_text = random.choice(chosung_list)
            t = random.choice(range(len(replace_text)))

            # 자음 모음 분리.
            try:
                decompose_text = decompose(replace_text[t])
                decompose_text = list(decompose_text)
                c = random.choice(range(len(decompose_text)))

                if decompose_text[c] in jongsung_list:
                    if c == 0:
                        decompose_text[c] = random.choice(chosung_list)
                        comp = compose(decompose_text[c], decompose_text[1], decompose_text[-1])
                    elif c == 2:
                        decompose_text[c] = random.choice(jongsung_list)
                        comp = compose(decompose_text[0], decompose_text[1], decompose_text[c])

                elif decompose_text[c] in jungsung_list:
                    if len(decompose_text) == 2:
                        decompose_text[c] = random.choice(jungsung_list)
                        comp = compose(decompose_text[0], decompose_text[c])
                    elif len(decompose_text) > 2:
                        decompose_text[c] = random.choice(jungsung_list)
                        comp = compose(decompose_text[0], decompose_text[c], decompose_text[-1])
            except:
                # print(text)
                pass

            text_l = list(replace_text)
            text_l[t] = comp
            replace_text = ''.join(text_l)
            split_t[n] = replace_text
        dist = ' '.join(split_t)
        return dist
    else:
        return text
        pass


def convert_jamo(df):
    df['error'] = df['tgt'].apply(lambda x: jamo_error_data(x))
    return df


if __name__ == '__main__':
    test_df = pd.read_csv('../data/train/20221124/colloquial_correct_test_data.csv')
    print(test_df)
    s_time = time.time()
    # t = parallel_dataframe(test_df, convert_jamo)
    t = [jamo_error_data(l) for l in test_df['tgt'].tolist()]
    print(time.time() - s_time)
    # print(t[:5])
    # 자음 모음 치환의 경우, 단일 코어로도 문제 없이 돌아가나 데이터 갯수가 늘어나면 늘어날수록 병렬 코어가 속도 향상에 효과가 있음을 확인.
    # 160만개의 데이터 기준 단일=34초, 병렬=16초로 동일하게 2배 이상의 효과를 보임.
