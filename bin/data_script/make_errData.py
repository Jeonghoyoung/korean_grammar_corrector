import re
import random
from tqdm import tqdm
import pandas as pd
from g2pk import G2p


random.seed(111)

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


def jamo_error_data(target_list:list):
    result = []
    for i in range(len(target_list)):
        if len(target_list[i]) >= 1:
            # 띄어쓰기를 기준으로 split
            split_t = target_list[i].split()

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
                    # print(target_list[i])
                    pass

                text_l = list(replace_text)
                text_l[t] = comp
                replace_text = ''.join(text_l)
                split_t[n] = replace_text
            dist = ' '.join(split_t)
            result.append(dist)
        else:
            pass
    return result


async def create_g2p_data(text_list):
    g2p = G2p()
    return [g2p(text_list[i]) for i in tqdm(range(len(text_list)))]


def main():
    df = pd.read_csv('../../data/train/20221130/korean_corpus_test_20221130.csv', names=['tgt'])
    df['tgt'] = df.apply(lambda x: x['tgt'].strip(), axis=1)
    print(len(df))
    print(df.head())

    disc_cnt = len(df) // 2
    t_idx = random.sample(range(len(df)), disc_cnt)

    gtp = df[['tgt']].loc[t_idx]
    edit_dist = df[['tgt']].drop(t_idx)

    gtp.reset_index(inplace=True, drop=True)
    edit_dist.reset_index(inplace=True, drop=True)
    #
    # print(len(df))
    # print(len(gtp))
    # print(len(edit_dist))


    g2p = G2p()
    g2p_data = [g2p(gtp['tgt'][k]) for k in tqdm(range(len(gtp)))]

    raw_gtp = gtp['tgt'].tolist()
    gtp_df = pd.DataFrame({'src':g2p_data, 'tgt':raw_gtp})
    gtp_df.to_csv('../../data/colloquial_g2p_data.csv', encoding='utf-8-sig', index=False)

    # gtp_df = pd.read_csv('../data/colloquial_g2p_data.csv')
    g2p_data = gtp_df['src'].tolist()
    raw_gtp = gtp_df['tgt'].tolist()

    raw_dist = edit_dist['tgt'].tolist()
    print(len(raw_dist))
    jamo_error_list = jamo_error_data(raw_dist)
    print(len(jamo_error_list))

    jamo_err_df = pd.DataFrame({'src': jamo_error_list, 'tgt': raw_dist})
    jamo_err_df.to_csv('../../data/colloquial_jamo_error_data.csv', encoding='utf-8-sig', index=False)

    src = g2p_data + jamo_error_list
    tgt = raw_gtp + raw_dist

    t_df = pd.DataFrame({'src': src, 'tgt': tgt})

    print(t_df.head())

    t_df.to_csv('../../data/train/korean_corpus_test_20221201.csv', encoding='utf-8-sig', index=False)


if __name__ == '__main__':
    main()