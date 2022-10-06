from mecab import MeCab

if __name__ == '__main__':
    m = MeCab()
    t = '안녕하세요. 반갑습니다.'

    print(m.morphs(t))