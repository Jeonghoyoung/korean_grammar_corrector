from konlpy.tag import Mecab

def correct_spacing(text, verbose=False):
    """
    Mecab 품사 태그를 기반으로 한국어 띄어쓰기를 교정하는 함수.
    
    Args:
        text (str): 띄어쓰기를 교정할 입력 문장.
    
    Returns:
        str: 교정된 문장.
    """
    mecab = Mecab()  # Mecab 객체 생성
    words = mecab.pos(text)  # 품사 태깅
    if verbose:
        print(words)
    corrected_sentence = []
    temp_buffer = []  # 임시 버퍼: 결합해야 할 단어 저장

    for i, (word, tag) in enumerate(words):
        temp_buffer.append(word)
        
        if i < len(words) - 1:
            next_word, next_tag = words[i + 1]
            
            # 1. 현재 단어와 다음 단어가 결합되는 경우
            # (1) 형용사(VA) + 연결 어미(EC)
            # (2) 보조 용언(VX) + 연결 어미(EC)
            if (tag.startswith("VA") and next_tag.startswith("EC")) or \
               (tag.startswith("VX") and next_tag.startswith("EC")):
                continue  # 결합 처리 후 넘어감
            
            if tag.startswith("EC") and next_tag.startswith("VX"):
                continue
            if next_tag.startswith("VCP"):
                continue
        
        # 임시 버퍼에 쌓인 단어들을 하나로 합쳐 저장
        corrected_sentence.append("".join(temp_buffer))
        temp_buffer = []  # 버퍼 초기화
        
        # 2. 다음 단어와 분리 여부 판단
        if i < len(words) - 1:
            next_word, next_tag = words[i + 1]
            
            # 조사(J), 어미(E), 접속사(X)는 앞 단어와 붙임
            if next_tag.startswith("J") or next_tag.startswith("E") or next_tag.startswith("X"):
                continue
            
            # 기본적으로 띄어쓰기 추가
            corrected_sentence.append(" ")
    
    # 남은 단어 처리
    if temp_buffer:
        corrected_sentence.append("".join(temp_buffer))
    
    return "".join(corrected_sentence).strip()

if __name__ == '__main__':
  # 테스트
  text1 = "안녕하세요잘지내시나요오늘은날씨가참좋네요"
  text2 = "좋아해요"
  print(correct_spacing(text1))  # 안녕하세요 잘 지내시나요 오늘은 날씨가 참 좋네요
  print(correct_spacing(text2))  # 좋아해요
