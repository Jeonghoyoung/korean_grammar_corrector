import os, sys
import tensorflow_datasets as tfds
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))) # 상위 폴더 내 모듈 참조
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))) # 상위 상위 폴더 내 모듈 참조
from model.transformer_model import *
from model.predict import *
# Ignore warning message
import warnings
warnings.filterwarnings('ignore')
from silence_tensorflow import silence_tensorflow
silence_tensorflow()

# 주석 추가 필요.
class Corrector:

    def __init__(self, model_path, tokenizer_path, max_length=70, d_model=128, num_layers=4, num_heads=4, dff=256, dropout=0.1):
        '''
        :param model_path: pretrained model 의 확장자를 제외한 파일명과 경로
        :param tokenizer_path: tokenizer의 확장자(.subwords)를 제외한 파일명과 경로
        :param max_length: 훈련된 모델의 MAX LENGTH
        :param d_model: 인코더와 디코더에서의 정해진 입력과 출력의 크기, 임베딩 벡터 차원의 크기
        :param num_layers: 인코더와 디코더의 각각의 층 수
        :param num_heads: MultiheadAttention 의 Head 개수
        :param dff: hidden layer dimension
        :param dropout: DropOut
        '''

        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.max_length = max_length

        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dff = dff
        self.dropout = dropout

    def load(self):
        token = self.__tokenizer()
        model = Transformer_Model(vocab_size=token.vocab_size+2,
                                  d_model=self.d_model,
                                  num_layers=self.num_layers,
                                  num_heads=self.num_heads,
                                  dff=self.dff,
                                  dropout=self.dropout,
                                  name='Corpus-Repair')

        model.load_weights(self.model_path)
        return model

    def __tokenizer(self):
        return tfds.deprecated.text.SubwordTextEncoder.load_from_file(self.tokenizer_path)

    def run_repair(self, text:str, debug=False):
        load_model = self.load()
        predictor = Predictor(self.max_length, load_model, self.__tokenizer())
        return predictor.predict(text, load_model, debug)


if __name__ == '__main__':
    m_path = '../../checkpoint_maxlength70/cp.ckpt'
    t_path = '../../tokenizer/tokenizer_maxlength_70.tok'

    corrector = Corrector(m_path, t_path, 70)
    test_model = corrector.load()

    # print(test_model.summary())
    t = '연방정부느 니와 함께 현재 판타나우에서 소방대를 통해 이뤄지는 진화자거블 확때하기 위한 재정지원 방안도 검토하고 읻따'
    print(corrector.run_repair(t, debug=True))



