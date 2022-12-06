import sys
sys.path.append('/Users/hoyoung/Desktop/pycharm_work/korean_grammar_corrector/utils')
sys.path.append('/Users/hoyoung/Desktop/pycharm_work/korean_grammar_corrector/bin/model')

import pandas as pd
from tensorflow_preprocess import *
from transformer_model import *
from predict import *

if __name__ == '__main__':
    testset = pd.read_csv('../data/train/20221130/korean_corpus_test_20221130.csv', names=['src', 'ref'])
    print(testset.head(3))
    model_path = '../checkpoint_20221205/cp.ckpt'
    
    tokenizer = load_tokenizer(path='../' ,filename='tokenizer.tok')
    model = Transformer_Model(vocab_size=tokenizer.vocab_size+2, d_model=128, num_layers=4, num_heads=4, dff=256, dropout=0.1, name='test')
    model.load_weights(model_path)

    predictor = Predictor(24, model, tokenizer)

    print('Start ----')
    pred = testset['src'].apply(lambda x: predictor.predict(x, model))
    print(len(pred))
    