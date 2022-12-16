import os, sys

import pandas as pd
import tensorflow as tf

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))  # 상위 폴더 내 모듈 참조
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))  # 상위 상위 폴더 내 모듈 참조
from utils.tensorflow_preprocess import *
from model.transformer_model import *
# from keras_model.keras_tensorflow_cust import *
from keras_model.keras_tensorflow_cust_v3 import *


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


class Trainer:
    def __init__(self, save_path=None, tokenizer_name=None, model_name=None, d_model=32, num_layers=4, num_heads=4, dff=64, dropout=0.1, max_length=10,
                 batch_size=16, epochs=30):
        self.save_path = save_path
        self.tokenizer_name = tokenizer_name
        self.model_name = model_name

        # Model Parameters
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dff = dff
        self.dropout = dropout
        self.max_length = max_length
        self.batch_size = batch_size
        self.epochs = epochs

    def accuracy(self, y_true, y_pred):
        # 레이블의 크기 : (batch_size, MAX_LENGTH - 1)
        y_true = tf.reshape(y_true, shape=(-1, self.max_length - 1))
        return tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred)

    def save_checkpoint(self):
        return tf.keras.callbacks.ModelCheckpoint(filepath=self.save_path + '/' + self.model_name, save_weights_only=True, verbose=1)

    def loss_function(self, y_true, y_pred):
        y_true = tf.reshape(y_true, shape=(-1, self.max_length - 1))

        loss = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction='none')(y_true, y_pred)

        mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
        loss = tf.multiply(loss, mask)

        return tf.reduce_mean(loss)

    def __load_data(self, data, d_type):
        # type() 통해 문자열인 경우 경로로 판단하여 READ 부터 시행.
        if d_type == 'path':
            extension = os.path.splitext(data)[1]
            if extension == '.csv':
                df = pd.read_csv(data)
                return df[['src', 'tgt']]
            elif extension == '.xlsx':
                df = pd.read_excel(data)
                return df[['src', 'tgt']]
            else:
                print('Only Supported CSV or XLSX File ---')
                return None
        elif d_type == 'dataframe':
            return data[['src', 'tgt']]
        else:
            print('Not Supported Data Type ---')
            return None

    def train(self, data, d_type, model=None):
        df = self.__load_data(data, d_type)
        print(df.head())

        print('!!!!!!!!!!!!!!!!! Convert data !!!!!!!!!!!!!!!!!')
        dataset, tokenizer = data2tensor(df['src'].tolist(), df['tgt'].tolist(), self.max_length, self.batch_size)

        if model is not None:
            transformer_model = model
        else:
            transformer_model = Model(vocab_size=tokenizer.vocab_size + 2,
                                                  d_model=self.d_model,
                                                  num_layers=self.num_layers,
                                                  num_heads=self.num_heads,
                                                  dff=self.dff,
                                                  dropout=self.dropout,
                                                  name='Corpus-Repair')

        learning_rate = CustomSchedule(d_model=self.d_model)
        optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

        transformer_model.compile(optimizer=optimizer, loss=self.loss_function, metrics=[self.accuracy])

        cp_callback = self.save_checkpoint()

        if self.save_path is not None:
            assert self.tokenizer_name is not None or self.model_name is not None, '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Write your tokenizer name !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'
            transformer_model.fit(dataset, epochs=self.epochs, callbacks=[cp_callback])
            save_tokenizer(tokenizer, self.save_path, self.tokenizer_name)
        else:
            transformer_model.fit(dataset, epochs=self.epochs)
        return transformer_model


if __name__ == '__main__':
    path = '../../data/testsets_sample100.xlsx'

    transformer_trainer = Trainer(save_path='../..', tokenizer_name='test_tok', model_name='cpk_test', epochs=3)

    tesT_model = transformer_trainer.train(path, d_type='path', model=None)
