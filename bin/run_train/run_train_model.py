import tensorflow as tf
import sys
sys.path.append('/Users/hoyoung/Desktop/pycharm_work/korean_grammar_corrector/bin/model')
sys.path.append('/Users/hoyoung/Desktop/pycharm_work/korean_grammar_corrector/utils')
import tensorflow_preprocess as tp
from model.transformer_model import *
import pandas as pd
import argparse


MAX_LENGTH = 24


def args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab_size', required=False, default=9000, help='Vocab Size')
    parser.add_argument('--num_layers', required=False, default=3, help='Number of layers ')
    parser.add_argument('--dff', required=False, default=256, help='Hidden layer size')
    parser.add_argument('--d_model', required=False, default=128, help='Embedding vector dimension size')
    parser.add_argument('--num_heads', required=False, default=4, help= 'Number of Multi-head attention heads')
    parser.add_argument('--dropout', required=False, default=0.1, help='Drop out prob')
    parser.add_argument('--model_name', required=False, default='transformer', help= 'Model Name')

    parser.add_argument('--epoch', required=False, default=1, help='Epochs')

    parser.add_argument('--input', required=False, default='../data/train/corpus_repair_test.csv')
    parser.add_argument('--save_path', required=False, default= "../../../checkpoint/cp.ckpt")
    return parser.parse_args()


def loss_function(y_true, y_pred):
    y_true = tf.reshape(y_true, shape=(-1, MAX_LENGTH - 1))

    loss = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')(y_true, y_pred)

    mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
    loss = tf.multiply(loss, mask)

    return tf.reduce_mean(loss)


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps**-1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


def accuracy(y_true, y_pred):
    # 레이블의 크기 : (batch_size, MAX_LENGTH - 1)
    y_true = tf.reshape(y_true, shape = (-1, MAX_LENGTH-1))
    return tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred)

def save_checkpoint(save_path):
    return tf.keras.callbacks.ModelCheckpoint(filepath=save_path, save_weights_only=True, verbose=1)

def main():
    configs = args()

    # Load Dataset
    df = pd.read_csv(configs.input)
    print(df.head())

    src = df['src'].apply(lambda x: tp.full_stop_filter(x))
    tgt = df['tgt'].apply(lambda x: tp.full_stop_filter(x))

    inputs, outputs, tokenizer = tp.tokenize_and_filter(src, tgt, max_length=24)
    dataset = tp.create_train_dataset(inputs, outputs, batch_size=64, buffer_size=len(src)+10000)

    # 모델 초기화
    tf.keras.backend.clear_session()

    model = transformer(
        vocab_size=configs.vocab_size,
        num_layers=configs.num_layers,
        dff=configs.dff,
        d_model=configs.d_model,
        num_heads=configs.num_heads,
        dropout=configs.dropout,
        name=configs.model_name)

    learning_rate = CustomSchedule(d_model=configs.d_model)

    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    model.compile(optimizer=optimizer, loss=loss_function, metrics=[accuracy])

    cp_callback = save_checkpoint(configs.save_path)

    if configs.save_model is not None:
        model.fit(dataset, epochs=configs.epoch, callbacks=[cp_callback])
    else:
        model.fit(dataset, epochs=configs.epoch)


if __name__ == '__main__':
    main()