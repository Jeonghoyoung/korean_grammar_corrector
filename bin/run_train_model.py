import tensorflow as tf
from model.transformer_model import *
import argparse


MAX_LENGTH = 24


def args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab_size', required=False, default=9000, help='Vocab Size')
    parser.add_argument('--num_layers', required=False, default=5, help='Number of layers ')
    parser.add_argument('--dff', required=False, default=512, help='Hidden layer size')
    parser.add_argument('--d_model', required=False, default=256, help='Embedding vector dimension size')
    parser.add_argument('--num_heads', required=False, default=8, help= 'Number of Multi-head attention heads')
    parser.add_argument('--dropout', required=False, default=0.1, help='Drop out prob')
    parser.add_argument('--model_name', required=False, default='transformer', help= 'Model Name')

    parser.add_argument('--epoch', required=False, default=30, help='Epochs')

    parser.add_argument('--input', required=False, default='../data/raw')
    parser.add_argument('--save_model', required=False, default=None)
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


def main():
    config = args()

    # Load Dataset
    df = pd.read_csv(config.input)

    src = df['src'].apply(lambda x: tp.full_stop_filter(x))
    tgt = df['tgt'].apply(lambda x: tp.full_stop_filter(x))

    inputs, outputs, tokenizer = tp.tokenize_and_filter(src, tgt, max_length=24)
    dataset = tp.create_train_dataset(inputs, outputs, batch_size=64, buffer_size=60000)

    # 모델 초기화
    tf.keras.backend.clear_session()

    model = transformer(
        vocab_size=config.vocab_size,
        num_layers=config.num_layers,
        dff=config.dff,
        d_model=config.d_model,
        num_heads=config.num_heads,
        dropout=config.dropout,
        name=config.model_name)

    learning_rate = CustomSchedule(d_model)

    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    model.compile(optimizer=optimizer, loss=loss_function, metrics=[accuracy])
    model.fit(dataset, epochs=config.epoch)

    if config.output is not None:
        model.save(f'{config.model_name}_model_{config.output}')
        model.save_weights(f'{config.model_name}_model_{config.output}_{config.epoch}_weights')


if __name__ == '__main__':
    main()