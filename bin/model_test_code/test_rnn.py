from collections import Counter

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential
from keras.layers import Dropout, GRU,Dense, SimpleRNN, TimeDistributed, Activation, RepeatVector,Bidirectional,\
    Embedding

from keras.callbacks import EarlyStopping, ModelCheckpoint

from keras.layers import Activation
from keras.optimizers import Adam
from keras.losses import sparse_categorical_crossentropy


def tokenize(x):
    """
    Tokenize x
    :param x: List of sentences/strings to be tokenized
    :return: Tuple of (tokenized x data, tokenizer used to tokenize x)
    """
    # TODO: Implement
    x_tk = Tokenizer()
    x_tk.fit_on_texts(x)

    return x_tk.texts_to_sequences(x), x_tk


def pad(x, length=None):
    """
    Pad x
    :param x: List of sequences.
    :param length: Length to pad the sequence to.  If None, use length of longest sequence in x.
    :return: Padded numpy array of sequences
    """
    # TODO: Implement
    if length is None:
        length = max([len(sentence) for sentence in x])
    return pad_sequences(x, maxlen=length, padding='post', truncating='post')


def preprocess(x, y):
    preprocess_x, x_tk = tokenize(x)
    preprocess_y, y_tk = tokenize(y)

    preprocess_x = pad(preprocess_x)
    preprocess_y = pad(preprocess_y)

    preprocess_y = preprocess_y.reshape(*preprocess_y.shape, 1)

    return preprocess_x, preprocess_y, x_tk, y_tk


def logits_to_text(logits, tokenizer):
    """
    Turn logits from a neural network into text using the tokenizer
    :param logits: Logits from a neural network
    :param tokenizer: Keras Tokenizer fit on the labels
    :return: String that represents the text of the logits
    """
    index_to_words = {idx: word for word, idx in tokenizer.word_index.items()}
    index_to_words[0] = '<PAD>'

    return ' '.join([index_to_words[prediction] for prediction in np.argmax(logits, 1)])


def simple_model(input_shape, output_sequence_length, english_vocab_size, tgt_vocab_size):
    """
    Build and train a basic RNN on x and y
    :param input_shape: Tuple of input shape
    :param output_sequence_length: Length of output sequence
    :param english_vocab_size: Number of unique English words in the dataset
    :param french_vocab_size: Number of unique French words in the dataset
    :return: Keras model built, but not trained
    """
    # TODO: Build the layers
    learning_rate = 1e-3
    model = Sequential()
    model.add(SimpleRNN(64, input_shape=input_shape[1:], return_sequences=True))
    model.add(Dropout(0.5))
    model.add(SimpleRNN(64, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(tgt_vocab_size, activation='softmax'))

    model.compile(loss=sparse_categorical_crossentropy,
                  optimizer=Adam(learning_rate),
                  metrics=['accuracy'])
    return model


if __name__ == '__main__':
    df = pd.read_csv('../data/sample/korean_correct_train_data_100000.csv')
    x_train, x_test, y_train, y_test = train_test_split(df['src'], df['tgt'], test_size=0.3, shuffle=True,
                                                        random_state=34)

    src_words_counter = Counter([word for s in x_train for word in s.split()])
    tgt_words_counter = Counter([word for s in y_train for word in s.split()])

    prep_x, prep_y, x_tok, y_tok = preprocess(x_train, y_train)

    max_src_sequence_length = prep_x.shape[1]
    max_tgt_sequence_length = prep_y.shape[1]
    src_vocab_size = len(x_tok.word_index)
    tgt_vocab_size = len(y_tok.word_index)

    tmpx = pad(prep_x, max_tgt_sequence_length)
    tmpx = tmpx.reshape((-1, prep_y.shape[-2], 1))

    simple_rnn_model = simple_model(
        tmpx.shape,
        max_tgt_sequence_length,
        src_vocab_size,
        tgt_vocab_size)

    simple_rnn_model.fit(tmpx, prep_y, batch_size=128, epochs=1, validation_split=0.2, verbose=1)

    simple_rnn_model.save('../rnn_model.h5')