import pandas as pd
import tensorflow as tf

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.layers import Input, LSTM, Embedding, Dense
from tensorflow.keras.models import Model


df = pd.read_csv(f'../data/sample/korean_correct_train_data_100000.csv')

df = df.iloc[:20000]
df['tgt'] = df.tgt.apply(lambda x: '\t ' + x + ' \n')
print(df['tgt'])

src_vocab = set()

for line in df['src'].tolist():
    for c in line:
        src_vocab.add(c)

tgt_vocab = set()

for line in df['tgt'].tolist():
    for c in line:
        tgt_vocab.add(c)


src_vocab_size = len(src_vocab) + 1
tgt_vocab_size = len(tgt_vocab) + 1

src_to_index = dict([(word, i+1) for i , word in enumerate(src_vocab)])
tgt_to_index = dict([(word, i+1) for i , word in enumerate(tgt_vocab)])

encoder_input = []

for line in df['src'].tolist():
    encoded_line = []
    for c in line:
        encoded_line.append(src_to_index[c])
    encoder_input.append(encoded_line)

decoder_input = []

for line in df['tgt'].tolist():
    encoded_line = []
    for c in line:
        encoded_line.append(tgt_to_index[c])
    decoder_input.append(encoded_line)

decoder_tgt = []

for line in df['tgt'].tolist():
    ts = 0
    encoded_line = []
    for c in line:
        if ts > 0:
            encoded_line.append(tgt_to_index[c])
        ts += 1
    decoder_tgt.append(encoded_line)

max_src_len = max([len(line) for line in df['src'].tolist()])
max_tgt_len = max([len(line) for line in df['tgt'].tolist()])

encoder_input = pad_sequences(encoder_input, maxlen=max_src_len, padding='post')
decoder_input = pad_sequences(decoder_input, maxlen=max_tgt_len, padding='post')
decoder_tgt = pad_sequences(decoder_tgt, maxlen=max_tgt_len, padding='post')

encoder_input = to_categorical(encoder_input)
print(1)
decoder_input = to_categorical(decoder_input)
print(2)
decoder_tgt = to_categorical(decoder_tgt)
print(3)
print(decoder_tgt)