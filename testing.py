import re
import tensorflow as tf
from keras.layers import Bidirectional
from keras.models import Sequential
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from keras import layers
from config.definitions import ROOT_DIR
from matplotlib import pyplot as plt
import pandas as pd
import os
import seaborn as sns
import string

stop = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you',
        "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself',
        'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her',
        'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them',
        'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this',
        'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were',
        'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
        'did', 'doing', 'a', 'an', 'the', 'and', 'if', 'or', 'because',
        'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about',
        'between', 'into', 'through', 'to', 'from', 'up', 'down',
        'in', 'out', 'on', 'off', 'again', 'further', 'then', 'once', 'here',
        'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each',
        'most', 'other', 'some', 'such', 'nor' 'own', 'same', 'so',
        'than', 'too', 'very', 'will', 'just', 'would',
        'ain', 'aren', 'doesn', 'hadn', "hadn't", 'hasn', "hasn't", 'haven',
        "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',
        "mustn't", 'needn', "needn't", 'shan', "shan't",
        'wasn', "wasn't", 'weren', "weren't", "won't", 'wouldn']


def dropZeros(df):
    return df[df['score'] != 0]


def scoreConfig(score):
    if type(score) is str:
        if score == 'positive':
            return 2
        elif score == 'neutral':
            return 1
        elif score == 'negative':
            return 0
    elif type(score) is int:
        if score == 4:
            return 2
        elif score == 2:
            return 1
        elif score == 0:
            return 0


def clean(text):
    txt = str(text)
    pun = set(string.punctuation)
    lower = txt.lower()
    nopun = ''.join(letter for letter in lower if letter not in pun)
    noascii = nopun.encode("ascii", "ignore").decode()
    nostop = ' '.join([word for word in noascii.split(' ') if word not in stop])
    result = re.sub('@[^\s]+', '', nostop)
    min = 10
    max = 300
    # if ((len(result.split(' ')) < min) or (len(result.split(' ')) > max)):
    #     result = None

    return nostop


stanfordcsv = pd.read_csv(
    os.path.join(ROOT_DIR, 'data', 'stanford.csv'), encoding='latin-1').dropna()
stanfordcsv.drop(columns=stanfordcsv.columns[1:5], inplace=True)
stanfordcsv.columns = ['score', 'text']

airlinecsv = pd.read_csv(
    os.path.join(ROOT_DIR, 'data', 'Tweets.csv'))[
    ['airline_sentiment', 'text']].dropna()
airlinecsv.columns = ['score', 'text']

stanfordcsv['text'] = stanfordcsv['text'].apply(lambda r: clean(r))
stanfordcsv['score'] = stanfordcsv['score'].apply(scoreConfig)
airlinecsv['text'] = airlinecsv['text'].apply(lambda r: clean(r))
airlinecsv['score'] = airlinecsv['score'].apply(scoreConfig)

data = stanfordcsv.append(airlinecsv, ignore_index=True)
print(data)
data.dropna(inplace=True)
data = data.sample(frac=1)

x = data['text']
y = data['score']
y = pd.get_dummies(y)

maxLen = 200
embSize = 64
tk = Tokenizer()
tk.fit_on_texts(x)
vocabSize = len(tk.word_index) + 1
xSeq = tk.texts_to_sequences(x)
xPad = pad_sequences(xSeq, maxlen=maxLen)
xTrain, xTest, yTrain, yTest = train_test_split(xPad, y, test_size=0.2)

model = Sequential()
model.add(layers.Embedding(vocabSize, embSize, input_length=maxLen))
model.add(layers.SpatialDropout1D(0.2))
model.add(Bidirectional(layers.LSTM(embSize, return_sequences=True)))
model.add(Bidirectional(layers.LSTM(embSize, return_sequences=True)))
model.add(layers.Conv1D(32, embSize, activation='relu', padding='same',
                        input_shape=(maxLen, embSize)))

model.add(layers.Dense(3, activation='softmax'))

model.summary()

opt = tf.keras.optimizers.Adam(learning_rate=0.01, decay=0.9)

model.compile(
    optimizer=opt,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    xTrain,
    yTrain,
    epochs=10,
    batch_size=256,
    validation_split=0.3,
)
