import datetime
import re
import tensorflow as tf
from keras.layers import Bidirectional
from keras.models import Sequential
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from keras import layers
from config.definitions import ROOT_DIR
import pandas as pd
import os
import seaborn as sns
from matplotlib import pyplot as plt
import string
import json

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


def load_dataset(*src_filenames, labels=None):
    data = []
    for filename in src_filenames:
        with open(os.path.join(ROOT_DIR, 'data', 'dynasent-v1.1', 'dynasent'
                                                                  '-v1.1-round01-yelp-train.jsonl')) as f:
            for line in f:
                d = json.loads(line)
                if labels is None or d['gold_label'] in labels:
                    data.append(d)
    return data


def keepNeutrals(df):
    return df[df['score'] == 2]


def scoreConfig(score, negatives=False):
    if type(score) is str:
        if score == 'positive':
            return 3
        elif score == 'neutral':
            return 2
        elif score == 'negative':
            return 1
    elif type(score) is int:
        if score == 4:
            return 3
        elif score == 2:
            return
        elif score == 0:
            return 1


def clean(text):
    txt = str(text)
    pun = set(string.punctuation)
    lower = txt.lower()
    nopun = ''.join(letter for letter in lower if letter not in pun)
    rere = re.sub('@[^\s]+', '', nopun)
    rere = re.sub('$[^\s]+', '', rere)
    noascii = rere.encode("ascii", "ignore").decode()
    result = ' '.join([word for word in noascii.split(' ') if word not in stop])
    return result


stanfordcsv = pd.read_csv(
    os.path.join(ROOT_DIR, 'data', 'stanford.csv'), encoding='latin-1').dropna()
stanfordcsv.drop(columns=stanfordcsv.columns[1:5], inplace=True)
stanfordcsv.columns = ['score', 'text']
stanfordcsv = stanfordcsv.sample(frac=1)


airlinecsv = pd.read_csv(
    os.path.join(ROOT_DIR, 'data', 'Tweets.csv'))[
    ['airline_sentiment', 'text']].dropna()
airlinecsv.columns = ['score', 'text']

financecsv = pd.read_csv(
    os.path.join(ROOT_DIR, 'data', 'data.csv')).dropna()
financecsv.columns = ['text', 'score']

dyna = load_dataset('dynasent-v1.1-round01-yelp-train.jsonl', labels=('positive', 'negative', 'neutral'))
dyna = pd.DataFrame(dyna)
dyna = pd.concat((dyna['sentence'], dyna['gold_label']), axis=1)
dyna.columns = ['text', 'score']

headlines = pd.read_csv(
    os.path.join(ROOT_DIR, 'data', 'all-data.csv'), encoding='latin-1').dropna()
headlines.columns = ['score', 'text']

data = financecsv.append(headlines, ignore_index=True)
data = data.append(dyna[:30000], ignore_index=True)
data = data.append(stanfordcsv[:50000], ignore_index=True)

data['text'] = data['text'].apply(lambda r: clean(r))
data['score'] = data['score'].apply(scoreConfig)
data.dropna(inplace=True)
data = data.sample(frac=1)

x = data['text']
y = data['score']
sns.set()
sns.displot(data=y)
plt.show()
y = pd.get_dummies(y)

maxLen = 200
embSize = 64
vocabSize = 80000

tk = Tokenizer(num_words=vocabSize, oov_token='oov_tok')
tk.fit_on_texts(x)
xSeq = tk.texts_to_sequences(x)
xPad = pad_sequences(xSeq, maxlen=maxLen)
xTrain, xTest, yTrain, yTest = train_test_split(xPad, y, test_size=0.2)

model = Sequential()
model.add(layers.Embedding(vocabSize, embSize, input_length=maxLen))
model.add(layers.SpatialDropout1D(0.3))
model.add(layers.Conv1D(16, 32, padding='same', input_shape=(maxLen, embSize)))
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling1D())
model.add(layers.SpatialDropout1D(0.4))
model.add(Bidirectional(layers.LSTM(16, return_sequences=True)))
model.add(Bidirectional(layers.LSTM(16)))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(3, activation='softmax'))

model.summary()

adam = tf.keras.optimizers.Adam(learning_rate=0.01, decay=0.85)
rms = tf.keras.optimizers.RMSprop(learning_rate=0.001)

model.compile(
    optimizer=adam,
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)

checkpointDir = os.path.join(ROOT_DIR, 'checkpoints', 'training.ckpt')
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpointDir,
                                                 save_weights_only=True,
                                                 monitor='val_accuracy',
                                                 verbose=1)

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                                      histogram_freq=1)

modelDir = os.path.join(ROOT_DIR, 'models', 'sentiment')

history = model.fit(
    xTrain,
    yTrain,
    epochs=30,
    batch_size=512,
    validation_split=0.2,
    callbacks=[cp_callback, tensorboard_callback],
)

loss, acc = model.evaluate(xTest, yTest, verbose=2)
print(acc)
