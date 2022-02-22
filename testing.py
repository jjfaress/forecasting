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
from sklearn.utils import resample

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
        if negatives:
            if score == 1:
                return 3
            if score == 0:
                return 2
            if score == -1:
                return 1
        else:
            if score == 4:
                return 3
            elif score == 2:
                return 2
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

airlinecsv = pd.read_csv(
    os.path.join(ROOT_DIR, 'data', 'Tweets.csv'))[
    ['airline_sentiment', 'text']].dropna()
airlinecsv.columns = ['score', 'text']

financecsv = pd.read_csv(
    os.path.join(ROOT_DIR, 'data', 'data.csv')).dropna()
financecsv.columns = ['text', 'score']

reddit = pd.read_csv(
    os.path.join(ROOT_DIR, 'data', 'Reddit_Data.csv')).dropna()
reddit.columns = ['text', 'score']

twitter = pd.read_csv(
    os.path.join(ROOT_DIR, 'data', 'Twitter_Data.csv')).dropna()
twitter.columns = ['text', 'score']

headlines = pd.read_csv(
    os.path.join(ROOT_DIR, 'data', 'all-data.csv'), encoding='latin-1').dropna()
headlines.columns = ['score', 'text']

neutrals = reddit.append(twitter, ignore_index=True)

neutrals = neutrals[neutrals['score'] == 0]
neutrals['score'].replace({0: 2}, inplace=True)

stanfordcsv['text'] = stanfordcsv['text'].apply(lambda r: clean(r))
stanfordcsv['score'] = stanfordcsv['score'].apply(scoreConfig)
airlinecsv['text'] = airlinecsv['text'].apply(lambda r: clean(r))
airlinecsv['score'] = airlinecsv['score'].apply(scoreConfig)
financecsv['text'] = financecsv['text'].apply(lambda r: clean(r))
financecsv['score'] = financecsv['score'].apply(scoreConfig)
headlines['text'] = headlines['text'].apply(lambda r: clean(r))
headlines['score'] = headlines['score'].apply(scoreConfig)

headlines = headlines[headlines['score'] == 2]

data = stanfordcsv.append(airlinecsv, ignore_index=True)
data = data.append(financecsv, ignore_index=True)
data = data.append(neutrals, ignore_index=True)
data = data.append(headlines, ignore_index=True)
data.dropna(inplace=True)
nSample = data[data['score'] == 1]
neutralSample = data[data['score'] == 2]
pSample = data[data['score'] == 3]
balanced = pd.concat(
    [neutralSample, nSample[:len(neutralSample)], pSample[:len(neutralSample)]],
    axis=0)

x = balanced['text']
y = balanced['score']

sns.set()
sns.displot(data=y)
plt.show()
y = pd.get_dummies(y)

maxLen = 200
embSize = 42
vocabSize = 100000
filters = 28

tk = Tokenizer(num_words=vocabSize, oov_token='oov_tok')
tk.fit_on_texts(x)
xSeq = tk.texts_to_sequences(x)
xPad = pad_sequences(xSeq, maxlen=maxLen)
xTrain, xTest, yTrain, yTest = train_test_split(xPad, y, test_size=0.2)
model = Sequential()
model.add(layers.Embedding(vocabSize, embSize, input_length=maxLen))
model.add(layers.SpatialDropout1D(0.3))
model.add(layers.MaxPooling1D())
model.add(Bidirectional(layers.LSTM(16, dropout=0.3)))
model.add(layers.Dropout(0.1))
model.add(layers.Dense(3))
model.add(layers.Activation('softmax'))

model.summary()

rms = tf.keras.optimizers.RMSprop(learning_rate=0.005)

model.compile(
    optimizer=rms,
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
    epochs=5,
    batch_size=256,
    validation_split=0.2,
    callbacks=[cp_callback, tensorboard_callback],
)

loss, acc = model.evaluate(xTest, yTest, verbose=2)
print(acc)
