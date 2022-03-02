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
from official.nlp import optimization
import seaborn as sns
from tensorflow import keras
from matplotlib import pyplot as plt
import tensorflow_text
import string
import json
import nltk
from nltk.stem import WordNetLemmatizer
import tensorflow_hub as hub

stop = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
        'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'i', 'me',
        'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you',
        "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself',
        'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her',
        'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them',
        'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this',
        'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were',
        'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
        'did', 'doing', 'a', 's', 'an', 'the', 'and', 'if', 'or', 'because',
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

wnl = WordNetLemmatizer()


def load_dataset(*src_filenames, labels=None):
    data = []
    for filename in src_filenames:
        with open(os.path.join(ROOT_DIR, 'data', 'dynasent-v1.1', 'dynasent'
                                                                  '-v1.1'
                                                                  '-round01-yelp-train.jsonl')) as f:
            for line in f:
                d = json.loads(line)
                if labels is None or d['gold_label'] in labels:
                    data.append(d)
    return data


def keepNeutrals(df):
    return df[df['score'] == 2]


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
    # pun = set(string.punctuation)
    # lower = txt.lower()
    # nopun = ''.join(letter for letter in lower if letter not in pun)
    rere = re.sub('@[^\s]+', '', txt)
    rere = re.sub('$[^\s]+', '', rere)
    # noascii = rere.encode("ascii", "ignore").decode()
    # nostop = ' '.join([word for word in noascii.split(' ') if word not in stop])
    # result = ' '.join(
    #     wnl.lemmatize(word) for word in nltk.word_tokenize(nostop))
    # if len(result.split(' ')) <= 3:
    #     return None
    return rere


def schedule(epoch, lr):
    if epoch >= 1:
        return lr * 0.1
    return lr


def build():
    text_input = layers.Input(shape=(), dtype=tf.string, name='text')
    preprocessor = hub.KerasLayer(
        "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
        name='preprocessing')
    encoder_inputs = preprocessor(text_input)
    encoder = hub.KerasLayer(
        "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-128_A-2/2",
        trainable=True,
        name='bert')
    outputs = encoder(encoder_inputs)
    net = outputs['pooled_output']
    net = layers.Dropout(0.1)(net)
    net = layers.Dense(3, activation='softmax', name='classifier')(net)
    return tf.keras.Model(text_input, net)


stanfordcsv = pd.read_csv(
    os.path.join(ROOT_DIR, 'data', 'stanford.csv'), encoding='latin-1').dropna()
stanfordcsv.drop(columns=stanfordcsv.columns[1:5], inplace=True)
stanfordcsv.columns = ['score', 'text']
stanfordcsv = stanfordcsv.sample(frac=1)


financecsv = pd.read_csv(
    os.path.join(ROOT_DIR, 'data', 'data.csv')).dropna()
financecsv.columns = ['text', 'score']

dyna = pd.DataFrame(load_dataset('dynasent-v1.1-round01-yelp-train.jsonl',
                                 labels=('positive', 'negative', 'neutral')))
dyna = pd.concat((dyna['sentence'], dyna['gold_label']), axis=1)
dyna.columns = ['text', 'score']

dyna2 = pd.DataFrame(load_dataset('dynasent-v1.1-round02-dynabench-train.jsonl',
                                  labels=('positive', 'negative', 'neutral')))
dyna2 = pd.concat((dyna2['sentence'], dyna2['gold_label']), axis=1)
dyna2.columns = ['text', 'score']

dyna = dyna.append(dyna2, ignore_index=True)

headlines = pd.read_csv(
    os.path.join(ROOT_DIR, 'data', 'all-data.csv'), encoding='latin-1').dropna()
headlines.columns = ['score', 'text']

data = financecsv.append(headlines, ignore_index=True)
data = data.append(dyna, ignore_index=True)
data = data.append(stanfordcsv[:120000], ignore_index=True)

data['text'] = data['text'].apply(lambda r: clean(r))
data['score'] = data['score'].apply(scoreConfig)

data.dropna(inplace=True)
data.drop_duplicates(inplace=True)
data = data.sample(frac=1)

sns.set()
sns.displot(data['score'])
plt.show()

train, test = train_test_split(data, test_size=0.2)
train, val = train_test_split(train, test_size=0.2)

BATCH_SIZE = 256

trainraw = (
    tf.data.Dataset.from_tensor_slices(
        (
            tf.cast(train['text'].values, tf.string, name='text'),
            tf.cast(train['score'].values, tf.int64, name='score')
        )
    ).batch(BATCH_SIZE)
)
train = trainraw.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

testraw = (
    tf.data.Dataset.from_tensor_slices(
        (
            tf.cast(test['text'].values, tf.string, name='text'),
            tf.cast(test['score'].values, tf.int64, name='score')
        )
    ).batch(BATCH_SIZE)
)
test = testraw.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

valraw = (
    tf.data.Dataset.from_tensor_slices(
        (
            tf.cast(val['text'].values, tf.string, name='text'),
            tf.cast(val['score'].values, tf.int64, name='score')
        )
    ).batch(BATCH_SIZE)
)
val = valraw.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

model = build()
model.summary()
stepsEpoch = tf.data.experimental.cardinality(train).numpy()
epochs = 5
trainSteps = stepsEpoch * epochs
warmup = int(0.1 * trainSteps)
init_lr = 3e-5
opt = optimization.create_optimizer(
    init_lr=init_lr,
    num_train_steps=trainSteps,
    num_warmup_steps=warmup,
    optimizer_type='adamw'
)

model.compile(
    optimizer=opt,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    x=train,
    validation_data=val,
    epochs=epochs
)

# preprocess = hub.load('https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3')
# encoder = hub.load('https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-128_A-2/2')
# input = preprocess('this is a test.')
# print(encoder(input))
#
# x = data['text']
# y = data['score']
#
# trainx, testx, trainy, testy = train_test_split(x, y, test_size=0.2)
# trainx, valx, trainy, valy = train_test_split(trainx, trainy, test_size=0.2)
#
# raw_trainx = tf.data.Dataset.from_tensor_slices(trainx)
# raw_trainx = raw_trainx.batch(32, drop_remainder=True)
#
# raw_testx = tf.data.Dataset.from_tensor_slices(testx)
# raw_testx = raw_testx.batch(32, drop_remainder=True)
#
# raw_valx = tf.data.Dataset.from_tensor_slices(valx)
# raw_valx = raw_valx.batch(32)
#
# for i in raw_trainx:
#     print(i)
#
# sns.set()
# sns.displot(data=y)
# plt.show()
#
#
#
# maxLen = 200
# embSize = 64
# vocabSize = 80000
#
# tk = Tokenizer(oov_token='oov_tok')
# tk.fit_on_texts(x)
# xSeq = tk.texts_to_sequences(x)
# xPad = pad_sequences(xSeq, maxlen=maxLen, padding='post')
# xTrain, xTest, yTrain, yTest = train_test_split(xPad, y, test_size=0.2)
#
#
# model = Sequential()
# model.add(layers.Embedding(200000, embSize, input_length=maxLen))
# model.add(layers.SpatialDropout1D(0.5))
# model.add(Bidirectional(layers.LSTM(embSize)))
# model.add(layers.Dense(embSize, activation='relu'))
# model.add(layers.Dense(3, activation='softmax'))
#
# model.summary()
#
# adam = tf.keras.optimizers.Adam(learning_rate=0.01, decay=0.1)
#
# model.compile(
#     optimizer=adam,
#     loss='sparse_categorical_crossentropy',
#     metrics=['accuracy'],
# )
#
# checkpointDir = os.path.join(ROOT_DIR, 'checkpoints', 'training.ckpt')
#
# cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpointDir,
#                                                  save_weights_only=True,
#                                                  monitor='val_accuracy',
#                                                  verbose=1)
#
# log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
#
# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
#                                                       histogram_freq=1)
#
# reducer = tf.keras.callbacks.ReduceLROnPlateau(
#     monitor='val_loss',
#     factor=0.7,
#     patience=2,
#     verbose=1
# )
#
# history = model.fit(
#     xTrain,
#     yTrain,
#     epochs=30,
#     batch_size=64,
#     validation_split=0.2,
#     callbacks=[cp_callback, tensorboard_callback, reducer],
# )
