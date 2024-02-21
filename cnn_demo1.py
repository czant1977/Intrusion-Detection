

import time
import warnings

import keras.callbacks as callbacks
import numpy as np
import pandas as pd
from keras import backend as K
from keras.callbacks import EarlyStopping, CSVLogger
from keras.layers import Dropout, Dense, Activation, Dropout, BatchNormalization, Flatten, Conv1D
from keras.models import Sequential
from keras.utils import np_utils
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer, LabelEncoder
from collections import Counter
from imblearn.over_sampling import SMOTE
from keras.models import load_model


warnings.filterwarnings("ignore")

np.random.seed(1337)  # for reproducibility

df = pd.read_csv('IOT.csv')

print(df.type.value_counts())
labelencoder = LabelEncoder()
df.iloc[:, -1] = labelencoder.fit_transform(df.iloc[:, -1])
print(df.type.value_counts())


X = df.drop(['ts', 'label', 'type'], axis=1).values
y = df.iloc[:, -2].values.reshape(-1, 1)
y = np.ravel(y)
X_train, X_test, Y_train, Y_test = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                    random_state=1337, stratify=y)
# SMOTE
# smote = SMOTE(sampling_strategy='auto', random_state=1337)
# X_smotesampled, y_smotesampled = smote.fit_resample(X_train, Y_train)
# print(Counter(y_smotesampled))
#
# X_train = X_smotesampled
# Y_train = y_smotesampled

scaler = Normalizer().fit(X_train)
X_train = scaler.transform(X_train)

scaler = Normalizer().fit(X_test)
X_test = scaler.transform(X_test)

X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
Y_train = Y_train.reshape((Y_train.shape[0], 1))
Y_test = Y_test.reshape((Y_test.shape[0], 1))


class TimeHistory(callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []
        self.totaltime = time.time()

    def on_train_end(self, logs={}):
        self.totaltime = time.time() - self.totaltime

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)


time_callback = TimeHistory()
accuracy = []
precision = []
recall = []
f1score = []
# learning_rate = 0.0001
Time = []

# 1. define the network
def cnn(X_train, Y_train, X_test, Y_test, num_class, batch_size, epochs):
    model = Sequential()
    model.add(Conv1D(16, kernel_size=3, strides=1, padding='same', input_shape=(X_train.shape[1], 1)))
    model.add(BatchNormalization())
    model.add(Conv1D(16, kernel_size=3, strides=1, padding='same'))
    model.add(Conv1D(64, kernel_size=3, strides=1, padding='valid'))
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(32, kernel_initializer='glorot_uniform', activation='Relu'))
    model.add(Dense(16, kernel_initializer='glorot_uniform', activation='Relu'))
    model.add(Dense(num_class, kernel_initializer='glorot_uniform', activation='Sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    # early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, verbose=1, mode='auto')
    history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2,
                        callbacks=[time_callback], verbose=1)
    # model.save("results/cnn/cnn_multiply_model.hdf5")

    # plot the training and validation loss
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'g', label='Validating loss')
    plt.title('Training and Validating loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    # plt.savefig(f'results/cnn/cnn_multiply_loss.png')
    plt.show()

    # plot the training and validation accuracy
    plt.clf()
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    plt.plot(epochs, acc, 'g', label='Training acc')
    plt.plot(epochs, val_acc, 'r', label='Validating acc')
    plt.title('Training and Validating accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('accuracy')
    plt.legend()
    # plt.savefig(f'results/cnn/cnn_multiply_acc.png')
    plt.show()

    # test

    Y_train = np_utils.to_categorical(Y_train, num_class)
    Y_test = np_utils.to_categorical(Y_test, num_class)

    y_predict = model.predict(X_test, batch_size=512, verbose=2)

    # y_predict = (y_predict > 0.007).astype(int)
    y_predict = (y_predict > 0.01).astype(int)
    y_true = np.reshape(Y_test, [-1])
    y_pred = np.reshape(y_predict, [-1])

    accuracy.append(accuracy_score(y_true, y_pred))
    precision.append(precision_score(y_true, y_pred, average='weighted'))
    recall.append(recall_score(y_true, y_pred, average='weighted'))
    f1score.append(f1_score(y_true, y_pred, average='weighted'))

    print('accuracy:', accuracy)
    print('precision:', precision)
    print('recall:', recall)
    print('f1score:', f1score)
    # print('Macro-F1: {}'.format(macro_f1))
    # print('Micro-F1: {}'.format(micro_f1))


if __name__ == '__main__':
    cnn(X_train, Y_train, X_test, Y_test, num_class=2, batch_size=512, epochs=80)
    print('times:', time_callback.times)
    print('totaltime:', time_callback.totaltime)
