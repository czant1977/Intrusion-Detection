
from __future__ import print_function

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from keras.callbacks import Callback
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, LSTM
from collections import Counter
from imblearn.over_sampling import SMOTE
from keras import callbacks
from keras.callbacks import EarlyStopping, CSVLogger
import warnings
warnings.filterwarnings("ignore")

np.random.seed(1337)  # for reproducibility
df = pd.read_csv('IOT.csv')

print(df.type.value_counts())

labelencoder = LabelEncoder()
df.iloc[:, -1] = labelencoder.fit_transform(df.iloc[:, -1])
print( df.type.value_counts())


X = df.drop(['ts', 'label', 'type'], axis=1).values
y = df.iloc[:, -1].values.reshape(-1, 1)
y = np.ravel(y)
X_train, X_test, Y_train, Y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=1337, stratify=y)

# # SMOTE
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

X_train = np.array(X_train)
X_test = np.array(X_test)

Y_train = np.array(Y_train)
Y_test = np.array(Y_test)

X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test .shape[0], 1, X_test .shape[1]))
print('X_train', X_train.shape)
print('Y_train', Y_train.shape)
print('X_test', X_test.shape)
print('Y_test', Y_test.shape)

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



class LearningRateScheduler(Callback):
    def __init__(self, n_epochs, verbose=0):
        self.epochs = n_epochs
        self.lrates = list()

    def lr_scheduler(self, epoch, n_epochs):
        initial_lrate = 0.1
        lrate = initial_lrate * np.exp(-0.1 * epoch)
        return lrate

    def on_epoch_begin(self, epoch, logs={}):
        lr = self.lr_scheduler(epoch, self.epochs)
        print(f'epoch {epoch + 1}, lr {lr}')
        K.set_value(self.model.optimizer.lr, lr)
        self.lrates.append(lr)


time_callback = TimeHistory()


accuracy = []
precision = []
recall = []
f1score = []
# learning_rate = 0.0001
Time = []


# 1. define the network
def lstm(X_train, Y_train, X_test, Y_test, num_class, batch_size, epochs):
    lstm = Sequential()  # initializing model

    lstm.add(LSTM(units=128, input_dim=18, activation='tanh', return_sequences=True))
    lstm.add(LSTM(units=100, activation='tanh', return_sequences=True))
    lstm.add(LSTM(units=64,  activation='tanh', return_sequences=True))
    lstm.add(Dropout(0.2))
    lstm.add(Dense(units=num_class, activation='softmax'))

    # defining loss function, optimizer, metrics and then compiling model

    lstm.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # summary of model layers
    lstm.summary()

    history = lstm.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2,
                        callbacks=[time_callback])
    # lstm.save("results/lstm/lstm_multipy.hdf5")

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
    #plt.savefig(f'results/lstm/lstm_multipy_loss.png')
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
    #plt.savefig(f'results/lstm/lstm_multipy_acc.png')
    plt.show()


    Y_train = np_utils.to_categorical(Y_train, num_class)
    Y_test = np_utils.to_categorical(Y_test, num_class)

    y_predict = lstm.predict(X_test, batch_size=512, verbose=2)
    # y_predict = (y_predict > 0.007).astype(int)
    y_predict = (y_predict > 0.01).astype(int)
    y_true = np.reshape(Y_test, [-1])
    print('y_true:', y_true.shape)
    y_pred = np.reshape(y_predict, [-1])
    print('y_pred:', y_pred.shape)

    accuracy.append(accuracy_score(y_true, y_pred))
    precision.append(precision_score(y_true, y_pred, average='weighted'))
    recall.append(recall_score(y_true, y_pred, average='weighted'))
    f1score.append(f1_score(y_true, y_pred, average='weighted'))

    print('accuracy:', accuracy)
    print('precision:', precision)
    print('recall:', recall)
    print('f1score:', f1score)



if __name__ == '__main__':
    lstm(X_train, Y_train, X_test, Y_test, num_class=8, batch_size=512, epochs=80)
    print('times:', time_callback.times)
    print('totaltime:', time_callback.totaltime)













