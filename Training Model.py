import numpy as np
import pandas as pds
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, BatchNormalization
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from keras.callbacks import EarlyStopping
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPooling1D
from keras.optimizers import Adam
import keras
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.utils import plot_model
from keras.callbacks import TensorBoard

from keras.datasets import cifar10
from keras.utils import np_utils
import sys
import threading

import os
import pandas as pd
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
import itertools
from sklearn import model_selection
from keras.optimizers import Adam
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from sklearn.model_selection import StratifiedKFold
import seaborn as sns
from keras import backend as K
import tensorflow as tf

sns.set(style="whitegrid", color_codes=True)

#######################################
#
# please insert code here to prepare input data(x) and supervised data(y)
# or please run the program "3 Dimensional Data.py" first to fill in x and y

# NEED TO BE CHANGE
subject = ["Edo1", "Edo2", "Hyo", "Kei", "Michael", "Ren", "Seiji", "Takuya"]
num_of_subject = 8
num_of_data = 150  # number of data for each cut data excel file
movement = ["gentlestroke", "notouch", "poke", "press", "scratch"]
num_of_movement = 5 # number of movement types
total_file = [30, 5, 30, 30, 30]  # number of excel files for each corresponding movement

# STATIC VAR
main_directory = "D:/TUAT/GV Lab Research Internship/Machine Learning/Data/Normalized_Cut_Data/Sensor1/" #dont forget to change the sensor's name!!!

# DYNAMIC VAR
coordinate = 0  # 0 for X, 1 for Y, 2 for Z
i = 0  # counting number of data in each file
counter = 0  # counting total file for EACH movement types
move = 0  # counting number of movement types
file = 0  # counting number of total file

# calculating total number of file
N = 0
for j in range(0, num_of_movement):
    N = N + total_file[j]*num_of_subject

#3 dimensional matrix to be filled up
x = np.zeros((N, num_of_data, 3)) #(z,y,x)
#matrix to be filled up
y = np.zeros((N, 1))

# Iteration to fill up the 3 dimensional matrix input & the supervised data
for people in range (0, num_of_subject): #subject change
    directory = main_directory + subject[people] + "/" + str(num_of_data)
    while move < num_of_movement:  # movement change
        while counter < total_file[move]:  # file change
            y[file, 0] = move
            file_read = pd.read_excel(
                directory + "/" + movement[move] + "n_" + str(num_of_data) + "_" + str(counter) + ".xlsx", "Sheet")
            while coordinate < 3:  # coordinate change
                for i in range(0, (num_of_data)):  # data change
                    x[file, i, coordinate] = file_read[coordinate][i]
                coordinate += 1
                i = 0
            counter += 1
            file += 1
            coordinate = 0
        move += 1
        counter = 0
    move = 0

print (x)
print  (y)
print ("3 Dimensional Data-------end")

#######################################

#YOU CAN CHANGE THIS PART
#test_size = percentage of the data that will be used for testing the model
(X_train, X_test, y_train, y_test) = train_test_split(x, y, test_size=0.3)
print('split out')
nb_classes = 5 #CHANGE NUMBER OF CLASSES HERE
fold_num = 5
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)
# Y_train = np.reshape(np.array(y_train),(420))
# Y_test = np.reshape(np.array(y_test),(1,180))

train_X = np.reshape(np.array(X_train), (X_train.shape[0], 450))
train_X = np.reshape(np.array(train_X), (X_train.shape[0], 450, 1))

test_X = np.reshape(np.array(X_test), (X_test.shape[0], 450))
test_X = np.reshape(np.array(test_X), (X_test.shape[0], 450, 1))

# input_x = K.placeholder(shape=(None, train_X.shape[1], train_X.shape[2]), name='X')
# input_y = K.placeholder(shape=(None, nb_classes), name='Y')
print('ready')

def plot_confusion_matrix(cm, classes,
                         normalize=False,
                         title='Confusion matrix',
                         cmap=plt.cm.Blues):
   """
   This function prints and plots the confusion matrix.
   Normalization can be applied by setting `normalize=True`.
   """
   if normalize:
       cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
       print("Normalized confusion matrix")
   else:
       print('Confusion matrix, without normalization')

   print(cm)

   plt.imshow(cm, interpolation='nearest', cmap=cmap)
   plt.title(title)
   plt.colorbar()
   tick_marks = np.arange(len(classes))
   plt.xticks(tick_marks, classes, rotation=45)
   plt.yticks(tick_marks, classes)

   fmt = '.2f' if normalize else 'd'
   thresh = cm.max() / 2.
   for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
       plt.text(j, i, format(cm[i, j], fmt),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")

   plt.tight_layout()
   plt.ylabel('True label')
   plt.xlabel('Predicted label')
# → UPPER (edited)
# モデルの定義
model = Sequential()

model.add(Conv1D(64, 3, padding='same', input_shape=(450, 1)))
model.add(Activation('relu'))
model.add(Conv1D(64, 3, padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling1D(2, padding='same'))

model.add(Conv1D(128, 3, padding='same'))
model.add(Activation('relu'))
model.add(Conv1D(128, 3, padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling1D(2, padding='same'))

model.add(Conv1D(256, 3, padding='same'))
model.add(Activation('relu'))
model.add(Conv1D(256, 3, padding='same'))
model.add(Activation('relu'))
model.add(Conv1D(256, 3, padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling1D(2, padding='same'))

model.add(Conv1D(512, 3, padding='same'))
model.add(Activation('relu'))
model.add(Conv1D(512, 3, padding='same'))
model.add(Activation('relu'))
model.add(Conv1D(512, 3, padding='same'))
model.add(Activation('relu'))

model.add(Flatten())
model.add(Dense(4096))
model.add(Dense(4096))
model.add(Dropout(0.5))
model.add(Dense(5)) #CHANGE HERE TO CHANGE THE NUMBER OF CLASSES
model.add(Activation('softmax'))

adam = Adam(lr=1e-4)

model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=["accuracy"])

model.summary()

epochs = 100

#YOU CAN CHANGE THIS PART
#this is for the early stops when the loss start to increase instead of decreasing
#early_stopping = EarlyStopping(patience=2,atience=5, verbose=1, mode verbose=1)
# es_cb = EarlyStopping(monitor='val_loss', p='auto')

#bacth_size = changing the number of file will be used before updating the weights and bias in 1 epoch
#validation_split = the percentage of the data that will be used for the validation during the training
history = model.fit(x=train_X, y=Y_train, batch_size=10, validation_split=0.3, epochs=epochs)

plt.plot(range(len(history.history['loss'])), history.history['loss'], label='loss')
plt.plot(range(len(history.history['val_loss'])), history.history['val_loss'], label='val_loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.show()

#YOU CAN CHANGE THIS PART
#this is the file name to save the CNN Model (cnn_model.json) and the weights (cnn_model_weights.hdf5)
json_string = model.to_json()
open(os.path.join('./', 'cnn_model.json'), 'w').write(json_string)

model.save_weights(os.path.join('./', 'cnn_model_weight.hdf5'))

score = model.evaluate(test_X, Y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

pred_y = model.predict(test_X)
pred_Y = np.argmax(pred_y, axis=1)
pred_Y = pred_Y[:, np.newaxis]  # 縦ベクトル

confusion_matrix(y_test, pred_Y)
print(f1_score(y_test, pred_Y, average='macro'))

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, pred_Y)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
class_names = ["", "", "", ""]
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                     title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                     title='Normalized confusion matrix')

plt.show()

# →LOWER (edited)