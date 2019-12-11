#!/usr/bin/env python
#-*- coding:utf-8 -*-

'''Trains a simple convnet on the MNIST dataset.
Gets to 99.03% test accuracy after 20 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import os
import time
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, Reshape
os.environ["TF_CPP_MIN_LOG_LEVEL"]='2'



###====================================模型的训练====================================###
def train(x_train, y_train, x_test, y_test, nb_epochs_num):
    begin_time = time.time()

    batch_size = 128
    num_classes = 10
    epochs = nb_epochs_num


    # x_train = x_train.reshape(60000, 784)
    # x_test = x_test.reshape(10000, 784)
    # x_train = x_train.astype('float32')
    # x_test = x_test.astype('float32')
    # x_train /= 255
    # x_test /= 255
    # y_train = keras.utils.to_categorical(y_train, num_classes)
    # y_test = keras.utils.to_categorical(y_test, num_classes)


    model = Sequential()
    model.add(Reshape((28, 28, 1), input_shape=(784,)))
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=(28, 28, 1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    # model.summary()



    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=0,
              validation_data=(x_test, y_test))



    # Save model and weights
    if not os.path.exists("./model"):
        os.system("mkdir ./model")
    model.save("./model/mnist_CNN")
    os.system("cp ./model/mnist_CNN ./model/origin_model")



    end_time = time.time()
    if nb_epochs_num != 0:
        score = model.evaluate(x_test, y_test, verbose=0)
        # print('Test loss:', score[0])
        # print('Test accuracy:', score[1])
        # print("model's training time: " + str(end_time - begin_time))

    return 0



###=================================================================================###

if __name__ == "__main__":
    train(20)
