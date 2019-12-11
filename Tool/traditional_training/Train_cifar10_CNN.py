#!/usr/bin/env python
#-*- coding:utf-8 -*-

'''
#Train a simple deep CNN on the CIFAR10 small images dataset.
It gets to 75.62% validation accuracy in 25 epochs, and 79% after 50 epochs.
(it's still underfitting at that point, though).
'''

from __future__ import print_function
import os
import time
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
os.environ["TF_CPP_MIN_LOG_LEVEL"]='2'


###====================================模型的训练====================================###
def train(x_train, y_train, x_test, y_test, nb_epochs_num):

    begin_time = time.time()

    batch_size = 32
    num_classes = 10
    epochs = nb_epochs_num


    # x_train = x_train.astype('float32')
    # x_test = x_test.astype('float32')
    # x_train /= 255
    # x_test /= 255
    # y_train = keras.utils.to_categorical(y_train, num_classes)
    # y_test = keras.utils.to_categorical(y_test, num_classes)


    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=x_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    # model.summary()


    # initiate RMSprop optimizer
    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

    # Let's train the model using RMSprop
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True, verbose=0)

    # Save model and weights
    if not os.path.exists("./model"):
        os.system("mkdir ./model")
    model.save("./model/cifar10_CNN")
    os.system("cp ./model/cifar10_CNN ./model/origin_model")



    end_time = time.time()
    if nb_epochs_num != 0:
        scores = model.evaluate(x_test, y_test, verbose=0)
        # print('Test loss:', scores[0])
        # print('Test accuracy:', scores[1])
        # print("model's training time: " + str(end_time-begin_time))

    return 0



###=================================================================================###

if __name__ == "__main__":
    train(25)