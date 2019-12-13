#!/usr/bin/env python
#-*- coding:utf-8 -*-

import os
import sys
import time
import keras
sys.path.append("../")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from keras.datasets import mnist, fashion_mnist, cifar10



if __name__ == "__main__":

    # ========================0.准备参数===================================== #
    # 准备模型参数
    data_type = sys.argv[1]                         # "mnist"
    model_type = sys.argv[2]                        # "MLP"
    train_epoch = int(sys.argv[3])                  # 20

    tra_begin_main = time.time()

    if os.path.exists("./model"):
        os.system("rm -r model")

    tra_begin_main = time.time()
    if data_type == "mnist" and model_type == "MLP":
        from Train_mnist_MLP import train
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train.reshape(60000, 784)
        x_test = x_test.reshape(10000, 784)
    elif data_type == "mnist" and model_type == "CNN":
        from training.Train_mnist_CNN import train
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train.reshape(60000, 784)
        x_test = x_test.reshape(10000, 784)
    elif data_type == "fmnist" and model_type == "MLP":
        from training.Train_fmnist_MLP import train
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        x_train = x_train.reshape(60000, 784)
        x_test = x_test.reshape(10000, 784)
    elif data_type == "fmnist" and model_type == "CNN":
        from training.Train_fmnist_CNN import train
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        x_train = x_train.reshape(60000, 784)
        x_test = x_test.reshape(10000, 784)
    elif data_type == "cifar10" and model_type == "CNN":
        from traditional_training.Train_cifar10_CNN import train
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    train(x_train, y_train, x_test, y_test, train_epoch)

    tra_end_main = time.time()
    print("traditional_training completed!")
    # print("tra_model generation's timecost: " + str(tra_end_main - tra_begin_main))
    os.system("echo " + data_type + "_" + model_type + ": " + str(tra_end_main - tra_begin_main) + " > ../evaluation/tra_time.txt")