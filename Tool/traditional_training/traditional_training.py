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
    model_type = sys.argv[1]                        # "mnist_MLP"
    train_epoch = int(sys.argv[2])                  # 20

    model_name = model_type[:model_type.find("_")]
    tra_begin_main = time.time()



    if os.path.exists("./model") and os.listdir("./model"):  #判断文件夹存在且不为空
        print("Please abandon the 'model' file and start training.")
        exit()
    else:
        tra_begin_main = time.time()
        if model_type == "mnist_MLP":
            from Train_mnist_MLP import train
            (x_train, y_train), (x_test, y_test) = mnist.load_data()
            x_train = x_train.reshape(60000, 784)
            x_test = x_test.reshape(10000, 784)
        elif model_type == "mnist_CNN":
            from Train_mnist_CNN import train
            (x_train, y_train), (x_test, y_test) = mnist.load_data()
            x_train = x_train.reshape(60000, 784)
            x_test = x_test.reshape(10000, 784)
        elif model_type == "fmnist_MLP":
            from Train_fmnist_MLP import train
            (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
            x_train = x_train.reshape(60000, 784)
            x_test = x_test.reshape(10000, 784)
        elif model_type == "fmnist_CNN":
            from Train_fmnist_CNN import train
            (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
            x_train = x_train.reshape(60000, 784)
            x_test = x_test.reshape(10000, 784)
        elif model_type == "cifar10_CNN":
            from Train_cifar10_CNN import train
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
    os.system("echo 'tra: '" + str(tra_end_main - tra_begin_main) + " >> ../evaluation/time.txt")