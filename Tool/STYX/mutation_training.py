#!/usr/bin/env python
#-*- coding:utf-8 -*-

import os
import sys
import time
import keras
import random
import numpy as np
sys.path.append("../")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


from Mutate import Mutate_mnist, Mutate_cifar10
from keras.datasets import mnist, fashion_mnist, cifar10



if __name__ == "__main__":

    # ========================0.准备参数===================================== #
    # 准备模型参数
    model_type = sys.argv[1]                        # "mnist_MLP"
    train_epoch = int(sys.argv[2])                  # 20
    # 准备变异参数
    # mutation_ratio = float(sys.argv[3])             # 0.5
    # pixel_ratio = float(sys.argv[4])                # 0.1
    mutation_ratio = 0.5
    pixel_ratio = 0.1

    model_name = model_type[:model_type.find("_")]


    # ========================1.判断是否存在初始网络模型======================== #
    if os.path.exists("./model") and os.listdir("./model"):  #判断文件夹存在且不为空
        print("Please abandon the 'model' file and start training.")
        exit()
    else:
        tra_begin_main = time.time()
        if model_type == "mnist_MLP":
            from traditional_training.Train_mnist_MLP import train
            (x_train, y_train), (x_test, y_test) = mnist.load_data()
            x_train = x_train.reshape(60000, 784)
            x_test = x_test.reshape(10000, 784)
        elif model_type == "mnist_CNN":
            from traditional_training.Train_mnist_CNN import train
            (x_train, y_train), (x_test, y_test) = mnist.load_data()
            x_train = x_train.reshape(60000, 784)
            x_test = x_test.reshape(10000, 784)
        elif model_type == "fmnist_MLP":
            from traditional_training.Train_fmnist_MLP import train
            (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
            x_train = x_train.reshape(60000, 784)
            x_test = x_test.reshape(10000, 784)
        elif model_type == "fmnist_CNN":
            from traditional_training.Train_fmnist_CNN import train
            (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
            x_train = x_train.reshape(60000, 784)
            x_test = x_test.reshape(10000, 784)
        elif model_type == "cifar10_CNN":
            from traditional_training.Train_cifar10_CNN import train
            (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255
        y_train = keras.utils.to_categorical(y_train, 10)
        y_test = keras.utils.to_categorical(y_test, 10)


    # ========================2.生成新的网络模型======================== #
    mut_begin_main = time.time()

    if model_name == "mnist":
        pixel_num = int(784 * pixel_ratio)
        pixel_lst = random.sample(set(np.arange(0, 784, 1)), pixel_num)
        input_lst = random.sample(set(np.arange(0, x_train.shape[0], 1)), int(x_train.shape[0]*mutation_ratio))

        # mutation 1
        new_train_data_1 = Mutate_mnist(model_name, x_train, 1, pixel_lst, input_lst)
        train(new_train_data_1, y_train, x_test, y_test, train_epoch)
        os.system("mv ./model/" + model_type + " ./model/retrained_1")
        # print("mutation 1 end.\n")

        # mutation 2
        new_train_data_2 = Mutate_mnist(model_name, x_train, 2, pixel_lst, input_lst)
        train(new_train_data_2, y_train, x_test, y_test, train_epoch)
        os.system("mv ./model/" + model_type + " ./model/retrained_2")
        # print("mutation 2 end.\n")

        # mutation 3
        new_train_data_3 = Mutate_mnist(model_name, x_train, 3, pixel_lst, input_lst)
        train(new_train_data_3, y_train, x_test, y_test, train_epoch)
        os.system("mv ./model/" + model_type + " ./model/retrained_3")
        # print("mutation 3 end.\n")

        # mutation 4
        new_train_data_4 = Mutate_mnist(model_name, x_train, 4, pixel_lst, input_lst)
        train(new_train_data_4, y_train, x_test, y_test, train_epoch)
        os.system("mv ./model/" + model_type + " ./model/retrained_4")
        # print("mutation 4 end.\n")

    elif model_name == "fmnist":
        pixel_num = int(784 * pixel_ratio)
        pixel_lst = random.sample(set(np.arange(0, 784, 1)), pixel_num)
        input_lst = random.sample(set(np.arange(0, x_train.shape[0], 1)), int(x_train.shape[0]*mutation_ratio))

        # mutation 1
        new_train_data_1 = Mutate_mnist(model_name, x_train, 1, pixel_lst, input_lst)
        train(new_train_data_1, y_train, x_test, y_test, train_epoch)
        os.system("mv ./model/" + model_type + " ./model/retrained_1")
        # print("mutation 1 end.\n")

        # mutation 2
        new_train_data_2 = Mutate_mnist(model_name, x_train, 2, pixel_lst, input_lst)
        train(new_train_data_2, y_train, x_test, y_test, train_epoch)
        os.system("mv ./model/" + model_type + " ./model/retrained_2")
        # print("mutation 2 end.\n")

        # mutation 3
        new_train_data_3 = Mutate_mnist(model_name, x_train, 3, pixel_lst, input_lst)
        train(new_train_data_3, y_train, x_test, y_test, train_epoch)
        os.system("mv ./model/" + model_type + " ./model/retrained_3")
        # print("mutation 3 end.\n")

        # mutation 4
        new_train_data_4 = Mutate_mnist(model_name, x_train, 4, pixel_lst, input_lst)
        train(new_train_data_4, y_train, x_test, y_test, train_epoch)
        os.system("mv ./model/" + model_type + " ./model/retrained_4")
        # print("mutation 4 end.\n")

    elif model_name == "cifar10":
        pixel_num = int(3072 * pixel_ratio)
        pixel_lst = random.sample(set(np.arange(0, 3072, 1)), pixel_num)
        input_lst = random.sample(set(np.arange(0, x_train.shape[0], 1)), int(x_train.shape[0] * mutation_ratio))

        # mutation 1
        new_train_data_1 = Mutate_cifar10(model_name, x_train, 1, pixel_lst, input_lst)
        train(new_train_data_1, y_train, x_test, y_test, train_epoch)
        os.system("mv ./model/" + model_type + " ./model/retrained_1")
        # print("mutation 1 end.\n")

        # mutation 2
        new_train_data_2 = Mutate_cifar10(model_name, x_train, 2, pixel_lst, input_lst)
        train(new_train_data_2, y_train, x_test, y_test, train_epoch)
        os.system("mv ./model/" + model_type + " ./model/retrained_2")
        # print("mutation 2 end.\n")

        # mutation 3
        new_train_data_3 = Mutate_cifar10(model_name, x_train, 3, pixel_lst, input_lst)
        train(new_train_data_3, y_train, x_test, y_test, train_epoch)
        os.system("mv ./model/" + model_type + " ./model/retrained_3")
        # print("mutation 3 end.\n")

        # mutation 4
        new_train_data_4 = Mutate_cifar10(model_name, x_train, 4, pixel_lst, input_lst)
        train(new_train_data_4, y_train, x_test, y_test, train_epoch)
        os.system("mv ./model/" + model_type + " ./model/retrained_4")
        # print("mutation 4 end.\n")
    os.system("rm ./model/origin_model")
    print("STYX completed!")

    mut_end_main = time.time()
    # print("mut_model generation's timecost: " + str((mut_end_main - mut_begin_main)/4))
    os.system("echo 'mut: '" + str((mut_end_main - mut_begin_main)/4) + " >> ../evaluation/time.txt")