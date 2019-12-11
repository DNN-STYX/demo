#!/usr/bin/env python
#-*- coding:utf-8 -*-

from __future__ import absolute_import, division, print_function, unicode_literals
### 这样的做法的作用就是将新版本的特性引进当前版本中，也就是说我们可以在当前版本使用新版本的一些特性


import os
import sys
import copy
import random
import numpy as np
sys.path.append("../")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'



# mutation操作的具体定义
def mutation_1(im, idx):  # 置零
    im[idx] = 0
    return im


def mutation_2(im, idx, model_name):   #  取平均
    if model_name == "mnist" or model_name == "fmnist":
        gross_pixel = 0
        gross_n = 0
        if idx - 29 >= 0:
            gross_pixel += im[idx - 29]
            gross_n += 1
        if idx - 28 >= 0:
            gross_pixel += im[idx - 28]
            gross_n += 1
        if idx - 27 >= 0:
            gross_pixel += im[idx - 27]
            gross_n += 1
        if idx - 1 >= 0:
            gross_pixel += im[idx - 1]
            gross_n += 1
        if idx + 1 <= 783:
            gross_pixel += im[idx + 1]
            gross_n += 1
        if idx + 27 <= 783:
            gross_pixel += im[idx + 27]
            gross_n += 1
        if idx + 28 <= 783:
            gross_pixel += im[idx + 28]
            gross_n += 1
        if idx + 29 <= 783:
            gross_pixel += im[idx + 29]
            gross_n += 1
        im[idx] = gross_pixel / gross_n
        return im
    elif model_name == "cifar10":
        gross_pixel = 0
        gross_n = 0
        if idx < 1024:      # 在第一层
            lay_idx = 0
        elif idx > 2047:    # 在第三层
            lay_idx = 2048
        else:               # 在第二层
            lay_idx = 1024

        if idx - 33 >= 0 + lay_idx:
            gross_pixel += im[idx - 33]
            gross_n += 1
        if idx - 32 >= 0 + lay_idx:
            gross_pixel += im[idx - 32]
            gross_n += 1
        if idx - 31 >= 0 + lay_idx:
            gross_pixel += im[idx - 31]
            gross_n += 1
        if idx - 1 >= 0 + lay_idx:
            gross_pixel += im[idx - 1]
            gross_n += 1
        if idx + 1 <= 1023 + lay_idx:
            gross_pixel += im[idx + 1]
            gross_n += 1
        if idx + 31 <= 1023 + lay_idx:
            gross_pixel += im[idx + 31]
            gross_n += 1
        if idx + 32 <= 1023 + lay_idx:
            gross_pixel += im[idx + 32]
            gross_n += 1
        if idx + 33 <= 1023 + lay_idx:
            gross_pixel += im[idx + 33]
            gross_n += 1
        im[idx] = gross_pixel / gross_n
        return im


def mutation_3(im, idx):    # 随机值替换
    im[idx] = random.random()
    return im


def mutation_4(im, idx):    # 加入高斯噪声
    perturbation = np.random.normal(0, 0.5)
    if im[idx] + perturbation > 1:
        im[idx] = 1
    elif im[idx] + perturbation < 0:
        im[idx] = 0
    else:
        im[idx] += perturbation
    return im




def Mutate_mnist(model_name, x_train, flag, pixel_lst, input_lst):

    new_train_data = copy.deepcopy(x_train)
    for i in input_lst:
        im = x_train[[i]].reshape((784, 1))
        if flag == 1:
            for idx in pixel_lst:
                im = mutation_1(im, idx)
        elif flag == 2:
            for idx in pixel_lst:
                im = mutation_2(im, idx, model_name)
        elif flag == 3:
            for idx in pixel_lst:
                im = mutation_3(im, idx)
        elif flag == 4:
            for idx in pixel_lst:
                im = mutation_4(im, idx)
        new_train_data[[i]] = np.array(im).reshape(784)

    return new_train_data



def Mutate_cifar10(model_name, x_train, flag, pixel_lst, input_lst):

    new_train_data = copy.deepcopy(x_train)
    for i in input_lst:
        im = x_train[[i]].reshape((3072, 1))
        if flag == 1:
            for idx in pixel_lst:
                im = mutation_1(im, idx)
        elif flag == 2:
            for idx in pixel_lst:
                im = mutation_2(im, idx, model_name)
        elif flag == 3:
            for idx in pixel_lst:
                im = mutation_3(im, idx)
        elif flag == 4:
            for idx in pixel_lst:
                im = mutation_4(im, idx)
        new_train_data[[i]] = np.array(im).reshape((32, 32, 3))

    return new_train_data

        
        
###=================================================================================###

if __name__ == "__main__":
    pass

