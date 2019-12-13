#!/usr/bin/env python
#-*- coding:utf-8 -*-

from __future__ import absolute_import, division, print_function, unicode_literals
### 这样的做法的作用就是将新版本的特性引进当前版本中，也就是说我们可以在当前版本使用新版本的一些特性


import os
import sys
import time
import keras
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
sys.path.append("../")

from keras.models import load_model
from art.classifiers import KerasClassifier
from keras.datasets import mnist, fashion_mnist, cifar10
from art.defences.adversarial_trainer import AdversarialTrainer
## 攻击方法
from art.attacks.deepfool import DeepFool
from art.attacks.saliency_map import SaliencyMapMethod
from art.attacks.fast_gradient import FastGradientMethod
from art.attacks.iterative_method import BasicIterativeMethod
from art.attacks.projected_gradient_descent import ProjectedGradientDescent




if __name__ == "__main__":

    # ========================0.获取参数====================================== #
    data_type = sys.argv[1]                                 # "mnist"
    model_type = sys.argv[2]                                # "MLP"
    adv_train_num = int(sys.argv[3])                        # 20
    # ratio_value = float(sys.argv[4])                        # 0.5
    # adv_train_attack = sys.argv[5]                          # "BIM"
    # attack_par_lst = sys.argv[6][1:-1].split(', ')          # "[norm_type, np.inf, epsilon, 0.3, epsilon_step, 0.01, max_iteration, 40]"
    ratio_value = 0.5
    adv_train_attack = "BIM"
    attack_par_lst = ["norm_type", "np.inf", "epsilon", "0.3", "epsilon_step", "0.01", "max_iteration", "40"]

    attack_par = {}
    # FGM       : [epsilon, norm_type]
    # BIM       : [norm_type, epsilon, epsilon_step, max_iteration]
    # JSMA      : [theta, gamma]
    # DeepFool  : [max_iteration, epsilon]
    # CW-L2     : [confidence_value, max_iter_value]
    # CW-Linf   : [confidence_value, max_iter_value]
    for i in range(int(len(attack_par_lst) / 2)):
        if "norm" in attack_par_lst[2 * i] or "iter" in attack_par_lst[2 * i]:
            if attack_par_lst[2 * i + 1] == "np.inf":
                attack_par[attack_par_lst[2 * i]] = np.inf
            else:
                attack_par[attack_par_lst[2 * i]] = int(attack_par_lst[2 * i + 1])
        else:
            attack_par[attack_par_lst[2 * i]] = float(attack_par_lst[2 * i + 1])

    # ==========================1.准备用于做对抗训练的随机模型 ======================= #
    # valid model name: "mnist", "cifar10"
    if os.path.exists("./model"):
        os.system("rm -r model")
    tra_begin_main = time.time()
    if data_type == "mnist" and model_type == "MLP":
        from traditional_training.Train_mnist_MLP import train
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train.reshape(60000, 784)
        x_test = x_test.reshape(10000, 784)
    elif data_type == "mnist" and model_type == "CNN":
        from traditional_training.Train_mnist_CNN import train
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train.reshape(60000, 784)
        x_test = x_test.reshape(10000, 784)
    elif data_type == "fmnist" and model_type == "MLP":
        from traditional_training.Train_fmnist_MLP import train
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        x_train = x_train.reshape(60000, 784)
        x_test = x_test.reshape(10000, 784)
    elif data_type == "fmnist" and model_type == "CNN":
        from traditional_training.Train_fmnist_CNN import train
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
    train(x_train, y_train, x_test, y_test, 0)
    os.system("mv ./model/" + data_type + "_" + model_type + " ./model/adv_model")


    
    # ==============================2.进行对抗训练=================================== #
    begin_time = time.time()
    robust_classifier_model = load_model("./model/adv_model")
    robust_classifier = KerasClassifier((0., 1.), robust_classifier_model)
    # ==================================2-1.准备对抗训练的攻击方法 =============================== #
    """
    The `ratio` determines how many of the clean samples in each batch are replaced with their adversarial counterpart.
    warning: Both successful and unsuccessful adversarial samples are used for training. In the case of unbounded attacks
            (e.g., DeepFool), this can result in invalid (very noisy) samples being included.
    """
    if adv_train_attack == "FGM":
        attacks = FastGradientMethod(robust_classifier, eps=attack_par["epsilon"], norm=attack_par["norm_type"])
    elif adv_train_attack == "BIM":
        attacks = BasicIterativeMethod(robust_classifier, norm=attack_par["norm_type"], eps=attack_par["epsilon"],
                                              eps_step=attack_par["epsilon_step"], max_iter=attack_par["max_iteration"])
    elif adv_train_attack == "PGD":
        attacks = ProjectedGradientDescent(robust_classifier, norm=attack_par["norm_type"], eps=attack_par["epsilon"],
                                          eps_step=attack_par["epsilon_step"], max_iter=attack_par["max_iteration"])
    elif adv_train_attack == "JSMA":
        attacks = SaliencyMapMethod(robust_classifier, theta=attack_par["theta"], gamma=attack_par["gamma"])
    elif adv_train_attack == "DeepFool":
        attacks = DeepFool(robust_classifier, max_iter=attack_par["max_iteration"], epsilon=attack_par["epsilon"])


    # ==================================2-2.开始对抗训练 =============================== #
    trainer = AdversarialTrainer(robust_classifier, attacks, ratio=ratio_value)
    trainer.fit(x_train, y_train, nb_epochs=adv_train_num, batch_size=128, verbose=2)
    robust_classifier_model.save("./model/adv_model")
    end_time = time.time()
    model = load_model("./model/adv_model")
    scores = model.evaluate(x_test, y_test, verbose=0)
    # print('Test loss:', scores[0])
    # print('Test accuracy:', scores[1])
    # print("adv_model generation's timecost: " + str(end_time - begin_time))
    print("adversarial_training completed!")
    os.system("echo " + data_type + "_" + model_type + ": " + str(end_time - begin_time) + " > ../evaluation/adv_time.txt")
    os.system("rm -r ./model/origin_model")