#!/usr/bin/env python
#-*- coding:utf-8 -*-

import os
import sys
import time
import numpy as np
sys.path.append("../")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from Evaluation import evaluation_attack, stat

if __name__ == "__main__":

    # begin_main = time.time()

    # ========================0.挑选攻击方法================================= #
    evaluation_lst = sys.argv[1][1:-1].split(', ')  # "[FGM, JSMA]"


    # ========================1.判断是否存在初始网络模型======================== #
    model_names = os.listdir("./")
    for i in range(len(model_names)):
        if "model" in model_names[i]:
            os.system("rm -r *_model")
            os.system("rm evaluation.xls")
            break
    os.system("cp -r ../traditional_training/model  ./tra_model")
    os.system("cp -r ../adversarial_training/model  ./adv_model")
    os.system("cp -r ../STYX/model  ./mut_model")

    if os.path.exists("./tra_model") and os.listdir("./tra_model"):  #判断文件夹存在且不为空
        model_names = os.listdir("./tra_model/")
        for i in range(len(model_names)):
            if "fmnist" in model_names[i]:
                model_name = "fmnist"
            elif "mnist" in model_names[i]:
                model_name = "mnist"
            elif "cifar10" in model_names[i]:
                model_name = "cifar10"
                break
    else:
        print("Please provide a model name.")



    # ========================2.对网络模型进行攻击评估======================== #
    print("++++++++++++++++++++++++++++++++++++++")
    print("++++++++++++++++++++++++++++++++++++++")
    print("            Model Information            ")
    print("Benchmark: " + model_names[i][:model_names[i].find("_")])
    print("Structure: " + model_names[i][model_names[i].find("_") + 1:])
    print("Evaluation attack: " + evaluation_lst[0])
    print("++++++++++++++++++++++++++++++++++++++")
    print("++++++++++++++++++++++++++++++++++++++\n")
    evaluation_attack(model_name, evaluation_lst)
    stat(model_name, evaluation_lst)
    os.system("rm time.txt")
    os.system("rm -r ../adversarial_training/model")
    os.system("rm -r ../traditional_training/model")
    os.system("rm -r ../STYX/model")


    # end_main = time.time()
    # print("whole timecost: " + str(end_main - begin_main))