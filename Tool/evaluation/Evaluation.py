#!/usr/bin/env python
#-*- coding:utf-8 -*-

import os
import sys
import time
import numpy as np
sys.path.append("../")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from Attack_Evaluation import evaluation_attack, stat

if __name__ == "__main__":

    # ========================0.挑选攻击方法================================= #
    data_type = sys.argv[1]         # "mnist"
    model_type = sys.argv[2]        # "MLP"
    training_type = sys.argv[3]     # "styx"
    attack_type = sys.argv[4]       # "FGM"


    # ========================1.判断是否存在初始网络模型======================== #

    if training_type == "traditional_training":
        os.system("cp -r ../traditional_training/model  ./model")
    elif training_type == "adversarial_training":
        os.system("cp -r ../adversarial_training/model  ./model")
    elif training_type == "STYX":
        os.system("cp -r ../STYX/model  ./model")



    # ========================2.对网络模型进行攻击评估======================== #
    print("++++++++++++++++++++++++++++++++++++++")
    print("++++++++++++++++++++++++++++++++++++++")
    print("            Model Information            ")
    print("Benchmark: " + data_type)
    print("Structure: " + model_type)
    print("Training method: " + training_type)
    print("Evaluation attack: " + attack_type)
    print("++++++++++++++++++++++++++++++++++++++")
    print("++++++++++++++++++++++++++++++++++++++\n")
    evaluation_attack(training_type, data_type, attack_type)
    stat(training_type)
    os.system("rm -r ./model")
