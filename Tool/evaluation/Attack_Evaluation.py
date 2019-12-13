#!/usr/bin/env python
#-*- coding:utf-8 -*-

import os
import sys
import time
import keras
import numpy as np
import pandas as pd
sys.path.append("../")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from xlutils.copy import copy
from xlrd import open_workbook
from keras.datasets import mnist
from keras.datasets import fashion_mnist
from keras.datasets import cifar10
from keras.models import load_model
from art.classifiers import KerasClassifier
from art.attacks.deepfool import DeepFool
from art.attacks.carlini import CarliniL2Method
from art.attacks.carlini import CarliniLInfMethod
from art.attacks.saliency_map import SaliencyMapMethod
from art.attacks.fast_gradient import FastGradientMethod
from art.attacks.iterative_method import BasicIterativeMethod

def stat(training_type):
    filename1 = open_workbook("./evaluation.xls")
    sheet1 = filename1.sheets()[0]
    line_num = 1
    if training_type == "STYX":
        mut1 = []
        mut2 = []
        mut3 = []
        mut4 = []
        while True:
            try:
                sheet1.cell_value(line_num, 0)
            except IndexError:
                break
            else:
                mut1.append(float(sheet1.cell_value(line_num, 4)))
                mut2.append(float(sheet1.cell_value(line_num, 7)))
                mut3.append(float(sheet1.cell_value(line_num, 10)))
                mut4.append(float(sheet1.cell_value(line_num, 13)))
                line_num += 1
        mut1_test_value = float(sheet1.cell_value(1, 3))
        mut1_robust_value = sum(mut1) / (line_num - 1)
        mut2_test_value = float(sheet1.cell_value(1, 6))
        mut2_robust_value = sum(mut2) / (line_num - 1)
        mut3_test_value = float(sheet1.cell_value(1, 9))
        mut3_robust_value = sum(mut3) / (line_num - 1)
        mut4_test_value = float(sheet1.cell_value(1, 12))
        mut4_robust_value = sum(mut4) / (line_num - 1)
        mut_test_value = (mut1_test_value + mut2_test_value + mut3_test_value + mut4_test_value) / 4
        mut_robust_value = (mut1_robust_value + mut2_robust_value + mut3_robust_value + mut4_robust_value) / 4
        mut_value = [mut_test_value, mut_robust_value]
        print("The accuracy: ")
        print("Zero Mutation: " + str(mut1_test_value))
        print("Average Mutation: " + str(mut2_test_value))
        print("Random Noise Mutation: " + str(mut3_test_value))
        print("Gaussian Noise Mutation: " + str(mut4_test_value))
        print("Average Accuracy: " + str(mut_test_value))
        print("--------------------------------------\n")
        print("The robustness: ")
        print("Zero Mutation: " + str(mut1_robust_value))
        print("Average Mutation: " + str(mut2_robust_value))
        print("Random Noise Mutation: " + str(mut3_robust_value))
        print("Gaussian Noise Mutation: " + str(mut4_robust_value))
        print("Average Robustness: " + str(mut_robust_value))
        print("--------------------------------------\n")
        file = open("./styx_time.txt")
        list_time = file.readlines()
        item = list_time[0]
        print("The average time-cost: " + item[item.find(":") + 2:-2] + "s")
    elif training_type == "traditional_training":
        tra = []
        while True:
            try:
                sheet1.cell_value(line_num, 0)
            except IndexError:
                break
            else:
                tra.append(float(sheet1.cell_value(line_num, 4)))
                line_num += 1
        tra_test_value = float(sheet1.cell_value(1, 3))
        tra_robust_value = sum(tra) / (line_num - 1)
        print("The accuracy: " + str(tra_test_value))
        # print("--------------------------------------\n")
        print("The robustness: " + str(tra_robust_value))
        # print("--------------------------------------\n")
        file = open("./tra_time.txt")
        list_time = file.readlines()
        item = list_time[0]
        print("The time-cost: " + item[item.find(":") + 2:-2] + "s")
    elif training_type == "adversarial_training":
        adv = []
        while True:
            try:
                sheet1.cell_value(line_num, 0)
            except IndexError:
                break
            else:
                adv.append(float(sheet1.cell_value(line_num, 4)))
                line_num += 1
        adv_test_value = float(sheet1.cell_value(1, 3))
        adv_robust_value = sum(adv) / (line_num - 1)
        print("The accuracy: " + str(adv_test_value))
        # print("--------------------------------------\n")
        print("The robustness: " + str(adv_robust_value))
        # print("--------------------------------------\n")
        file = open("./adv_time.txt")
        list_time = file.readlines()
        item = list_time[0]
        print("The time-cost: " + item[item.find(":") + 2:-2] + "s")



def evaluation(x_test, y_test, classify_idx_lst, model, test_acc, ws, current_line, attack_name, flag, column_i):

    classifier = KerasClassifier((0., 1.), model=model)

    if attack_name == "FGM":
        # ===========================参数设置========================= #
        # Maximum perturbation
        # Order of the norm
        parameter_lst = [[10, 1],
                         [20, 1],
                         [30, 1],
                         [40, 1],
                         [50, 1],
                         [60, 1],
                         [70, 1],
                         [80, 1],
                         [90, 1],
                         [100, 1],
                         [1, 2],
                         [2, 2],
                         [3, 2],
                         [4, 2],
                         [5, 2],
                         [6, 2],
                         [7, 2],
                         [8, 2],
                         [9, 2],
                         [10, 2],
                         [0.05, np.inf],
                         [0.10, np.inf],
                         [0.15, np.inf],
                         [0.20, np.inf],
                         [0.25, np.inf],
                         [0.30, np.inf],
                         [0.35, np.inf],
                         [0.40, np.inf],
                         [0.45, np.inf],
                         [0.50, np.inf]
                         ]
        # ===========================进行攻击========================= #
        for [epsilon, norm_type] in parameter_lst:
            # print("current parameter: " + str(epsilon) + ", " + str(norm_type))
            adv_crafter = FastGradientMethod(classifier)
            x_test_adv = adv_crafter.generate(x=x_test[classify_idx_lst], eps=epsilon, norm=norm_type)
            score = model.evaluate(x_test_adv, y_test[classify_idx_lst], verbose=0)
            acc = score[1]
            ws.write(current_line, 0, attack_name)
            ws.write(current_line, 1, "(" + str(round(epsilon, 4)) + ", " + str(norm_type) + ")")
            if flag == "ori":
                ws.write(current_line, 3, test_acc)
                ws.write(current_line, 4, acc)
            elif flag == "adv":
                ws.write(current_line, 3, test_acc)
                ws.write(current_line, 4, acc)
            else:
                ws.write(current_line, 0+3*column_i, test_acc)
                ws.write(current_line, 1+3*column_i, acc)
            current_line += 1

    elif attack_name == "BIM":
        # ===========================参数设置========================= #
        # Order of the norm
        # Maximum perturbation that the attacker can introduce
        # Attack step size (input variation) at each iteration
        # The maximum number of iterations.
        parameter_lst = [[1, 20.0, 2.0, 10],
                         [1, 20.0, 4.0, 10],
                         [1, 20.0, 6.0, 10],
                         [1, 20.0, 8.0, 10],
                         [1, 20.0, 10.0, 10],
                         [1, 20.0, 2.0, 50],
                         [1, 20.0, 4.0, 50],
                         [1, 20.0, 6.0, 50],
                         [1, 20.0, 8.0, 50],
                         [1, 20.0, 10.0, 50],
                         [2, 2.0, 0.2, 10],
                         [2, 2.0, 0.4, 10],
                         [2, 2.0, 0.6, 10],
                         [2, 2.0, 0.8, 10],
                         [2, 2.0, 1.0, 10],
                         [2, 2.0, 0.2, 50],
                         [2, 2.0, 0.4, 50],
                         [2, 2.0, 0.6, 50],
                         [2, 2.0, 0.8, 50],
                         [2, 2.0, 1.0, 50],
                         [np.inf, 0.1, 0.002, 10],
                         [np.inf, 0.1, 0.004, 10],
                         [np.inf, 0.1, 0.006, 10],
                         [np.inf, 0.1, 0.008, 10],
                         [np.inf, 0.1, 0.010, 10],
                         [np.inf, 0.1, 0.002, 50],
                         [np.inf, 0.1, 0.004, 50],
                         [np.inf, 0.1, 0.006, 50],
                         [np.inf, 0.1, 0.008, 50],
                         [np.inf, 0.1, 0.010, 50]
                         ]
        # ===========================进行攻击========================= #
        for [norm_type, epsilon, epsilon_step, max_iteration] in parameter_lst:
            # print("current parameter: " + str(norm_type) + ", " + str(epsilon) + ", " + str(epsilon_step) + ", " + str(
            #     max_iteration))
            adv_crafter = BasicIterativeMethod(classifier)
            x_test_adv = adv_crafter.generate(x=x_test[classify_idx_lst], norm=norm_type, eps=epsilon, eps_step=epsilon_step, max_iter=max_iteration)
            score = model.evaluate(x_test_adv, y_test[classify_idx_lst], verbose=0)
            acc = score[1]
            ws.write(current_line, 0, attack_name)
            ws.write(current_line, 1, "(" + str(norm_type) + ", " + str(round(epsilon, 4)) + ", " + str(
                round(epsilon_step, 4)) + ", " + str(max_iteration) + ")")
            if flag == "ori":
                ws.write(current_line, 3, test_acc)
                ws.write(current_line, 4, acc)
            elif flag == "adv":
                ws.write(current_line, 3, test_acc)
                ws.write(current_line, 4, acc)
            else:
                ws.write(current_line, 0 + 3 * column_i, test_acc)
                ws.write(current_line, 1 + 3 * column_i, acc)
            current_line += 1

    elif attack_name == "JSMA":
        # ===========================参数设置========================= #
        # Perturbation introduced to each modified feature per step (can be positive or negative).
        # Maximum percentage of perturbed features (between 0 and 1).
        parameter_lst = [[0.5, 0.5],
                         [0.4, 0.5],
                         [0.3, 0.5],
                         [0.2, 0.5],
                         [0.1, 0.5],
                         [-0.1, 0.5],
                         [-0.2, 0.5],
                         [-0.3, 0.5],
                         [-0.4, 0.5],
                         [-0.5, 0.5]
                         ]
        # ===========================进行攻击========================= #
        for [theta, gamma] in parameter_lst:
            # print("current parameter: " + str(theta) + ", " + str(gamma))
            adv_crafter = SaliencyMapMethod(classifier)
            x_test_adv = adv_crafter.generate(x=x_test[classify_idx_lst], theta=theta, gamma=gamma)
            score = model.evaluate(x_test_adv, y_test[classify_idx_lst], verbose=0)
            acc = score[1]
            ws.write(current_line, 0, attack_name)
            ws.write(current_line, 1, "(" + str(round(theta, 4)) + ", " + str(round(gamma, 4)) + ")")

            if flag == "ori":
                ws.write(current_line, 3, test_acc)
                ws.write(current_line, 4, acc)
            elif flag == "adv":
                ws.write(current_line, 3, test_acc)
                ws.write(current_line, 4, acc)
            else:
                ws.write(current_line, 0 + 3 * column_i, test_acc)
                ws.write(current_line, 1 + 3 * column_i, acc)
            current_line += 1

    elif attack_name == "DeepFool":
        # ===========================参数设置========================= #
        # The maximum number of iterations.
        # Overshoot parameter.
        parameter_lst = [[2, 0.10],
                         [4, 0.10],
                         [6, 0.10],
                         [8, 0.10],
                         [10, 0.10],
                         [12, 0.10],
                         [14, 0.10],
                         [16, 0.10],
                         [18, 0.10],
                         [20, 0.10]
                         ]
        # ===========================进行攻击========================= #
        for [max_iteration, epsilon] in parameter_lst:
            # print("current parameter: " + str(max_iteration) + ", " + str(epsilon))
            adv_crafter = DeepFool(classifier)
            x_test_adv = adv_crafter.generate(x=x_test[classify_idx_lst], max_iter=max_iteration, epsilon=epsilon)
            score = model.evaluate(x_test_adv, y_test[classify_idx_lst], verbose=0)
            acc = score[1]
            ws.write(current_line, 0, attack_name)
            ws.write(current_line, 1, "(" + str(max_iteration) + ", " + str(round(epsilon, 4)) + ")")
            if flag == "ori":
                ws.write(current_line, 3, test_acc)
                ws.write(current_line, 4, acc)
            elif flag == "adv":
                ws.write(current_line, 3, test_acc)
                ws.write(current_line, 4, acc)
            else:
                ws.write(current_line, 0 + 3 * column_i, test_acc)
                ws.write(current_line, 1 + 3 * column_i, acc)
            current_line += 1

    elif attack_name == "CW-L2":
        # ===========================参数设置========================= #
        # confidence: Confidence of adversarial examples: a higher value produces examples that are farther away,
        #         from the original input, but classified with higher confidence as the target class.
        # The maximum number of iterations.
        parameter_lst = [[0, 1],
                         [0, 2],
                         [0, 3],
                         [0, 4],
                         [0, 5]
                         ]
        # ===========================进行攻击========================= #
        for [confidence_value, max_iter_value] in parameter_lst:
            # print("current parameter: " + str(confidence_value) + ", " + str(max_iter_value))
            adv_crafter = CarliniL2Method(classifier)
            sum_adv_acc = 0
            for adv_label in range(0, 10):
                one_hot_label = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                one_hot_label[adv_label] = 1
                x_test_adv = adv_crafter.generate(x=x_test[classify_idx_lst], confidence=confidence_value, targeted=True,
                                                  max_iter=max_iter_value,
                                                  y=np.array([one_hot_label] * x_test[classify_idx_lst].shape[0]))
                score = model.evaluate(x_test_adv, y_test[classify_idx_lst], verbose=0)
                acc = score[1]
                sum_adv_acc += acc
            ws.write(current_line, 0, attack_name)
            ws.write(current_line, 1, "(" + str(round(confidence_value, 4)) + ", " + str(max_iter_value) + ")")
            if flag == "ori":
                ws.write(current_line, 3, test_acc)
                ws.write(current_line, 4, sum_adv_acc / 10)
            elif flag == "adv":
                ws.write(current_line, 3, test_acc)
                ws.write(current_line, 4, sum_adv_acc / 10)
            else:
                ws.write(current_line, 0 + 3 * column_i, test_acc)
                ws.write(current_line, 1 + 3 * column_i, sum_adv_acc / 10)
            current_line += 1

    elif attack_name == "CW-Linf":
        # ===========================参数设置========================= #
        # confidence: Confidence of adversarial examples: a higher value produces examples that are farther away,
        #         from the original input, but classified with higher confidence as the target class.
        # The maximum number of iterations.
        parameter_lst = [[0, 1],
                         [0, 2],
                         [0, 3],
                         [0, 4],
                         [0, 5]
                         ]
        # ===========================进行攻击========================= #
        for [confidence_value, max_iter_value] in parameter_lst:
            # print("current parameter: " + str(confidence_value) + ", " + str(max_iter_value))
            adv_crafter = CarliniLInfMethod(classifier)
            sum_adv_acc = 0
            for adv_label in range(0, 10):
                one_hot_label = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                one_hot_label[adv_label] = 1
                x_test_adv = adv_crafter.generate(x=x_test[classify_idx_lst], confidence=confidence_value, targeted=True,
                                                  max_iter=max_iter_value,
                                                  y=np.array([one_hot_label] * x_test[classify_idx_lst].shape[0]))
                score = model.evaluate(x_test_adv, y_test[classify_idx_lst], verbose=0)
                acc = score[1]
                sum_adv_acc += acc
            ws.write(current_line, 0, attack_name)
            ws.write(current_line, 1, "(" + str(round(confidence_value, 4)) + ", " + str(max_iter_value) + ")")
            if flag == "ori":
                ws.write(current_line, 3, test_acc)
                ws.write(current_line, 4, sum_adv_acc / 10)
            elif flag == "adv":
                ws.write(current_line, 3, test_acc)
                ws.write(current_line, 4, sum_adv_acc / 10)
            else:
                ws.write(current_line, 0 + 3 * column_i, test_acc)
                ws.write(current_line, 1 + 3 * column_i, sum_adv_acc / 10)
            current_line += 1

    current_line += 1
    # print("\n------------------------------------------------")
    return ws, current_line




def evaluation_attack(training_type, data_type, attack_type):

    # ========================0.将数据统一放入excel文件======================== #
    if os.path.exists("./evaluation.xls"):
        os.system("rm evaluation.xls")
    empty_excel = pd.DataFrame()
    empty_excel.to_excel("./evaluation.xls")

    rb = open_workbook(u'evaluation.xls')
    # 通过sheet_by_index()获取的sheet没有write()方法
    rs = rb.sheet_by_index(0)
    wb = copy(rb)
    # 通过get_sheet()获取的sheet有write()方法
    ws = wb.get_sheet(0)

    # ========================1.获取数据集======================== #
    if data_type == "mnist":
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train.reshape(60000, 784)
        x_test = x_test.reshape(10000, 784)
    elif data_type == "fmnist":
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        x_train = x_train.reshape(60000, 784)
        x_test = x_test.reshape(10000, 784)
    elif data_type == "cifar10":
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    num_classes = 10
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)



    # ========================2.对网络模型进行攻击评估======================== #
    ws.write(0, 0, "Attacking_mode")
    ws.write(0, 1, "Parameters")

    if training_type == "STYX":

        ws.write(0, 3, "styx_1_test_acc")
        ws.write(0, 4, "styx_1_adv_acc")

        ws.write(0, 6, "styx_2_test_acc")
        ws.write(0, 7, "styx_2_adv_acc")

        ws.write(0, 9, "styx_3_test_acc")
        ws.write(0, 10, "styx_3_adv_acc")

        ws.write(0, 12, "styx_4_test_acc")
        ws.write(0, 13, "styx_4_adv_acc")
        for mutation_i in [1, 2, 3, 4]:
            repeat_model_path = "./model/styx_model_" + str(mutation_i)
            repeat_model = load_model(repeat_model_path)
            repeat_classifier = KerasClassifier((0., 1.), model=repeat_model)
            preds = np.argmax(repeat_classifier.predict(x_test), axis=1)
            acc = np.sum(preds == np.argmax(y_test, axis=1)) / y_test.shape[0]
            # print("(mut_model_" + str(mutation_i) + ")Test accuracy on test sample: %.2f%%" % (acc * 100))

            # 获取分类正确的元素下标
            repeat_classify_idx_lst = []
            for i in range(len(preds)):
                if preds[i] == np.argmax(y_test, axis=1)[i]:
                    repeat_classify_idx_lst.append(i)

            current_line = 1
            if "FGM" in attack_type:
                [ws, current_line] = evaluation(x_test, y_test, repeat_classify_idx_lst, repeat_model, acc, ws,
                                                current_line, "FGM", "robust", mutation_i)
            if "BIM" in attack_type:
                [ws, current_line] = evaluation(x_test, y_test, repeat_classify_idx_lst, repeat_model, acc, ws,
                                                current_line, "BIM", "robust", mutation_i)
            if "JSMA" in attack_type:
                [ws, current_line] = evaluation(x_test, y_test, repeat_classify_idx_lst, repeat_model, acc, ws,
                                                current_line, "JSMA", "robust", mutation_i)
            if "DeepFool" in attack_type:
                [ws, current_line] = evaluation(x_test, y_test, repeat_classify_idx_lst, repeat_model, acc, ws,
                                                current_line, "DeepFool", "robust", mutation_i)
            if "CW-L2" in attack_type:
                [ws, current_line] = evaluation(x_test, y_test, repeat_classify_idx_lst, repeat_model, acc, ws,
                                                current_line, "CW-L2", "robust", mutation_i)
            if "CW-Linf" in attack_type:
                [ws, current_line] = evaluation(x_test, y_test, repeat_classify_idx_lst, repeat_model, acc, ws,
                                                current_line, "CW-Linf", "robust", mutation_i)


    elif training_type == "adversarial_training":
        ws.write(0, 3, "adv_test_acc")
        ws.write(0, 4, "adv_adv_acc")
        ori_model_path = "./model/adv_model"
        ori_model = load_model(ori_model_path)
        ori_classifier = KerasClassifier((0., 1.), model=ori_model)
        preds = np.argmax(ori_classifier.predict(x_test), axis=1)
        acc = np.sum(preds == np.argmax(y_test, axis=1)) / y_test.shape[0]
        # print("(adv_model)Test accuracy on test sample: %.2f%%" % (acc * 100))

        # 获取分类正确的元素下标
        origin_classify_idx_lst = []
        for i in range(len(preds)):
            if preds[i] == np.argmax(y_test, axis=1)[i]:
                origin_classify_idx_lst.append(i)

        current_line = 1
        if "FGM" in attack_type:
            [ws, current_line] = evaluation(x_test, y_test, origin_classify_idx_lst, ori_model, acc, ws, current_line,
                                            "FGM", "adv", 0)
        if "BIM" in attack_type:
            [ws, current_line] = evaluation(x_test, y_test, origin_classify_idx_lst, ori_model, acc, ws, current_line,
                                            "BIM", "adv", 0)
        if "JSMA" in attack_type:
            [ws, current_line] = evaluation(x_test, y_test, origin_classify_idx_lst, ori_model, acc, ws, current_line,
                                            "JSMA", "adv", 0)
        if "DeepFool" in attack_type:
            [ws, current_line] = evaluation(x_test, y_test, origin_classify_idx_lst, ori_model, acc, ws, current_line,
                                            "DeepFool", "adv", 0)
        if "CW-L2" in attack_type:
            [ws, current_line] = evaluation(x_test, y_test, origin_classify_idx_lst, ori_model, acc, ws, current_line,
                                            "CW-L2", "adv", 0)
        if "CW-Linf" in attack_type:
            [ws, current_line] = evaluation(x_test, y_test, origin_classify_idx_lst, ori_model, acc, ws, current_line,
                                            "CW-Linf", "adv", 0)
        # print("\n***************************************************\n")
        # print("\n***************************************************\n")

    elif training_type == "traditional_training":
        ws.write(0, 3, "tra_test_acc")
        ws.write(0, 4, "tra_adv_acc")
        ori_model_path = "./model/origin_model"
        ori_model = load_model(ori_model_path)
        ori_classifier = KerasClassifier((0., 1.), model=ori_model)
        preds = np.argmax(ori_classifier.predict(x_test), axis=1)
        acc = np.sum(preds == np.argmax(y_test, axis=1)) / y_test.shape[0]
        # print("(tra_model)Test accuracy on test sample: %.2f%%" % (acc * 100))

        # 获取分类正确的元素下标
        origin_classify_idx_lst = []
        for i in range(len(preds)):
            if preds[i] == np.argmax(y_test, axis=1)[i]:
                origin_classify_idx_lst.append(i)

        current_line = 1
        if "FGM" in attack_type:
            [ws, current_line] = evaluation(x_test, y_test, origin_classify_idx_lst, ori_model, acc, ws, current_line,
                                            "FGM", "ori", 0)
        if "BIM" in attack_type:
            [ws, current_line] = evaluation(x_test, y_test, origin_classify_idx_lst, ori_model, acc, ws, current_line,
                                            "BIM", "ori", 0)
        if "JSMA" in attack_type:
            [ws, current_line] = evaluation(x_test, y_test, origin_classify_idx_lst, ori_model, acc, ws, current_line,
                                            "JSMA", "ori", 0)
        if "DeepFool" in attack_type:
            [ws, current_line] = evaluation(x_test, y_test, origin_classify_idx_lst, ori_model, acc, ws, current_line,
                                            "DeepFool", "ori", 0)
        if "CW-L2" in attack_type:
            [ws, current_line] = evaluation(x_test, y_test, origin_classify_idx_lst, ori_model, acc, ws, current_line,
                                            "CW-L2", "ori", 0)
        if "CW-Linf" in attack_type:
            [ws, current_line] = evaluation(x_test, y_test, origin_classify_idx_lst, ori_model, acc, ws, current_line,
                                            "CW-Linf", "ori", 0)
        # print("\n***************************************************\n")
        # print("\n***************************************************\n")


    wb.save(u'evaluation.xls')




###=================================================================================###

if __name__ == "__main__":
    pass