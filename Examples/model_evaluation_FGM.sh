#!/bin/sh

mkdir evaluation
mkdir evaluation_result
mkdir evaluation/mnist_MLP
cd mnist_MLP/evaluation/
python -u evaluation.py "mnist" "MLP" "traditional_training" "FGM" 	    > ../../evaluation_result/mnist_MLP_tra_FGM.txt 
cp evaluation.xls     	   					../../evaluation/mnist_MLP/FGM_tra.xls
python -u evaluation.py "mnist" "MLP" "adversarial_training" "FGM" 	    > ../../evaluation_result/mnist_MLP_adv_FGM.txt 
cp evaluation.xls     	   					../../evaluation/mnist_MLP/FGM_adv.xls
python -u evaluation.py "mnist" "MLP" "STYX" "FGM" 	    > ../../evaluation_result/mnist_MLP_styx_FGM.txt 
cp evaluation.xls     	   					../../evaluation/mnist_MLP/FGM_styx.xls
cd ../../
rm -r mnist_MLP
mv evaluation_result ./evaluation