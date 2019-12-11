#!/bin/sh


mkdir evaluation
mkdir evaluation_result


mkdir evaluation/mnist_MLP
cd mnist_MLP/evaluation/
python -u main_evaluation.py "[FGM]" 	    > ../../evaluation_result/mnist_MLP_evaluation.txt 
cp evaluation.xls     	   					../../evaluation/mnist_MLP
cd ../../
rm -r mnist_MLP
mv evaluation_result ./evaluation