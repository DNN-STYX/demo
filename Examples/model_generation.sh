#!/bin/sh

mkdir model
mkdir result
mkdir model/mnist_MLP
cp -r ../Tool mnist_MLP
#==================================
cd mnist_MLP/traditional_training/
python -u traditional_training.py "mnist" "MLP" 20 > ../../result/mnist_MLP_tra.txt
cp -r ./model ../../model/mnist_MLP/tra_model
#==================================
cd ../../
cd mnist_MLP/STYX/
python -u styx.py "mnist" "MLP" 20 	 > ../../result/mnist_MLP_mut.txt
cp -r ./model ../../model/mnist_MLP/mut_model
#==================================
cd ../../
cd mnist_MLP/adversarial_training/
python -u adversarial_training.py "mnist" "MLP" 20 > ../../result/mnist_MLP_adv.txt
cp -r ./model ../../model/mnist_MLP/adv_model
mv result ./model