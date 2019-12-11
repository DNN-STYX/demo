#!/bin/sh

Nproc=5			# the limit number of processes
total_mission=5	# the number of missions

# "[FGM, BIM, JSMA, DeepFool, CW-L2, CW-Linf]"

mkdir evaluation
mkdir evaluation_result


#mission
function mission(){
	i=$1
	if [ "$i" -eq 1 ]; then
		mkdir evaluation/mnist_MLP
		cd mnist_MLP/evaluation/
		python -u main_evaluation.py "[FGM]" 	    > ../../evaluation_result/mnist_MLP_evaluation.txt 
		cp evaluation.xls     	   					../../evaluation/mnist_MLP
		cd ../../
		rm -r mnist_MLP
	elif [ "$i" -eq 2 ]; then
		mkdir evaluation/mnist_CNN
		cd mnist_CNN/evaluation/
		python -u main_evaluation.py "[FGM]" 	    > ../../evaluation_result/mnist_CNN_evaluation.txt 
		cp evaluation.xls     	   					../../evaluation/mnist_CNN
		cd ../../
		rm -r mnist_CNN
	elif [ "$i" -eq 3 ]; then
		mkdir evaluation/fmnist_MLP
		cd fmnist_MLP/evaluation/
		python -u main_evaluation.py "[FGM]" 	    > ../../evaluation_result/fmnist_MLP_evaluation.txt 
		cp evaluation.xls     	   					../../evaluation/fmnist_MLP
		cd ../../
		rm -r fmnist_MLP
	elif [ "$i" -eq 4 ]; then
		mkdir evaluation/fmnist_CNN
		cd fmnist_CNN/evaluation/
		python -u main_evaluation.py "[FGM]" 	    > ../../evaluation_result/fmnist_CNN_evaluation.txt 
		cp evaluation.xls     	   					../../evaluation/fmnist_CNN
		cd ../../
		rm -r fmnist_CNN
	elif [ "$i" -eq 5 ]; then
		mkdir evaluation/cifar10_CNN
		cd cifar10_CNN/evaluation/
		python -u main_evaluation.py "[FGM]" 	    > ../../evaluation_result/cifar10_CNN_evaluation.txt 
		cp evaluation.xls     	   					../../evaluation/cifar10_CNN
		cd ../../
		rm -r cifar10_CNN
	fi
}


Pfifo="/tmp/$$.fifo"    # create a fifo type file
mkfifo $Pfifo     		# create a named pipe
exec 6<>$Pfifo     		# fd is 6
rm -f $Pfifo

# Initialize the pipe
for((i=1; i<=$Nproc; i=i+1)); do
    echo
done >&6


for ((j=1; j<=$total_mission; j=j+1))
do
    read -u6
    {
        echo mission $j
        mission $j
        echo mission $j completed.
        echo >&6
    } &
done

wait     # waiting for all the background processes finished
