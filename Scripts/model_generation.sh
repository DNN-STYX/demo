#!/bin/sh

Nproc=5			# the limit number of processes
total_mission=5	# the number of missions

mkdir model
mkdir result


#mission
function mission(){
	i=$1
	if [ "$i" -eq 1 ]; then
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
	elif [ "$i" -eq 2 ]; then
		mkdir model/mnist_CNN
		cp -r ../Tool mnist_CNN
		#==================================
		cd mnist_CNN/traditional_training/
		python -u traditional_training.py "mnist" "CNN" 20 > ../../result/mnist_CNN_tra.txt
		cp -r ./model ../../model/mnist_CNN/tra_model
		#==================================
		cd ../../
		cd mnist_CNN/STYX/
		python -u styx.py "mnist" "CNN" 20	 > ../../result/mnist_CNN_mut.txt
		cp -r ./model ../../model/mnist_CNN/mut_model
		#==================================
		cd ../../
		cd mnist_CNN/adversarial_training/
		python -u adversarial_training.py "mnist" "CNN" 20 > ../../result/mnist_CNN_adv.txt
		cp -r ./model ../../model/mnist_CNN/adv_model
	elif [ "$i" -eq 3 ]; then
		mkdir model/fmnist_MLP
		cp -r ../Tool fmnist_MLP
		#==================================
		cd fmnist_MLP/traditional_training/
		python -u traditional_training.py "fmnist" "MLP" 20 > ../../result/fmnist_MLP_tra.txt
		cp -r ./model ../../model/fmnist_MLP/tra_model
		#==================================
		cd ../../
		cd fmnist_MLP/STYX/
		python -u styx.py "fmnist" "MLP" 20	  > ../../result/fmnist_MLP_mut.txt
		cp -r ./model ../../model/fmnist_MLP/mut_model
		#==================================
		cd ../../
		cd fmnist_MLP/adversarial_training/
		python -u adversarial_training.py "fmnist" "MLP" 20 > ../../result/fmnist_MLP_adv.txt
		cp -r ./model ../../model/fmnist_MLP/adv_model
	elif [ "$i" -eq 4 ]; then
		mkdir model/fmnist_CNN
		cp -r ../Tool fmnist_CNN
		#==================================
		cd fmnist_CNN/traditional_training/
		python -u traditional_training.py "fmnist" "CNN" 20 > ../../result/fmnist_CNN_tra.txt
		cp -r ./model ../../model/fmnist_CNN/tra_model
		#==================================
		cd ../../
		cd fmnist_CNN/STYX/
		python -u styx.py "fmnist" "CNN" 20 	  > ../../result/fmnist_CNN_mut.txt
		cp -r ./model ../../model/fmnist_CNN/mut_model
		#==================================
		cd ../../
		cd fmnist_CNN/adversarial_training/
		python -u adversarial_training.py "fmnist" "CNN" 20 > ../../result/fmnist_CNN_adv.txt
		cp -r ./model ../../model/fmnist_CNN/adv_model
	elif [ "$i" -eq 5 ]; then
		mkdir model/cifar10_CNN
		cp -r ../Tool cifar10_CNN
		#==================================
		cd cifar10_CNN/traditional_training/
		python -u traditional_training.py "cifar10_CNN" 25 > ../../result/cifar10_CNN_tra.txt
		cp -r ./model ../../model/cifar10_CNN/tra_model
		#==================================
		cd ../../
		cd cifar10_CNN/STYX/
		python -u styx.py "cifar10_CNN" 25    > ../../result/cifar10_CNN_mut.txt
		cp -r ./model ../../model/cifar10_CNN/mut_model
		#==================================
		cd ../../
		cd cifar10_CNN/adversarial_training/
		python -u adversarial_training.py "cifar10_CNN" 25  > ../../result/cifar10_CNN_adv.txt
		cp -r ./model ../../model/cifar10_CNN/adv_model
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
