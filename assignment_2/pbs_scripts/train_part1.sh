#!/bin/sh
#PBS -lwalltime=2:00:00
#PBS -lnodes=1:ppn=12
#PBS -lmem=250GB

module load Python/3.6.3-foss-2017b cuDNN/7.0.5-CUDA-9.0.176 OpenMPI/2.1.1-GCC-6.4.0-2.28 NCCL
export LD_LIBRARY_PATH=/hpc/sw/NCCL/2.0.5/lib:/hpc/eb/Debian9/cuDNN/7.0.5-CUDA-9.0.176/lib64:/hpc/eb/Debian9/CUDA/9.0.176/lib64:$LD_LIBRARY_PATH

pip3 install -r ~/uvadlc_practicals_2018/requirements.txt --user --no-cache
cd ~/uvadlc_practicals_2018/assignment_2/part1/
for T in 16 18 20 22 24
do
	python train.py --model_type RNN --input_length $T --input_dim 10 --train_steps 400
	sleep 5
done
# for T in 20
# do
# 	python train.py --model_type LSTM --input_length $T --input_dim 10
# done
