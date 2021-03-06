#!/bin/sh
#PBS -lwalltime=8:00:00
#PBS -lnodes=1:ppn=12
#PBS -lmem=250GB

module load Python/3.6.3-foss-2017b cuDNN/7.0.5-CUDA-9.0.176 OpenMPI/2.1.1-GCC-6.4.0-2.28 NCCL
export LD_LIBRARY_PATH=/hpc/sw/NCCL/2.0.5/lib:/hpc/eb/Debian9/cuDNN/7.0.5-CUDA-9.0.176/lib64:/hpc/eb/Debian9/CUDA/9.0.176/lib64:$LD_LIBRARY_PATH

pip3 install -r ~/uva_deeplearning/assignment_1/code/requirements.txt --user --no-cache

python ~/uva_deeplearning/assignment_1/code/gridsearch.py >> ~/uva_deeplearning/assignment_1/code/output_${PBS_JOBID}.txt
