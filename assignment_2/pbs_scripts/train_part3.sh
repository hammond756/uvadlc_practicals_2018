#!/bin/sh
#PBS -lwalltime=02:00:00
#PBS -lnodes=1:ppn=12
#PBS -lmem=250GB

module load Python/3.6.3-foss-2017b cuDNN/7.0.5-CUDA-9.0.176 OpenMPI/2.1.1-GCC-6.4.0-2.28 NCCL
export LD_LIBRARY_PATH=/hpc/sw/NCCL/2.0.5/lib:/hpc/eb/Debian9/cuDNN/7.0.5-CUDA-9.0.176/lib64:/hpc/eb/Debian9/CUDA/9.0.176/lib64:$LD_LIBRARY_PATH

pip3 install -r ~/uvadlc_practicals_2018/requirements.txt --user --no-cache
cd ~/uvadlc_practicals_2018/assignment_2/part3
python train.py --txt_file assets/book_EN_grimms_fairy_tails.txt --epochs 25  --sample_every 100
