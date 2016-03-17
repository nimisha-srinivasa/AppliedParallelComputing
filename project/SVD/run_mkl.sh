#!/bin/bash
#SBATCH --job-name="part1"
#SBATCH --output="part1.%j.%N.out"
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --export=ALL
#SBATCH -t 01:30:00

#SET the number of openmp threads
#export OMP_NUM_THREADS=4


./mkl_svd
