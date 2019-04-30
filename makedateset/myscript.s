#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --mem=8GB
#SBATCH --job-name=GPUDemo3
#SBATCH --mail-type=END
#SBATCH --mail-user=lx643@nyu.edu
#SBATCH --output=slurm_%j.out


python makeDataset.py 
#python readpcap.py
