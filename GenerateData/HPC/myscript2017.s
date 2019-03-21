#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=10:00:00
#SBATCH --mem=2GB
#SBATCH --job-name=2018_2
#SBATCH --mail-type=END
#SBATCH --mail-user=lx643@nyu.edu
#SBATCH --output=slurm_%j.out
  

cd /scratch/lx643/w_2018_2
python test.py
