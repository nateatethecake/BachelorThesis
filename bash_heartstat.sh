#!/bin/bash

echo "Loading the shell script..."
#SBATCH -J Heartstatlog
#SBATCH -o ./%x.%j.%N.out
#SBATCH -D ./
#SBATCH --get-user-env
#SBATCH --clusters=hlai
#SBATCH --partition=hlai_std
#SBATCH --mail-type=begin
#SBATCH --mail-user=N.Weiss@campus.lmu.de

# ---- load modules
module load python/3.8.11-extended


# ---- run the python code below
echo "Running Python script..."
python TabPFN_heartstatlog.py
