#!/bin/bash

# Ensure the script exists if any command fails
set -e

echo "Loading the shell script..."
#SBATCH -J <name_of_your_job>
#SBATCH -o <dir_to_save_your_output>
#SBATCH -D <working_dir>
#SBATCH --get-user-env
SBATCH --clusters=hlai
SBATCH --partition=hlai_std
SBATCH --nodelist=hpdar06c01s01
SBATCH  --output=tabpfn_breastw.log
#SBATCH --mail-type=end
#SBATCH --mail-type=begin
#SBATCH --mail-user=N.Weiss@campus.lmu.de

# ---- load modules
module load python/3.8.11-extended

# ---- check if pip is installed, if not, install it
if ! command -v pip &> /dev/null;
then
    echo "Pip could not be found. Installing pip..."
    sudo apt-get update
    sudo apt-get install -y python3-pip
fi

# ---- install pip libraries if necessary
echo "Installing required Python packages..."
pip install tabpfn
pip install torch
pip install pandas
pip install openml
pip install matplotlib
pip install seaborn

# ---- run the python code below
echo "Running Python script..."
python TabPFN_hepatitis.py
