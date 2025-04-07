#!/bin/bash
#SBATCH -N 1                # number of nodes
#SBATCH -c 4                # number of cores
#SBATCH -t 0-04:00:00       # time in d-hh:mm:ss
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH -p general          # partition
#SBATCH -q class            # QOS
#SBATCH -A class_cse476spring2025  # Updated to your actual account
#SBATCH --export=NONE       # Purge the job-submitting shell environment

# Load modules
module load mamba/latest
module load cuda-12.5.0-gcc-12.1.0

# Create virtual environment if it doesn't exist
if [ ! -d "$HOME/llama_env" ]; then
    mamba create -p $HOME/llama_env python=3.10 -y
fi

# Activate environment
source activate $HOME/llama_env

# Install required packages if not already installed
pip install flask torch transformers accelerate bitsandbytes

# Navigate to the API directory
cd "$(dirname "$0")"

# Run the API
python app.py
