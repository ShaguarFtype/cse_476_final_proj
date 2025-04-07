#!/bin/bash
#SBATCH -N 1                # number of nodes
#SBATCH -c 8                # number of cores
#SBATCH -t 0-08:00:00       # time in d-hh:mm:ss
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH -p general          # partition
#SBATCH -q class            # QOS
#SBATCH -A class_asu101spring2025
#SBATCH --mail-type=ALL
#SBATCH --mail-user="your_email@asu.edu"
#SBATCH --export=NONE       # Purge the job-submitting shell environment

# Load modules
module load mamba/latest
module load cuda-12.5.0-gcc-12.1.0
source activate llama_assistant

# Run training
cd "$(dirname "$0")/.."
python scripts/train.py
