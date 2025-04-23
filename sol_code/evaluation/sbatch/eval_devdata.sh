#!/bin/bash
#SBATCH -N 1                # number of nodes
#SBATCH -c 4                # number of cores
#SBATCH -t 0-04:00:00       # time in d-hh:mm:ss
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH -p general          # partition
#SBATCH -q class            # QOS
#SBATCH -A class_cse476spring2025
#SBATCH --output=logs/devdata_eval_%j.log
#SBATCH --error=logs/devdata_eval_%j.err

# Load modules
module load mamba/latest
module load cuda-12.5.0-gcc-12.1.0

# Activate environment
source activate $HOME/llama_env

# Navigate to project directory
cd $HOME/cse_476_final_proj/sol_code

# Run evaluation script
python evaluation/eval.py \
    --model_id devdata-tuned \
    --data_path data/dev_data.json \
    --output_dir evaluation/results 