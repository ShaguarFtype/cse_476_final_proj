#!/bin/bash
#SBATCH -N 1                # number of nodes
#SBATCH -c 4                # number of cores
#SBATCH -t 0-12:00:00       # time in d-hh:mm:ss
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH -p general          # partition
#SBATCH -q class            # QOS
#SBATCH -A class_cse476spring2025
#SBATCH --output=logs/eval_all_%j.log
#SBATCH --error=logs/eval_all_%j.err

# Create logs directory if it doesn't exist
mkdir -p evaluation/logs

# Load modules
module load mamba/latest
module load cuda-12.5.0-gcc-12.1.0

# Activate environment
source activate $HOME/llama_env

# Navigate to project directory
cd $HOME/cse_476_final_proj/sol_code

# Run evaluation on all models
python evaluation/eval.py \
    --evaluate_all \
    --data_path data/dev_data.json \
    --output_dir evaluation/results \
    --config evaluation/configs/default_eval_config.yaml 