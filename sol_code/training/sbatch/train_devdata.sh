#!/bin/bash
#SBATCH -N 1                # number of nodes
#SBATCH -c 4                # number of cores
#SBATCH -t 0-08:00:00       # time in d-hh:mm:ss
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH -p general          # partition
#SBATCH -q class            # QOS
#SBATCH -A class_cse476spring2025
#SBATCH --output=logs/training/devdata_train_%j.log
#SBATCH --error=logs/training/devdata_train_%j.err

# Load modules
module load mamba/latest
module load cuda-12.5.0-gcc-12.1.0

# Activate the environment
ENV_NAME="llama_training_env"
ENV_PATH="$HOME/$ENV_NAME"
source activate $ENV_PATH

# Install compatible versions of packages
pip install 'numpy<2.0.0' --no-deps  # Install numpy 1.x
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.36.0 datasets==2.16.0

# Navigate to project directory
cd $HOME/cse_476_final_proj

# Check if symlink for model exists, if not create it
if [ ! -L "sol_code/models/base/llama-3.2-3b-base" ]; then
    echo "Creating symbolic link for model path..."
    rm -rf sol_code/models/base/llama-3.2-3b-base
    mkdir -p sol_code/models/base
    ln -s $HOME/cse_476_final_proj/sol_code/models/llama-3.2-3b-base $HOME/cse_476_final_proj/sol_code/models/base/llama-3.2-3b-base
fi

# Create logs directory if it doesn't exist
mkdir -p sol_code/logs/training
rm -rf sol_code/logs/training/*

# First, process the data if it doesn't exist
if [ ! -f sol_code/data/processed/devdata_instruction.json ]; then
    echo "Processing dev data..."
    mkdir -p sol_code/data/processed
    python sol_code/data/process_math_data.py --input_file sol_code/data/raw/dev_data.json --output_file sol_code/data/processed/devdata_instruction.json
fi

# Create config file for this training run
mkdir -p sol_code/training/configs
if [ ! -f sol_code/training/configs/devdata_config.yaml ]; then
    echo "Creating config file for dev data training..."
    cat > sol_code/training/configs/devdata_config.yaml << EOF
model_id: "devdata-tuned"
description: "Llama-3.2-3B model fine-tuned on the development dataset"
base_model_path: "models/base/llama-3.2-3b-base"
dataset_path: "data/processed/devdata_instruction.json"
output_dir: "models/variants/devdata-tuned"
format_type: "instruction"
batch_size: 1
gradient_accumulation: 8
learning_rate: 2e-5
epochs: 3
max_length: 512
EOF
fi

# Run training script with the generated config
echo "Starting training..."
python sol_code/training/train.py --config sol_code/training/configs/devdata_config.yaml 