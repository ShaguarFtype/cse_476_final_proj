#!/bin/bash
#SBATCH -N 1                # number of nodes
#SBATCH -c 4                # number of cores
#SBATCH -t 0-08:00:00       # time in d-hh:mm:ss
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH -p general          # partition
#SBATCH -q class            # QOS
#SBATCH -A class_cse476spring2025
#SBATCH --output=logs/devdata_train_%j.log
#SBATCH --error=logs/devdata_train_%j.err

# Load modules
module load mamba/latest
module load cuda-12.5.0-gcc-12.1.0

# Activate environment
source activate $HOME/llama_env

# Navigate to project directory
cd $HOME/cse_476_final_proj/sol_code

# First, process the data if it doesn't exist
if [ ! -f data/processed/devdata_instruction.json ]; then
    echo "Processing dev data..."
    python data/process_math_data.py --input_file data/dev_data.json --output_file data/processed/devdata_instruction.json
fi

# Create config file for this training run
cat > training/configs/devdata_config.yaml << EOF
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

# Run training script with the generated config
python training/train.py --config training/configs/devdata_config.yaml 