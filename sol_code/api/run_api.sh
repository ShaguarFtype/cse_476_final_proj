#!/bin/bash
#SBATCH -N 1                # number of nodes
#SBATCH -c 4                # number of cores
#SBATCH -t 0-12:00:00       # time in d-hh:mm:ss
#SBATCH --mem=16G           # memory required
#SBATCH --gres=gpu:1        # GPUs required
#SBATCH -p general          # partition
#SBATCH -q class            # QOS
#SBATCH -A class_cse476spring2025
#SBATCH --output=api_output_%j.log
#SBATCH --error=api_error_%j.err

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
pip install flask torch transformers accelerate bitsandbytes requests python-dotenv

# Navigate to project directory
cd $HOME/cse_476_final_proj/sol_code

# Start the Flask API
export PYTHONPATH=$PYTHONPATH:/home/$USER/cse_476_final_proj/sol_code
cd api
python app.py
