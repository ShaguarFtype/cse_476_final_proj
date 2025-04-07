#!/bin/bash
#SBATCH -N 1                # number of nodes
#SBATCH -c 4                # number of cores
#SBATCH -t 0-02:00:00       # time in d-hh:mm:ss
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH -p general          # partition
#SBATCH -q class            # QOS
#SBATCH -A class_cse476spring2025
#SBATCH --output=api_test_output.log  # Output log file
#SBATCH --error=api_test_error.log    # Error log file

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
pip install flask torch transformers accelerate bitsandbytes requests

# Navigate to the API directory - Full path to avoid errors
cd $HOME/cse_476_final_proj/sol_code/api

# Run the API with test
echo "Starting API with self-test..."
python app_with_test.py

# Run for 20 minutes, then exit
sleep 1200
echo "Test complete. Check api.log for results."
