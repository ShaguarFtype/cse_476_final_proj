#!/bin/bash

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

# Navigate to the API directory - Full path to avoid errors
cd $HOME/cse_476_final_proj/sol_code/api

# Check if model exists before starting
MODEL_PATH="$HOME/cse_476_final_proj/sol_code/models/llama-3.2-3b-base"
if [ ! -d "$MODEL_PATH" ] || [ ! -f "$MODEL_PATH/config.json" ] || [ ! -f "$MODEL_PATH/tokenizer.json" ]; then
    echo "Error: Model files not found at $MODEL_PATH"
    echo "Please run the download_model.py script first."
    exit 1
fi

# Run the API
echo "Starting API..."
python app.py

# Note: Since this is running in your terminal directly, you can press Ctrl+C to stop it
# No need for the sleep command as in the Slurm job
