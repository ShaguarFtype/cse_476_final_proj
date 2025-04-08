#!/bin/bash
# setup_env.sh - Environment setup script for Llama-3.2-3B work

# Print header
echo "======================================================"
echo "  Llama-3.2-3B Development Environment Setup"
echo "  ASU SOL Cluster"
echo "======================================================"

# Load required modules
echo "[1/4] Loading required modules..."
module load cuda-12.5.0-gcc-12.1.0
module load mamba/latest

# Environment name and path
ENV_NAME="llama_env"
ENV_PATH="$HOME/$ENV_NAME"

# Check if environment exists as a directory
if [ -d "$ENV_PATH" ]; then
    echo "[2/4] Found existing environment at '$ENV_PATH', activating..."
    source activate "$ENV_PATH"
else
    echo "[2/4] Creating new environment at '$ENV_PATH'..."
    mamba create -p "$ENV_PATH" python=3.10 -y
    source activate "$ENV_PATH"
    
    echo "[3/4] Installing required packages..."
    pip install flask torch transformers accelerate bitsandbytes requests python-dotenv pandas numpy matplotlib jupyter
fi

# Verify CUDA availability
echo "[4/4] Verifying CUDA availability..."
python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('CUDA Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

# Set up working directory
WORK_DIR="$HOME/cse_476_final_proj/sol_code"
if [ -d "$WORK_DIR" ]; then
    cd $WORK_DIR
    echo "Changed to working directory: $WORK_DIR"
else
    echo "Working directory not found: $WORK_DIR"
    echo "Please create your project directory or modify this script."
fi

# Print environment information
echo "======================================================"
echo "Environment '$ENV_NAME' is ready!"
echo "Python version: $(python --version)"
echo "Packages installed:"
pip list | grep -E 'torch|transformers|flask|numpy|pandas|jupyter'
echo "======================================================"
echo "To use the Llama-3.2-3B client:"
echo "  cd $WORK_DIR/api"
echo "  python client.py --node <node_name> --interactive"
echo "======================================================"

# Keep the environment active
exec $SHELL
