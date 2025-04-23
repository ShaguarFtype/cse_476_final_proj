#!/bin/bash

# Load modules
module load mamba/latest
module load cuda-12.5.0-gcc-12.1.0

# Create and activate a fresh environment
ENV_NAME="llama_training_env"
ENV_PATH="$HOME/$ENV_NAME"

# Remove existing environment if it exists
if [ -d "$ENV_PATH" ]; then
    echo "Removing existing environment: $ENV_NAME"
    rm -rf $ENV_PATH
fi

# Create new environment
echo "Creating new environment: $ENV_NAME"
mamba create -y -p $ENV_PATH python=3.10

# Activate the environment
source activate $ENV_PATH

# Install dependencies in the correct order
echo "Installing dependencies..."

# First install numpy with correct version
pip install numpy==1.23.5

# Install PyTorch with CUDA support
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

# Install accelerate first to ensure it's available
pip install accelerate==0.21.0

# Install transformers and related libraries
pip install transformers==4.30.2 datasets==2.13.1 peft==0.4.0 bitsandbytes==0.40.2

# Install additional dependencies
pip install pyyaml tqdm

# Verify installations
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
python -c "import transformers; print('Transformers version:', transformers.__version__)"
python -c "import accelerate; print('Accelerate version:', accelerate.__version__)"

echo "Environment setup complete. You can now run the training script." 