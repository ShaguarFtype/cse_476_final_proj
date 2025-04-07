#!/bin/bash

if [ $# -ne 1 ]; then
    echo "Usage: $0 <node_name> (e.g., $0 sg039)"
    exit 1
fi

NODE=$1
echo "Running evaluation against API on node $NODE..."

# Load modules
module load mamba/latest

# Activate environment (use the same environment as in run_fixed_test.sh)
source activate $HOME/llama_env

# Install pandas if not already installed
pip install pandas requests

# Run the evaluation script
python ./api/eval_model.py --node $NODE

echo "Evaluation complete. Check model_eval.log for details."
