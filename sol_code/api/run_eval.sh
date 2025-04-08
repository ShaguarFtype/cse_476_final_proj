#!/bin/bash
#SBATCH -N 1                # number of nodes
#SBATCH -c 4                # number of cores
#SBATCH -t 0-01:00:00       # time in d-hh:mm:ss
#SBATCH --mem=16G           # increased memory
#SBATCH -p general          # partition
#SBATCH -q class            # QOS
#SBATCH -A class_cse476spring2025
#SBATCH --output=eval_output_%j.log  # Output log with job ID
#SBATCH --error=eval_error_%j.log    # Error log with job ID

# Node to evaluate
NODE=$1
TIMEOUT=${2:-60}  # Default timeout of 60 seconds if not specified

# Load modules
module load mamba/latest

# Activate environment
source activate $HOME/llama_env

# Install required packages
pip install pandas requests

# Run the evaluation script
python $HOME/cse_476_final_proj/sol_code/api/eval_model.py --node $NODE --timeout $TIMEOUT

echo "Evaluation complete. Check model_eval.log and eval_output_${SLURM_JOB_ID}.log for details."
