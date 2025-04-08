#!/bin/bash
#SBATCH -N 1                # number of nodes
#SBATCH -c 4                # number of cores
#SBATCH -t 0-00:30:00       # time in d-hh:mm:ss
#SBATCH --mem=8G            # less memory needed for evaluation
#SBATCH -p general          # partition
#SBATCH -q class            # QOS
#SBATCH -A class_cse476spring2025
#SBATCH --output=eval_output.log  # Output log file
#SBATCH --error=eval_error.log    # Error log file

# Node to evaluate
NODE=$1

# Load modules
module load mamba/latest

# Activate environment
source activate $HOME/llama_env

# Install required packages
pip install pandas requests dotenv

# Run the evaluation script
python $HOME/cse_476_final_proj/sol_code/api/eval_model.py --node $NODE

echo "Evaluation complete. Check model_eval.log for details."
