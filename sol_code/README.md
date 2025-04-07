# AI Assistant Project

This project aims to build a general-purpose AI assistant using the Llama-3.2-3B base model, 
fine-tuned with instruction data for the midpoint report due 04/08.

## Project Structure
llama-assistant/
├── api/              # API code
├── configs/          # Configuration files
├── data/             # Data storage
│   └── processed/    # Processed datasets
├── models/           # Model storage
├── notebooks/        # Jupyter notebooks for exploration
├── scripts/          # Training and utility scripts
└── utils/            # Utility functions
Copy
## Setup Instructions

1. Clone the repository:
git clone git@github.com/your-repo-name.git
cd your-repo-name
Copy
2. Login to SOL and request an interactive session:
ssh your_ASURITE@login.sol.rc.asu.edu
interactive -q class -A class_asu101spring2025 --gres=gpu:1 -t 0-4:00 --mem=16G
Copy
3. Set up the environment:
module load mamba/latest
mamba create -n llama_assistant python=3.10 -y
source activate llama_assistant
pip install -r requirements.txt
Copy
4. Download the base model:
python scripts/download_model.py
Copy
5. Prepare the dataset:
python scripts/prepare_dataset.py
Copy
6. Run the training (using SBATCH):
sbatch scripts/train.sh
Copy
7. Test the model:
python scripts/test_model.py
Copy
8. Run the API (using SBATCH):
sbatch api/run_api.sh
Copy
## API Usage

Once the API is running, you can interact with the model using:
curl -X POST http://localhost:5000/generate 
-H "Content-Type: application/json" 
-d '{"query": "Explain how transformers work"}'
Copy
## Evaluation

The model is being optimized for the evaluation benchmarks specified in the project requirements.
