# Multi-Model Training and Experimentation Framework

This repository contains a flexible architecture for training, evaluating, and serving multiple fine-tuned versions of the Llama-3.2-3B model.

## Directory Structure

```
sol_code/
├── models/
│   ├── base/                      # Base model
│   │   └── llama-3.2-3b-base/     # Downloaded base model
│   └── variants/                  # Fine-tuned variants
│       ├── math-specialized/      # Math-focused model
│       ├── reasoning-enhanced/    # Chain-of-thought model
│       └── general-assistant/     # General instruction model
├── api/
│   ├── app.py                     # Enhanced API with model selection
│   ├── client.py                  # Enhanced client with model selection
│   ├── model_registry.py          # Model registry and management
│   ├── model_manager.py           # Model loading and switching
│   └── run_api.sh                 # SBATCH script to run the API
├── training/
│   ├── configs/                   # Training configurations
│   │   ├── math_config.yaml
│   │   ├── reasoning_config.yaml
│   │   └── general_config.yaml
│   ├── train.py                   # Main parameterized training script
│   └── sbatch/                    # Training job scripts
│       ├── train_math.sh
│       ├── train_reasoning.sh
│       └── train_general.sh
├── data/
│   ├── raw/                       # Original datasets
│   │   └── dev_data.json
│   ├── process_math_data.py       # Data processing script
│   └── processed/                 # Processed datasets
│       ├── math_instruction.json
│       ├── reasoning_instruction.json
│       └── general_instruction.json
└── evaluation/
    ├── eval.py                    # Multi-model evaluation script
    ├── configs/                   # Evaluation configurations
    │   └── default_eval_config.yaml
    ├── run_eval_all.sh            # Script to evaluate all models
    └── results/                   # Evaluation results by model
```

## Usage

### 1. Process the Data

First, preprocess the dev data into different formats for specific model training:

```bash
cd sol_code
python data/process_math_data.py
```

This will create:
- `data/processed/math_instruction.json` - For math-specialized training
- `data/processed/reasoning_instruction.json` - For reasoning-enhanced training
- `data/processed/general_instruction.json` - For general assistant training

### 2. Train Models

Submit the training jobs:

```bash
cd sol_code
sbatch training/sbatch/train_math.sh
sbatch training/sbatch/train_reasoning.sh
sbatch training/sbatch/train_general.sh
```

Monitor the training progress:

```bash
squeue -u $USER
tail -f training/logs/math_*.log
```

### 3. Evaluate Models

After training is complete, evaluate the models:

```bash
cd sol_code
sbatch evaluation/run_eval_all.sh
```

This will evaluate all trained models on the test data and save the results in `evaluation/results/`.

### 4. Start the API Server

Start the API with the model manager:

```bash
cd sol_code
sbatch api/run_api.sh
```

### 5. Interact with the API

Use the client to interact with the API:

```bash
cd sol_code/api
python client.py --interactive
```

Available commands:
- Type 'models' to list available models
- Type 'switch <model_id>' to switch models (example: 'switch math-specialized')
- Type 'clear' to clear the screen
- Type 'exit' to end the session

## Model Registry

Models are tracked in a registry system that stores metadata about each model, including:
- Training parameters
- Dataset information
- Evaluation metrics

The registry is stored in `models/model_registry.json`.

## Creating Custom Models

1. Create a new configuration file in `training/configs/`
2. Process your data in the required format
3. Submit a training job pointing to your configuration
4. Evaluate the model using the evaluation script
5. Access the model through the API using the model_id

## Contributing

To add new model architectures or training approaches:
1. Update the `training/train.py` script
2. Create appropriate configuration files
3. Add any new data processing scripts to `data/`
4. Document your changes
