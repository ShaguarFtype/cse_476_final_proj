# Project Architecture

This document outlines the consolidated architecture of our model training and evaluation framework.

## Overview

The project has been designed with a focus on modularity, maintainability, and reusability. The architecture supports:

1. Processing different types of data
2. Training multiple model variants with different configurations
3. Evaluating models with consistent metrics
4. Serving models through a unified API

## Key Components

### 1. Data Processing (`data/process_math_data.py`)

The data processing pipeline is unified through a single script that can:

- Process data for different model variants (math, reasoning, general assistant)
- Apply custom instructions to datasets
- Support subsetting for faster development cycles
- Generate instruction-tuning format data

Usage:
```bash
# Process data for all model variants
python data/process_math_data.py --process_all

# Process data for a specific format
python data/process_math_data.py --format_type math_specialized --output_file data/processed/math.json
```

### 2. Model Training (`training/train.py`)

The training system uses a unified script with configuration files:

- Supports different training configurations through YAML config files
- Automatically registers models in the model registry
- Uses consistent logging and checkpointing

Usage:
```bash
# Train a model with a specific config file
python training/train.py --config training/configs/math_config.yaml
```

### 3. Model Evaluation (`evaluation/eval.py`)

The evaluation system:

- Evaluates models using consistent metrics
- Can evaluate specific models or all models in the registry
- Updates model metrics in the registry
- Generates detailed results reports

Usage:
```bash
# Evaluate all models
python evaluation/eval.py --evaluate_all --data_path data/test_data.json --output_dir evaluation/results

# Evaluate a specific model
python evaluation/eval.py --model_id math-specialized --data_path data/test_data.json --output_dir evaluation/results
```

### 4. Model Registry (`api/model_registry.py`)

The model registry:

- Tracks all trained models and their metadata
- Stores training parameters, dataset information, and evaluation metrics
- Provides a unified interface for accessing model information

### 5. API Service (`api/app.py` and `api/model_manager.py`)

The API service:

- Serves models through a REST API
- Supports model switching
- Provides a client for easy interaction

## Configuration System

All configurations are stored in YAML files:

- Training configurations: `training/configs/`
- Evaluation configurations: `evaluation/configs/`

## Batch Processing Scripts

The `sbatch/` directories in `training/` and `evaluation/` contain job scripts for running on HPC systems.

## How It All Fits Together

1. Data is processed using `process_math_data.py` into instruction-tuning format
2. Training configs are created in `training/configs/`
3. Models are trained using `train.py` with the appropriate config
4. Models are automatically registered in the model registry
5. Models are evaluated using `eval.py`
6. Evaluation metrics are stored in the model registry
7. Models are served through the API service

## Migration from Previous Scripts

If you have been using the previous standalone scripts:

- `finetune_devdata.sh` -> use `training/sbatch/train_devdata.sh`
- `train_devdata.py` -> replaced by `training/train.py` with configs
- `eval_devdata.py` -> replaced by `evaluation/eval.py`
- `preprocess.py` -> replaced by `data/process_math_data.py`

The consolidated architecture maintains all the functionality of the previous scripts while improving maintainability and flexibility. 