import os
import json
import yaml
import argparse
import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    Trainer, 
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import sys
import logging

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Add the parent directory to sys.path to import the model_registry
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from api.model_registry import ModelRegistry

def train_model(config):
    """Train a model with the specified configuration."""
    logging.info(f"Starting training with config: {config['model_id']}")
    
    # Load dataset
    logging.info(f"Loading dataset from {config['dataset_path']}")
    with open(config["dataset_path"], 'r') as f:
        data = json.load(f)
    
    logging.info(f"Dataset contains {len(data)} examples")
    
    # Setup tokenizer and model
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", config["base_model_path"]))
    logging.info(f"Loading base model from {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        load_in_4bit=True,
        local_files_only=True
    )
    
    # Format examples with prompt template
    def create_prompt(example):
        if config.get("format_type") == "chat":
            return f"<human>: {example['input']}\n<assistant>: {example['output']}"
        else:
            # Default to instruction format
            return f"""### Instruction:
{example['instruction']}

### Input:
{example['input']}

### Response:
{example['output']}"""
    
    # Tokenize function
    def tokenize_function(examples):
        prompts = [create_prompt(ex) for ex in examples]
        return tokenizer(
            prompts, 
            truncation=True,
            padding="max_length",
            max_length=config.get("max_length", 512)
        )
    
    # Create dataset
    logging.info("Creating and tokenizing dataset")
    dataset = Dataset.from_list(data)
    tokenized_dataset = dataset.map(
        lambda examples: tokenize_function([examples]),
        batched=False
    )
    
    # Output directory
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", config["output_dir"]))
    logging.info(f"Output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Training arguments
    logging.info("Setting up training arguments")
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=config.get("batch_size", 1),
        gradient_accumulation_steps=config.get("gradient_accumulation", 8),
        learning_rate=config.get("learning_rate", 2e-5),
        num_train_epochs=config.get("epochs", 3),
        logging_steps=10,
        save_steps=100,
        fp16=True,
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    # Initialize trainer
    logging.info("Initializing trainer")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )
    
    # Train
    logging.info("Starting training")
    trainer.train()
    
    # Save model
    logging.info(f"Training complete, saving model to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Register model in registry
    logging.info("Registering model in registry")
    registry = ModelRegistry()
    registry.register_model(
        model_id=config["model_id"],
        description=config["description"],
        training_params={
            "learning_rate": config.get("learning_rate", 2e-5),
            "batch_size": config.get("batch_size", 1),
            "epochs": config.get("epochs", 3),
            "max_length": config.get("max_length", 512),
            "gradient_accumulation": config.get("gradient_accumulation", 8),
        },
        dataset_info={
            "path": config["dataset_path"],
            "format_type": config.get("format_type", "instruction"),
            "examples": len(data)
        }
    )
    
    logging.info(f"Model '{config['model_id']}' has been trained and registered")
    return output_dir

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()
    
    config_path = os.path.abspath(args.config)
    logging.info(f"Loading config from {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    train_model(config)

if __name__ == "__main__":
    main() 