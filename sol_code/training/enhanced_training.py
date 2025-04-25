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
    DataCollatorForSeq2Seq,
    LlamaForCausalLM,
    LlamaConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel,
    prepare_model_for_kbit_training
)
from datasets import Dataset
import sys
import logging
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Add the parent directory to sys.path to import the model_registry
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from api.model_registry import ModelRegistry

def train_model(config):
    """Train a model with the specified configuration using PEFT/LoRA."""
    logging.info(f"Starting enhanced training with PEFT/LoRA for: {config['model_id']}")
    
    # Load dataset - ensure path is absolute
    dataset_path = config["dataset_path"]
    if not os.path.isabs(dataset_path):
        dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", dataset_path))
    logging.info(f"Loading dataset from {dataset_path}")
    
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    logging.info(f"Dataset contains {len(data)} examples")
    
    # Setup tokenizer and model
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", config["base_model_path"]))
    logging.info(f"Loading base model from {model_path}")
    
    # Load tokenizer with basic settings
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logging.info("Setting pad_token to eos_token")
    except Exception as e:
        logging.error(f"Error loading tokenizer: {e}")
        raise
    
    # Load model with proper handling for Llama 3's GQA architecture
    try:
        # First, manually load and fix the configuration
        # Load the raw config file
        config_path = os.path.join(model_path, "config.json")
        with open(config_path, 'r') as f:
            raw_config = json.load(f)
        
        # Fix the rope_scaling format before creating the config object
        if "rope_scaling" in raw_config:
            logging.info(f"Original rope_scaling: {raw_config['rope_scaling']}")
            raw_config["rope_scaling"] = {
                "type": "dynamic",
                "factor": raw_config["rope_scaling"].get("factor", 32.0)
            }
            logging.info(f"Updated rope_scaling: {raw_config['rope_scaling']}")
            
            # Save the modified config temporarily
            temp_config_path = os.path.join(model_path, "config_fixed.json")
            with open(temp_config_path, 'w') as f:
                json.dump(raw_config, f)
            
            # Load the fixed config
            model_config = LlamaConfig.from_pretrained(temp_config_path, local_files_only=True)
            os.remove(temp_config_path)  # Clean up
        else:
            model_config = LlamaConfig.from_pretrained(model_path, local_files_only=True)
        
        # Log the current configuration for debugging
        logging.info(f"Model config: num_attention_heads={model_config.num_attention_heads}, num_key_value_heads={model_config.num_key_value_heads}")
        
        # Load the model with the fixed configuration and quantization
        use_4bit = config.get("use_4bit", True)
        
        # Use 4-bit quantization if specified
        if use_4bit:
            from transformers import BitsAndBytesConfig
            
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
            )
            
            model = LlamaForCausalLM.from_pretrained(
                model_path,
                config=model_config,
                quantization_config=bnb_config,
                device_map="auto",
                local_files_only=True
            )
            
            # Prepare model for PEFT training
            model = prepare_model_for_kbit_training(model)
        else:
            # Load without quantization
            model = LlamaForCausalLM.from_pretrained(
                model_path,
                config=model_config,
                local_files_only=True,
                torch_dtype=torch.float16,
                device_map="auto"
            )
        
        # Setup LoRA configuration
        target_modules = config.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"])
        
        lora_config = LoraConfig(
            r=config.get("lora_r", 16),
            lora_alpha=config.get("lora_alpha", 32),
            lora_dropout=config.get("lora_dropout", 0.05),
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules
        )
        
        # Apply LoRA to model
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        raise
    
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
    
    # Split dataset for training and validation
    train_val_split = dataset.train_test_split(test_size=0.05, seed=42)
    train_dataset = train_val_split["train"]
    val_dataset = train_val_split["test"]
    
    # Tokenize datasets
    tokenized_train = train_dataset.map(
        lambda examples: tokenize_function([examples]),
        batched=False
    )
    
    tokenized_val = val_dataset.map(
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
        per_device_eval_batch_size=config.get("batch_size", 1),
        gradient_accumulation_steps=config.get("gradient_accumulation", 8),
        learning_rate=config.get("learning_rate", 2e-5),
        num_train_epochs=config.get("epochs", 3),
        logging_steps=10,
        save_steps=100,
        eval_steps=100,
        evaluation_strategy="steps",
        save_strategy="steps",
        fp16=True,
        lr_scheduler_type=config.get("lr_scheduler", "cosine"),
        warmup_steps=config.get("warmup_steps", 100),
        weight_decay=config.get("weight_decay", 0.01),
        save_total_limit=3,
        load_best_model_at_end=True,
        report_to=["tensorboard"],
        remove_unused_columns=False,
    )
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, 
        padding=True,
        return_tensors="pt"
    )
    
    # Initialize trainer
    logging.info("Initializing trainer")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # Train
    logging.info("Starting training")
    trainer.train()
    
    # Save model (LoRA adapter only)
    adapter_path = os.path.join(output_dir, "adapter_model")
    os.makedirs(adapter_path, exist_ok=True)
    logging.info(f"Training complete, saving adapter to {adapter_path}")
    model.save_pretrained(adapter_path)
    
    # Save tokenizer with the main model
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
            "lora_r": config.get("lora_r", 16),
            "lora_alpha": config.get("lora_alpha", 32),
            "lora_dropout": config.get("lora_dropout", 0.05),
            "enhanced": True
        },
        dataset_info={
            "path": config["dataset_path"],
            "format_type": config.get("format_type", "instruction"),
            "examples": len(data)
        }
    )
    
    logging.info(f"Model '{config['model_id']}' has been trained with PEFT/LoRA and registered")
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