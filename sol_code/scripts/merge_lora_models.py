import os
import argparse
import torch
import json
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import sys

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Add parent directory to path to import model_registry
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from api.model_registry import ModelRegistry

def merge_lora_model(base_model_path, lora_model_path, output_path, model_id=None):
    """Merge a LoRA model with its base model."""
    logging.info(f"Loading base model from {base_model_path}")
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        local_files_only=True,
        device_map="auto"
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        local_files_only=True
    )
    
    logging.info(f"Loading LoRA model from {lora_model_path}")
    
    # Load PEFT (LoRA) model
    model = PeftModel.from_pretrained(
        base_model,
        lora_model_path,
        torch_dtype=torch.float16,
        local_files_only=True
    )
    
    logging.info("Merging models...")
    
    # Merge LoRA adapter with base model
    merged_model = model.merge_and_unload()
    
    # Save merged model
    os.makedirs(output_path, exist_ok=True)
    logging.info(f"Saving merged model to {output_path}")
    merged_model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    
    # Update registry if model_id is provided
    if model_id:
        try:
            registry = ModelRegistry()
            registry.update_model_path(model_id, output_path)
            logging.info(f"Updated registry entry for model '{model_id}'")
        except Exception as e:
            logging.error(f"Error updating registry: {e}")
    
    logging.info("Model merging complete!")
    return output_path

def main():
    parser = argparse.ArgumentParser(description="Merge LoRA model with base model")
    parser.add_argument("--base", required=True, help="Path to base model")
    parser.add_argument("--lora", required=True, help="Path to LoRA adapter model")
    parser.add_argument("--output", required=True, help="Output path for merged model")
    parser.add_argument("--model-id", help="Model ID in registry to update")
    args = parser.parse_args()
    
    # Convert paths to absolute if needed
    if not os.path.isabs(args.base):
        args.base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", args.base))
    if not os.path.isabs(args.lora):
        args.lora = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", args.lora))
    if not os.path.isabs(args.output):
        args.output = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", args.output))
    
    merge_lora_model(args.base, args.lora, args.output, args.model_id)

if __name__ == "__main__":
    main()