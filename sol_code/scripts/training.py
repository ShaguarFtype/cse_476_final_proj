import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
import datasets

def main():
    # Load configuration
    with open("configs/base_config.json", "r") as f:
        config = json.load(f)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_id"],
        cache_dir=config["model_path"]
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        config["model_id"],
        cache_dir=config["model_path"],
        torch_dtype=torch.bfloat16,
        device_map="auto",
        load_in_4bit=True
    )
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=config["lora"]["r"],
        lora_alpha=config["lora"]["lora_alpha"],
        target_modules=config["lora"]["target_modules"],
        lora_dropout=config["lora"]["lora_dropout"],
        bias=config["lora"]["bias"],
        task_type=config["lora"]["task_type"]
    )
    
    model = get_peft_model(model, lora_config)
    
    # Load dataset
    with open("data/processed/instruction_dataset.json", "r") as f:
        data = json.load(f)
    
    # Format data for training
    def format_instruction(example):
        instruction = example["instruction"]
        input_text = example["input"]
        output = example["output"]
        
        if input_text:
            prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
        else:
            prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
            
        return {
            "prompt": prompt,
            "response": output
        }
    
    formatted_data = [format_instruction(item) for item in data]
    train_dataset = datasets.Dataset.from_list(formatted_data)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="models/llama-3.2-3b-ft",
        per_device_train_batch_size=config["training"]["per_device_train_batch_size"],
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        learning_rate=config["training"]["learning_rate"],
        num_train_epochs=config["training"]["num_train_epochs"],
        save_steps=500,
        logging_steps=100,
        fp16=True,
    )
    
    # Set up trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        max_seq_length=config["training"]["max_seq_length"],
        packing=True
    )
    
    # Train model
    trainer.train()
    
    # Save model
    trainer.save_model("models/llama-3.2-3b-ft")
    tokenizer.save_pretrained("models/llama-3.2-3b-ft")
    
    print("Model training complete and saved!")

if __name__ == "__main__":
    main()
