import os
import json
import re
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
import logging
import yaml
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Add the parent directory to sys.path to import the model_registry
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from api.model_registry import ModelRegistry

def extract_answer(text):
    """Extract the final answer from model output"""
    match = re.search(r'####\s*(.+)', text)
    if match:
        return match.group(1).strip()
    return None

def evaluate_model(model_path, test_data, model_id=None, config=None):
    """Evaluate a model on test data"""
    logging.info(f"Loading model from {model_path}")
    
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
    
    results = []
    correct = 0
    total = 0
    
    # Configuration for generation and evaluation
    temperature = config.get("temperature", 0.7) if config else 0.7
    max_new_tokens = config.get("max_new_tokens", 512) if config else 512
    format_type = config.get("format_type", "instruction") if config else "instruction"
    
    for i, example in enumerate(test_data):
        logging.info(f"Processing example {i+1}/{len(test_data)}")
        
        question = example.get("question", "")
        answer = example.get("answer", "")
        true_answer = extract_answer(answer)
        
        if not true_answer:
            logging.info(f"Skipping example {i+1} - no valid answer found")
            continue
        
        # Prepare the prompt based on the format type
        if format_type == "chat":
            prompt = f"<human>: {question}\n<assistant>: "
        else:
            # Default to instruction format
            prompt = f"""### Instruction:
Solve this problem step by step, showing your work and calculations. End with #### followed by the final answer.

### Input:
{question}

### Response:
"""
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=0.9,
                do_sample=True
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the model's response
        if format_type == "chat" and "<assistant>:" in response:
            response = response.split("<assistant>:")[-1].strip()
        else:
            response = response.replace(prompt, "").strip()
        
        predicted_answer = extract_answer(response)
        
        is_correct = False
        if predicted_answer and predicted_answer.strip() == true_answer.strip():
            correct += 1
            is_correct = True
        
        total += 1
        
        results.append({
            "question": question,
            "true_answer": true_answer,
            "predicted_answer": predicted_answer,
            "is_correct": is_correct,
            "full_response": response[:1000]  # Truncate for readability
        })
        
        # Log progress
        if (i+1) % 10 == 0 or i+1 == len(test_data):
            current_accuracy = correct / total if total > 0 else 0
            logging.info(f"Progress: {i+1}/{len(test_data)} examples | Current accuracy: {current_accuracy:.2%}")
    
    # Calculate final metrics
    accuracy = correct / total if total > 0 else 0
    
    logging.info(f"\nFinal Results for {model_id or 'base model'}:")
    logging.info(f"Accuracy: {accuracy:.2%}")
    logging.info(f"Correct: {correct}/{total}")
    
    metrics = {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "evaluation_timestamp": datetime.now().isoformat()
    }
    
    # If model_id is provided, update registry
    if model_id:
        try:
            registry = ModelRegistry()
            registry.update_model_metrics(model_id, metrics)
            logging.info(f"Updated metrics for model '{model_id}' in registry")
        except Exception as e:
            logging.error(f"Error updating registry: {e}")
    
    return metrics, results

def evaluate_all_models(test_data_path, output_dir, config_path=None):
    """Evaluate all models in the registry"""
    logging.info("Starting evaluation of all models")
    
    # Load test data
    with open(test_data_path, 'r') as f:
        test_data = json.load(f)
    
    # Load evaluation config if provided
    config = None
    if config_path:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    
    # Get all models from registry
    registry = ModelRegistry()
    models = registry.list_models()
    
    # Add base model
    base_model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models/base/llama-3.2-3b-base"))
    
    # Results container
    all_results = {
        "base": {
            "model_id": "base",
            "description": "Original Llama-3.2-3B base model (no fine-tuning)",
            "metrics": None,
            "results": None
        }
    }
    
    # Evaluate base model first
    logging.info("Evaluating base model")
    base_metrics, base_results = evaluate_model(base_model_path, test_data, None, config)
    all_results["base"]["metrics"] = base_metrics
    all_results["base"]["results"] = base_results
    
    # Evaluate each fine-tuned model
    for model in models:
        model_id = model["model_id"]
        model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", model["path"]))
        
        logging.info(f"Evaluating model: {model_id}")
        
        try:
            metrics, results = evaluate_model(model_path, test_data, model_id, config)
            
            all_results[model_id] = {
                "model_id": model_id,
                "description": model["description"],
                "metrics": metrics,
                "results": results
            }
            
        except Exception as e:
            logging.error(f"Error evaluating model {model_id}: {e}")
            all_results[model_id] = {
                "model_id": model_id,
                "description": model["description"],
                "error": str(e)
            }
    
    # Save all results
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    logging.info(f"All evaluation results saved to {output_file}")
    
    # Print comparison table
    print("\n=== Model Comparison ===")
    print(f"{'Model':30} | {'Accuracy':10} | {'Correct/Total':15}")
    print("-" * 60)
    
    for model_id, model_data in all_results.items():
        if "metrics" in model_data and model_data["metrics"]:
            metrics = model_data["metrics"]
            print(f"{model_id:30} | {metrics['accuracy']:.2%}    | {metrics['correct']}/{metrics['total']}")
    
    return all_results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", help="Path to a specific model to evaluate")
    parser.add_argument("--model_id", help="ID of the model in the registry")
    parser.add_argument("--data_path", required=True, help="Path to test data")
    parser.add_argument("--output_dir", default="evaluation/results", help="Directory for saving results")
    parser.add_argument("--config", help="Path to evaluation config file")
    parser.add_argument("--evaluate_all", action="store_true", help="Evaluate all models in registry")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load configuration if provided
    config = None
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    
    if args.evaluate_all:
        # Evaluate all models in the registry
        evaluate_all_models(args.data_path, args.output_dir, args.config)
    else:
        # Evaluate a single model
        with open(args.data_path, 'r') as f:
            test_data = json.load(f)
        
        if not args.model_path and not args.model_id:
            logging.error("Either --model_path or --model_id must be provided")
            return
        
        # If model_id is provided but not model_path, get path from registry
        if args.model_id and not args.model_path:
            try:
                registry = ModelRegistry()
                model_info = registry.get_model(args.model_id)
                args.model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", model_info["path"]))
            except Exception as e:
                logging.error(f"Error getting model path from registry: {e}")
                return
        
        # Evaluate the model
        metrics, results = evaluate_model(args.model_path, test_data, args.model_id, config)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = args.model_id or "unknown"
        output_file = os.path.join(args.output_dir, f"{model_name}_results_{timestamp}.json")
        
        with open(output_file, 'w') as f:
            json.dump({
                "model_id": args.model_id,
                "model_path": args.model_path,
                "metrics": metrics,
                "results": results
            }, f, indent=2)
        
        logging.info(f"Results saved to {output_file}")

if __name__ == "__main__":
    main() 