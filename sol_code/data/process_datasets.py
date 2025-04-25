import os
import json
import argparse
import random
from tqdm import tqdm

def format_example_for_instruction(example):
    """Format example for instruction-tuning format."""
    # For standard instruction format
    return {
        "instruction": f"Solve this problem step by step, showing your work and calculations. End with #### followed by the final answer.",
        "input": example.get("question", ""),
        "output": example.get("answer", "")
    }

def format_example_for_chat(example):
    """Format example for chat format."""
    # For chat format
    return {
        "instruction": "You are a helpful assistant.",
        "input": example.get("question", ""),
        "output": example.get("answer", "")
    }

def format_example_for_cot(example):
    """Format example with explicit chain-of-thought prompting."""
    question = example.get("question", "")
    answer = example.get("answer", "")
    
    # Extract any step-by-step reasoning from the answer
    steps = []
    final_answer = ""
    
    lines = answer.split('\n')
    for line in lines:
        if "####" in line:
            final_answer = line.strip()
        elif any(x in line.lower() for x in ["=", "+", "-", "*", "/", "let", "first", "then", "so", "therefore"]):
            steps.append(line.strip())
    
    # If we found clear reasoning steps, use them
    return {
        "instruction": "Solve this problem by breaking it down into steps. Show each calculation clearly and arrive at the final answer.",
        "input": question,
        "output": "\n".join(steps) + "\n" + final_answer if steps else answer
    }

def process_datasets(input_files, output_format, output_path, sample_size=None):
    """Process and combine multiple dataset files."""
    all_data = []
    
    # Process each input file
    for input_file in input_files:
        if not os.path.exists(input_file):
            print(f"Warning: File {input_file} does not exist. Skipping.")
            continue
            
        print(f"Processing {input_file}...")
        
        # Load the data
        with open(input_file, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                print(f"  Loaded {len(data)} examples as JSON.")
            except json.JSONDecodeError:
                # Try as JSONL
                f.seek(0)
                data = []
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
                print(f"  Loaded {len(data)} examples as JSONL.")
        
        # Process the data based on format
        formatted_data = []
        for example in tqdm(data, desc=f"Formatting {input_file}"):
            if output_format == "instruction":
                formatted = format_example_for_instruction(example)
            elif output_format == "chat":
                formatted = format_example_for_chat(example)
            elif output_format == "cot":
                formatted = format_example_for_cot(example)
            else:
                raise ValueError(f"Unknown format: {output_format}")
            
            formatted_data.append(formatted)
        
        all_data.extend(formatted_data)
    
    # Sample if needed
    if sample_size and sample_size < len(all_data):
        print(f"Sampling {sample_size} examples from {len(all_data)} total examples")
        all_data = random.sample(all_data, sample_size)
    
    # Save the output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, indent=2)
    
    print(f"Saved {len(all_data)} examples to {output_path}")
    return len(all_data)

def main():
    parser = argparse.ArgumentParser(description="Process datasets for fine-tuning")
    parser.add_argument("--input", nargs="+", required=True, help="Input dataset paths")
    parser.add_argument("--output", required=True, help="Output path")
    parser.add_argument("--format", choices=["instruction", "chat", "cot"], default="instruction", 
                        help="Output format (instruction, chat, or chain-of-thought)")
    parser.add_argument("--sample", type=int, help="Sample size (if provided)")
    args = parser.parse_args()
    
    process_datasets(args.input, args.format, args.output, args.sample)

if __name__ == "__main__":
    main()