import json
import os
import argparse
import logging

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def format_for_instruction_tuning(example, instruction=None):
    """Convert example to instruction format"""
    default_instruction = "Solve this problem step by step, showing your work and calculations. End with #### followed by the final answer."
    
    return {
        "instruction": instruction or default_instruction,
        "input": example["question"],
        "output": example["answer"]
    }

def process_dataset(input_file, output_file, format_type="instruction", instruction=None, subset_size=None):
    """Process a dataset and save it in the specified format."""
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Load the input data
    logging.info(f"Loading data from {input_file}")
    with open(input_file, "r") as f:
        data = json.load(f)
    
    # Basic data analysis
    logging.info(f"Total examples in raw data: {len(data)}")
    
    # Apply subset limit if specified
    if subset_size and subset_size < len(data):
        logging.info(f"Using subset of {subset_size} examples")
        data = data[:subset_size]
    
    # Convert to the required format
    if format_type == "instruction":
        logging.info(f"Converting to instruction format")
        processed_data = [format_for_instruction_tuning(ex, instruction) for ex in data]
    elif format_type == "math_specialized":
        logging.info(f"Converting to math specialized format")
        processed_data = [format_for_instruction_tuning(ex, "Solve this mathematical problem step-by-step. Show all your work clearly. End with #### followed by the final answer.") for ex in data]
    elif format_type == "reasoning_enhanced":
        logging.info(f"Converting to reasoning enhanced format")
        processed_data = [format_for_instruction_tuning(ex, "Think through this problem carefully. Break it down into logical steps. Explain your reasoning for each step. End with #### followed by the final answer.") for ex in data]
    elif format_type == "general_assistant":
        logging.info(f"Converting to general assistant format")
        processed_data = [format_for_instruction_tuning(ex, "You are a helpful assistant. Solve this problem clearly and accurately. End with #### followed by the final answer.") for ex in data]
    else:
        raise ValueError(f"Unknown format type: {format_type}")
    
    # Save processed data
    logging.info(f"Saving {len(processed_data)} processed examples to {output_file}")
    with open(output_file, "w") as f:
        json.dump(processed_data, f, indent=2)
    
    return len(processed_data)

def main():
    parser = argparse.ArgumentParser(description="Process data for model training")
    parser.add_argument("--input_file", default="data/raw/dev_data.json", help="Input data file")
    parser.add_argument("--output_file", help="Output file for processed data")
    parser.add_argument("--format_type", default="instruction", 
                        choices=["instruction", "math_specialized", "reasoning_enhanced", "general_assistant"],
                        help="Format type for processing")
    parser.add_argument("--instruction", help="Custom instruction for the dataset")
    parser.add_argument("--subset_size", type=int, help="Limit dataset to this many examples")
    parser.add_argument("--process_all", action="store_true", 
                      help="Process all format types and save to default locations")
    
    args = parser.parse_args()
    
    if args.process_all:
        # Process for all model variants
        formats = ["math_specialized", "reasoning_enhanced", "general_assistant"]
        for format_type in formats:
            output_file = f"data/processed/{format_type.replace('-', '_')}_instruction.json"
            process_dataset(args.input_file, output_file, format_type, None, args.subset_size)
        
        # Also process the default dev data format
        process_dataset(args.input_file, "data/processed/devdata_instruction.json", 
                      "instruction", None, args.subset_size)
    else:
        # Process a single format
        if not args.output_file:
            args.output_file = f"data/processed/{args.format_type}_instruction.json"
        
        process_dataset(args.input_file, args.output_file, args.format_type, args.instruction, args.subset_size)

if __name__ == "__main__":
    main()