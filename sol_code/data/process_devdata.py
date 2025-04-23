import json
import os
import argparse
import logging

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def format_for_instruction_tuning(example, instruction=None):
    """Convert example to instruction format based on source type"""
    source = example.get("source", "unknown")
    
    # Default instruction for each source type
    source_instructions = {
        "gsm8k": "Solve this math problem step by step, showing your work and calculations. End with #### followed by the final answer.",
        "math": "Solve this mathematical problem step-by-step. Show all your work clearly.",
        "hotpot_qa": "Answer the following question based on the given information.",
        "mmlu": "Answer the following multiple-choice question. Choose the most accurate response.",
        "hhh_alignment": "Respond to the following query in a helpful, harmless, and honest way.",
        "strategy_qa": "Answer the following yes/no question with 'true' or 'false' based on the given facts.",
        "trivia_qa": "Answer this trivia question with a concise and accurate response.",
        "truthful_qa": "Provide the most truthful answer to the following question."
    }
    
    # Use provided instruction or fallback to source-specific default
    default_instruction = source_instructions.get(source, "Solve this problem step by step, showing your work and calculations.")
    instruction = instruction or default_instruction
    
    # Prepare output based on source type
    if source == "hhh_alignment":
        # For hhh_alignment, get the highest rated (label=1) choice
        choices = example.get("answer", {}).get("choices", [])
        labels = example.get("answer", {}).get("labels", [])
        if choices and labels:
            for choice, label in zip(choices, labels):
                if label == 1:
                    output = choice
                    break
            else:
                output = choices[0] if choices else "Sorry, I cannot help with that."
        else:
            output = "Sorry, I cannot help with that."
    elif source == "strategy_qa":
        # For strategy_qa, format boolean answer with explanation
        answer = example.get("answer", False)
        facts = example.get("facts", "")
        output = f"{'Yes' if answer else 'No'}. Based on these facts: {facts}"
    else:
        # For all other sources, use the answer field directly
        output = example.get("answer", "")
    
    return {
        "instruction": instruction,
        "input": example.get("question", ""),
        "output": output
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
    source_counts = {}
    for item in data:
        source = item.get("source", "unknown")
        source_counts[source] = source_counts.get(source, 0) + 1
    
    logging.info(f"Source distribution: {source_counts}")
    
    # Apply subset limit if specified
    if subset_size and subset_size < len(data):
        logging.info(f"Using subset of {subset_size} examples")
        # Stratified sampling to maintain source distribution
        subset_data = []
        for source in source_counts.keys():
            source_items = [item for item in data if item.get("source") == source]
            source_proportion = source_counts[source] / len(data)
            source_subset_size = int(subset_size * source_proportion)
            subset_data.extend(source_items[:source_subset_size])
        
        # If we didn't get enough due to rounding, add more from any source
        if len(subset_data) < subset_size:
            remaining = [item for item in data if item not in subset_data]
            subset_data.extend(remaining[:subset_size - len(subset_data)])
        
        data = subset_data[:subset_size]  # Ensure we don't exceed the limit
    
    # Convert to the required format
    format_instructions = {
        "instruction": None,  # Use source-specific defaults
        "math_specialized": "Solve this mathematical problem step-by-step. Show all your work clearly. End with #### followed by the final answer.",
        "reasoning_enhanced": "Think through this problem carefully. Break it down into logical steps. Explain your reasoning for each step. End with #### followed by the final answer.",
        "general_assistant": "You are a helpful assistant. Solve this problem clearly and accurately. End with #### followed by the final answer.",
        "chat_format": "Answer the following query in a helpful, accurate, and clear manner."
    }
    
    selected_instruction = instruction or format_instructions.get(format_type)
    
    logging.info(f"Converting to {format_type} format")
    processed_data = [format_for_instruction_tuning(ex, selected_instruction) for ex in data]
    
    # Save processed data
    logging.info(f"Saving {len(processed_data)} processed examples to {output_file}")
    with open(output_file, "w") as f:
        json.dump(processed_data, f, indent=2)
    
    # Print a sample of the processed data
    if processed_data:
        logging.info(f"Sample processed example: {json.dumps(processed_data[0], indent=2)}")
    
    return len(processed_data)

def main():
    parser = argparse.ArgumentParser(description="Process data for model training")
    parser.add_argument("--input_file", default="data/raw/dev_data.json", help="Input data file")
    parser.add_argument("--output_file", help="Output file for processed data")
    parser.add_argument("--format_type", default="instruction", 
                        choices=["instruction", "math_specialized", "reasoning_enhanced", "general_assistant", "chat_format"],
                        help="Format type for processing")
    parser.add_argument("--instruction", help="Custom instruction for the dataset")
    parser.add_argument("--subset_size", type=int, help="Limit dataset to this many examples")
    parser.add_argument("--process_all", action="store_true", 
                      help="Process all format types and save to default locations")
    
    args = parser.parse_args()
    
    if args.process_all:
        # Process for all model variants
        formats = ["math_specialized", "reasoning_enhanced", "general_assistant", "chat_format"]
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