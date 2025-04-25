import os
import argparse
import subprocess
import logging

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def run_command(cmd, desc=None):
    """Run a shell command and log output."""
    if desc:
        logging.info(f"Running: {desc}")
    logging.info(f"Command: {cmd}")
    
    try:
        process = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        # Stream output
        for stdout_line in iter(process.stdout.readline, ""):
            if stdout_line:
                logging.info(stdout_line.strip())
        
        for stderr_line in iter(process.stderr.readline, ""):
            if stderr_line:
                logging.error(stderr_line.strip())
        
        process.stdout.close()
        process.stderr.close()
        return_code = process.wait()
        
        if return_code != 0:
            logging.error(f"Command failed with return code {return_code}")
            return False
        
        return True
    except Exception as e:
        logging.error(f"Error running command: {e}")
        return False

def download_alpaca():
    """Download and prepare Alpaca dataset."""
    output_dir = "data/raw/alpaca.json"
    if os.path.exists(output_dir):
        logging.info(f"Alpaca dataset already exists at {output_dir}")
        return True
    
    logging.info("Downloading Alpaca dataset...")
    cmd = (
        "wget https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json "
        "-O data/raw/alpaca.json"
    )
    return run_command(cmd, "Downloading Alpaca dataset")

def download_gsm8k():
    """Download the GSM8K dataset for math problems."""
    output_dir = "data/raw/gsm8k.json"
    if os.path.exists(output_dir):
        logging.info(f"GSM8K dataset already exists at {output_dir}")
        return True
    
    # Use the Hugging Face datasets library to download GSM8K
    cmd = (
        "python -c \"from datasets import load_dataset; "
        "dataset = load_dataset('gsm8k', 'main', split='train'); "
        "import json; "
        "with open('data/raw/gsm8k.json', 'w') as f: "
        "    json.dump([{'question': item['question'], 'answer': item['answer']} "
        "              for item in dataset], f, indent=2)\""
    )
    return run_command(cmd, "Downloading GSM8K dataset")

def process_datasets():
    """Process all datasets into the required formats."""
    # Create directories
    os.makedirs("data/processed", exist_ok=True)
    
    # Process dev data for general instruction format
    cmd1 = "python data/process_datasets.py --input data/raw/dev_data.json --output data/processed/general_instruction.json --format instruction",
    cmd2 = "python data/process_datasets.py --input data/raw/dev_data.json --output data/processed/math_cot.json --format cot",
    cmd3 = "python data/process_datasets.py --input data/raw/dev_data.json --output data/processed/chat_format.json --format chat",
    
    # Process Alpaca data
    cmd4 = "python data/process_datasets.py --input data/raw/alpaca.json --output data/processed/alpaca_instruction.json --format instruction",
    
    # Process GSM8K data
    cmd5 = "python data/process_datasets.py --input data/raw/gsm8k.json --output data/processed/gsm8k_cot.json --format cot",
    
    # Create combined datasets
    cmd6 = "python data/process_datasets.py --input data/processed/general_instruction.json data/processed/alpaca_instruction.json --output data/processed/combined_general.json --format instruction --sample 5000",
    cmd7 = "python data/process_datasets.py --input data/processed/math_cot.json data/processed/gsm8k_cot.json --output data/processed/reasoning_cot.json --format cot --sample 5000",
    
    # Run all commands
    success = True
    success &= run_command(cmd1, "Processing dev data for general instruction format")
    success &= run_command(cmd2, "Processing dev data for math with chain-of-thought")
    success &= run_command(cmd3, "Processing dev data for chat format")
    success &= run_command(cmd4, "Processing Alpaca data")
    success &= run_command(cmd5, "Processing GSM8K data")
    success &= run_command(cmd6, "Creating combined general dataset")
    success &= run_command(cmd7, "Creating combined reasoning dataset")
    
    return success

def main():
    parser = argparse.ArgumentParser(description="Set up datasets for fine-tuning")
    parser.add_argument("--skip-download", action="store_true", help="Skip downloading external datasets")
    args = parser.parse_args()
    
    # Create raw data directory
    os.makedirs("data/raw", exist_ok=True)
    
    # Download datasets if needed
    if not args.skip_download:
        download_alpaca()
        download_gsm8k()
    
    # Process all datasets
    process_datasets()
    
    logging.info("Dataset setup complete!")

if __name__ == "__main__":
    main()