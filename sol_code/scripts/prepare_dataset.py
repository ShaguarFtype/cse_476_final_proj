import datasets
import json
import os

def prepare_datasets():
    # Directory for saving datasets
    os.makedirs("data/processed", exist_ok=True)
    
    # Load datasets from Hugging Face
    alpaca_dataset = datasets.load_dataset("tatsu-lab/alpaca")
    
    # Process datasets into a unified format
    processed_data = []
    
    # Process Alpaca data
    for item in alpaca_dataset["train"]:
        processed_data.append({
            "instruction": item["instruction"],
            "input": item["input"] if item["input"] else "",
            "output": item["output"],
            "source": "alpaca"
        })
    
    # Save processed data
    with open("data/processed/instruction_dataset.json", "w") as f:
        json.dump(processed_data, f, indent=2)
    
    print(f"Saved {len(processed_data)} processed examples")

if __name__ == "__main__":
    prepare_datasets()
