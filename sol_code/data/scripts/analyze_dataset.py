#!/usr/bin/env python3
import json
import os
from collections import Counter
import argparse

def analyze_dataset(file_path):
    """
    Analyze the dataset to identify different sources and provide examples.
    """
    print(f"Analyzing dataset: {file_path}")
    
    # Load the JSON data
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    print(f"Total number of examples: {len(data)}")
    
    # Check the structure of the first example
    if data:
        print("\nFirst example structure:")
        first_example = data[0]
        print(json.dumps(first_example, indent=2))
        print(f"\nKeys available: {list(first_example.keys())}")
    
    # Count sources
    sources = Counter()
    for example in data:
        if 'source' in example:
            sources[example['source']] += 1
        else:
            sources['unknown'] += 1
    
    print("\nSources distribution:")
    for source, count in sources.items():
        print(f"  - {source}: {count} examples ({count/len(data)*100:.1f}%)")
    
    # Get one example from each source
    print("\nExample from each source:")
    examples_by_source = {}
    for example in data:
        source = example.get('source', 'unknown')
        if source not in examples_by_source:
            examples_by_source[source] = example
    
    for source, example in examples_by_source.items():
        print(f"\n=== Source: {source} ===")
        print(json.dumps(example, indent=2))
        print("=" * 50)

def main():
    parser = argparse.ArgumentParser(description='Analyze a dataset JSON file')
    parser.add_argument('--file', required=True, help='Path to the JSON dataset file')
    args = parser.parse_args()
    
    if not os.path.exists(args.file):
        print(f"Error: File {args.file} does not exist")
        return
    
    analyze_dataset(args.file)

if __name__ == "__main__":
    main()