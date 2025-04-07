import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

def test_model():
    # Load the fine-tuned model and tokenizer
    model_path = "models/llama-3.2-3b-ft"
    
    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        load_in_4bit=True
    )
    print("Model loaded successfully!")
    
    # Test queries
    test_queries = [
        "Explain how to implement a binary search algorithm",
        "What are the advantages and disadvantages of transformer models?",
        "Write a short story about a robot learning to paint",
        "Summarize the key ideas of reinforcement learning"
    ]
    
    results = []
    
    for query in test_queries:
        print(f"\nTesting query: {query}")
        
        # Format the query as instruction
        formatted_prompt = f"### Instruction:\n{query}\n\n### Response:\n"
        
        # Generate response
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
        
        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9,
                num_return_sequences=1
            )
        end_time = time.time()
        
        # Decode and extract only the response part
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = full_response[len(formatted_prompt):]
        
        # Calculate time taken
        time_taken = end_time - start_time
        
        print(f"Response: {response[:100]}...")
        print(f"Time taken: {time_taken:.2f} seconds")
        
        results.append({
            "query": query,
            "response": response,
            "time_taken_seconds": time_taken
        })
    
    # Save results
    with open("data/test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\nTest results saved to data/test_results.json")

if __name__ == "__main__":
    test_model()
