# api/test_api.py
import requests
import json
import time

def test_api():
    base_url = "http://localhost:5000"
    
    # Health check
    print("Checking API health...")
    try:
        response = requests.get(f"{base_url}/health")
        health_data = response.json()
        print(f"Health status: {health_data['status']}")
        print(f"Model loaded: {health_data['model_loaded']}")
        if 'device' in health_data:
            print(f"Device: {health_data['device']}")
    except Exception as e:
        print(f"Error checking health: {e}")
        return
    
    # Test queries
    test_queries = [
        "What are neural networks?",
        "Explain how transformers work in machine learning",
        "Write a simple Python function to calculate the factorial of a number",
        "What's the capital of France?"
    ]
    
    print("\nTesting response generation...")
    for i, query in enumerate(test_queries):
        print(f"\nQuery {i+1}: {query}")
        try:
            start_time = time.time()
            response = requests.post(
                f"{base_url}/generate",
                json={"query": query}
            )
            total_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                print(f"Response: {result['response'][:100]}...")
                print(f"Model processing time: {result['response_time_seconds']:.2f} seconds")
                print(f"Total round-trip time: {total_time:.2f} seconds")
            else:
                print(f"Error: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"Error making request: {e}")
    
    print("\nAPI test completed!")

if __name__ == "__main__":
    test_api()
