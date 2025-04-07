# api/client.py
import requests
import argparse
import time
import sys

def query_model(query, api_url="http://localhost:5000"):
    """Send a query to the model API and return the response."""
    try:
        response = requests.post(
            f"{api_url}/generate",
            json={"query": query}
        )
        
        if response.status_code != 200:
            return f"Error: {response.status_code} - {response.text}"
        
        result = response.json()
        return result["response"]
    except Exception as e:
        return f"Error: {e}"

def interactive_mode(api_url="http://localhost:5000"):
    """Run an interactive session with the model."""
    print("Interactive Mode: Type 'exit' or 'quit' to end the session")
    print("Checking API health...")
    
    try:
        health_response = requests.get(f"{api_url}/health")
        health_data = health_response.json()
        if health_data.get("status") == "healthy":
            print(f"API is healthy! Model: {health_data.get('model', 'Unknown')}")
            if 'device' in health_data:
                print(f"Running on device: {health_data['device']}")
        else:
            print("Warning: API health check failed. API may not be fully operational.")
    except Exception as e:
        print(f"Warning: Could not check API health - {e}")
    
    print("\nEnter your query below:")
    
    while True:
        try:
            user_input = input("\n> ")
            if user_input.lower() in ["exit", "quit"]:
                print("Exiting interactive mode.")
                break
                
            print("\nGenerating response...")
            start_time = time.time()
            response = query_model(user_input, api_url)
            end_time = time.time()
            
            print(f"\nResponse (generated in {end_time - start_time:.2f} seconds):")
            print(response)
            
        except KeyboardInterrupt:
            print("\nExiting interactive mode.")
            break
        except Exception as e:
            print(f"Error: {e}")

def main():
    parser = argparse.ArgumentParser(description="Client for the Llama-3.2-3B API")
    parser.add_argument("--query", "-q", type=str, help="Query to send to the model")
    parser.add_argument("--interactive", "-i", action="store_true", help="Run in interactive mode")
    parser.add_argument("--api-url", type=str, default="http://localhost:5000", help="API URL")
    
    args = parser.parse_args()
    
    if args.interactive:
        interactive_mode(args.api_url)
    elif args.query:
        print("Sending query:", args.query)
        print("\nResponse:")
        print(query_model(args.query, args.api_url))
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
