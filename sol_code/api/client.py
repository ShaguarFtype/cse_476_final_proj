import requests
import argparse
import time
import sys
import json
import os
import re
import logging

# Set up logging
logging.basicConfig(
    filename='client.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    """Print an elegant, minimal ASCII art header."""
    header = """
╭───────────────────────────────────────────╮
│           LLAMA 3.2 MODEL CLIENT          │
╰───────────────────────────────────────────╯
"""
    print(header)
    logging.info("Client started")

def print_divider():
    """Print an elegant divider line."""
    print("─" * 55)

def clean_response(response, query):
    """Clean up the response text to handle common issues."""
    logging.info("Cleaning response text")
    
    # Log the original response for debugging
    logging.info(f"Original response (first 100 chars): {response[:100]}")
    
    # Remove any prompt prefixes
    prefixes = [
        "You are a helpful, accurate, and friendly assistant.",
        "You are a helpful, accurate, and friendly",
        "You are a helpful assistant",
        "<human>:", 
        "<assistant>:", 
        f"<human>: {query}\n<assistant>:",
        f"<human>: {query}",
        "system",
        "user",
        "assistant"
    ]
    
    for prefix in prefixes:
        if response.startswith(prefix):
            response = response[len(prefix):].strip()
    
    # Remove any tag artifacts that might still be there
    for token in ["<|begin_of_text|>", "<|start_header_id|>", "<|end_header_id|>", 
                  "<|eot_id|>", "<human>:", "<assistant>:"]:
        response = response.replace(token, "")
    
    # Remove the query if it somehow got included in the response
    response = response.replace(query, "").strip()
    
    # Sometimes responses start with the query again - remove it
    if response.lower().startswith(query.lower()):
        response = response[len(query):].strip()
    
    # Remove any leading/trailing whitespace and newlines
    response = response.strip()
    
    # Remove any repeated newlines (more than 2)
    response = re.sub(r'\n{3,}', '\n\n', response)
    
    # Check if the response seems to be about an unrelated topic
    # This helps catch cases where the model outputs preset content
    unrelated_topics = ["Berkshire Hathaway", "Warren Buffet", "Warren Buffett"]
    for topic in unrelated_topics:
        if topic in response and topic.lower() not in query.lower():
            logging.warning(f"Response contains unrelated topic: {topic}")
            response = "[Note: The model generated unrelated content. Please try again with a more specific query.]"
            break
    
    logging.info(f"Cleaned response, length: {len(response)} characters")
    return response

def query_model(query, api_url="http://localhost:5000", timeout=120):
    """Send a query to the model API and return the response."""
    logging.info(f"Sending query to {api_url}: '{query}'")
    try:
        print(f"Sending query to {api_url}...")
        start_time = time.time()

        response = requests.post(
            f"{api_url}/generate",
            json={"query": query},
            timeout=timeout
        )

        total_time = time.time() - start_time
        logging.info(f"Received response after {total_time:.2f} seconds")

        if response.status_code != 200:
            error_msg = f"Error: {response.status_code} - {response.text}"
            logging.error(error_msg)
            return error_msg, 0, 0

        result = response.json()
        model_time = result.get("response_time_seconds", 0)
        network_time = total_time - model_time
        
        logging.info(f"Model processing time: {model_time:.2f}s, Network time: {network_time:.2f}s")

        # Clean up the response text
        raw_response = result["response"]
        logging.info(f"Raw API response: {raw_response[:100]}...")
        model_response = clean_response(raw_response, query)
        
        # Check for specific issues
        if "Berkshire" in model_response:
            model_response = "The model generated an unrelated response about Berkshire Hathaway. This is a known issue with some base LLaMA models. Please try asking your question again with more detail."
        
        # Check if response seems invalid or too short
        if len(model_response) < 10:
            warning_msg = "Warning: Model generated a very short response"
            logging.warning(warning_msg)
            model_response = f"{model_response}\n\n[Note: The model generated a very short response. You may want to try rephrasing your query.]"
        
        return model_response, model_time, network_time
    except requests.exceptions.Timeout:
        error_msg = f"Error: Request timed out after {timeout} seconds"
        logging.error(error_msg)
        return error_msg, 0, 0
    except Exception as e:
        error_msg = f"Error: {e}"
        logging.error(error_msg)
        return error_msg, 0, 0

def check_health(api_url="http://localhost:5000", timeout=10):
    """Check if the API is healthy and responsive."""
    logging.info(f"Checking health of API at {api_url}")
    try:
        response = requests.get(f"{api_url}/health", timeout=timeout)
        if response.status_code == 200:
            health_data = response.json()
            logging.info(f"Health check successful: {health_data}")
            return True, health_data
        else:
            error_msg = {"error": f"Status code: {response.status_code}"}
            logging.error(f"Health check failed: {error_msg}")
            return False, error_msg
    except Exception as e:
        error_msg = {"error": str(e)}
        logging.error(f"Health check failed with exception: {e}")
        return False, error_msg

def suggest_rewrite(query):
    """Suggest a better version of the query if it's too vague or simple."""
    logging.info(f"Checking if query can be improved: '{query}'")
    suggestions = {
        "what is": "Could you explain in detail what",
        "who is": "Could you provide information about",
        "how to": "Could you give me step-by-step instructions on how to",
        "define": "Could you provide a comprehensive definition and explanation of",
        "tell me about": "Could you provide detailed information about"
    }
    
    for phrase, replacement in suggestions.items():
        if query.lower().startswith(phrase):
            improved = f"{replacement} {query[len(phrase):].strip()}?"
            logging.info(f"Suggested improvement: '{improved}'")
            return improved
    
    # For very short queries (less than 5 words)
    if len(query.split()) < 5 and not query.endswith("?"):
        improved = f"Could you explain in detail about {query}?"
        logging.info(f"Suggested improvement for short query: '{improved}'")
        return improved
    
    logging.info("No improvement suggestion generated")
    return None

def interactive_mode(api_url="http://localhost:5000", timeout=120):
    """Run an interactive session with the model."""
    logging.info("Starting interactive mode")
    clear_screen()
    print_header()
    print("Interactive Mode | Type 'exit' to end the session")
    print("                 | Type 'clear' to clear the screen")
    print_divider()

    try:
        is_healthy, health_data = check_health(api_url)
        if is_healthy:
            model_info = health_data.get('model', 'Unknown')
            print(f"Connected to {model_info}")
            logging.info(f"Connected to {model_info}")
            if 'device' in health_data:
                device_info = health_data['device']
                print(f"Running on {device_info}")
                logging.info(f"Running on {device_info}")
        else:
            warning_msg = f"Warning: API check failed - {health_data.get('error', 'Unknown error')}"
            print(warning_msg)
            logging.warning(warning_msg)
    except Exception as e:
        warning_msg = f"Warning: Health check failed - {e}"
        print(warning_msg)
        logging.warning(warning_msg)

    print_divider()
    print("Enter your query:")

    query_count = 0
    while True:
        try:
            user_input = input("\n> ")
            logging.info(f"User input: '{user_input}'")
            
            if user_input.lower() in ["exit", "quit"]:
                logging.info("Session ended by user")
                print("Session ended.")
                break
            
            if user_input.lower() == "clear":
                logging.info("Screen cleared by user")
                clear_screen()
                print_header()
                print("Interactive Mode | Type 'exit' to end the session")
                print("                 | Type 'clear' to clear the screen")
                print_divider()
                continue
            
            # Skip empty input
            if not user_input.strip():
                logging.info("Empty input detected, skipping")
                continue
            
            # Get a suggested rewrite for better results
            suggestion = suggest_rewrite(user_input)
            if suggestion and len(user_input) < 20:
                print(f"\nSuggestion: You might get better results with:\n> {suggestion}")
                print("\nUse this suggestion? (y/n)")
                use_suggestion = input("> ").lower()
                logging.info(f"User response to suggestion: '{use_suggestion}'")
                if use_suggestion.startswith('y'):
                    user_input = suggestion
                    print(f"\nUsing: {user_input}")
                    logging.info(f"Using suggested query: '{user_input}'")

            query_count += 1
            logging.info(f"Processing query #{query_count}")
            print("\nGenerating response...")
            response, model_time, network_time = query_model(user_input, api_url, timeout)

            print("\nResponse:")
            print_divider()
            print(response)
            print_divider()
            total_time = model_time + network_time
            print(f"Time: {model_time:.2f}s (model) + {network_time:.2f}s (network) = {total_time:.2f}s total")
            logging.info(f"Response displayed, total time: {total_time:.2f}s")

        except KeyboardInterrupt:
            logging.info("Session interrupted by user (KeyboardInterrupt)")
            print("\nSession ended.")
            break
        except Exception as e:
            error_msg = f"Error: {e}"
            print(error_msg)
            logging.error(f"Exception in interactive mode: {e}")

def main():
    parser = argparse.ArgumentParser(description="Client for the Llama-3.2-3B API")
    parser.add_argument("--query", "-q", type=str, help="Query to send to the model")
    parser.add_argument("--interactive", "-i", action="store_true", help="Run in interactive mode")
    parser.add_argument("--api-url", type=str, default="http://localhost:5000", help="API URL")
    parser.add_argument("--node", type=str, help="Node name where API is running (e.g., sg039)")
    parser.add_argument("--timeout", type=int, default=120, help="Timeout for API requests in seconds")
    parser.add_argument("--save", "-s", type=str, help="Save response to specified file")
    parser.add_argument("--log-level", type=str, default="INFO", 
                       choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                       help="Set logging level")

    args = parser.parse_args()
    
    # Set log level based on argument
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    logging.info(f"Starting client with log level: {args.log_level}")

    # If node is provided, update the API URL
    if args.node:
        args.api_url = f"http://{args.node}:5000"
        logging.info(f"Using node {args.node}, API URL set to {args.api_url}")

    if args.interactive:
        interactive_mode(args.api_url, args.timeout)
    elif args.query:
        logging.info(f"Running in single query mode with query: '{args.query}'")
        print_header()
        print(f"Query: {args.query}")
        print_divider()
        response, model_time, network_time = query_model(args.query, args.api_url, args.timeout)

        print("\nResponse:")
        print_divider()
        print(response)
        print_divider()
        total_time = model_time + network_time
        print(f"Time: {model_time:.2f}s (model) + {network_time:.2f}s (network) = {total_time:.2f}s total")
        logging.info(f"Single query completed in {total_time:.2f}s")

        # Save response to file if requested
        if args.save:
            logging.info(f"Saving response to file: {args.save}")
            try:
                with open(args.save, 'w') as f:
                    output = {
                        "query": args.query,
                        "response": response,
                        "model_time": model_time,
                        "network_time": network_time,
                        "total_time": model_time + network_time,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                    }
                    json.dump(output, f, indent=2)
                print(f"Response saved to {args.save}")
                logging.info(f"Response successfully saved to {args.save}")
            except Exception as e:
                error_msg = f"Error saving response: {e}"
                print(error_msg)
                logging.error(error_msg)
    else:
        logging.info("No query or interactive mode specified, showing help")
        print_header()
        parser.print_help()

    logging.info("Client execution completed")

if __name__ == "__main__":
    main()
