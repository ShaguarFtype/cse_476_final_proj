# api/eval_model.py

import requests
import json
import time
import logging
import argparse
import pandas as pd
import os
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("model_eval.log"),
        logging.StreamHandler()
    ]
)

def setup_eval_directory():
    """Create directory structure for evaluation results"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_dir = f"eval_results_{timestamp}"
    os.makedirs(eval_dir, exist_ok=True)
    os.makedirs(f"{eval_dir}/responses", exist_ok=True)
    return eval_dir

def test_api_health(api_url):
    """Test if the API is healthy and responsive"""
    try:
        start_time = time.time()
        response = requests.get(f"{api_url}/health", timeout=10)
        latency = time.time() - start_time
        
        if response.status_code == 200:
            health_data = response.json()
            logging.info(f"API Health: {health_data['status']}")
            logging.info(f"Model Loaded: {health_data.get('model_loaded', 'Unknown')}")
            logging.info(f"Device: {health_data.get('device', 'Unknown')}")
            logging.info(f"Health Check Latency: {latency:.4f} seconds")
            return True, health_data
        else:
            logging.error(f"Health check failed with status code: {response.status_code}")
            return False, None
    except Exception as e:
        logging.error(f"Error connecting to API: {e}")
        return False, None

def generate_response(api_url, query, max_retries=3, timeout=30):
    """Generate a response from the model API with retry logic"""
    logging.info(f"Sending query: {query}")
    
    for attempt in range(max_retries):
        try:
            start_time = time.time()
            response = requests.post(
                f"{api_url}/generate",
                json={"query": query},
                timeout=timeout
            )
            total_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                model_time = result.get('response_time_seconds', 0)
                network_time = total_time - model_time
                
                logging.info(f"Response generated successfully")
                logging.info(f"Model processing time: {model_time:.2f} seconds")
                logging.info(f"Network latency: {network_time:.2f} seconds")
                logging.info(f"Total response time: {total_time:.2f} seconds")
                
                return True, result, total_time
            else:
                logging.warning(f"Attempt {attempt+1}: API returned error {response.status_code}: {response.text}")
                time.sleep(2)  # Wait before retrying
        except requests.exceptions.Timeout:
            logging.warning(f"Attempt {attempt+1}: Request timed out after {timeout} seconds")
            time.sleep(2)  # Wait before retrying
        except Exception as e:
            logging.warning(f"Attempt {attempt+1}: Error: {e}")
            time.sleep(2)  # Wait before retrying
    
    logging.error(f"Failed to generate response after {max_retries} attempts")
    return False, None, 0

def evaluate_model(api_url, eval_dir):
    """Run a comprehensive evaluation of the model"""
    logging.info("Starting model evaluation")
    
    # Define evaluation tasks
    eval_tasks = [
        {
            "category": "Understanding",
            "query": "What is a language model and how does it work?",
            "expected_elements": ["prediction", "statistical", "text generation"]
        },
        {
            "category": "Math",
            "query": "What is 24 multiplied by 15?",
            "expected_elements": ["360"]
        },
        {
            "category": "Reasoning",
            "query": "If a train travels at 60 miles per hour, how far will it go in 2.5 hours?",
            "expected_elements": ["150", "miles"]
        },
        {
            "category": "Coding",
            "query": "Write a Python function that checks if a number is prime.",
            "expected_elements": ["def", "prime", "return"]
        },
        {
            "category": "Summarization",
            "query": "Summarize the key features of a language model in a single paragraph.",
            "expected_elements": ["text", "predict", "training"]
        }
    ]
    
    results = []
    
    # Test each task
    for i, task in enumerate(eval_tasks):
        logging.info(f"\n[{i+1}/{len(eval_tasks)}] Testing {task['category']} capability")
        
        success, response_data, total_time = generate_response(api_url, task["query"])
        
        if success:
            response_text = response_data.get("response", "")
            
            # Save detailed response to a file
            with open(f"{eval_dir}/responses/{task['category'].lower()}_response.txt", "w") as f:
                f.write(f"Query: {task['query']}\n\n")
                f.write(f"Response:\n{response_text}\n\n")
                f.write(f"Model processing time: {response_data.get('response_time_seconds', 0):.2f} seconds\n")
                f.write(f"Total response time: {total_time:.2f} seconds\n")
            
            # Check for expected elements
            elements_found = sum(1 for elem in task["expected_elements"] if elem.lower() in response_text.lower())
            success_rate = elements_found / len(task["expected_elements"]) if task["expected_elements"] else 0
            
            result = {
                "category": task["category"],
                "query": task["query"],
                "success": success,
                "response_length": len(response_text),
                "processing_time": response_data.get("response_time_seconds", 0),
                "total_time": total_time,
                "expected_elements_found": f"{elements_found}/{len(task['expected_elements'])}",
                "success_rate": f"{success_rate:.2f}"
            }
        else:
            result = {
                "category": task["category"],
                "query": task["query"],
                "success": False,
                "response_length": 0,
                "processing_time": 0,
                "total_time": 0,
                "expected_elements_found": "0/0",
                "success_rate": "0.00"
            }
        
        results.append(result)
        
        # Short pause between requests
        time.sleep(1)
    
    return results

def save_results(results, eval_dir):
    """Save evaluation results to CSV and generate summary"""
    # Create DataFrame and save to CSV
    df = pd.DataFrame(results)
    csv_path = f"{eval_dir}/evaluation_results.csv"
    df.to_csv(csv_path, index=False)
    logging.info(f"Results saved to {csv_path}")
    
    # Calculate overall stats
    successful_queries = sum(1 for r in results if r["success"])
    avg_processing_time = sum(r["processing_time"] for r in results if r["success"]) / max(successful_queries, 1)
    avg_total_time = sum(r["total_time"] for r in results if r["success"]) / max(successful_queries, 1)
    
    # Generate summary report
    summary = {
        "total_queries": len(results),
        "successful_queries": successful_queries,
        "success_rate": f"{successful_queries/len(results):.2f}",
        "avg_processing_time": f"{avg_processing_time:.2f}",
        "avg_total_time": f"{avg_total_time:.2f}"
    }
    
    # Save summary to JSON
    with open(f"{eval_dir}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    # Log summary
    logging.info("\nEvaluation Summary:")
    logging.info(f"Total queries: {summary['total_queries']}")
    logging.info(f"Successful queries: {summary['successful_queries']}")
    logging.info(f"Success rate: {summary['success_rate']}")
    logging.info(f"Average processing time: {summary['avg_processing_time']} seconds")
    logging.info(f"Average total time: {summary['avg_total_time']} seconds")
    
    return summary

def main():
    parser = argparse.ArgumentParser(description="Evaluate Llama-3.2-3B API")
    parser.add_argument("--api-url", type=str, default="http://localhost:5000", 
                      help="URL of the API (default: http://localhost:5000)")
    parser.add_argument("--node", type=str, help="Node name where API is running (e.g., sg039)")
    
    args = parser.parse_args()
    
    # If node is provided, update the API URL
    if args.node:
        args.api_url = f"http://{args.node}:5000"
    
    logging.info(f"Evaluating model API at: {args.api_url}")
    
    # Setup directory for results
    eval_dir = setup_eval_directory()
    
    # Test API health
    api_healthy, health_data = test_api_health(args.api_url)
    
    if not api_healthy:
        logging.error("API health check failed. Exiting evaluation.")
        return
    
    # Run evaluation
    results = evaluate_model(args.api_url, eval_dir)
    
    # Save and summarize results
    summary = save_results(results, eval_dir)
    
    logging.info(f"Evaluation complete. Results saved to {eval_dir}/")

if __name__ == "__main__":
    main()
