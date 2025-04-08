# eval_model.py - Updated version

import requests
import json
import time
import logging
import argparse
import pandas as pd
import os
from datetime import datetime
import traceback

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

def test_api_health(api_url, timeout=15):
    """Test if the API is healthy and responsive"""
    try:
        start_time = time.time()
        response = requests.get(f"{api_url}/health", timeout=timeout)
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
            logging.error(f"Response: {response.text}")
            return False, None
    except Exception as e:
        logging.error(f"Error connecting to API: {e}")
        logging.error(traceback.format_exc())
        return False, None

def is_response_complete(text):
    """Check if a response appears to be complete"""
    if not text or len(text) < 20:
        return False
    
    # Check for abrupt endings
    if text.endswith((' ', ',', ';', '...')):
        return False
    
    # Check for incomplete sentences (this is a simple heuristic)
    last_sentence = text.split('.')[-1].strip()
    if len(last_sentence) > 0 and len(last_sentence) < 15:
        # Last bit is short and doesn't end with punctuation
        if not any(last_sentence.endswith(p) for p in ['.', '!', '?', '"', "'", ')', ']', '}']):
            return False
    
    return True

def generate_response(api_url, query, max_retries=5, timeout=60):
    """Generate a response from the model API with retry logic and better error handling"""
    logging.info(f"Sending query: {query}")
    logging.info(f"Query length: {len(query)} characters")

    for attempt in range(max_retries):
        try:
            start_time = time.time()
            logging.info(f"Attempt {attempt+1}: Starting request")
            
            response = requests.post(
                f"{api_url}/generate",
                json={"query": query},
                timeout=timeout
            )
            
            total_time = time.time() - start_time
            logging.info(f"Attempt {attempt+1}: Request completed in {total_time:.2f} seconds")

            if response.status_code == 200:
                try:
                    result = response.json()
                    response_text = result.get('response', '')
                    model_time = result.get('response_time_seconds', 0)
                    network_time = total_time - model_time

                    # Log response details for debugging
                    logging.info(f"Response generated successfully")
                    logging.info(f"Response length: {len(response_text)} characters")
                    logging.info(f"Model processing time: {model_time:.2f} seconds")
                    logging.info(f"Network latency: {network_time:.2f} seconds")
                    
                    # Check if response seems complete
                    if not is_response_complete(response_text):
                        logging.warning(f"Response may be incomplete: {response_text[:100]}...")
                        if attempt < max_retries - 1:
                            logging.info("Retrying to get a complete response...")
                            time.sleep(2)  # Wait before retrying
                            continue
                    
                    return True, result, total_time
                except json.JSONDecodeError:
                    logging.error(f"Failed to decode JSON response: {response.text[:500]}...")
                    if attempt < max_retries - 1:
                        time.sleep(2)  # Wait before retrying
                        continue
            else:
                logging.warning(f"Attempt {attempt+1}: API returned error {response.status_code}: {response.text}")
                if attempt < max_retries - 1:
                    time.sleep(3)  # Longer wait for server errors
                    continue
        except requests.exceptions.Timeout:
            logging.warning(f"Attempt {attempt+1}: Request timed out after {timeout} seconds")
            if attempt < max_retries - 1:
                # Increase timeout for subsequent attempts
                timeout = min(timeout * 1.5, 120)  # Cap at 2 minutes
                logging.info(f"Increasing timeout to {timeout} seconds for next attempt")
                time.sleep(2)
                continue
        except Exception as e:
            logging.warning(f"Attempt {attempt+1}: Error: {e}")
            logging.warning(traceback.format_exc())
            if attempt < max_retries - 1:
                time.sleep(3)
                continue

    logging.error(f"Failed to generate response after {max_retries} attempts")
    return False, None, 0

def evaluate_model(api_url, eval_dir):
    """Run a comprehensive evaluation of the model with improved error handling"""
    logging.info("Starting model evaluation")

    # Define evaluation tasks - expanded with more variety
    eval_tasks = [
        {
            "category": "Understanding",
            "query": "What is a language model and how does it work?",
            "expected_elements": ["prediction", "statistical", "text generation", "training", "neural"]
        },
        {
            "category": "Short_Answer",
            "query": "What is a bird?",
            "expected_elements": ["animal", "wings", "feathers", "fly", "vertebrate"]
        },
        {
            "category": "Math_Simple",
            "query": "What is 24 multiplied by 15?",
            "expected_elements": ["360"]
        },
        {
            "category": "Math_Complex",
            "query": "If a train travels at 60 miles per hour, how far will it go in 2.5 hours?",
            "expected_elements": ["150", "miles"]
        },
        {
            "category": "Coding",
            "query": "Write a Python function that checks if a number is prime.",
            "expected_elements": ["def", "prime", "return", "range", "divisor"]
        },
        {
            "category": "Summarization",
            "query": "Summarize the key features of a language model in a single paragraph.",
            "expected_elements": ["text", "predict", "training", "neural", "generation"]
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
            response_file_path = f"{eval_dir}/responses/{task['category'].lower()}_response.txt"
            try:
                with open(response_file_path, "w") as f:
                    f.write(f"Query: {task['query']}\n\n")
                    f.write(f"Response:\n{response_text}\n\n")
                    f.write(f"Response length: {len(response_text)} characters\n")
                    f.write(f"Model processing time: {response_data.get('response_time_seconds', 0):.2f} seconds\n")
                    f.write(f"Total response time: {total_time:.2f} seconds\n")
                    f.write(f"Response seems complete: {is_response_complete(response_text)}\n")
                
                logging.info(f"Response saved to {response_file_path}")
            except Exception as e:
                logging.error(f"Error saving response to file: {e}")
                logging.error(traceback.format_exc())

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
                "success_rate": f"{success_rate:.2f}",
                "seems_complete": is_response_complete(response_text)
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
                "success_rate": "0.00",
                "seems_complete": False
            }

        results.append(result)

        # Short pause between requests - longer to reduce server load
        time.sleep(3)

    return results

def calculate_perplexity_score(api_url, eval_dir):
    """Calculate a simple approximation of perplexity using a standard test set"""
    # This is a simplified approach since true perplexity calculation would require model internals
    
    logging.info("Starting perplexity estimation test")
    
    # Short validation sentences for perplexity estimation
    validation_sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning models require large amounts of training data.",
        "Neural networks have revolutionized natural language processing.",
        "The capital of France is Paris and it is known for its cuisine.",
        "Artificial intelligence aims to mimic human cognitive functions."
    ]
    
    results = []
    
    for i, sentence in enumerate(validation_sentences):
        logging.info(f"Testing sentence {i+1}/{len(validation_sentences)}")
        
        # For this simplified approach, we'll use response time as a proxy for difficulty
        # Real perplexity would need log probabilities from the model
        success, response_data, total_time = generate_response(
            api_url, 
            f"Complete this sentence: {sentence[:-1]}",  # Remove the period to test completion
            timeout=45
        )
        
        if success:
            response_text = response_data.get("response", "")
            processing_time = response_data.get("response_time_seconds", 0)
            
            # Check if completion contains the correct ending
            # This is a very simplified metric
            correct_ending = sentence.split()[-1].strip(".,!?")
            contains_correct = correct_ending.lower() in response_text.lower()
            
            result = {
                "sentence": sentence,
                "processing_time": processing_time,
                "contains_correct_ending": contains_correct
            }
            results.append(result)
            
            # Save to a file
            with open(f"{eval_dir}/perplexity_test.txt", "a") as f:
                f.write(f"Sentence: {sentence}\n")
                f.write(f"Response: {response_text[:100]}...\n")
                f.write(f"Processing time: {processing_time:.2f}s\n")
                f.write(f"Contains correct ending: {contains_correct}\n\n")
        
        time.sleep(2)
    
    # Calculate summary stats
    if results:
        avg_time = sum(r["processing_time"] for r in results) / len(results)
        correct_rate = sum(1 for r in results if r["contains_correct_ending"]) / len(results)
        
        with open(f"{eval_dir}/perplexity_summary.txt", "w") as f:
            f.write(f"Average processing time: {avg_time:.2f}s\n")
            f.write(f"Correct ending rate: {correct_rate:.2f}\n")
        
        logging.info(f"Perplexity estimation complete: avg_time={avg_time:.2f}s, correct_rate={correct_rate:.2f}")
        return {"avg_time": avg_time, "correct_rate": correct_rate}
    
    return None

def save_results(results, perplexity_results, eval_dir):
    """Save evaluation results to CSV and generate summary"""
    # Create DataFrame and save to CSV
    df = pd.DataFrame(results)
    csv_path = f"{eval_dir}/evaluation_results.csv"
    df.to_csv(csv_path, index=False)
    logging.info(f"Results saved to {csv_path}")

    # Calculate overall stats
    successful_queries = sum(1 for r in results if r["success"])
    complete_responses = sum(1 for r in results if r.get("seems_complete", False))
    avg_processing_time = sum(r["processing_time"] for r in results if r["success"]) / max(successful_queries, 1)
    avg_total_time = sum(r["total_time"] for r in results if r["success"]) / max(successful_queries, 1)
    avg_response_length = sum(r["response_length"] for r in results if r["success"]) / max(successful_queries, 1)

    # Generate summary report
    summary = {
        "total_queries": len(results),
        "successful_queries": successful_queries,
        "success_rate": f"{successful_queries/len(results):.2f}",
        "complete_responses": complete_responses,
        "completion_rate": f"{complete_responses/max(successful_queries, 1):.2f}",
        "avg_processing_time": f"{avg_processing_time:.2f}",
        "avg_total_time": f"{avg_total_time:.2f}",
        "avg_response_length": f"{avg_response_length:.2f}",
    }
    
    # Add perplexity results if available
    if perplexity_results:
        summary["perplexity_avg_time"] = f"{perplexity_results['avg_time']:.2f}"
        summary["perplexity_correct_rate"] = f"{perplexity_results['correct_rate']:.2f}"

    # Save summary to JSON
    with open(f"{eval_dir}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Log summary
    logging.info("\nEvaluation Summary:")
    logging.info(f"Total queries: {summary['total_queries']}")
    logging.info(f"Successful queries: {summary['successful_queries']}")
    logging.info(f"Success rate: {summary['success_rate']}")
    logging.info(f"Complete responses: {summary['complete_responses']}")
    logging.info(f"Completion rate: {summary['completion_rate']}")
    logging.info(f"Average processing time: {summary['avg_processing_time']} seconds")
    logging.info(f"Average total time: {summary['avg_total_time']} seconds")
    logging.info(f"Average response length: {summary['avg_response_length']} characters")
    
    if perplexity_results:
        logging.info(f"Perplexity avg time: {summary['perplexity_avg_time']} seconds")
        logging.info(f"Perplexity correct rate: {summary['perplexity_correct_rate']}")

    return summary

def main():
    parser = argparse.ArgumentParser(description="Evaluate Llama-3.2-3B API")
    parser.add_argument("--api-url", type=str, default="http://localhost:5000",
                      help="URL of the API (default: http://localhost:5000)")
    parser.add_argument("--node", type=str, help="Node name where API is running (e.g., sg039)")
    parser.add_argument("--skip-perplexity", action="store_true", help="Skip perplexity estimation")
    parser.add_argument("--timeout", type=int, default=60, help="Timeout for API requests in seconds")

    args = parser.parse_args()

    # If node is provided, update the API URL
    if args.node:
        args.api_url = f"http://{args.node}:5000"

    logging.info(f"Evaluating model API at: {args.api_url}")
    logging.info(f"Request timeout: {args.timeout} seconds")

    # Setup directory for results
    eval_dir = setup_eval_directory()

    # Test API health
    api_healthy, health_data = test_api_health(args.api_url)

    if not api_healthy:
        logging.error("API health check failed. Exiting evaluation.")
        return

    # Run evaluation
    results = evaluate_model(args.api_url, eval_dir)
    
    # Calculate perplexity if not skipped
    perplexity_results = None
    if not args.skip_perplexity:
        perplexity_results = calculate_perplexity_score(args.api_url, eval_dir)

    # Save and summarize results
    summary = save_results(results, perplexity_results, eval_dir)

    logging.info(f"Evaluation complete. Results saved to {eval_dir}/")

if __name__ == "__main__":
    main()
