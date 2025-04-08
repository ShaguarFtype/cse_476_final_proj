from flask import Flask, request, jsonify
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import sys
import time
import requests
import threading
import logging

# Set up logging
logging.basicConfig(filename='api.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

# Global variables for model and tokenizer
model = None
tokenizer = None

def load_model():
    global model, tokenizer

    # Force offline mode
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    logging.info("Loading the base Llama-3.2-3B model...")
    model_id = "meta-llama/Llama-3.2-3B"

    # Adjust path to look for locally downloaded model
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "llama-3.2-3b-base"))

    # Check if model exists locally
    if not os.path.exists(model_path):
        logging.error(f"Local model path not found: {model_path}")
        logging.error("Please run the download_model.py script first to download the model.")
        sys.exit(1)  # Exit with error code

    # Additional verification of model files
    required_files = ["config.json", "tokenizer.json"]
    missing_files = [f for f in required_files if not os.path.exists(os.path.join(model_path, f))]

    if missing_files:
        logging.error(f"Missing required model files: {missing_files}")
        logging.error("Model download appears incomplete. Please re-run download_model.py.")
        sys.exit(1)

    logging.info(f"Loading model from local path: {model_path}")
    try:
        # Set local_files_only=True to prevent downloading from HF
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            load_in_4bit=True,
            local_files_only=True
        )
        logging.info("Model loaded successfully from local path!")
    except Exception as e:
        logging.error(f"Error loading model from local path: {e}")
        logging.error("Please ensure the model has been properly downloaded.")
        sys.exit(1)  # Exit with error code

    # Ensure we have a pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

@app.route("/generate", methods=["POST"])
def generate():
    if not model or not tokenizer:
        return jsonify({"error": "Model not loaded"}), 500

    data = request.json
    query = data.get("query", "")
    logging.info(f"Received query: {query}")

    if not query:
        return jsonify({"error": "Empty query"}), 400

    # Simple prompt format - works better with base models
    prompt = f"<human>: {query}\n<assistant>: "

    # Track response time
    start_time = time.time()

    # Generate response
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    try:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        # Get the full response
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the assistant's response
        # This handles the case where the output contains the input prompt
        if "<assistant>:" in full_response:
            response = full_response.split("<assistant>:")[-1].strip()
        else:
            # If for some reason the expected token isn't there
            response = full_response.replace(prompt, "").strip()
        
        # Clean up the response - remove any extra prompt artifacts
        response = response.replace("<human>:", "").strip()
        
        # Log the raw and processed response for debugging
        logging.info(f"Raw response: {full_response[:100]}...")
        logging.info(f"Processed response: {response[:100]}...")

    except Exception as e:
        logging.error(f"Error generating response: {e}")
        return jsonify({"error": f"Generation error: {str(e)}"}), 500

    # Calculate response time
    response_time = time.time() - start_time
    logging.info(f"Generated response in {response_time:.2f} seconds")

    return jsonify({
        "response": response,
        "response_time_seconds": response_time,
        "model": "Llama-3.2-3B (base)",
        "status": "success"
    })

@app.route("/health", methods=["GET"])
def health_check():
    if model and tokenizer:
        return jsonify({
            "status": "healthy",
            "model_loaded": True,
            "model": "Llama-3.2-3B (base)",
            "device": str(next(model.parameters()).device)
        })
    else:
        return jsonify({"status": "unhealthy", "model_loaded": False})

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Llama-3.2-3B API is running",
        "endpoints": [
            {"path": "/", "method": "GET", "description": "Home page with API information"},
            {"path": "/health", "method": "GET", "description": "Check if the API and model are running properly"},
            {"path": "/generate", "method": "POST", "description": "Generate a response for a given query"}
        ],
        "model": "Llama-3.2-3B (base)"
    })

def run_self_test():
    logging.info("Starting self-test after API initialization...")
    time.sleep(10)  # Wait for Flask to start

    try:
        # Test health endpoint
        health_response = requests.get("http://localhost:5000/health")
        health_data = health_response.json()
        logging.info(f"Health check: {health_data}")

        # Test generation endpoint
        test_query = "What is a language model?"
        generate_response = requests.post(
            "http://localhost:5000/generate",
            json={"query": test_query}
        )
        result = generate_response.json()
        logging.info(f"Test query: {test_query}")
        logging.info(f"Response: {result['response']}")
        logging.info(f"Response time: {result['response_time_seconds']} seconds")

        logging.info("Self-test completed successfully")
    except Exception as e:
        logging.error(f"Error during self-test: {e}")

if __name__ == "__main__":
    logging.info("Starting API...")
    load_model()

    # Start self-test in a separate thread
    threading.Thread(target=run_self_test).start()

    # Start Flask app
    app.run(host="0.0.0.0", port=5000)
