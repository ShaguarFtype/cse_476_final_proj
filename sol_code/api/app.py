from flask import Flask, request, jsonify
import torch
import os
import sys
import time
import requests
import threading
import logging

# Import our model manager
from model_manager import ModelManager

# Set up logging
logging.basicConfig(filename='api.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

# Global variables for model manager
model_manager = None

def initialize():
    global model_manager
    logging.info("Initializing model manager...")
    model_manager = ModelManager()
    # Load the base model by default
    model_manager.load_model()
    logging.info("Model manager initialized successfully")

@app.route("/generate", methods=["POST"])
def generate():
    global model_manager
    
    if model_manager is None:
        return jsonify({"error": "Model manager not initialized"}), 500
    
    data = request.json
    query = data.get("query", "")
    model_id = data.get("model_id")  # Optional model ID to switch to
    
    logging.info(f"Received query: {query}")
    
    if not query:
        return jsonify({"error": "Empty query"}), 400
    
    # Switch model if requested
    if model_id and model_id != model_manager.current_model_id:
        try:
            model, tokenizer = model_manager.load_model(model_id)
            logging.info(f"Switched to model: {model_id}")
        except Exception as e:
            logging.error(f"Error switching models: {e}")
            return jsonify({"error": f"Error switching models: {str(e)}"}), 500
    
    # Simple prompt format - works better with base models
    prompt = f"<human>: {query}\n<assistant>: "

    # Track response time
    start_time = time.time()

    # Get the current model and tokenizer
    model = model_manager.current_model
    tokenizer = model_manager.current_tokenizer

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

    # Get current model info
    model_info = model_manager.get_current_model_info()
    model_name = model_info.get("model_id", "unknown")
    
    return jsonify({
        "response": response,
        "response_time_seconds": response_time,
        "model": model_name,
        "status": "success"
    })

@app.route("/models", methods=["GET"])
def list_models():
    """List all available models."""
    if model_manager is None:
        return jsonify({"error": "Model manager not initialized"}), 500
    
    try:
        models = model_manager.list_available_models()
        return jsonify({
            "models": models,
            "current_model": model_manager.current_model_id or "base"
        })
    except Exception as e:
        logging.error(f"Error listing models: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/models/switch", methods=["POST"])
def switch_model():
    """Switch to a different model."""
    if model_manager is None:
        return jsonify({"error": "Model manager not initialized"}), 500
    
    data = request.json
    model_id = data.get("model_id")
    
    try:
        model, tokenizer = model_manager.load_model(model_id)
        return jsonify({
            "status": "success",
            "model": model_id or "base",
            "message": f"Switched to model: {model_id or 'base'}"
        })
    except Exception as e:
        logging.error(f"Error switching models: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health_check():
    if model_manager is None or model_manager.current_model is None:
        return jsonify({"status": "unhealthy", "model_loaded": False})
    
    model_info = model_manager.get_current_model_info()
    
    return jsonify({
        "status": "healthy",
        "model_loaded": True,
        "model": model_info.get("model_id", "unknown"),
        "device": str(next(model_manager.current_model.parameters()).device)
    })

@app.route("/", methods=["GET"])
def home():
    model_info = "unknown"
    if model_manager and model_manager.current_model:
        model_info = model_manager.get_current_model_info().get("model_id", "unknown")
    
    return jsonify({
        "message": "Llama API is running",
        "endpoints": [
            {"path": "/", "method": "GET", "description": "Home page with API information"},
            {"path": "/health", "method": "GET", "description": "Check if the API and model are running properly"},
            {"path": "/generate", "method": "POST", "description": "Generate a response for a given query"},
            {"path": "/models", "method": "GET", "description": "List all available models"},
            {"path": "/models/switch", "method": "POST", "description": "Switch to a different model"}
        ],
        "current_model": model_info
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

        # Test models endpoint
        models_response = requests.get("http://localhost:5000/models")
        models_data = models_response.json()
        logging.info(f"Available models: {len(models_data.get('models', []))}")
        
        logging.info("Self-test completed successfully")
    except Exception as e:
        logging.error(f"Error during self-test: {e}")

if __name__ == "__main__":
    logging.info("Starting API...")
    initialize()

    # Start self-test in a separate thread
    threading.Thread(target=run_self_test).start()

    # Start Flask app
    app.run(host="0.0.0.0", port=5000)
