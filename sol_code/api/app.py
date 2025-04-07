# api/app.py

from flask import Flask, request, jsonify
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import sys
import time

app = Flask(__name__)

# Global variables for model and tokenizer
model = None
tokenizer = None

def load_model():
    global model, tokenizer
    
    print("Loading the base Llama-3.2-3B model...")
    model_id = "meta-llama/Llama-3.2-3B"
    
    # Adjust path to look for locally downloaded model
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "llama-3.2-3b-base"))
    
    # Check if model exists locally
    if os.path.exists(model_path):
        print(f"Loading model from local path: {model_path}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                load_in_4bit=True
            )
            print("Model loaded successfully from local path!")
        except Exception as e:
            print(f"Error loading model from local path: {e}")
            print("Attempting to load model directly from Hugging Face...")
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                load_in_4bit=True
            )
            print("Model loaded successfully from Hugging Face!")
    else:
        print(f"Local model path not found: {model_path}")
        print("Loading model directly from Hugging Face...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            load_in_4bit=True
        )
        print("Model loaded successfully from Hugging Face!")
    
    # Ensure we have a pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

@app.route("/generate", methods=["POST"])
def generate():
    if not model or not tokenizer:
        return jsonify({"error": "Model not loaded"}), 500
    
    data = request.json
    query = data.get("query", "")
    
    if not query:
        return jsonify({"error": "Empty query"}), 400
    
    # Format the prompt
    # Simple prompt template for the base model
    prompt = f"<human>: {query}\n<assistant>:"
    
    # Track response time
    start_time = time.time()
    
    # Generate response
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            num_return_sequences=1
        )
    
    # Calculate response time
    response_time = time.time() - start_time
    
    # Decode the full response
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract just the assistant's response
    response = full_response.split("<assistant>:")[-1].strip()
    
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

if __name__ == "__main__":
    print("Loading model...")
    load_model()
    app.run(host="0.0.0.0", port=5000)
