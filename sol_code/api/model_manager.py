import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import logging
from .model_registry import ModelRegistry

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

class ModelManager:
    def __init__(self, base_model_path="models/base/llama-3.2-3b-base"):
        self.registry = ModelRegistry()
        self.base_model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", base_model_path))
        self.current_model = None
        self.current_tokenizer = None
        self.current_model_id = None
        
        # Force offline mode
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    
    def load_model(self, model_id=None):
        """Load a model by ID. If no ID is provided, load the base model."""
        # If model is already loaded, return it
        if self.current_model_id == model_id:
            return self.current_model, self.current_tokenizer
        
        if model_id is None or model_id == "base":
            # Load base model
            model_path = self.base_model_path
            model_type = "base"
            logging.info(f"Loading base model from {model_path}")
        else:
            # Get model from registry
            try:
                model_info = self.registry.get_model(model_id)
                model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", model_info["path"]))
                model_type = "variant"
                logging.info(f"Loading model variant '{model_id}' from {model_path}")
            except ValueError as e:
                logging.error(f"Model ID '{model_id}' not found in registry, falling back to base model")
                model_path = self.base_model_path
                model_type = "base"
                model_id = None
        
        # Check if model exists locally
        if not os.path.exists(model_path):
            logging.error(f"Local model path not found: {model_path}")
            if model_type == "variant":
                logging.info("Falling back to base model")
                model_path = self.base_model_path
                model_type = "base"
                model_id = None
            else:
                raise FileNotFoundError(f"Base model not found at {model_path}")
        
        # Unload current model to free up GPU memory
        if self.current_model is not None:
            del self.current_model
            torch.cuda.empty_cache()
            logging.info("Unloaded previous model and cleared cache")
        
        # Load the model and tokenizer
        try:
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
            
            # Ensure we have a pad token
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            self.current_model = model
            self.current_tokenizer = tokenizer
            self.current_model_id = model_id
            
            logging.info(f"Successfully loaded model: {model_id or 'base'}")
            return model, tokenizer
            
        except Exception as e:
            logging.error(f"Error loading model from {model_path}: {e}")
            if model_type == "variant":
                logging.info("Attempting to fall back to base model")
                return self.load_model(None)  # Recursively try to load base model
            raise
    
    def list_available_models(self):
        """List all available models with their descriptions."""
        models = self.registry.list_models()
        model_list = [
            {
                "model_id": model["model_id"],
                "description": model["description"],
                "metrics": model["metrics"]
            }
            for model in models
        ]
        
        # Add base model
        model_list.insert(0, {
            "model_id": "base",
            "description": "Original Llama-3.2-3B base model (no fine-tuning)",
            "metrics": {}
        })
        
        return model_list
        
    def get_current_model_info(self):
        """Get information about the currently loaded model."""
        if self.current_model is None:
            return {"error": "No model currently loaded"}
            
        if self.current_model_id is None:
            return {
                "model_id": "base",
                "description": "Original Llama-3.2-3B base model (no fine-tuning)",
                "metrics": {}
            }
        
        try:
            model_info = self.registry.get_model(self.current_model_id)
            return {
                "model_id": model_info["model_id"],
                "description": model_info["description"],
                "metrics": model_info["metrics"]
            }
        except ValueError:
            return {
                "model_id": self.current_model_id,
                "description": "Unknown model (not in registry)",
                "metrics": {}
            } 