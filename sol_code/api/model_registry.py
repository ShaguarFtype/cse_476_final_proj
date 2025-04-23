import json
import os
from datetime import datetime

class ModelRegistry:
    def __init__(self, registry_file="models/model_registry.json"):
        self.registry_file = registry_file
        self.registry = self._load_registry()
    
    def _load_registry(self):
        """Load the model registry from disk."""
        if os.path.exists(self.registry_file):
            with open(self.registry_file, 'r') as f:
                return json.load(f)
        return {"models": []}
    
    def _save_registry(self):
        """Save the model registry to disk."""
        os.makedirs(os.path.dirname(self.registry_file), exist_ok=True)
        with open(self.registry_file, 'w') as f:
            json.dump(self.registry, f, indent=2)
    
    def register_model(self, model_id, description, training_params, dataset_info, metrics=None):
        """Register a new model."""
        for model in self.registry["models"]:
            if model["model_id"] == model_id:
                raise ValueError(f"Model ID '{model_id}' already exists")
        
        model_entry = {
            "model_id": model_id,
            "description": description,
            "training_params": training_params,
            "dataset_info": dataset_info,
            "metrics": metrics or {},
            "timestamp": datetime.now().isoformat(),
            "path": f"models/variants/{model_id}"
        }
        
        self.registry["models"].append(model_entry)
        self._save_registry()
        return model_entry
    
    def update_model_metrics(self, model_id, metrics):
        """Update metrics for an existing model."""
        for model in self.registry["models"]:
            if model["model_id"] == model_id:
                model["metrics"] = metrics
                model["last_updated"] = datetime.now().isoformat()
                self._save_registry()
                return model
        raise ValueError(f"Model ID '{model_id}' not found")
    
    def get_model(self, model_id):
        """Get a model by ID."""
        for model in self.registry["models"]:
            if model["model_id"] == model_id:
                return model
        raise ValueError(f"Model ID '{model_id}' not found")
    
    def list_models(self):
        """List all registered models."""
        return self.registry["models"] 