#!/usr/bin/env python3
"""
Model registry for tracking champion models
"""

import json
import joblib
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional


class ModelRegistry:
    """
    Registry for tracking champion models and their metadata
    """
    
    def __init__(self, registry_path: str = "models/model_registry.json"):
        self.registry_path = Path(registry_path)
        self.registry = self._load_registry()
    
    def _load_registry(self) -> Dict[str, Any]:
        """Load the model registry from file"""
        if self.registry_path.exists():
            try:
                with open(self.registry_path, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                return {"models": {}, "champion": None, "last_updated": None}
        return {"models": {}, "champion": None, "last_updated": None}
    
    def _save_registry(self):
        """Save the model registry to file"""
        self.registry_path.parent.mkdir(exist_ok=True)
        with open(self.registry_path, 'w') as f:
            json.dump(self.registry, f, indent=2)
    
    def register_model(self, model_name: str, model_type: str, metrics: Dict[str, Any], 
                      model_path: str, preprocessor_path: str, label_mapping_path: str):
        """Register a new model in the registry"""
        model_id = f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        model_info = {
            "model_id": model_id,
            "model_name": model_name,
            "model_type": model_type,
            "metrics": metrics,
            "paths": {
                "model": model_path,
                "preprocessor": preprocessor_path,
                "label_mapping": label_mapping_path
            },
            "created_at": datetime.now().isoformat(),
            "is_champion": False
        }
        
        self.registry["models"][model_id] = model_info
        self.registry["last_updated"] = datetime.now().isoformat()
        
        self._save_registry()
        return model_id
    
    def set_champion(self, model_id: str, reason: str = "Best test F1-score"):
        """Set a model as the champion"""
        if model_id not in self.registry["models"]:
            raise ValueError(f"Model {model_id} not found in registry")
        
        # Remove champion status from all models
        for mid, model_info in self.registry["models"].items():
            model_info["is_champion"] = False
        
        # Set new champion
        self.registry["models"][model_id]["is_champion"] = True
        self.registry["champion"] = {
            "model_id": model_id,
            "reason": reason,
            "set_at": datetime.now().isoformat()
        }
        
        self._save_registry()
    
    def get_champion(self) -> Optional[Dict[str, Any]]:
        """Get the current champion model info"""
        if not self.registry["champion"]:
            return None
        
        champion_id = self.registry["champion"]["model_id"]
        if champion_id in self.registry["models"]:
            return self.registry["models"][champion_id]
        return None
    
    def get_champion_paths(self) -> Optional[Dict[str, str]]:
        """Get the file paths for the champion model"""
        champion = self.get_champion()
        if champion:
            return champion["paths"]
        return None
    
    def list_models(self) -> Dict[str, Any]:
        """List all registered models"""
        return self.registry["models"]
    
    def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific model"""
        return self.registry["models"].get(model_id)
    
    def validate_model_files(self, model_id: str) -> bool:
        """Validate that all model files exist"""
        model_info = self.get_model_info(model_id)
        if not model_info:
            return False
        
        paths = model_info["paths"]
        for path_type, path in paths.items():
            if not Path(path).exists():
                return False
        return True
    
    def get_available_models(self) -> Dict[str, Dict[str, Any]]:
        """Get all models with valid files"""
        available = {}
        for model_id, model_info in self.registry["models"].items():
            if self.validate_model_files(model_id):
                available[model_id] = model_info
        return available


def create_model_registry():
    """Create a new model registry instance"""
    return ModelRegistry()


def register_champion_model(model_name: str, model_type: str, metrics: Dict[str, Any],
                          model_dir: str = "models") -> str:
    """
    Register a champion model in the registry
    
    Args:
        model_name: Name of the model (e.g., "logistic_regression")
        model_type: Type of the model (e.g., "Logistic Regression")
        metrics: Dictionary of model metrics
        model_dir: Directory containing model files
    
    Returns:
        Model ID of the registered model
    """
    registry = ModelRegistry()
    
    # Construct file paths
    model_path = f"{model_dir}/{model_name}_model.joblib"
    preprocessor_path = f"{model_dir}/{model_name}_preprocessor.joblib"
    label_mapping_path = f"{model_dir}/{model_name}_label_mapping.joblib"
    
    # Register the model
    model_id = registry.register_model(
        model_name=model_name,
        model_type=model_type,
        metrics=metrics,
        model_path=model_path,
        preprocessor_path=preprocessor_path,
        label_mapping_path=label_mapping_path
    )
    
    # Set as champion
    registry.set_champion(model_id, "Best test F1-score")
    
    return model_id


if __name__ == "__main__":
    # Test the registry
    registry = ModelRegistry()
    
    # Example usage
    print("Model Registry Test")
    print("=" * 50)
    
    # List available models
    available = registry.get_available_models()
    print(f"Available models: {len(available)}")
    
    for model_id, info in available.items():
        print(f"  {model_id}: {info['model_name']} ({info['model_type']})")
        if info['is_champion']:
            print(f"    🏆 CHAMPION")
    
    # Get champion info
    champion = registry.get_champion()
    if champion:
        print(f"\nCurrent Champion: {champion['model_name']} ({champion['model_type']})")
        print(f"Champion metrics: {champion['metrics']}")
    else:
        print("\nNo champion model set")
