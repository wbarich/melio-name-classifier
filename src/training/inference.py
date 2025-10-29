#!/usr/bin/env python3
"""
KServe-compatible inference module for name classification
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, List, Union, Any, Optional
import logging
from model_registry import ModelRegistry

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NameClassificationModel:
    """
    KServe-compatible model class for name classification
    """
    
    def __init__(self, model_name: Optional[str] = None, 
                 model_dir: str = "models", 
                 use_champion: bool = True,
                 use_embeddings: bool = True):  # New parameter
        self.model_name = model_name
        self.model_dir = Path(model_dir)
        self.use_champion = use_champion
        self.use_embeddings = use_embeddings  # New
        self.model = None
        self.preprocessor = None
        self.label_mappings = None
        self.is_loaded = False
        self.registry = ModelRegistry()
        self.model_info = None
        
    def load(self):
        """Load the trained model and preprocessor"""
        try:
            if self.use_champion:
                # Load champion model from registry
                champion = self.registry.get_champion()
                if not champion:
                    raise ValueError("No champion model found in registry. Please train a model first.")
                
                self.model_info = champion
                self.model_name = champion['model_name']
                paths = champion['paths']
                
                logger.info(f"Loading champion model: {champion['model_name']} ({champion['model_type']})")
                logger.info(f"Champion metrics: Test F1={champion['metrics'].get('test_f1', 'N/A'):.4f}, "
                           f"Test Acc={champion['metrics'].get('test_accuracy', 'N/A'):.4f}")
                
            else:
                # Load specific model
                if not self.model_name:
                    raise ValueError("model_name must be specified when use_champion=False")
                
                paths = {
                    'model': str(self.model_dir / f"{self.model_name}_model.joblib"),
                    'preprocessor': str(self.model_dir / f"{self.model_name}_preprocessor.joblib"),
                    'label_mapping': str(self.model_dir / f"{self.model_name}_label_mapping.joblib")
                }
                
                logger.info(f"Loading specific model: {self.model_name}")
            
            # Load model
            self.model = joblib.load(paths['model'])
            logger.info(f"Loaded model from {paths['model']}")
            
            # Load preprocessor
            self.preprocessor = joblib.load(paths['preprocessor'])
            logger.info(f"Loaded preprocessor from {paths['preprocessor']}")
            
            # If embeddings are enabled, ensure embedding model is available
            if self.use_embeddings and hasattr(self.preprocessor, 'embedding_extractor'):
                if self.preprocessor.embedding_extractor is not None:
                    logger.info("Loading embedding model for runtime inference...")
                    # Initialize embedding model with dummy data
                    dummy_df = pd.DataFrame({'dirty_name': ['dummy']})
                    self.preprocessor.embedding_extractor.fit(dummy_df)
                    logger.info("Embedding model loaded successfully")
            
            # Load label mappings
            self.label_mappings = joblib.load(paths['label_mapping'])
            logger.info(f"Loaded label mappings from {paths['label_mapping']}")
            
            self.is_loaded = True
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def preprocess(self, input_data: Union[str, List[str], Dict]) -> np.ndarray:
        """
        Preprocess input data for prediction
        
        Args:
            input_data: Can be:
                - Single string (name)
                - List of strings (names)
                - Dictionary with 'names' key
                - Dictionary with 'instances' key (KServe format)
        
        Returns:
            Preprocessed features as numpy array
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        # Extract names from different input formats
        if isinstance(input_data, str):
            names = [input_data]
        elif isinstance(input_data, list):
            names = input_data
        elif isinstance(input_data, dict):
            if 'names' in input_data:
                names = input_data['names']
            elif 'instances' in input_data:
                # KServe format
                instances = input_data['instances']
                if isinstance(instances, list) and len(instances) > 0:
                    if isinstance(instances[0], str):
                        names = instances
                    elif isinstance(instances[0], dict) and 'name' in instances[0]:
                        names = [instance['name'] for instance in instances]
                    else:
                        raise ValueError("Invalid instances format")
                else:
                    raise ValueError("Empty instances list")
            else:
                raise ValueError("Input dictionary must contain 'names' or 'instances' key")
        else:
            raise ValueError("Input must be string, list, or dictionary")
        
        # Convert to DataFrame for preprocessing
        df = pd.DataFrame({'dirty_name': names})
        
        # Transform using preprocessor
        features = self.preprocessor.transform(df)
        
        return features
    
    def predict(self, input_data: Union[str, List[str], Dict]) -> Dict[str, Any]:
        """
        Make predictions on input data
        
        Args:
            input_data: Input data in various formats
            
        Returns:
            Dictionary containing predictions and probabilities
        """
        if not self.is_loaded:
            self.load()
        
        # Preprocess input
        features = self.preprocess(input_data)
        
        # Make predictions
        predictions = self.model.predict(features)
        probabilities = self.model.predict_proba(features)
        
        # Convert numeric predictions back to labels
        inverse_mapping = self.label_mappings['inverse_mapping']
        predicted_labels = [inverse_mapping[pred] for pred in predictions]
        
        # Get class names for probabilities
        class_names = list(self.label_mappings['label_mapping'].keys())
        
        # Format results
        results = {
            'predictions': predicted_labels,
            'probabilities': probabilities.tolist(),
            'class_names': class_names
        }
        
        # If single prediction, return simplified format
        if len(predicted_labels) == 1:
            results = {
                'prediction': predicted_labels[0],
                'probabilities': dict(zip(class_names, probabilities[0].tolist())),
                'confidence': float(max(probabilities[0]))
            }
        
        return results
    
    def predict_proba(self, input_data: Union[str, List[str], Dict]) -> np.ndarray:
        """
        Get prediction probabilities
        
        Args:
            input_data: Input data in various formats
            
        Returns:
            Probability array
        """
        if not self.is_loaded:
            self.load()
        
        features = self.preprocess(input_data)
        return self.model.predict_proba(features)
    
    def get_model_info(self) -> Optional[Dict[str, Any]]:
        """Get information about the loaded model"""
        if not self.is_loaded:
            return None
        return self.model_info
    
    def get_champion_info(self) -> Optional[Dict[str, Any]]:
        """Get information about the current champion model"""
        return self.registry.get_champion()
    
    def list_available_models(self) -> Dict[str, Dict[str, Any]]:
        """List all available models in the registry"""
        return self.registry.get_available_models()


# KServe Model Server interface
class ModelServer:
    """
    KServe Model Server interface
    """
    
    def __init__(self, model_name: Optional[str] = None, model_dir: str = "models", use_champion: bool = True, use_embeddings: bool = True):
        self.model = NameClassificationModel(model_name, model_dir, use_champion, use_embeddings)
        self.model.load()
    
    def predict(self, request: Dict) -> Dict:
        """
        KServe predict endpoint
        
        Args:
            request: KServe request format
            
        Returns:
            KServe response format
        """
        try:
            # Extract instances from request
            instances = request.get('instances', [])
            
            if not instances:
                return {'error': 'No instances provided'}
            
            # Make predictions
            results = self.model.predict({'instances': instances})
            
            # Format response for KServe
            response = {
                'predictions': results['predictions'],
                'probabilities': results['probabilities']
            }
            
            return response
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return {'error': str(e)}


def create_inference_pipeline(model_name: Optional[str] = None, model_dir: str = "models", use_champion: bool = True, use_embeddings: bool = True):
    """
    Create a complete inference pipeline
    
    Args:
        model_name: Name of the model to load (if use_champion=False)
        model_dir: Directory containing model files
        use_champion: Whether to use the champion model (default: True)
        use_embeddings: Whether to use embeddings (default: True)
        
    Returns:
        Configured inference pipeline
    """
    return ModelServer(model_name, model_dir, use_champion, use_embeddings)


# Example usage and testing
if __name__ == "__main__":
    # Test the inference pipeline with champion model
    print("Testing Champion Model Inference Pipeline")
    print("=" * 50)
    
    try:
        # Load champion model
        model = NameClassificationModel(use_champion=True)
        model.load()
        
        # Show model info
        model_info = model.get_model_info()
        if model_info:
            print(f"Loaded Model: {model_info['model_name']} ({model_info['model_type']})")
            print(f"Model ID: {model_info['model_id']}")
            print(f"Test F1-Score: {model_info['metrics'].get('test_f1', 'N/A'):.4f}")
            print(f"Test Accuracy: {model_info['metrics'].get('test_accuracy', 'N/A'):.4f}")
            print()
        
        # Test single prediction
        test_name = "John Smith"
        result = model.predict(test_name)
        print(f"Single prediction for '{test_name}': {result}")
        
        # Test batch prediction
        test_names = ["John Smith", "ACME Corp", "University of California"]
        results = model.predict(test_names)
        print(f"Batch predictions: {results}")
        
        # Test KServe format
        kserve_request = {
            'instances': [
                {'name': 'John Smith'},
                {'name': 'ACME Corp'},
                {'name': 'University of California'}
            ]
        }
        kserve_results = model.predict(kserve_request)
        print(f"KServe format results: {kserve_results}")
        
        # Show available models
        print("\nAvailable Models:")
        available_models = model.list_available_models()
        for model_id, info in available_models.items():
            champion_marker = " 🏆" if info['is_champion'] else ""
            print(f"  {model_id}: {info['model_name']} ({info['model_type']}){champion_marker}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure to train the model first by running train_model.py")
