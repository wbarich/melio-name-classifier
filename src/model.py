import logging
import sys
from pathlib import Path
from typing import Dict

import kserve
from kserve import InferRequest, InferResponse, InferOutput

# Add training directory to path so we can import inference module
sys.path.insert(0, str(Path(__file__).parent / "training"))

from inference import NameClassificationModel

logger = logging.getLogger(__name__)


class NameClassifier(kserve.Model):
    """
    KServe Model for name classification using trained ML models.

    Automatically loads the champion model from the model registry.
    Supports the KServe V2 inference protocol.
    """

    def __init__(self, name: str):
        super().__init__(name)
        self.name = name
        self.ready = False
        self.classes = ["Person", "Company", "University"]

        # Load the champion model
        try:
            logger.info(f"Loading champion model for {name}...")
            self.ml_model = NameClassificationModel(use_champion=True, model_dir="models")
            self.ml_model.load()

            # Get model info
            model_info = self.ml_model.get_model_info()
            if model_info:
                logger.info(f"✅ Loaded champion: {model_info['model_name']} ({model_info['model_type']})")
                logger.info(f"   Test Accuracy: {model_info['metrics'].get('test_accuracy', 0):.2%}")
                logger.info(f"   Test F1-Score: {model_info['metrics'].get('test_f1', 0):.4f}")

            self.ready = True
            logger.info(f"Initialized NameClassifier model: {name}")

        except Exception as e:
            logger.error(f"Failed to load champion model: {e}")
            logger.error("Model will not be available for inference")
            raise

    def preprocess(self, request: InferRequest, headers: Dict[str, str] = None) -> Dict:
        """
        Extract the name string from the KServe V2 inference request.

        Expected input format:
        {
            "inputs": [
                {
                    "name": "name",
                    "shape": [1],
                    "datatype": "BYTES",
                    "data": ["Bob Immerman"]
                }
            ]
        }
        """
        logger.debug(f"Preprocessing request: {request}")

        # Extract the input data from the request
        if not request.inputs:
            raise ValueError("No inputs provided in request")

        # Get the first input (we expect only one input: the name)
        input_data = request.inputs[0]

        # Extract the name string from the data field
        if not input_data.data:
            raise ValueError("No data provided in input")

        name = input_data.data[0]
        logger.debug(f"Extracted name: {name}")

        return {"name": name}

    def predict(self, data: Dict, headers: Dict[str, str] = None) -> Dict:
        """
        Perform ML-based classification using the champion model.
        """
        name = data["name"]
        logger.debug(f"Predicting classification for: {name}")

        # Use the trained ML model for prediction
        result = self.ml_model.predict(name)
        classification = result['prediction']
        confidence = result['confidence']

        logger.debug(f"Classified '{name}' as '{classification}' (confidence: {confidence:.2%})")

        return {
            "classification": classification,
            "confidence": confidence,
            "probabilities": result['probabilities']
        }

    def postprocess(self, result: Dict, headers: Dict[str, str] = None) -> InferResponse:
        """
        Format the prediction result into KServe V2 response format.

        Expected output format:
        {
            "model_name": "name-classifier",
            "outputs": [
                {
                    "name": "classification",
                    "shape": [1],
                    "datatype": "BYTES",
                    "data": ["Person"]
                }
            ]
        }
        """
        logger.debug(f"Postprocessing result: {result}")

        classification = result["classification"]

        # Create InferOutput object
        output = InferOutput(
            name="classification",
            shape=[1],
            datatype="BYTES",
            data=[classification]
        )

        # Create and return InferResponse
        response = InferResponse(
            response_id="",  # Optional field
            model_name=self.name,
            infer_outputs=[output]
        )

        logger.debug(f"Returning response: {response}")
        return response
