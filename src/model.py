import logging
import random
from typing import Dict
import kserve
from kserve import InferRequest, InferResponse, InferOutput

logger = logging.getLogger(__name__)


class NameClassifier(kserve.Model):
    """
    KServe Model for name classification.

    Currently uses a naive random classifier that randomly selects from:
    - Person
    - Company
    - University

    This is a placeholder implementation to establish the infrastructure.
    The classifier will be improved with ML models in future iterations.
    """

    def __init__(self, name: str):
        super().__init__(name)
        self.name = name
        self.ready = True
        self.classes = ["Person", "Company", "University"]
        logger.info(f"Initialized NameClassifier model: {name}")

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
        Perform naive random classification.

        TODO: Replace with actual ML model (rule-based + sklearn classifier)
        """
        name = data["name"]
        logger.debug(f"Predicting classification for: {name}")

        # Naive random classification
        classification = random.choice(self.classes)
        logger.debug(f"Classified '{name}' as '{classification}' (random)")

        return {"classification": classification}

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
