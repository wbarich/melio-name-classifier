import logging
import kserve
from model import NameClassifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    # Create the model instance
    model_name = "name-classifier"
    model = NameClassifier(model_name)

    logger.info(f"Starting KServe ModelServer for {model_name}")

    # Start the KServe ModelServer
    # This will automatically:
    # - Expose V2 protocol endpoints on port 8080
    # - Handle /v2/health/live and /v2/health/ready
    # - Handle /v2/models/{model_name} metadata
    # - Handle /v2/models/{model_name}/infer for predictions
    kserve.ModelServer().start([model])
