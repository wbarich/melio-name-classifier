import logging
import kserve
from model import NameClassifier
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Create the model instance
model_name = "name-classifier"
model = NameClassifier(model_name)

# Monkey-patch to add UI routes to KServe's REST server
def add_ui_routes():
    """Add custom UI routes to KServe's internal REST server"""
    from fastapi import Request
    from fastapi.responses import HTMLResponse, FileResponse
    from kserve.protocol.rest.server import RESTServer

    # Get the original create_application method
    original_create_application = RESTServer.create_application

    def patched_create_application(self):
        """Patched method that adds UI routes to the FastAPI app"""
        # Call the original method to create the app
        app = original_create_application(self)

        # Add custom UI routes
        @app.get("/", response_class=HTMLResponse, include_in_schema=False)
        @app.get("/ui", response_class=HTMLResponse, include_in_schema=False)
        async def serve_ui():
            """Serve the frontend UI"""
            logger.debug("Serving frontend UI")
            ui_path = Path(__file__).parent / "templates" / "index.html"
            if ui_path.exists():
                return FileResponse(ui_path, media_type="text/html")
            else:
                return HTMLResponse(
                    content="<h1>UI not found</h1><p>Frontend UI is not available.</p>",
                    status_code=404
                )

        @app.get("/health", include_in_schema=False)
        async def simple_health():
            """Simple health check endpoint"""
            return {"status": "healthy", "model": model_name}

        logger.info("Custom UI routes registered at / and /ui")
        return app

    # Replace the method
    RESTServer.create_application = patched_create_application

if __name__ == "__main__":
    logger.info(f"Starting KServe ModelServer for {model_name}")

    # Add UI routes before starting the server
    add_ui_routes()

    logger.info("Frontend UI will be available at http://localhost:8080/ and http://localhost:8080/ui")
    logger.info("KServe API available at http://localhost:8080/v2/models/name-classifier/infer")

    # Start the KServe ModelServer with the model
    kserve.ModelServer().start([model])
