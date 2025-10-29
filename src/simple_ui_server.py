"""
Simple UI server that serves the frontend and proxies to KServe backend.
Run this alongside the KServe model server.
"""
import logging
import os
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import httpx
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(title="Name Classifier UI")

# Set up Jinja2 templates
templates = Jinja2Templates(directory="templates")

# KServe backend URL - use environment variable or default to localhost
KSERVE_URL = os.getenv("KSERVE_URL", "http://localhost:8080")


@app.get("/", response_class=HTMLResponse)
@app.get("/ui", response_class=HTMLResponse)
async def serve_ui(request: Request):
    """Serve the frontend UI"""
    logger.debug("Serving frontend UI")
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
async def health_check():
    """Simple health check endpoint"""
    return {"status": "healthy", "service": "ui-server"}


@app.api_route("/v2/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy_to_kserve(path: str, request: Request):
    """Proxy all /v2/* requests to KServe backend"""
    logger.debug(f"Proxying request to KServe: /v2/{path}")

    async with httpx.AsyncClient() as client:
        try:
            # Forward the request to KServe
            url = f"{KSERVE_URL}/v2/{path}"

            if request.method == "GET":
                response = await client.get(url)
            elif request.method == "POST":
                body = await request.body()
                response = await client.post(
                    url,
                    content=body,
                    headers={"Content-Type": request.headers.get("content-type", "application/json")}
                )
            else:
                return JSONResponse(
                    {"error": f"Method {request.method} not supported"},
                    status_code=405
                )

            return JSONResponse(
                content=response.json(),
                status_code=response.status_code
            )
        except httpx.ConnectError:
            logger.error(f"Failed to connect to KServe at {KSERVE_URL}")
            return JSONResponse(
                {"error": "KServe backend is not available. Make sure it's running on port 8080."},
                status_code=503
            )
        except Exception as e:
            logger.error(f"Error proxying to KServe: {e}")
            return JSONResponse(
                {"error": str(e)},
                status_code=500
            )


if __name__ == "__main__":
    logger.info("Starting UI Server on http://localhost:8000")
    logger.info("Make sure KServe is running on http://localhost:8080")
    logger.info("Frontend available at: http://localhost:8000/")

    uvicorn.run(app, host="0.0.0.0", port=8000)
