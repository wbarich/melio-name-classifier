# Local Development Guide

This guide walks you through running and testing the Name Classifier KServe server locally.

## Prerequisites

- Docker and Docker Compose installed
- Python 3.11+ (for non-Docker development)
- curl or any HTTP client for testing

## Option 1: Docker Compose (Recommended)

### Starting the Server

1. **Build and start the container** (detached mode):
   ```bash
   make docker-up
   ```

2. **View logs**:
   ```bash
   make docker-logs
   ```

3. **Check server status**:
   ```bash
   make status
   ```


### Stopping the Server

```bash
make docker-down
```

This stops and removes the containers.

### Rebuilding After Code Changes

If you modify the Python code in `src/`:

```bash
make docker-rebuild
```

This will stop containers, rebuild the image from scratch, and start them again.

### Quick Development Commands

- **Start development environment**: `make dev`
- **Check server health**: `make health`
- **Test inference**: `make test-inference`
- **View all available commands**: `make help`

---

## Option 2: Local Python Development (No Docker)

### Setup

1. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Starting the Server

```bash
cd src
python server.py
```

You should see output like:
```
2025-10-28 09:48:58,261 - model - INFO - Initialized NameClassifier model: name-classifier
2025-10-28 09:48:58,261 - __main__ - INFO - Starting KServe ModelServer for name-classifier
...
INFO:     Uvicorn running on http://0.0.0.0:8080 (Press CTRL+C to quit)
```

### Testing

Use the same curl commands as in Option 1 above.

### Stopping the Server

Press `Ctrl+C` in the terminal.

---

## Advanced Testing

### Using Python Requests

Create a test script `test_api.py`:

```python
import requests
import json

# Server URL
BASE_URL = "http://localhost:8080"

# Test health endpoint
def test_health():
    response = requests.get(f"{BASE_URL}/v2/health/live")
    print("Health check:", response.json())

# Test inference
def test_inference(name):
    url = f"{BASE_URL}/v2/models/name-classifier/infer"
    payload = {
        "inputs": [
            {
                "name": "name",
                "shape": [1],
                "datatype": "BYTES",
                "data": [name]
            }
        ]
    }
    response = requests.post(url, json=payload)
    result = response.json()
    classification = result['outputs'][0]['data'][0]
    print(f"Name: '{name}' → Classification: '{classification}'")

# Run tests
if __name__ == "__main__":
    test_health()

    # Test multiple names
    names = [
        "Bob Immerman",
        "Microsoft Corporation",
        "Harvard University",
        "Dr. Jane Smith",
        "Apple Inc.",
        "Stanford University"
    ]

    for name in names:
        test_inference(name)
```

Run it:
```bash
python test_api.py
```

### Using Postman or Insomnia

Import this request:

**POST** `http://localhost:8080/v2/models/name-classifier/infer`

Headers:
```
Content-Type: application/json
```

Body (JSON):
```json
{
  "inputs": [
    {
      "name": "name",
      "shape": [1],
      "datatype": "BYTES",
      "data": ["Your Name Here"]
    }
  ]
}
```

---

## Troubleshooting

### Port 8080 Already in Use

**Check what's using the port**:
```bash
# Linux/Mac
lsof -i :8080

# Windows
netstat -ano | findstr :8080
```

**Option 1: Stop the other process**

**Option 2: Change the port in docker-compose.yml**:
```yaml
ports:
  - "8081:8080"  # Map host port 8081 to container port 8080
```

Then access the server at `http://localhost:8081`

### Container Won't Start

**View logs**:
```bash
docker-compose logs
```

**Common issues**:
- Missing dependencies → Rebuild: `docker-compose up --build`
- Port conflict → Change port in `docker-compose.yml`

### Server Returns 500 Error

Check the logs:
```bash
docker-compose logs -f
```

Look for Python exceptions or stack traces.

### Can't Connect to Server

1. **Verify server is running**:
   ```bash
   docker ps
   ```
   You should see a container with port `0.0.0.0:8080->8080/tcp`

2. **Test with curl verbose mode**:
   ```bash
   curl -v http://localhost:8080/v2/health/live
   ```

3. **Try 127.0.0.1 instead of localhost**:
   ```bash
   curl http://127.0.0.1:8080/v2/health/live
   ```

### Docker Build is Slow

The first build downloads all dependencies (~100MB). Subsequent builds use Docker's cache and are much faster.

To force a clean rebuild:
```bash
docker-compose build --no-cache
docker-compose up
```

---

## Monitoring & Logs

### View Real-Time Logs

```bash
docker-compose logs -f name-classifier
```

### Check Container Status

```bash
docker ps
```

### Inspect Container

```bash
# Get container ID
docker ps

# Exec into container
docker exec -it <container-id> /bin/bash

# Check Python version
docker exec -it <container-id> python --version
```

### Check Resource Usage

```bash
docker stats
```

This shows CPU and memory usage in real-time.

---

## Quick Reference

| Command | Description |
|---------|-------------|
| `make docker-up` | Build and start server (detached) |
| `make docker-down` | Stop and remove containers |
| `make docker-logs` | View logs |
| `make status` | Check container status |
| `make health` | Test health endpoint |
| `make test-inference` | Test inference API |
| `make dev` | Start development environment |
| `make help` | Show all available commands |

---

## Next Steps

Once you've verified the server works locally:

1. **Modify the classifier** in `src/model.py` to add your ML logic
2. **Test your changes** by rebuilding and running tests
3. **Iterate** on the model until you're satisfied
4. **Deploy** to production (see main README.md)

For more details, see the main [README.md](README.md).
