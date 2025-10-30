# Name Classifier - KServe V2 Inference Server

A KServe-compliant HTTP API for classifying full names into one of three categories: `Person`, `Company`, or `University`.

## Project Overview

This project implements a production-ready KServe V2 inference server that classifies names. The server loads a trained champion machine learning model selected via an internal model registry.

### Architecture

- **Framework**: KServe Python SDK (v0.13.0)
- **Protocol**: KServe V2 Inference Protocol (REST)
- **Containerization**: Docker + Docker Compose
- **Python Version**: 3.11

### Classification Categories

- `Person` - Individual names (e.g., "Bob Immerman", "Dr. Jane Smith")
- `Company` - Business entities (e.g., "Microsoft Corporation", "Acme Inc.")
- `University` - Educational institutions (e.g., "Harvard University", "MIT")

## Quick Start

### Using Make (Recommended)

This project includes a Makefile for easy command execution:

```bash
# See all available commands
make help

# Install dependencies
make install

# Run tests
make test

# Start server locally
make run

# Start with Docker
make docker-up

# Stop Docker
make docker-down
```

### Option 1: Local Development (Without Docker)

1. **Install dependencies**:
   ```bash
   make install
   # or: pip install -r requirements.txt
   ```

2. **Start the server**:
   ```bash
   make run
   # or: cd src && python server.py
   ```

3. **Server will run on**: `http://localhost:8080`

### Option 2: Docker Compose (Recommended)

1. **Build and start the container**:
   ```bash
   make docker-up
   # or: docker-compose up --build
   ```

2. **Server will run on**: `http://localhost:8080`

3. **Stop the container**:
   ```bash
   make docker-down
   # or: docker-compose down
   ```

## API Usage

### Health Endpoints

```bash
# Check if server is live
curl http://localhost:8080/v2/health/live

# Check if server is ready
curl http://localhost:8080/v2/health/ready
```

### Model Metadata

```bash
curl http://localhost:8080/v2/models/name-classifier
```

### Inference (Name Classification)

```bash
curl -X POST http://localhost:8080/v2/models/name-classifier/infer \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": [
      {
        "name": "name",
        "shape": [1],
        "datatype": "BYTES",
        "data": ["Bob Immerman"]
      }
    ]
  }'
```

**Response**:
```json
{
  "model_name": "name-classifier",
  "model_version": null,
  "id": "",
  "parameters": null,
  "outputs": [
    {
      "name": "classification",
      "shape": [1],
      "datatype": "BYTES",
      "parameters": null,
      "data": ["Person"]
    }
  ]
}
```

## Project Structure

```
melio/
├── src/
│   ├── model.py           # NameClassifier class (KServe Model)
│   └── server.py          # KServe ModelServer entrypoint
├── requirements.txt       # Python dependencies
├── Dockerfile            # KServe v0.10 compliant container
├── docker-compose.yml    # Local development with Docker
├── .dockerignore        # Docker build optimization
└── README.md            # This file
```

## KServe V2 Protocol Compliance

This server implements the KServe V2 Inference Protocol:

✅ **Health Endpoints**:
- `GET /v2/health/live` - Liveness probe
- `GET /v2/health/ready` - Readiness probe

✅ **Metadata Endpoints**:
- `GET /v2/models/{model_name}` - Model metadata

✅ **Inference Endpoints**:
- `POST /v2/models/{model_name}/infer` - Synchronous inference

## Resource Constraints

The Docker container is configured to match KServe requirements:

- **CPU**: 1 vCPU (max)
- **Memory**: 2GB RAM (max)
- **Port**: 8080 (HTTP), 8081 (gRPC)

## Current Implementation

The server loads the current champion model from a lightweight registry in `models/model_registry.json`. The trained pipeline includes feature engineering (lexical features, character n-grams, optional embeddings) and a standard classifier (e.g., Logistic Regression, Random Forest, or an ensemble), chosen based on evaluation metrics.

### Model Pipeline

```python
Input (name string)
  ↓
preprocess()   # Extract name from V2 request
  ↓
predict()      # Champion ML model inference
  ↓
postprocess()  # Format to V2 response
  ↓
Output (classification)
```

## Development Notes

### Dependencies

Core dependencies (see `requirements.txt`):
- `kserve==0.13.0` - KServe Python SDK
- `numpy==1.26.4` - Numerical computing

The KServe SDK includes:
- FastAPI - Web framework
- Uvicorn - ASGI server
- Ray - Distributed computing (for scaling)

### Testing

**Test all endpoints**:
```bash
# Health
curl http://localhost:8080/v2/health/live
curl http://localhost:8080/v2/health/ready

# Metadata
curl http://localhost:8080/v2/models/name-classifier

# Inference
curl -X POST http://localhost:8080/v2/models/name-classifier/infer \
  -H "Content-Type: application/json" \
  -d '{"inputs": [{"name": "name", "shape": [1], "datatype": "BYTES", "data": ["Test Name"]}]}'
```

### Docker Build

**Build manually**:
```bash
docker build -t name-classifier:latest .
```

**Run manually**:
```bash
docker run -p 8080:8080 name-classifier:latest
```

**Check image size**:
```bash
docker images name-classifier:latest
```

## Assumptions & Trade-offs

### Assumptions
1. Input names are in English (primarily)
2. Single classification per request (batch inference not implemented)
3. No authentication/authorization required (demo/dev environment)
4. Labels in training data may have some noise (mentioned in assignment)

### Trade-offs
1. KServe SDK for native V2 protocol support and future scalability
2. CPU-only deployment keeps container lightweight

## Troubleshooting

### Port 8080 already in use
```bash
# Check what's using port 8080
lsof -i :8080

# Or change the port in docker-compose.yml:
ports:
  - "8081:8080"  # Map host port 8081 to container port 8080
```

### Container won't start
```bash
# Check logs
docker-compose logs

# Rebuild from scratch
docker-compose down
docker-compose up --build
```

### Dependencies conflict
```bash
# Use a clean virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Makefile Quick Reference

| Command | Description |
|---------|-------------|
| `make help` | Show all available commands |
| `make install` | Install Python dependencies |
| `make test` | Run all unit tests |
| `make test-cov` | Run tests with coverage report |
| `make run` | Run server locally (without Docker) |
| `make docker-up` | Build and start Docker containers |
| `make docker-down` | Stop Docker containers |
| `make docker-logs` | View container logs |
| `make clean` | Clean up generated files |
| `make health` | Check server health |
| `make test-inference` | Test the inference endpoint |

See `make help` for the complete list of commands.

## Testing

Run the test suite:

```bash
# Run all unit tests
make test

# Run with coverage
make test-cov

# Run specific test file
python3 -m pytest src/tests/test_model.py -v
```

See [TESTING.md](TESTING.md) for detailed testing documentation.

## License

This project is created for the Melio ML Engineer technical assessment.

## Next Steps

1. Expand evaluation metrics (accuracy, precision, recall)
2. Add monitoring and observability
3. Enable model versioning and A/B testing
4. Push Docker image to registry for deployment
