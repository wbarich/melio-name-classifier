# KServe v0.10 compliant Dockerfile for Name Classifier
# Base image: Python 3.11 slim for minimal size
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements first for Docker layer caching
COPY requirements.txt .

# Install CPU-only PyTorch first to avoid downloading huge CUDA version
# sentence-transformers depends on PyTorch, so we install CPU version explicitly
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install Python dependencies
# --no-cache-dir reduces image size
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ .

# Copy trained models and embedding models
COPY models/ models/

# Create directories for templates and static files
RUN mkdir -p templates static

# Create directory for embedding model cache
RUN mkdir -p models/embedding_models

# Expose port 8080 (KServe default HTTP port)
EXPOSE 8080

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV SENTENCE_TRANSFORMERS_HOME=/app/models/embedding_models

# Run the KServe model server
# KServe will automatically bind to 0.0.0.0:8080
CMD ["python", "server.py"]
