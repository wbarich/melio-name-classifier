# Makefile for Name Classifier KServe Project

.PHONY: help install test run build clean lint format docker-up docker-down docker-logs health test-inference dev prod-check install-dev status restart cycle info

# Default target
help:
	@echo "Name Classifier - Available Commands"
	@echo "===================================="
	@echo ""
	@echo "Development:"
	@echo "  make install       - Install Python dependencies"
	@echo "  make run           - Run the server locally (without Docker)"
	@echo "  make test          - Run all tests (unit + integration with server)"
	@echo "  make dev           - Start development environment (docker-up + helpful info)"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-up     - Build and start Docker containers (detached)"
	@echo "  make docker-down   - Stop and remove Docker containers"
	@echo "  make docker-logs   - View Docker container logs"
	@echo "  make docker-rebuild- Rebuild Docker image from scratch"
	@echo "  make build         - Build Docker image only"
	@echo ""
	@echo "Testing & Health:"
	@echo "  make health        - Check server health"
	@echo "  make test-inference- Test inference endpoint"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint          - Run code linting"
	@echo "  make format        - Format code with black"
	@echo "  make clean         - Clean up generated files"
	@echo "  make prod-check    - Run production checks (test + lint)"
	@echo ""
	@echo "Utilities:"
	@echo "  make status        - Check Docker container status"
	@echo "  make restart       - Restart the server"
	@echo "  make cycle         - Full development cycle (clean + rebuild + test)"
	@echo "  make info          - Show server information and available commands"
	@echo ""

# Installation
install:
	@echo "Installing dependencies..."
	pip install -r requirements.txt
	@echo "✅ Dependencies installed"

# Running the server
run:
	@echo "Starting KServe server locally..."
	cd src && python3 server.py

# Docker commands
docker-up:
	@echo "Building and starting Docker containers..."
	docker-compose up --build -d
	@echo "✅ Server running at http://localhost:8080"
	@echo "Run 'make docker-logs' to view logs"

docker-down:
	@echo "Stopping Docker containers..."
	docker-compose down
	@echo "✅ Containers stopped"

docker-logs:
	@echo "Streaming Docker logs (Ctrl+C to exit)..."
	docker-compose logs -f

docker-rebuild:
	@echo "Rebuilding Docker image from scratch..."
	docker-compose down
	docker-compose build --no-cache
	docker-compose up -d
	@echo "✅ Image rebuilt and containers started"

build:
	@echo "Building Docker image..."
	docker-compose build
	@echo "✅ Build complete"

# Testing
test:
	@echo "Running all tests (unit + integration)..."
	@echo "Starting server in background..."
	docker-compose up -d
	@sleep 5
	@echo "Running unit tests..."
	python3 -m pytest src/tests/test_model.py src/tests/test_server.py -v
	@echo "Running API integration tests..."
	python3 -m pytest src/tests/test_api.py -v
	@echo "Stopping server..."
	docker-compose down
	@echo "✅ All tests complete"

# Code quality
lint:
	@echo "Running linting checks..."
	@if command -v ruff > /dev/null; then \
		ruff check src/; \
	else \
		echo "⚠️  ruff not installed. Install with: pip install ruff"; \
	fi

format:
	@echo "Formatting code with black..."
	@if command -v black > /dev/null; then \
		black src/; \
	else \
		echo "⚠️  black not installed. Install with: pip install black"; \
	fi

# Cleaning
clean:
	@echo "Cleaning up generated files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".coverage" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "✅ Cleanup complete"


# Health check
health:
	@echo "Checking server health..."
	@curl -s http://localhost:8080/v2/health/live | jq . || echo "❌ Server not responding"

# Test inference
test-inference:
	@echo "Testing inference endpoint..."
	@curl -s -X POST http://localhost:8080/v2/models/name-classifier/infer \
		-H "Content-Type: application/json" \
		-d '{"inputs": [{"name": "name", "shape": [1], "datatype": "BYTES", "data": ["Bob Immerman"]}]}' \
		| jq . || echo "❌ Inference failed"

# Development workflow
dev: docker-up
	@echo "🚀 Development environment started!"
	@echo "📡 Server: http://localhost:8080"
	@echo "📊 Health: http://localhost:8080/v2/health/live"
	@echo "🔬 Model: http://localhost:8080/v2/models/name-classifier/infer"
	@echo ""
	@echo "Next steps:"
	@echo "  make test          - Run all tests"
	@echo "  make docker-logs   - View server logs"
	@echo "  make health        - Check server status"
	@echo "  make test-inference- Test the API"
	@echo "  make docker-down   - Stop the server"

# Production workflow
prod-check: test lint
	@echo "✅ Production checks passed"
	@echo "Ready to deploy!"

# Install dev dependencies
install-dev: install
	@echo "Installing development dependencies..."
	pip install black ruff pytest-watch
	@echo "✅ Dev dependencies installed"

# Additional useful commands
status:
	@echo "Checking Docker container status..."
	@docker ps --filter "name=name-classifier" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

restart: docker-down docker-up
	@echo "✅ Server restarted"

# Quick development cycle
cycle: clean docker-rebuild test
	@echo "✅ Full development cycle complete"

# Show server info
info:
	@echo "Name Classifier Server Information"
	@echo "=================================="
	@echo "Server URL: http://localhost:8080"
	@echo "Health Check: http://localhost:8080/v2/health/live"
	@echo "Model Endpoint: http://localhost:8080/v2/models/name-classifier/infer"
	@echo ""
	@echo "Available Commands:"
	@echo "  make test          - Run all tests"
	@echo "  make health        - Check if server is running"
	@echo "  make test-inference- Test the classification API"
	@echo "  make docker-logs   - View server logs"
	@echo "  make docker-down   - Stop the server"
