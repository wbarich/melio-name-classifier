# Makefile for Name Classifier KServe Project

.PHONY: help start stop restart logs status test train train-docker clean test-aws

# Default target - make Docker the easiest option
help:
	@echo "╔════════════════════════════════════════════════════════════╗"
	@echo "║          Name Classifier - Quick Start Guide              ║"
	@echo "╚════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "🚀 QUICK START (Docker - Recommended):"
	@echo "  make start         - Start everything (build + run with Docker)"
	@echo "  make stop          - Stop all containers"
	@echo "  make restart       - Restart all containers"
	@echo "  make logs          - View live server logs"
	@echo ""
	@echo "📊 MONITORING:"
	@echo "  make status        - Check if containers are running"
	@echo ""
	@echo "🤖 TRAINING:"
	@echo "  make train-docker  - Train with embeddings in Docker (recommended)"
	@echo ""
	@echo "🌐 AWS DEPLOYMENT:"
	@echo "  make test-aws      - Test inference on AWS EC2 deployment"
	@echo ""
	@echo "📍 URLS (after running 'make start'):"
	@echo "  Frontend UI:  http://localhost:8000"
	@echo "  Backend API:  http://localhost:8080"
	@echo ""

# ============================================================
# SIMPLE DOCKER COMMANDS
# ============================================================

# Start everything with Docker (ONE COMMAND!)
start:
	@echo "🚀 Starting Name Classifier with Docker..."
	@echo ""
	docker-compose up --build -d
	@echo ""
	@echo "✅ Success! Everything is running!"
	@echo ""
	@echo "📍 Access your application:"
	@echo "   Frontend UI:  http://localhost:8000"
	@echo "   Backend API:  http://localhost:8080"
	@echo ""
	@echo "💡 Next steps:"
	@echo "   make logs      - View live logs"
	@echo "   make status    - Check if running"
	@echo "   make stop      - Stop everything"
	@echo ""

# Stop all containers
stop:
	@echo "🛑 Stopping all containers..."
	docker-compose down
	@echo "✅ All containers stopped"

# Restart everything
restart: stop start
	@echo "♻️  Restarted!"

# View logs (live streaming)
logs:
	@echo "📜 Streaming logs (Ctrl+C to exit)..."
	docker-compose logs -f

# Check status
status:
	@echo "📊 Container Status:"
	@docker ps --filter "name=name-classifier" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

# ============================================================
# TESTING & TRAINING
# ============================================================


# Install Python dependencies (run once before tests if needed)
install:
	@echo "📦 Installing Python dependencies..."
	python3 -m pip install -r requirements.txt
	@echo "✅ Dependencies installed"

# Train the model with embeddings in Docker (recommended)
train-docker:
	@echo "🚀 Training model with embeddings in Docker..."
	@echo "   This includes semantic embeddings for better accuracy!"
	@echo "   First run may take 5-8 minutes (downloading embedding model)"
	@echo ""
	@echo "📊 Progress monitoring enabled - you'll see real-time updates"
	@echo ""
	docker run --rm -v $(PWD):/app -w /app melio-name-classifier python src/training/train_model.py
	@echo ""
	@echo "✅ Training with embeddings complete!"
	@echo ""
	@echo "🎯 New model features:"
	@echo "   • 449 total features (65 original + 384 embeddings)"
	@echo "   • Semantic understanding via all-MiniLM-L6-v2"
	@echo "   • Better accuracy, especially for Company classification"
	@echo ""
	@echo "💡 To use the new model, restart the server:"
	@echo "   make restart"

# ============================================================
# AWS DEPLOYMENT TESTING
# ============================================================

# Test inference on AWS EC2 deployment
test-aws:
	@echo "🌐 Testing AWS EC2 deployment..."
	@echo ""
	@echo "📍 Endpoint: http://3.136.23.88:8000/v2/models/name-classifier/infer"
	@echo ""
	curl -X POST http://3.136.23.88:8000/v2/models/name-classifier/infer \
		-H "Content-Type: application/json" \
		-d '{"inputs": [{"name": "name", "shape": [1], "datatype": "BYTES", "data": ["Bob Immerman"]}]}' \
		--max-time 10
	@echo ""
	@echo ""

