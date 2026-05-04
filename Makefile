.PHONY: help install start stop restart logs clean test chunk index verify

help:
	@echo "GraphRAG Psychology Chatbot - Available Commands"
	@echo ""
	@echo "Setup & Installation:"
	@echo "  make quickstart     Run full setup (Docker + indexing)"
	@echo ""
	@echo "Docker Commands:"
	@echo "  make start          Start all services"
	@echo "  make stop           Stop all services"
	@echo "  make restart        Restart all services"
	@echo "  make logs           View logs from all services"
	@echo "  make logs-backend   View backend logs only"
	@echo "  make logs-ollama    View Ollama logs only"
	@echo ""
	@echo "Data Processing (Docker-based):"
	@echo "  make chunk          Process PDFs into chunks"
	@echo "  make index          Index chunks (requires chunks)"
	@echo "  make index-rechunk  Chunk + Index"
	@echo "  make verify         Verify graph database"
	@echo ""
	@echo "Development:"
	@echo "  make test           Run tests"
	@echo "  make clean          Clean generated files"
	@echo "  make shell          Open shell in backend container"
	@echo ""
	@echo "Examples:"
	@echo "  make quickstart     First time setup"
	@echo "  make logs-backend   Debug backend issues"

quickstart:
	@echo "Setting up GraphRAG Psychology Chatbot..."
	@echo ""
	@echo "1. Building and starting services..."
	docker-compose up -d --build
	@echo ""
	@echo "2. Waiting for services to be ready..."
	sleep 15
	@echo ""
	@echo "3. Pulling Qwen model..."
	docker exec psychology-ollama ollama pull qwen2.5:3b || true
	@echo ""
	@echo "4. Running chunking and indexing..."
	docker exec psychology-backend python -m scripts.index_data --clear-db --rechunk
	@echo ""
	@echo "✅ Setup complete!"
	@echo "  UI: http://localhost:3000"
	@echo "  API: http://localhost:8000/health"
	@echo ""
	@echo "To verify: make verify"

start:
	docker-compose up -d --build
	@echo "✅ Services started"
	@echo "  UI: http://localhost:3000"
	@echo "  API: http://localhost:8000"

stop:
	docker-compose down
	@echo "✅ Services stopped"

restart:
	docker-compose restart
	@echo "✅ Services restarted"

logs:
	docker-compose logs -f

logs-backend:
	docker-compose logs -f backend

logs-ollama:
	docker-compose logs -f ollama

logs-neo4j:
	docker-compose logs -f neo4j

logs-qdrant:
	docker-compose logs -f qdrant

chunk:
	@echo "Processing PDF documents into chunks..."
	docker exec psychology-backend python -m chunking.chunk_processor

index:
	@echo "Indexing chunks into databases..."
	docker exec psychology-backend python -m scripts.index_data

index-rechunk:
	@echo "Re-chunking and indexing..."
	docker exec psychology-backend python -m scripts.index_data --rechunk

index-clear:
	@echo "Clearing databases and re-indexing..."
	docker exec psychology-backend python -m scripts.index_data --clear-db --rechunk

verify:
	@echo "Verifying graph database..."
	docker exec psychology-backend python -m scripts.verify_graph

test:
	@echo "Running tests..."
	cd backend && python -m pytest tests/ -v

clean:
	@echo "Cleaning generated files..."
	rm -rf backend/data/
	rm -rf backend/__pycache__/
	rm -rf frontend/node_modules/
	rm -rf frontend/build/
	@echo "✅ Clean complete"

shell:
	docker-compose exec backend /bin/bash

health:
	@curl -s http://localhost:8000/health | python -m json.tool

pull-model:
	docker exec psychology-ollama ollama pull qwen2.5:3b
	@echo "✅ Model pulled"

reset-db:
	@echo "⚠️  WARNING: This will delete all database data"
	@read -p "Type 'YES' to confirm: " confirm; \
	if [ "$$confirm" = "YES" ]; then \
		docker-compose down; \
		docker volume rm psychology-chatbot_neo4j_data psychology-chatbot_qdrant_data; \
		docker-compose up -d; \
		echo "✅ Databases reset"; \
	else \
		echo "Cancelled"; \
	fi

status:
	@docker-compose ps