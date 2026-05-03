#!/bin/bash

# Quick Start Script for GraphRAG Psychology Chatbot
# Usage: ./quickstart.sh [--rechunk] [--use-llm] [--clear-db]

set -e  # Exit on error

echo "================================================"
echo "🚀 GraphRAG Psychology Chatbot - Quick Start"
echo "================================================"
echo ""

# Check if we're in the right directory
if [ ! -f "docker-compose.yml" ]; then
    echo "❌ Error: Please run this script from the project root directory"
    exit 1
fi

# Parse arguments
RECHUNK=false
USE_LLM=false
CLEAR_DB=false

for arg in "$@"; do
    case $arg in
        --rechunk)
            RECHUNK=true
            ;;
        --use-llm)
            USE_LLM=true
            ;;
        --clear-db)
            CLEAR_DB=true
            ;;
        *)
            echo "Unknown option: $arg"
            echo "Usage: $0 [--rechunk] [--use-llm] [--clear-db]"
            exit 1
            ;;
    esac
done

# Step 1: Start Docker services
echo "📦 Step 1/4: Starting Docker services..."
echo "   (Ollama, Qdrant, Neo4j, Backend, Frontend)"
docker-compose up -d

# Wait for services to be ready
echo ""
echo "⏳ Waiting for services to be ready..."
sleep 10

# Check Ollama
echo ""
echo "🔍 Checking Ollama..."
if ! docker exec psychology-ollama ollama list > /dev/null 2>&1; then
    echo "⚠️  Ollama not ready. Pulling qwen2.5:3b model..."
    docker exec psychology-ollama ollama pull qwen2.5:3b
fi

# Step 2: Run chunking if requested
if [ "$RECHUNK" = true ]; then
    echo ""
    echo "📄 Step 2/4: Chunking PDF documents..."
    docker exec psychology-backend python -m chunking.chunk_processor
else
    echo ""
    echo "⏭️  Step 2/4: Skipping chunking (use --rechunk to re-chunk)"
fi

# Step 3: Index data
echo ""
echo "🔍 Step 3/4: Indexing data into Qdrant and Neo4j..."
INDEX_ARGS="--clear-db --rechunk"
if [ "$USE_LLM" = true ]; then
    INDEX_ARGS="$INDEX_ARGS --use-llm"
fi
docker exec psychology-backend python -m scripts.index_data $INDEX_ARGS

# Step 4: Health check
echo ""
echo "🏥 Step 4/4: Health check..."
sleep 3
if curl -s http://localhost:8000/health > /dev/null; then
    echo "✅ Backend is healthy"
else
    echo "⚠️  Backend not responding. Check logs: docker-compose logs backend"
fi

# Summary
echo ""
echo "================================================"
echo "✅ SETUP COMPLETE"
echo "================================================"
echo ""
echo "Services running:"
echo "  📱 Frontend UI:    http://localhost:3000"
echo "  🔧 Backend API:    http://localhost:8000"
echo "  📚 API Docs:       http://localhost:8000/docs"
echo "  🗄️  Neo4j Browser: http://localhost:7474 (neo4j/password)"
echo "  🔎 Qdrant UI:      http://localhost:6333/dashboard"
echo ""
echo "Next steps:"
echo "  1. Open http://localhost:3000 in your browser"
echo "  2. Start chatting with the bot"
echo ""
echo "Useful commands:"
echo "  View logs:        docker-compose logs -f [service]"
echo "  Restart backend:  docker-compose restart backend"
echo "  Stop all:         docker-compose down"
echo "  Verify graph:     docker exec psychology-backend python -m scripts.verify_graph"
echo ""