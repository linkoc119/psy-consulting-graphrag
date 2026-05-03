@echo off
REM Quick Start Script for GraphRAG Psychology Chatbot (Windows)
REM Usage: quickstart.bat [--rechunk] [--use-llm] [--clear-db]

echo ================================================
echo 🚀 GraphRAG Psychology Chatbot - Quick Start
echo ================================================
echo.

REM Check if docker-compose.yml exists
if not exist "docker-compose.yml" (
    echo X Error: Please run this script from the project root directory
    exit /b 1
)

set RECHUNK=false
set USE_LLM=false
set CLEAR_DB=false

REM Parse arguments
:parse_args
if "%1"=="--rechunk" (
    set RECHUNK=true
    shift
    goto parse_args
)
if "%1"=="--use-llm" (
    set USE_LLM=true
    shift
    goto parse_args
)
if "%1"=="--clear-db" (
    set CLEAR_DB=true
    shift
    goto parse_args
)

REM Step 1: Start Docker services
echo 📦 Step 1/4: Starting Docker services...
docker-compose up -d

REM Wait for services
echo.
echo ⏳ Waiting for services to be ready...
timeout /t 10 /nobreak >nul

REM Check Ollama
echo.
echo 🔍 Checking Ollama...
docker exec psychology-ollama ollama list >nul 2>&1
if errorlevel 1 (
    echo ⚠️  Ollama not ready. Pulling qwen2.5:3b model...
    docker exec psychology-ollama ollama pull qwen2.5:3b
)

REM Step 2: Chunking
if "%RECHUNK%"=="true" (
    echo.
    echo 📄 Step 2/4: Chunking PDF documents...
    cd backend
    python -m chunking.chunk_processor
    cd ..
) else (
    echo.
    echo ⏭️  Step 2/4: Skipping chunking ^(use --rechunk to re-chunk^)
)

REM Step 3: Indexing
echo.
echo 🔍 Step 3/4: Indexing data into Qdrant and Neo4j...
cd backend
set INDEX_ARGS=
if "%USE_LLM%"=="true" set INDEX_ARGS=%INDEX_ARGS% --use-llm
if "%CLEAR_DB%"=="true" set INDEX_ARGS=%INDEX_ARGS% --clear-db
python -m scripts.index_data %INDEX_ARGS%
cd ..

REM Step 4: Health check (simplified)
echo.
echo 🏥 Step 4/4: Health check...
timeout /t 3 /nobreak >nul
curl -s http://localhost:8000/health >nul 2>&1
if errorlevel 1 (
    echo ⚠️  Backend not responding. Check logs: docker-compose logs backend
) else (
    echo ✅ Backend is healthy
)

REM Summary
echo.
echo ================================================
echo ✅ SETUP COMPLETE
echo ================================================
echo.
echo Services running:
echo   📱 Frontend UI:    http://localhost:3000
echo   🔧 Backend API:    http://localhost:8000
echo   📚 API Docs:       http://localhost:8000/docs
echo   🗄️  Neo4j Browser: http://localhost:7474 ^(neo4j/password^)
echo   🔎 Qdrant UI:      http://localhost:6333/dashboard
echo.
echo Next steps:
echo   1. Open http://localhost:3000 in your browser
echo   2. Start chatting with the bot
echo.
echo Useful commands:
echo   View logs:        docker-compose logs -f [service]
echo   Restart backend:  docker-compose restart backend
echo   Stop all:         docker-compose down
echo   Verify graph:     cd backend ^&^& python -m scripts.verify_graph
echo.
pause