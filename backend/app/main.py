"""
FastAPI Application for GraphRAG Psychology Chatbot
"""
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from config import settings
from .routers import chat, health
from .utils.prompts import SYSTEM_PROMPT_COUNSELING, SYSTEM_PROMPT_CRISIS

# Configure logging
logging.basicConfig(
    level=logging.INFO if not settings.DEBUG else logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan events
    - Startup: Initialize services, check connections, pre-load ML models
    - Shutdown: Clean up resources
    """
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")

    # Startup checks
    try:
        # Check Ollama connection
        from .services.llm_service import get_llm
        llm = get_llm()
        connected = await llm.check_connection()
        if not connected:
            logger.error("❌ Ollama is not available at startup")
            # Don't raise, let health check fail but start app anyway

        # Check Qdrant connection and create collection if needed
        from .services.qdrant_service import get_qdrant_service
        qdrant = get_qdrant_service()
        qdrant.create_collection(force=False)

        # Initialize Neo4j constraints
        from .services.neo4j_service import get_neo4j_service
        neo4j = await get_neo4j_service()
        await neo4j.create_constraints()

        # PRE-LOAD ML MODELS for instant first response
        logger.info("🔄 Pre-loading ML models (this may take 5-10 minutes)...")
        from .services.rag_service import GraphRAGService
        rag_service = GraphRAGService()
        await rag_service.initialize()
        app.state.rag_service = rag_service
        logger.info("✅ All services initialized and models loaded - ready for instant response!")

    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        # Continue startup, but log error

    yield

    # Shutdown
    logger.info("Shutting down application...")
    try:
        if hasattr(app.state, 'rag_service'):
            await app.state.rag_service.close()
        logger.info("✅ Services shut down cleanly")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")


# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="GraphRAG-based Psychology Chatbot for School Counseling",
    lifespan=lifespan,
    default_response_class=JSONResponse
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Middleware to ensure UTF-8 encoding in responses
@app.middleware("http")
async def utf8_middleware(request: Request, call_next):
    response = await call_next(request)
    response.headers["Content-Type"] = response.headers.get("Content-Type", "application/json") + "; charset=utf-8"
    return response


# Exception handlers
@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc) if settings.DEBUG else "An error occurred"
        }
    )


# Include routers
app.include_router(health.router)
app.include_router(chat.router)


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API info"""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running",
        "endpoints": {
            "health": "/health",
            "chat": "/chat/completion",
            "docs": "/docs"
        }
    }


@app.get("/info")
async def get_system_info():
    """Get system configuration and status"""
    return {
        "app": {
            "name": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "debug": settings.DEBUG
        },
        "llm": {
            "model": settings.LLM_MODEL,
            "base_url": settings.OLLAMA_BASE_URL
        },
        "embedding": {
            "model": settings.EMBEDDING_MODEL,
            "dimension": settings.EMBEDDING_DIMENSION
        },
        "reranker": {
            "model": settings.RERANKER_MODEL,
            "top_k": settings.RERANKER_TOP_K
        },
        "databases": {
            "qdrant": settings.QDRANT_URL,
            "neo4j": settings.NEO4J_URI
        },
        "triage": {
            "high_threshold": settings.TRIAGE_THRESHOLD_HIGH,
            "medium_threshold": settings.TRIAGE_THRESHOLD_MEDIUM
        }
    }


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level="info"
    )