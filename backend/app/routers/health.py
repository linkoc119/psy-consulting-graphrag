"""
Health check router - monitor system status
"""
import time
import logging
from typing import List, Dict
from fastapi import APIRouter, Depends, Request
from ..models.schemas import HealthCheckRequest, HealthCheckResponse, ServiceStatus

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/health", tags=["health"])


async def check_ollama() -> ServiceStatus:
    """Check Ollama service"""
    try:
        from ..services.llm_service import get_llm
        llm = get_llm()
        start = time.time()
        is_healthy = await llm.check_connection()
        response_time = (time.time() - start) * 1000
        
        return ServiceStatus(
            name="ollama",
            status="healthy" if is_healthy else "unhealthy",
            response_time_ms=response_time if is_healthy else None,
            details={"model": llm.model} if is_healthy else {"error": "Model not available"}
        )
    except Exception as e:
        logger.error(f"Ollama health check failed: {e}")
        return ServiceStatus(
            name="ollama",
            status="error",
            details={"error": str(e)}
        )


async def check_qdrant() -> ServiceStatus:
    """Check Qdrant service"""
    try:
        from ..services.qdrant_service import get_qdrant_service
        qdrant = get_qdrant_service()
        start = time.time()
        info = qdrant.get_collection_info()
        response_time = (time.time() - start) * 1000
        
        status = "healthy" if info else "unhealthy"
        return ServiceStatus(
            name="qdrant",
            status=status,
            response_time_ms=response_time,
            details=info
        )
    except Exception as e:
        logger.error(f"Qdrant health check failed: {e}")
        return ServiceStatus(
            name="qdrant",
            status="error",
            details={"error": str(e)}
        )


async def check_neo4j() -> ServiceStatus:
    """Check Neo4j service"""
    try:
        from ..services.neo4j_service import get_neo4j_service
        neo4j = await get_neo4j_service()
        start = time.time()
        # Simple query to check connection
        count = await neo4j.count_nodes_by_label("BenhLy")
        response_time = (time.time() - start) * 1000
        
        return ServiceStatus(
            name="neo4j",
            status="healthy",
            response_time_ms=response_time,
            details={"benhly_count": count}
        )
    except Exception as e:
        logger.error(f"Neo4j health check failed: {e}")
        return ServiceStatus(
            name="neo4j",
            status="error",
            details={"error": str(e)}
        )


async def check_embedding() -> ServiceStatus:
    """Check Embedding service"""
    try:
        from ..services.embedding_service import get_embedding_service
        embedding = get_embedding_service()
        if embedding.model is None:
            embedding.load_model()
        
        # Test encoding
        start = time.time()
        test_vec = embedding.encode_query("test")
        response_time = (time.time() - start) * 1000
        
        return ServiceStatus(
            name="embedding",
            status="healthy",
            response_time_ms=response_time,
            details={"dimension": len(test_vec)}
        )
    except Exception as e:
        logger.error(f"Embedding health check failed: {e}")
        return ServiceStatus(
            name="embedding",
            status="error",
            details={"error": str(e)}
        )


async def check_reranker() -> ServiceStatus:
    """Check Reranker service"""
    try:
        from ..services.reranker_service import get_reranker_service
        reranker = get_reranker_service()
        if reranker.model is None:
            reranker.load_model()
        
        # Quick test
        start = time.time()
        test_query = "test"
        test_docs = ["test document"]
        results = reranker.rerank(test_query, test_docs)
        response_time = (time.time() - start) * 1000
        
        return ServiceStatus(
            name="reranker",
            status="healthy",
            response_time_ms=response_time,
            details={"model": reranker.model_name}
        )
    except Exception as e:
        logger.error(f"Reranker health check failed: {e}")
        return ServiceStatus(
            name="reranker",
            status="error",
            details={"error": str(e)}
        )


@router.get("", response_model=HealthCheckResponse)
async def health_check(request: HealthCheckRequest = Depends()):
    """
    Health check endpoint
    
    Returns status of all services
    """
    services = []
    
    # Always check core services
    services.append(await check_ollama())
    services.append(await check_qdrant())
    services.append(await check_neo4j())
    
    # Check ML services if requested
    if request.include_services:
        services.append(await check_embedding())
        services.append(await check_reranker())
    
    # Determine overall status
    overall_status = "healthy"
    for service in services:
        if service.status != "healthy":
            overall_status = "unhealthy"
            break
    
    return HealthCheckResponse(
        status=overall_status,
        services=services
    )


@router.get("/ready")
async def readiness_check(request: Request) -> Dict[str, str]:
    """Lightweight readiness check for Docker/Kubernetes probes."""
    if hasattr(request.app.state, "rag_service"):
        return {"status": "ready"}
    return {"status": "not ready", "reason": "rag service not initialized"}
