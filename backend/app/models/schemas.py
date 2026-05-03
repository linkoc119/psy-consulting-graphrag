"""
Pydantic schemas for request/response validation
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime

# ============ Request Models ============

class ChatMessage(BaseModel):
    """Single chat message"""
    role: str = Field(..., description="Role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")
    timestamp: Optional[datetime] = Field(default_factory=datetime.now)
    
    class Config:
        json_schema_extra = {
            "example": {
                "role": "user",
                "content": "Tôi cảm thấy lo lắng về kỳ thi sắp tới..."
            }
        }


class ChatRequest(BaseModel):
    """Request for chat endpoint"""
    message: str = Field(..., description="User's message")
    conversation_id: Optional[str] = Field(None, description="Conversation ID for context")
    history: Optional[List[ChatMessage]] = Field(default=[], description="Previous conversation turns")
    user_id: Optional[str] = Field(None, description="User identifier (for logging/tracking)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "message": "Tôi rất căng thẳng về việc thi cử",
                "conversation_id": "conv_123",
                "history": [],
                "user_id": "user_456"
            }
        }


class HealthCheckRequest(BaseModel):
    """Health check request"""
    include_services: bool = Field(default=False, description="Check all dependent services")


# ============ Response Models ============

class ChatResponse(BaseModel):
    """Response from chat endpoint"""
    message: str = Field(..., description="Assistant's response")
    conversation_id: str = Field(..., description="Conversation identifier")
    severity_level: int = Field(..., description="Triage severity level (1-5)")
    is_crisis: bool = Field(..., description="Whether crisis mode was activated")
    sources: List[Dict[str, Any]] = Field(default=[], description="Sources used for response")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    timestamp: datetime = Field(default_factory=datetime.now)
    
    class Config:
        json_schema_extra = {
            "example": {
                "message": "Tôi hiểu bạn đang lo lắng về kỳ thi. Hãy thử các bài tập hít thở sâu...",
                "conversation_id": "conv_123",
                "severity_level": 2,
                "is_crisis": False,
                "sources": [
                    {"type": "document", "title": "Sổ tay tư vấn", "score": 0.85}
                ],
                "processing_time_ms": 1234.56
            }
        }


class ServiceStatus(BaseModel):
    """Status of a service"""
    name: str = Field(..., description="Service name")
    status: str = Field(..., description="Status: 'healthy', 'unhealthy', 'error'")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional details")
    response_time_ms: Optional[float] = Field(None, description="Response time in ms")


class HealthCheckResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Overall status: 'healthy' or 'unhealthy'")
    services: List[ServiceStatus] = Field(default=[], description="Status of all services")
    timestamp: datetime = Field(default_factory=datetime.now)
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "services": [
                    {
                        "name": "ollama",
                        "status": "healthy",
                        "response_time_ms": 150.5
                    }
                ],
                "timestamp": "2024-01-15T10:30:00"
            }
        }


class ErrorResponse(BaseModel):
    """Error response"""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    timestamp: datetime = Field(default_factory=datetime.now)


# ============ Internal Models (not exposed via API) ============

class RetrievedDocument(BaseModel):
    """Internal model for retrieved document"""
    text: str
    metadata: Dict[str, Any]
    score: float
    
    class Config:
        frozen = True  # Make hashable for set operations


class GraphNode(BaseModel):
    """Internal model for graph node"""
    id: str
    name: str
    node_type: str
    description: Optional[str] = None
    severity_level: Optional[int] = None
    properties: Dict[str, Any] = Field(default_factory=dict)
    score: Optional[float] = None
    rerank_score: Optional[float] = None
    
    class Config:
        frozen = True


class RetrievalContext(BaseModel):
    """Combined retrieval context"""
    documents: List[RetrievedDocument]
    graph_nodes: List[GraphNode]
    relationships: List[Dict[str, Any]]
    severity_indicators: Dict[str, Any]
    query: str
    retrieval_time_ms: float