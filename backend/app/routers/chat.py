"""
Chat router - handles chat requests
"""
import time
import logging
from typing import List, Dict
from fastapi import APIRouter, HTTPException, Depends, Request
from ..models.schemas import (
    ChatRequest, ChatResponse, ErrorResponse, ChatMessage
)
from ..services.rag_service import GraphRAGService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"])

# In-memory conversation store (in production, use Redis/DB)
_conversations: Dict[str, List[Dict[str, str]]] = {}


async def get_rag_service(request: Request) -> GraphRAGService:
    """Dependency to get pre-loaded RAG service instance from app.state"""
    if not hasattr(request.app.state, 'rag_service'):
        raise HTTPException(
            status_code=503,
            detail="RAG service not initialized. Please wait a moment and try again."
        )
    return request.app.state.rag_service


@router.post(
    "/completion",
    response_model=ChatResponse,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)
async def chat_completion(
    request: ChatRequest,
    rag_service: GraphRAGService = Depends(get_rag_service)
):
    """
    Get a response from the GraphRAG chatbot

    This endpoint processes user messages through:
    1. Context retrieval (vector + graph)
    2. Triage and prompt construction
    3. LLM generation with streaming

    Returns the complete response with metadata
    """
    start_time = time.time()

    try:
        # Get conversation history
        conversation_id = request.conversation_id or f"conv_{int(time.time())}"
        history = request.history or []

        # Ensure history is in correct format
        formatted_history = []
        for msg in history:
            if isinstance(msg, ChatMessage):
                formatted_history.append({
                    "role": msg.role,
                    "content": msg.content
                })
            else:
                formatted_history.append(msg)

        # Store conversation for future reference
        if conversation_id not in _conversations:
            _conversations[conversation_id] = []

        # Add current user message to history
        _conversations[conversation_id].append({
            "role": "user",
            "content": request.message
        })

        # Generate response (streaming internally)
        full_response = ""
        async for chunk in rag_service.process_query(
            query=request.message,
            user_id=request.user_id,
            conversation_history=formatted_history
        ):
            full_response += chunk

        # Get retrieval context for sources
        context = getattr(rag_service, 'last_context', None)
        sources = []
        if context:
            # Add document sources
            for doc in context.get('documents', [])[:5]:  # Top 5
                meta = doc.get('metadata', {})
                sources.append({
                    'type': 'document',
                    'title': meta.get('source_filename', 'Unknown'),
                    'doc_type': meta.get('doc_type', ''),
                    'score': float(doc.get('score', 0)),
                    'text_preview': doc.get('text', '')[:200]
                })
            # Add graph node sources
            for node in context.get('graph_nodes', [])[:5]:  # Top 5
                sources.append({
                    'type': 'graph_node',
                    'title': node.get('name', 'Unknown'),
                    'node_type': node.get('label', ''),
                    'score': float(node.get('rerank_score', node.get('severity_level', 0))),
                    'text_preview': node.get('text', '')[:200]
                })

        # Add assistant response to history
        _conversations[conversation_id].append({
            "role": "assistant",
            "content": full_response
        })

        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000

        # Keep conversation history limited (last 20 messages)
        if len(_conversations[conversation_id]) > 40:  # 20 exchanges
            _conversations[conversation_id] = _conversations[conversation_id][-40:]

        # Determine severity and crisis mode
        severity = 1
        if context:
            severity = context.get('severity_indicators', {}).get('level', 1)
        # Crisis mode: severity >= 4 OR query classified as crisis
        is_crisis = severity >= 4
        if context:
            query_type = context.get('query_type', '')
            if query_type == 'crisis':
                is_crisis = True

        # Build response
        response = ChatResponse(
            message=full_response,
            conversation_id=conversation_id,
            severity_level=severity,
            is_crisis=is_crisis,
            sources=sources,
            processing_time_ms=processing_time
        )

        return response

    except Exception as e:
        logger.error(f"Error in chat_completion: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process chat request: {str(e)}"
        )


@router.get("/conversation/{conversation_id}")
async def get_conversation(conversation_id: str) -> List[Dict[str, str]]:
    """Get conversation history"""
    return _conversations.get(conversation_id, [])


@router.delete("/conversation/{conversation_id}")
async def delete_conversation(conversation_id: str) -> Dict[str, str]:
    """Delete conversation history"""
    if conversation_id in _conversations:
        del _conversations[conversation_id]
        return {"message": f"Conversation {conversation_id} deleted"}
    else:
        raise HTTPException(status_code=404, detail="Conversation not found")


@router.post("/clear")
async def clear_all_conversations() -> Dict[str, str]:
    """Clear all conversation histories"""
    global _conversations
    _conversations.clear()
    return {"message": "All conversations cleared"}
