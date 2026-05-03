"""
GraphRAG Service - Core retrieval and generation logic
Combines vector search (Qdrant) and graph traversal (Neo4j)
"""
import logging
from typing import List, Dict, Any, Optional, AsyncGenerator
from .llm_service import OllamaLLM
from .embedding_service import EmbeddingService
from .reranker_service import RerankerService
from .qdrant_service import QdrantService
from .neo4j_service import Neo4jService
from ..utils.prompts import (
    SYSTEM_PROMPT_COUNSELING,
    SYSTEM_PROMPT_CRISIS,
    FEW_SHOT_COUNSELING,
    FEW_SHOT_CRISIS,
    TRIAGE_GUIDELINES
)
from config import settings

TRIAGE_THRESHOLD_HIGH = settings.TRIAGE_THRESHOLD_HIGH
TRIAGE_THRESHOLD_MEDIUM = settings.TRIAGE_THRESHOLD_MEDIUM

logger = logging.getLogger(__name__)

class GraphRAGService:
    """Main service for GraphRAG pipeline"""
    
    def __init__(self):
        self.llm = None
        self.embedding = None
        self.reranker = None
        self.qdrant = None
        self.neo4j = None
        
    async def initialize(self):
        """Initialize all services"""
        logger.info("Initializing GraphRAG services...")
        
        # Initialize LLM
        self.llm = OllamaLLM()
        connected = await self.llm.check_connection()
        if not connected:
            raise RuntimeError("Ollama is not available. Please start Ollama and pull the model.")
        
        # Initialize embedding service
        self.embedding = EmbeddingService()
        self.embedding.load_model()
        
        # Initialize reranker
        self.reranker = RerankerService()
        self.reranker.load_model()
        
        # Initialize Qdrant
        self.qdrant = QdrantService()
        self.qdrant.connect()
        
        # Initialize Neo4j
        self.neo4j = await Neo4jService()
        await self.neo4j.connect()
        
        logger.info("✅ All services initialized")
    
    async def process_query(
        self,
        query: str,
        user_id: Optional[str] = None,
        conversation_history: Optional[List[Dict]] = None
    ) -> AsyncGenerator[str, None]:
        """
        Main entry point: Process user query through GraphRAG pipeline
        
        Args:
            query: User's message
            user_id: Optional user identifier
            conversation_history: Previous conversation turns
            
        Yields:
            Streamed response chunks
        """
        try:
            # Phase 1: Retrieval (Qdrant + Neo4j)
            context = await self._retrieve_context(query)
            
            # Phase 2: Triage & Prompt Building
            prompt, system_prompt, severity = await self._build_prompt(query, context, conversation_history)
            
            # Phase 3: Generation
            async for chunk in self.llm.generate(
                prompt=prompt,
                stream=True,
                system_prompt=system_prompt
            ):
                yield chunk
                
        except Exception as e:
            logger.error(f"Error in process_query: {e}")
            yield f"❌ Xin lỗi, đã xảy ra lỗi: {str(e)}"
    
    async def _retrieve_context(self, query: str) -> Dict[str, Any]:
        """
        Retrieve relevant context from Qdrant and Neo4j
        
        Returns:
            Dict with keys: documents, graph_nodes, relationships, severity_indicators
        """
        logger.info(f"Retrieving context for query: {query[:100]}...")
        
        # Step 1: Generate query embedding
        query_embedding = self.embedding.encode_query(query)
        
        # Step 2: Vector search in Qdrant
        vector_results = self.qdrant.search(
            query_vector=query_embedding,
            limit=15,
            with_payload=True
        )
        
        documents = []
        for res in vector_results:
            payload = res.get("payload", {})
            documents.append({
                "text": payload.get("page_content", ""),
                "metadata": {
                    k: v for k, v in payload.items() if k != "page_content"
                },
                "score": res["score"]
            })
        
        # Step 3: Extract potential graph nodes from top documents
        # In a full implementation, we'd have node IDs stored in document payloads
        # For now, we'll extract keywords and find matching nodes in Neo4j
        graph_nodes = []
        node_ids_from_docs = []
        
        for doc in documents[:5]:  # Use top 5 docs to extract node candidates
            # This would ideally extract entity IDs from the document
            # For prototype, we'll use a simple keyword match approach
            pass
        
        # If we have node IDs, expand via graph traversal
        relationships = []
        if node_ids_from_docs:
            subgraph = await self.neo4j.get_subgraph(
                node_ids=node_ids_from_docs,
                depth=2
            )
            graph_nodes = subgraph.get("nodes", [])
            relationships = subgraph.get("relationships", [])
        
        # Step 4: Rerank all retrieved content
        # Combine documents and graph nodes into a unified text representation
        all_texts = [doc["text"] for doc in documents]
        all_metadata = [doc["metadata"] for doc in documents]
        
        # Add graph node texts
        for node in graph_nodes:
            node_text = f"{node.get('name', '')}: {node.get('description', '')}"
            all_texts.append(node_text)
            all_metadata.append({"source": "graph", "node_type": node.get("label")})
        
        # Rerank
        reranked = self.reranker.rerank(query, all_texts, all_metadata)
        
        # Separate back into documents and graph nodes
        final_documents = []
        final_graph_nodes = []
        
        for text, score, meta in reranked[:10]:  # Keep top 10
            if meta.get("source") == "graph":
                final_graph_nodes.append({
                    "text": text,
                    "metadata": meta,
                    "score": score
                })
            else:
                final_documents.append({
                    "text": text,
                    "metadata": meta,
                    "score": score
                })
        
        # Step 5: Detect severity indicators
        severity_indicators = self._detect_severity_indicators(query, final_documents, final_graph_nodes)
        
        return {
            "documents": final_documents,
            "graph_nodes": final_graph_nodes,
            "relationships": relationships,
            "severity_indicators": severity_indicators,
            "query": query
        }
    
    def _detect_severity_indicators(
        self,
        query: str,
        documents: List[Dict],
        graph_nodes: List[Dict]
    ) -> Dict[str, Any]:
        """
        Detect red flags and severity level from query and retrieved context
        
        Returns:
            Dict with keys: level (1-5), red_flags (list), crisis_keywords (list)
        """
        crisis_keywords = [
            "tự tử", "tự làm hại", "muốn chết", "kết thúc cuộc sống",
            "cắt tay", "tự đánh", "bỏ ăn", "bỏ học",
            "bị bạo lực", "bị xâm hại", "bị đánh", "bị làm nhục",
            "ảo giác", "nghe thấy tiếng nói", "hoang tưởng",
            "sẽ làm hại", "sẽ giết", "đe dọa"
        ]
        
        query_lower = query.lower()
        detected = []
        for keyword in crisis_keywords:
            if keyword in query_lower:
                detected.append(keyword)
        
        # Check graph nodes for high severity labels
        max_severity = 1
        for node in graph_nodes:
            severity = node.get("metadata", {}).get("severity_level", 1)
            if isinstance(severity, (int, float)):
                max_severity = max(max_severity, int(severity))
        
        # If any crisis keyword detected, bump severity
        if detected:
            max_severity = max(max_severity, 4)
        
        return {
            "level": max_severity,
            "red_flags": detected,
            "has_crisis_keywords": len(detected) > 0
        }
    
    async def _build_prompt(
        self,
        query: str,
        context: Dict,
        history: Optional[List[Dict]]
    ) -> tuple:
        """
        Build dynamic prompt based on severity and context
        
        Returns:
            (prompt, system_prompt, severity_level)
        """
        severity = context["severity_indicators"]["level"]
        documents = context["documents"]
        graph_nodes = context["graph_nodes"]
        
        # Build context string
        context_parts = []
        
        # Add document context
        if documents:
            context_parts.append("=== TÀI LIỆU THAM KHẢO ===")
            for i, doc in enumerate(documents[:5], 1):
                meta = doc["metadata"]
                source = meta.get("source", "Unknown")
                doc_type = meta.get("doc_type", "")
                context_parts.append(f"[{i}] ({doc_type}) {doc['text'][:500]}...")
        
        # Add graph context
        if graph_nodes:
            context_parts.append("\n=== MỐI LIÊN HỾT TRI THỨC ===")
            for i, node in enumerate(graph_nodes[:5], 1):
                node_type = node.get("metadata", {}).get("node_type", "Unknown")
                context_parts.append(f"[{i}] ({node_type}) {node['text'][:300]}...")
        
        context_str = "\n".join(context_parts) if context_parts else "Không có ngữ cảnh đặc biệt."
        
        # Determine which system prompt to use
        if severity >= TRIAGE_THRESHOLD_HIGH:
            # Crisis mode - PFA
            system_prompt = SYSTEM_PROMPT_CRISIS
            few_shot = FEW_SHOT_CRISIS
        else:
            # Normal counseling mode
            system_prompt = SYSTEM_PROMPT_COUNSELING
            few_shot = FEW_SHOT_COUNSELING
        
        # Build main prompt
        prompt_parts = [
            few_shot,
            "\n=== NGỮ CẢNH ===",
            context_str,
            "\n=== CÂU HỎI CỦA NGƯỜI DÙNG ===",
            query,
            "\n=== PHẢN HỒI ==="
        ]
        
        prompt = "\n".join(prompt_parts)
        
        logger.info(f"Built prompt with severity={severity}, using {'CRISIS' if severity>=4 else 'COUNSELING'} mode")
        
        return prompt, system_prompt, severity
    
    async def close(self):
        """Close all connections"""
        if self.llm:
            await self.llm.close()
        if self.neo4j:
            await self.neo4j.close()
        logger.info("GraphRAG service closed")


# Convenience function for single query
async def get_rag_response(
    query: str,
    history: Optional[List[Dict]] = None
) -> AsyncGenerator[str, None]:
    """
    Get streaming response for a query
    
    Usage:
        async for chunk in get_rag_response("Tôi cảm thấy lo lắng..."):
            print(chunk, end='', flush=True)
    """
    service = GraphRAGService()
    try:
        await service.initialize()
        async for chunk in service.process_query(query, conversation_history=history):
            yield chunk
    finally:
        await service.close()