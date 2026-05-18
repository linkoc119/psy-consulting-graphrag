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
from ..models.graph_schema import NODE_LABELS, get_schema_info

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

        # Initialize reranker and preload model
        self.reranker = RerankerService()
        self.reranker.load_model()

        # Initialize Qdrant
        self.qdrant = QdrantService()
        self.qdrant.connect()

        # Initialize Neo4j
        self.neo4j = Neo4jService()
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
            import time
            start_time = time.time()

            # Store retrieval context for later access (for sources)
            self.last_context = None

            # Phase 1: Retrieval (Qdrant + Neo4j)
            context = await self._retrieve_context(query, conversation_history)
            self.last_context = context  # Store for sources

            # Phase 2: Triage & Prompt Building
            prompt, system_prompt, severity = await self._build_prompt(query, context, conversation_history)

            # Phase 3: Generation
            import time
            gen_start_time = time.time()

            async for chunk in self.llm.generate(
                prompt=prompt,
                stream=True,
                system_prompt=system_prompt
            ):
                yield chunk

            gen_time = time.time() - gen_start_time
            logger.info(f"✅ LLM generation time: {gen_time*1000:.0f}ms")

            # Log total processing time
            total_time = time.time() - start_time
            retrieval_time = gen_start_time - start_time - (getattr(self, '_prompt_build_time', 0))
            logger.info(f"✅ Retrieval time: {retrieval_time*1000:.0f}ms")
            logger.info(f"✅ Total response time: {total_time*1000:.0f}ms")

        except Exception as e:
            logger.error(f"Error in process_query: {e}")
            yield f"❌ Xin lỗi, đã xảy ra lỗi: {str(e)}"

    async def _retrieve_context(self, query: str, conversation_history: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Retrieve relevant context from Qdrant and Neo4j using GraphRAG

        Pipeline:
        1. Vector search (Qdrant) → top K documents
        2. Extract entity IDs from document payloads
        3. Graph traversal (Neo4j) from those entity IDs
        4. RRF fusion of vector results + graph nodes
        5. Rerank final results

        Returns:
            Dict with keys: documents, graph_nodes, relationships, severity_indicators
        """
        import time
        logger.info(f"Retrieving context for query: {query[:100]}...")
        t0 = time.time()

        # Step 1: Generate query embedding
        t1 = time.time()
        query_embedding = self.embedding.encode_query(query)
        logger.info(f"✅ Embedding generated in {(time.time()-t1)*1000:.0f}ms")

        # Step 2: Classify query to determine appropriate document types
        # Use conversation history for better context
        query_type = self._classify_query_type(query, conversation_history)

        # Step 3: Vector search in Qdrant with optional filtering
        t1 = time.time()
        filter_conditions = None
        if query_type == "counseling":
            # For counseling queries, ONLY retrieve school_counseling docs
            # This prevents medical_guideline docs (with severity 4-5) from triggering crisis mode
            filter_conditions = {"metadata.doc_type": "school_counseling"}
            logger.info(f"Applying filter: metadata.doc_type=school_counseling")
        elif query_type == "crisis":
            # For crisis queries, retrieve all document types (first_aid and medical_guideline are helpful)
            logger.info(f"No filter applied for crisis query - retrieving all doc types")
        else:
            # General queries: retrieve all document types
            logger.info(f"No filter applied for general query - retrieving all doc types")

        vector_results = self.qdrant.search(
            query_vector=query_embedding,
            limit=8,  # Optimized: 8 docs for faster reranking
            filter_conditions=filter_conditions,
            with_payload=True
        )
        logger.info(f"✅ Qdrant search returned {len(vector_results)} docs in {(time.time()-t1)*1000:.0f}ms")

        # Debug: Log scores if no results with filter
        if len(vector_results) == 0 and filter_conditions:
            # Try search without filter to see if there are any results at all
            unfiltered = self.qdrant.search(
                query_vector=query_embedding,
                limit=3,
                filter_conditions=None,
                with_payload=True
            )
            if unfiltered:
                logger.warning(f"Filtered search returned 0, but unfiltered returned {len(unfiltered)}. Sample scores: {[r.get('score', 0) for r in unfiltered]}")
            else:
                logger.warning(f"No results even without filter - embedding may not match any documents")

        # Log document types retrieved
        doc_types_retrieved = set()
        for res in vector_results:
            payload = res.get("payload", {})
            metadata = payload.get("metadata", {})
            doc_type = metadata.get("doc_type", payload.get("doc_type", "unknown"))
            doc_types_retrieved.add(doc_type)
        logger.info(f"Document types retrieved: {doc_types_retrieved}")

        # Step 4: Fallback for counseling queries if filtered search returned too few results
        if query_type == "counseling" and len(vector_results) < 3 and filter_conditions:
            logger.warning(f"Filtered search returned only {len(vector_results)} results, falling back to unfiltered search")
            # Re-run search without filter to get more results
            fallback_results = self.qdrant.search(
                query_vector=query_embedding,
                limit=8,
                filter_conditions=None,  # No filter
                with_payload=True
            )
            # Merge results, deduplicate by chunk_id, keeping higher score
            results_by_chunk = {}
            for res in vector_results + fallback_results:
                chunk_id = res.get("payload", {}).get("chunk_id", "")
                if not chunk_id:
                    continue
                score = res.get("score", 0)
                if chunk_id not in results_by_chunk or score > results_by_chunk[chunk_id]["score"]:
                    results_by_chunk[chunk_id] = res
            # Rebuild vector_results sorted by score (descending)
            vector_results = sorted(results_by_chunk.values(), key=lambda x: x.get("score", 0), reverse=True)[:8]
            logger.info(f"After fallback: {len(vector_results)} total results")

        # Extract documents and entity IDs
        documents = []
        entity_ids_from_docs = set()

        for res in vector_results:
            payload = res.get("payload", {})
            doc = {
                "text": payload.get("page_content", ""),
                "metadata": {
                    k: v for k, v in payload.items()
                    if k not in ["page_content", "embedding", "entity_ids"]
                },
                "score": res["score"],
                "qdrant_rank": len(documents) + 1  # Track rank for RRF
            }
            documents.append(doc)

            # Collect entity IDs from payload
            if "entity_ids" in payload:
                for eid in payload["entity_ids"]:
                    entity_ids_from_docs.add(eid)

        logger.info(f"Vector search: {len(documents)} docs, {len(entity_ids_from_docs)} unique entity IDs")

        # Step 3: Graph traversal from entity IDs
        t1 = time.time()
        graph_nodes = []
        relationships = []

        # Initialize subgraph to avoid "variable not associated with a value" error
        subgraph = {"nodes": [], "relationships": []}

        if entity_ids_from_docs:
            # Only traverse meaningful relationships (exclude co-occurrence which is noisy)
            # These are the "golden" relationships from schema
            REL_TYPES_TO_TRAVERSE = [
                "CO_TRIEU_CHUNG",
                "BAO_HIEU_NGUY_HIEM",
                "YEU_CAU_HANH_DONG",
                "DIEU_TRI_BANG",
                "QUAN_LY_BANG",
                "AP_DUNG_CHO"
            ]

            # Limit entities to prevent graph explosion (depth=2 with many entities = thousands of relationships)
            MAX_ENTITIES_FOR_TRAVERSAL = 3
            entity_ids_list = list(entity_ids_from_docs)[:MAX_ENTITIES_FOR_TRAVERSAL]
            if len(entity_ids_from_docs) > MAX_ENTITIES_FOR_TRAVERSAL:
                logger.info(f"Limited entities for traversal: {len(entity_ids_from_docs)} → {MAX_ENTITIES_FOR_TRAVERSAL}")

            # Get subgraph with limited entities and depth=1 for speed + quality
            subgraph = await self.neo4j.get_subgraph(
                node_ids=entity_ids_list,
                depth=1,
                relationship_types=REL_TYPES_TO_TRAVERSE
            )
        raw_nodes = subgraph.get("nodes", [])
        relationships = subgraph.get("relationships", [])
        logger.info(f"✅ Graph traversal returned {len(raw_nodes)} nodes, {len(relationships)} rels in {(time.time()-t1)*1000:.0f}ms")

        # Normalize graph nodes
        t1 = time.time()
        for node in raw_nodes:
            # Extract label from labels set
            labels = node.get("labels", set())
            label = next(iter(labels)) if labels else "Unknown"

            # Build text representation for reranker
            node_text = f"{node.get('name', '')}"
            description = node.get("description", "")
            if description:
                node_text += f": {description}"

            graph_nodes.append({
                "id": node.get("id"),
                "name": node.get("name", ""),
                "description": description,
                "label": label,
                "text": node_text,
                "severity_level": node.get("severity_level", 1),
                "graph_rank": len(graph_nodes) + 1  # Track rank for RRF
            })

        # Limit graph nodes before fusion to avoid reranker overload
        # Keep top N by severity_level (higher = more important)
        MAX_GRAPH_NODES_BEFORE_RERANK = 15  # Optimized for speed: 15 instead of 25
        if len(graph_nodes) > MAX_GRAPH_NODES_BEFORE_RERANK:
            graph_nodes.sort(key=lambda n: n.get("severity_level", 1), reverse=True)
            graph_nodes = graph_nodes[:MAX_GRAPH_NODES_BEFORE_RERANK]
            # Re-assign ranks after limiting
            for i, node in enumerate(graph_nodes):
                node["graph_rank"] = i + 1
        logger.info(f"Graph nodes limited to {len(graph_nodes)} (max {MAX_GRAPH_NODES_BEFORE_RERANK})")

        # Step 3.5: Filter and format relationships
        # Only keep high-quality relationships to avoid token explosion
        # Pass nodes list to resolve node names from IDs
        filtered_relationships = self._filter_relationships(
            relationships,
            raw_nodes,
            max_rels=60  # Optimized: 60 relationships for speed
        )

        logger.info(f"Graph traversal returned {len(raw_nodes)} nodes, limited to {len(graph_nodes)} nodes, relationships filtered to {len(filtered_relationships)}")

        # Step 4: RRF Fusion
        # Convert graph nodes to text format for reranker
        k = 60  # RRF constant

        # Prepare items for fusion
        fused_items = []

        # Add vector documents
        for doc in documents:
            # Use chunk_id as unique identifier
            item_id = doc["metadata"].get("chunk_id", f"doc_{id(doc)}")
            # RRF score from vector search (lower rank = higher score)
            rrf_score = 1.0 / (k + doc["qdrant_rank"])
            fused_items.append({
                "type": "document",
                "item": doc,
                "rrf_score": rrf_score,
                "combined_score": rrf_score
            })

        # Add graph nodes (already normalized with text field)
        for node in graph_nodes:
            # Use node.id as unique identifier
            item_id = node.get("id", f"node_{id(node)}")
            rrf_score = 1.0 / (k + node.get("graph_rank", 999))
            # Build metadata for reranker
            node_metadata = {
                "source": "graph",
                "node_type": node.get("label", ""),
                "node_id": item_id,
                "severity_level": node.get("severity_level", 1)
            }
            # Create item with all node properties + metadata
            item = {
                **node,  # Include id, name, description, label, text, severity_level, graph_rank
                "metadata": node_metadata,
                "score": node.get("severity_level", 1)
            }
            fused_items.append({
                "type": "graph_node",
                "item": item,
                "rrf_score": rrf_score,
                "combined_score": rrf_score
            })

        # Step 5: Rerank all fused items with Vietnamese_Reranker
        # Prepare texts and metadata for reranker
        all_texts = []
        all_metadata = []
        for fused in fused_items:
            item = fused["item"]
            all_texts.append(item["text"])
            all_metadata.append(item["metadata"])

        # Rerank
        reranked = self.reranker.rerank(query, all_texts, all_metadata)

        # Map reranked results back to fused items
        reranked_items = []
        for text, score, meta in reranked:
            # Find the corresponding fused item
            for fused in fused_items:
                item = fused["item"]
                if item["text"] != text:
                    continue
                # For graph nodes, also match node_id in metadata
                if fused["type"] == "graph_node":
                    if item["metadata"].get("node_id") != meta.get("node_id"):
                        continue
                fused["rerank_score"] = score
                fused["combined_score"] = fused["rrf_score"] + score  # Simple sum, can tune
                reranked_items.append(fused)
                break

        # Sort by combined score
        reranked_items.sort(key=lambda x: x["combined_score"], reverse=True)

        # Separate back into documents and graph nodes
        final_documents = []
        final_graph_nodes = []

        for fused in reranked_items[:8]:  # Keep top 8 for speed
            item = fused["item"]
            if fused["type"] == "document":
                final_documents.append(item)
            else:
                final_graph_nodes.append(item)

        # Step 6: Detect severity indicators
        severity_indicators = self._detect_severity_indicators(
            query, final_documents, final_graph_nodes
        )

        total_time = time.time() - t0
        logger.info(f"✅ Retrieval complete in {total_time*1000:.0f}ms")
        logger.info(f"Final: {len(final_documents)} docs, {len(final_graph_nodes)} graph nodes, {len(filtered_relationships)} rels")

        return {
            "documents": final_documents,
            "graph_nodes": final_graph_nodes,
            "relationships": filtered_relationships,
            "severity_indicators": severity_indicators,
            "query": query,
            "query_type": query_type
        }

    def _filter_relationships(
        self,
        relationships: List[Dict],
        all_nodes: List[Dict],
        max_rels: int = 150
    ) -> List[str]:
        """
        Filter relationships to keep only high-quality ones and avoid token explosion

        Strategy:
        1. Keep only "golden" relationship types that encode medical knowledge
        2. Filter by severity (higher severity = more important)
        3. Remove duplicates (same source-target-type)
        4. Limit total count
        5. Format as text for prompt

        Args:
            relationships: List of relationship dicts with keys: type, start (node_id), end (node_id), properties
            all_nodes: List of all graph nodes from Neo4j traversal (used to look up names)
            max_rels: Maximum number of relationships to return

        Returns:
            List of formatted relationship strings
        """
        # Golden relationship types that represent actual medical knowledge
        GOLDEN_REL_TYPES = {
            "CO_TRIEU_CHUNG",      # Symptom co-occurrence
            "BAO_HIEU_NGUY_HIEM",  # Danger sign indicator
            "YEU_CAU_HANH_DONG",   # Required action
            "DIEU_TRI_BANG",       # Treated by
            "QUAN_LY_BANG",        # Managed by
            "AP_DUNG_CHO"          # Applicable to
        }

        # Severity threshold: only keep relationships with severity >= 1
        SEVERITY_THRESHOLD = 1

        logger.info(f"Filtering {len(relationships)} raw relationships")

        # Build node ID → name mapping for fast lookup
        node_id_to_name = {}
        for node in all_nodes:
            node_id = node.get("id", "")
            node_name = node.get("name", "")
            if node_id and node_name:
                node_id_to_name[node_id] = node_name

        logger.debug(f"Built node_id→name map with {len(node_id_to_name)} entries")

        # Deduplicate by (source_name, target_name, type)
        seen = set()
        filtered = []

        for rel in relationships:
            rel_type = rel.get("type", "")

            # Step 1: Filter by golden relationship types
            if rel_type not in GOLDEN_REL_TYPES:
                logger.debug(f"Skipping relationship type: {rel_type}")
                continue

            # Step 2: Filter by severity (if available)
            properties = rel.get("properties", {})
            severity = properties.get("severity_level", 1)
            if isinstance(severity, (int, float)) and severity < SEVERITY_THRESHOLD:
                logger.debug(f"Skipping low severity: {severity} for type {rel_type}")
                continue

            # Step 3: Get node names from ID mapping (not from rel dict directly!)
            start_id = rel.get("start", "")
            end_id = rel.get("end", "")
            start_name = node_id_to_name.get(start_id, "")
            end_name = node_id_to_name.get(end_id, "")

            # Debug logging for first few relationships
            if len(filtered) < 3:
                logger.debug(f"Rel: {rel_type}, start_id={start_id}, end_id={end_id}, start_name={start_name}, end_name={end_name}")

            # Skip if names missing
            if not start_name or not end_name:
                logger.debug(f"Skipping relationship with missing names: start={start_name}, end={end_name}")
                continue

            # Step 4: Deduplicate
            dedup_key = (start_name.lower().strip(), end_name.lower().strip(), rel_type)
            if dedup_key in seen:
                logger.debug(f"Duplicate relationship: {start_name} → {end_name} ({rel_type})")
                continue
            seen.add(dedup_key)

            # Step 5: Format as readable text
            # Format: "[REL_TYPE] Source → Target"
            formatted = f"[{rel_type}] {start_name} → {end_name}"
            filtered.append({
                "text": formatted,
                "type": rel_type,
                "source": start_name,
                "target": end_name,
                "severity": severity
            })

        logger.info(f"After filtering: kept {len(filtered)} relationships (skipped {len(relationships) - len(filtered)})")

        # Step 6: Sort by severity (desc) and limit
        filtered.sort(key=lambda x: x.get("severity", 1), reverse=True)
        limited = filtered[:max_rels]

        logger.info(f"Final relationships count: {len(limited)} (top {max_rels})")

        # Return only text for simplicity in prompt
        return [item["text"] for item in limited]

    def _classify_query_type(self, query: str, conversation_history: Optional[List[Dict]] = None) -> str:
        """
        Classify query type to determine which document types should be retrieved.
        Uses both current query AND conversation history for better context.

        Query types:
        - "crisis": Contains crisis keywords (self-harm, harm to others, abuse, hallucinations)
          Should retrieve: first_aid, medical_guideline (any doc type, no filter)
        - "counseling": School counseling topics (anxiety, stress, relationships, study, future)
          Should retrieve: school_counseling ONLY (exclude medical/first_aid to avoid severity contamination)
        - "general": Other queries, retrieve all document types

        Returns:
            "crisis", "counseling", or "general"
        """
        crisis_keywords = [
            "tự tử", "tự làm hại", "muốn chết", "kết thúc cuộc sống",
            "cắt tay", "tự đánh", "bỏ ăn", "bỏ học",
            "bị bạo lực", "bị xâm hại", "bị đánh", "bị làm nhục",
            "ảo giác", "nghe thấy tiếng nói", "hoang tưởng",
            "sẽ làm hại", "sẽ giết", "đe dọa", "tự sát",
            "cô lập", "bỏ nhà", "chạy trốn"
        ]

        counseling_keywords = [
            "lo lắng", "áp lực", "căng thẳng", "stress",
            "bạn bè", "bạn thân", "mâu thuẫn", "cãi nhau",
            "học tập", "thi cử", " điểm", "tương lai",
            "nghề nghiệp", "gia đình", "cha mẹ", "anh chị",
            "tự tin", "self-esteem", "khả năng", "cảm thấy"
        ]

        # Combine current query with conversation history for better context
        full_text = query.lower()
        if conversation_history:
            for msg in conversation_history:
                content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, 'content', '')
                if content:
                    full_text += " " + content.lower()

        # Check for crisis keywords first (higher priority)
        for keyword in crisis_keywords:
            if keyword in full_text:
                logger.info(f"Query classified as CRISIS due to keyword: '{keyword}'")
                return "crisis"

        # Check for counseling keywords
        for keyword in counseling_keywords:
            if keyword in full_text:
                logger.info(f"Query classified as COUNSELING due to keyword: '{keyword}'")
                return "counseling"

        logger.info(f"Query classified as GENERAL (no specific keywords detected)")
        return "general"

    def _detect_severity_indicators(
        self,
        query: str,
        documents: List[Dict],
        graph_nodes: List[Dict]
    ) -> Dict[str, Any]:
        """
        Detect red flags and severity level from query and retrieved context

        IMPORTANT: Graph nodes with high severity (4-5) should ONLY trigger crisis mode
        if the query itself contains crisis keywords. This prevents false positives
        from co-occurrence in the knowledge graph.

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

        # Check graph nodes for severity levels
        # NOTE: Only use graph severity if crisis keywords present in query
        # to avoid false positives from co-occurrence relationships
        max_graph_severity = 1
        for node in graph_nodes:
            severity = node.get("metadata", {}).get("severity_level", 1)
            if isinstance(severity, (int, float)):
                max_graph_severity = max(max_graph_severity, int(severity))

        # Determine final severity
        if detected:
            # Crisis keywords in query → severity at least 4
            severity_level = max(max_graph_severity, 4)
        else:
            # No crisis keywords → cap severity at 3 (moderate max)
            # Even if graph has severity 4-5 nodes, they're from co-occurrence
            severity_level = min(max_graph_severity, 3)

        return {
            "level": severity_level,
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
        relationships = context.get("relationships", [])

        # Build context string (balanced for quality vs speed)
        context_parts = []

        # Add document context (3 docs, 300 chars each)
        if documents:
            context_parts.append("=== TÀI LIỆU THAM KHẢO ===")
            for i, doc in enumerate(documents[:3], 1):
                meta = doc["metadata"]
                doc_type = meta.get("doc_type", "")
                context_parts.append(f"[{i}] ({doc_type}) {doc['text'][:300]}...")

        # Add graph nodes context (3 nodes, 200 chars each)
        if graph_nodes:
            context_parts.append("\n=== CÁC KHÁI NIỆM TRI THỨC LIÊN QUAN ===")
            for i, node in enumerate(graph_nodes[:3], 1):
                node_type = node.get("metadata", {}).get("node_type", "Unknown")
                context_parts.append(f"[{i}] ({node_type}) {node['text'][:200]}...")

        # Add relationships context (10 relationships)
        if relationships:
            context_parts.append("\n=== CÁC MỐI LIÊN KẾT TRI THỨC ===")
            context_parts.append("(Các mối liên hệ quan trọng):")
            for i, rel in enumerate(relationships[:10], 1):
                context_parts.append(f"  {i}. {rel}")

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
