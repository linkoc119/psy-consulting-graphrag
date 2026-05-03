from qdrant_client.http.models import PointStruct
"""
Graph Indexer Service
Extracts entities and relationships from document chunks and populates Neo4j
"""
import logging
import asyncio
from typing import List, Dict, Any, Optional, Set, Tuple
from .llm_service import OllamaLLM
from .embedding_service import EmbeddingService
from .neo4j_service import Neo4jService, get_neo4j_service
from .qdrant_service import QdrantService, get_qdrant_service
from .reranker_service import RerankerService
from ..models.graph_schema import (
    NODE_LABELS, RELATIONSHIPS, PREDEFINED_ENTITIES,
    get_schema_info
)

logger = logging.getLogger(__name__)

class GraphIndexer:
    """Service for indexing documents into Knowledge Graph"""
    
    def __init__(
        self,
        llm: Optional[OllamaLLM] = None,
        embedding: Optional[EmbeddingService] = None,
        reranker: Optional[RerankerService] = None,
        qdrant: Optional[QdrantService] = None,
        neo4j: Optional[Neo4jService] = None
    ):
        self.llm = llm
        self.embedding = embedding
        self.reranker = reranker
        self.qdrant = qdrant
        self.neo4j = neo4j
        
        # Load predefined entities for matching
        self.predefined_entities = PREDEFINED_ENTITIES
        self.schema_info = get_schema_info()
        
        # Entity type mapping (what we're looking for)
        self.entity_labels = {
            "BenhLy": ["bệnh", "rối loạn", "hội chứng", "bệnh lý", "rối tương tác"],
            "TrieuChung": ["triệu chứng", "dấu hiệu", "biểu hiện", "symptom", "triệu ứng"],
            "Thuoc": ["thuốc", "điều trị", "lớp thuốc", "medication", "drug"],
            "KyNangTuVan": ["kỹ năng", "kỹ thuật", "phương pháp", "cách tiếp cận"],
            "HanhDongPFA": ["hành động", "can thiệp", "sơ cứu", "ưu tiên"],
            "BuocTuVan": ["bước", "giai đoạn", "quy trình", "tiến trình"],
            "DauHieuNguyHiem": ["nguy hiểm", "khẩn cấp", "red flag", "dấu hiệu nguy"],
            "DoiTuong": ["đối tượng", "học sinh", "người dùng", "vị thành niên"]
        }
    
    async def initialize(self):
        """Initialize all services"""
        logger.info("Initializing GraphIndexer...")
        
        # Initialize services if not provided
        if self.llm is None:
            self.llm = OllamaLLM()
            await self.llm.check_connection()
        
        if self.embedding is None:
            self.embedding = EmbeddingService()
            self.embedding.load_model()
        
        if self.reranker is None:
            self.reranker = RerankerService()
            self.reranker.load_model()
        
        if self.qdrant is None:
            self.qdrant = QdrantService()
            self.qdrant.connect()
            self.qdrant.create_collection(force=False)
        
        if self.neo4j is None:
            self.neo4j = await get_neo4j_service()
            await self.neo4j.create_constraints()
        
        logger.info("✅ GraphIndexer initialized")
    
    async def index_chunks(
        self,
        chunks: List[Dict[str, Any]],
        batch_size: int = 100,
        use_llm_for_extraction: bool = False
    ) -> Dict[str, int]:
        """
        Main indexing pipeline: process all chunks
        
        Args:
            chunks: List of chunk dicts with 'page_content' and 'metadata'
            batch_size: Process in batches
            use_llm_for_extraction: Use LLM for NER (slower but more accurate)
            
        Returns:
            Stats dict with counts
        """
        stats = {
            "total_chunks": len(chunks),
            "chunks_indexed": 0,
            "entities_created": 0,
            "relationships_created": 0,
            "vectors_uploaded": 0
        }
        
        logger.info(f"Starting indexing of {len(chunks)} chunks")
        
        # Step 1: Generate embeddings for all chunks
        logger.info("Generating embeddings...")
        texts = [chunk['page_content'] for chunk in chunks]
        embeddings = self.embedding.encode_documents(texts)
        
        # Step 2: Upload chunks to Qdrant
        logger.info("Uploading to Qdrant...")
        points = self._prepare_qdrant_points(chunks, embeddings)
        self.qdrant.upload_points(points, batch_size=batch_size)
        stats["vectors_uploaded"] = len(points)
        
        # Step 3: Extract entities and build graph
        logger.info("Extracting entities and building graph...")
        entity_map = {}  # { (entity_type, normalized_name): node_id }
        relationships_to_create = []
        
        for i, chunk in enumerate(chunks):
            logger.debug(f"Processing chunk {i+1}/{len(chunks)}")
            
            # Extract entities from chunk
            if use_llm_for_extraction:
                entities = await self._extract_entities_with_llm(chunk['page_content'])
            else:
                entities = self._extract_entities_rule_based(chunk['page_content'])
            
            # Create/get nodes
            chunk_node_id = await self._create_document_chunk_node(chunk, embeddings[i])
            
            for entity in entities:
                entity_type = entity['type']
                entity_name = entity['name']
                normalized_name = self._normalize_entity_name(entity_name)
                
                # Check if node already exists
                node_id = await self._get_or_create_entity_node(
                    entity_type, entity_name, normalized_name, chunk['metadata']
                )
                
                if node_id:
                    entity_map[(entity_type, normalized_name)] = node_id
                    
                    # Create relationship: Entity -> DocumentChunk
                    relationships_to_create.append((
                        node_id,
                        chunk_node_id,
                        "NAM_TRONG_CHUNK",
                        {"relevance_score": entity.get('confidence', 0.8)}
                    ))
            
            # Create relationships between entities in same chunk (co-occurrence)
            if len(entities) >= 2:
                for j in range(len(entities)):
                    for k in range(j+1, len(entities)):
                        rel_type = self._infer_relationship_type(
                            entities[j]['type'], entities[k]['type']
                        )
                        if rel_type:
                            node1 = entity_map.get((entities[j]['type'], 
                                                  self._normalize_entity_name(entities[j]['name'])))
                            node2 = entity_map.get((entities[k]['type'], 
                                                  self._normalize_entity_name(entities[k]['name'])))
                            if node1 and node2:
                                relationships_to_create.append((
                                    node1, node2, rel_type, {"source": "co-occurrence"}
                                ))
            
            stats["chunks_indexed"] += 1
        
        # Step 4: Create all relationships in batch
        logger.info(f"Creating {len(relationships_to_create)} relationships...")
        for source_id, target_id, rel_type, props in relationships_to_create:
            await self.neo4j.create_relationship(source_id, target_id, rel_type, props)
            stats["relationships_created"] += 1
        
        # Step 5: Create additional relationships from predefined entities
        await self._create_predefined_relationships()
        
        logger.info(f"✅ Indexing complete. Stats: {stats}")
        return stats
    
    def _prepare_qdrant_points(
        self, 
        chunks: List[Dict], 
        embeddings: List[List[float]]
    ) -> List[Any]:
        """Prepare points for Qdrant upload"""
        from qdrant_client.http.models import PointStruct
        
        points = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            payload = chunk.copy()  # Includes page_content and metadata
            payload["embedding"] = embedding  # Optional: store in payload
            
            point = PointStruct(
                id=i,
                vector=embedding,
                payload=payload
            )
            points.append(point)
        
        return points
    
    def _extract_entities_rule_based(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract entities using rule-based matching against predefined entities
        This is fast but limited to known entities
        """
        entities = []
        text_lower = text.lower()
        
        # Check predefined entities
        for entity_type, entity_list in self.predefined_entities.items():
            for entity in entity_list:
                name = entity['name']
                aliases = entity.get('aliases', [])
                
                # Check if entity name or any alias appears in text
                search_terms = [name.lower()] + [alias.lower() for alias in aliases]
                for term in search_terms:
                    if term and term in text_lower:
                        entities.append({
                            'type': entity_type,
                            'name': name,
                            'confidence': 0.9,  # High confidence for exact match
                            'source': 'predefined'
                        })
                        break  # Only add once per entity
        
        # Additional keyword matching for unknown entities
        for entity_type, keywords in self.entity_labels.items():
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    # Try to extract the actual entity name
                    # This is simplified - in production use proper NER
                    # For now, just note that this type appears
                    entities.append({
                        'type': entity_type,
                        'name': keyword,  # Placeholder
                        'confidence': 0.5,
                        'source': 'keyword'
                    })
        
        # Deduplicate
        seen = set()
        unique_entities = []
        for ent in entities:
            key = (ent['type'], self._normalize_entity_name(ent['name']))
            if key not in seen:
                seen.add(key)
                unique_entities.append(ent)
        
        return unique_entities
    
    async def _extract_entities_with_llm(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract entities using LLM (more accurate but slower)
        Uses Qwen to perform NER on Vietnamese medical text
        """
        prompt = f"""Phân tích văn bản y tế/ tâm lý sau và liệt kê các thực thể quan trọng.

Văn bản:
{text[:2000]}  # Limit context

Hãy trả về JSON với format:
{{
  "entities": [
    {{"type": "BenhLy|TrieuChung|Thuoc|KyNangTuVan|HanhDongPFA|BuocTuVan|DauHieuNguyHiem|DoiTuong", 
      "name": "tên thực thể", 
      "confidence": 0.0-1.0}}
  ]
}}

Chỉ liệt kê những thực thể có trong văn bản. Tối đa 10 thực thể.
"""
        
        try:
            response = await self.llm.generate_text(prompt)
            
            # Parse JSON from response
            import json
            # Extract JSON from response (might have extra text)
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start != -1 and json_end != -1:
                json_str = response[json_start:json_end]
                data = json.loads(json_str)
                return data.get('entities', [])
            
        except Exception as e:
            logger.warning(f"LLM NER failed: {e}, falling back to rule-based")
        
        # Fallback to rule-based
        return self._extract_entities_rule_based(text)
    
    def _normalize_entity_name(self, name: str) -> str:
        """Normalize entity name for deduplication"""
        # Remove extra spaces, lowercase, remove diacritics for comparison?
        # Keep Vietnamese diacritics, just normalize whitespace
        return ' '.join(name.strip().lower().split())
    
    async def _get_or_create_entity_node(
        self,
        entity_type: str,
        entity_name: str,
        normalized_name: str,
        chunk_metadata: Dict[str, Any]
    ) -> Optional[str]:
        """
        Get existing entity node or create new one
        Returns node ID
        """
        # Check if node exists by name and type
        query = f"""
        MATCH (n:{entity_type} {{name: $name}})
        RETURN n.id as id
        LIMIT 1
        """
        
        async with self.neo4j.driver.session() as session:
            result = await session.run(query, name=entity_name)
            record = await result.single()
            
            if record:
                return record["id"]
            
            # Create new node
            properties = {
                "id": f"{entity_type}_{normalized_name}_{hash(entity_name) % 10000}",
                "name": entity_name,
                "source_domain": self._infer_source_domain(entity_type, chunk_metadata),
                "created_from_chunk": True
            }
            
            # Add severity if applicable
            if entity_type in ["BenhLy", "TrieuChung", "DauHieuNguyHiem"]:
                severity = self._get_default_severity(entity_type, entity_name)
                properties["severity_level"] = severity
            
            # Add aliases if from predefined
            for entity_list in self.predefined_entities.values():
                for entity in entity_list:
                    if entity['name'] == entity_name:
                        properties["aliases"] = entity.get('aliases', [])
                        break
            
            node_id = await self.neo4j.create_node(entity_type, properties)
            logger.debug(f"Created node: {entity_type} - {entity_name}")
            return node_id
    
    def _infer_source_domain(self, entity_type: str, chunk_metadata: Dict) -> str:
        """Infer source domain based on chunk metadata and entity type"""
        doc_type = chunk_metadata.get("doc_type", "unknown")
        
        mapping = {
            "medical_guideline": "Psychiatry_PhacDo",
            "first_aid": "PFA_SoCuu",
            "school_counseling": "Counseling_SoTay"
        }
        
        base_domain = mapping.get(doc_type, "Unknown")
        return f"{base_domain}_{entity_type}"
    
    def _get_default_severity(self, entity_type: str, entity_name: str) -> int:
        """Get default severity for entity (could be from predefined)"""
        # Check predefined entities
        for entity_list in self.predefined_entities.values():
            for entity in entity_list:
                if entity['name'] == entity_name and 'severity_level' in entity:
                    return entity['severity_level']
        
        # Default severities by type
        defaults = {
            "BenhLy": 3,
            "TrieuChung": 2,
            "DauHieuNguyHiem": 5,
            "Thuoc": 1,
            "KyNangTuVan": 1,
            "HanhDongPFA": 4,
            "BuocTuVan": 1,
            "DoiTuong": 1
        }
        
        return defaults.get(entity_type, 1)
    
    def _infer_relationship_type(
        self, 
        entity1_type: str, 
        entity2_type: str
    ) -> Optional[str]:
        """
        Infer what relationship might exist between two entity types
        Based on co-occurrence in same chunk
        """
        # Define possible relationships based on type pairs
        relationship_map = {
            ("BenhLy", "TrieuChung"): "CO_TRIEU_CHUNG",
            ("TrieuChung", "BenhLy"): "CO_TRIEU_CHUNG",  # Undirected
            ("TrieuChung", "DauHieuNguyHiem"): "BAO_HIEU_NGUY_HIEM",
            ("BenhLy", "DauHieuNguyHiem"): "BAO_HIEU_NGUY_HIEM",
            ("DauHieuNguyHiem", "HanhDongPFA"): "YEU_CAU_HANH_DONG",
            ("BenhLy", "Thuoc"): "DIEU_TRI_BANG",
            ("TrieuChung", "KyNangTuVan"): "QUAN_LY_BANG",
            ("BenhLy", "KyNangTuVan"): "QUAN_LY_BANG",
            ("HanhDongPFA", "DoiTuong"): "AP_DUNG_CHO",
            ("KyNangTuVan", "DoiTuong"): "AP_DUNG_CHO",
            ("BuocTuVan", "BuocTuVan"): "BAO_GOM_BUOC"
        }
        
        return relationship_map.get((entity1_type, entity2_type))
    
    async def _create_document_chunk_node(
        self, 
        chunk: Dict[str, Any], 
        embedding: List[float]
    ) -> str:
        """Create DocumentChunk node in Neo4j"""
        properties = {
            "id": chunk['metadata']['chunk_id'],
            "text": chunk['page_content'],
            "source": chunk['metadata']['source'],
            "doc_type": chunk['metadata']['doc_type'],
            "risk_priority": chunk['metadata']['risk_priority'],
            "section": chunk['metadata']['section'],
            "page_no": chunk['metadata']['page_no'],
            "embedding": embedding  # Store embedding in node (or use separate index)
        }
        
        # Check if already exists
        query = "MATCH (n:DocumentChunk {id: $id}) RETURN n.id as id LIMIT 1"
        async with self.neo4j.driver.session() as session:
            result = await session.run(query, id=properties['id'])
            record = await result.single()
            if record:
                return record["id"]
        
        # Create new
        node_id = await self.neo4j.create_node("DocumentChunk", properties)
        return node_id
    
    async def _create_predefined_relationships(self):
        """Create relationships between predefined entities based on design doc"""
        logger.info("Creating predefined relationships...")
        
        # This would create the core graph structure from the design
        # For now, we'll create a subset based on PREDEFINED_ENTITIES
        
        # Example: Connect BenhLy to TrieuChung (would need to match by name)
        # In full implementation, you'd have explicit mappings
        
        # For demo, we'll create a few key relationships manually
        # TODO: Build a proper relationship mapping from the design doc
        
        pass
    
    async def close(self):
        """Cleanup"""
        if self.neo4j:
            await self.neo4j.close()
        logger.info("GraphIndexer closed")


async def index_all_chunks(
    chunks_file: str = "backend/data/processed_chunks.jsonl",
    use_llm: bool = False
) -> Dict[str, int]:
    """
    Convenience function to index all chunks from file
    
    Args:
        chunks_file: Path to JSONL file with chunks
        use_llm: Use LLM for entity extraction (slower)
        
    Returns:
        Stats dictionary
    """
    import json
    
    # Load chunks
    logger.info(f"Loading chunks from {chunks_file}")
    chunks = []
    with open(chunks_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    
    logger.info(f"Loaded {len(chunks)} chunks")
    
    # Initialize indexer
    indexer = GraphIndexer()
    await indexer.initialize()
    
    # Index
    try:
        stats = await indexer.index_chunks(chunks, use_llm_for_extraction=use_llm)
        return stats
    finally:
        await indexer.close()


if __name__ == "__main__":
    import asyncio
    
    async def main():
        # Test indexing
        stats = await index_all_chunks(use_llm=False)
        print(f"Indexing stats: {stats}")
    
    asyncio.run(main())