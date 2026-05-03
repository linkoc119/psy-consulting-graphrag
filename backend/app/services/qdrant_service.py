"""
Qdrant Vector Database Service
"""
import logging
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance, VectorParams, PointStruct, 
    Filter, FieldCondition, MatchValue
)
from config import settings

QDRANT_URL = settings.QDRANT_URL
QDRANT_COLLECTION_NAME = settings.QDRANT_COLLECTION_NAME
QDRANT_VECTOR_SIZE = settings.QDRANT_VECTOR_SIZE

logger = logging.getLogger(__name__)

class QdrantService:
    """Service for interacting with Qdrant vector database"""
    
    def __init__(self, url: str = QDRANT_URL, collection_name: str = QDRANT_COLLECTION_NAME):
        self.url = url
        self.collection_name = collection_name
        self.client = None
        
    def connect(self):
        """Connect to Qdrant server"""
        try:
            self.client = QdrantClient(url=self.url)
            logger.info(f"✅ Connected to Qdrant at {self.url}")
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            raise
    
    def create_collection(self, vector_size: int = QDRANT_VECTOR_SIZE, force: bool = False):
        """Create collection if it doesn't exist"""
        try:
            # Check if collection exists
            collections = self.client.get_collections().collections
            exists = any(c.name == self.collection_name for c in collections)
            
            if exists:
                if force:
                    logger.info(f"Collection {self.collection_name} exists, recreating...")
                    self.client.delete_collection(self.collection_name)
                else:
                    logger.info(f"Collection {self.collection_name} already exists")
                    return
            
            # Create new collection
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE
                )
            )
            logger.info(f"✅ Created collection {self.collection_name} with vector size {vector_size}")
            
        except Exception as e:
            logger.error(f"Failed to create collection: {e}")
            raise
    
    def upload_points(
        self, 
        points: List[PointStruct], 
        batch_size: int = 100
    ):
        """Upload points to collection in batches"""
        try:
            total = len(points)
            for i in range(0, total, batch_size):
                batch = points[i:i+batch_size]
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch
                )
                logger.debug(f"Uploaded batch {i//batch_size + 1}/{(total+batch_size-1)//batch_size}")
            logger.info(f"✅ Uploaded {total} points to collection {self.collection_name}")
        except Exception as e:
            logger.error(f"Failed to upload points: {e}")
            raise
    
    def search(
        self,
        query_vector: List[float],
        limit: int = 10,
        filter_conditions: Optional[Dict] = None,
        with_payload: bool = True,
        with_vectors: bool = False
    ) -> List[Dict]:
        """
        Search for similar vectors
        
        Args:
            query_vector: The query embedding vector
            limit: Number of results to return
            filter_conditions: Optional filters (e.g., {"doc_type": "medical_guideline"})
            with_payload: Include payload in results
            with_vectors: Include vectors in results
            
        Returns:
            List of search results with score and payload
        """
        try:
            # Build filter if provided
            qdrant_filter = None
            if filter_conditions:
                conditions = []
                for field, value in filter_conditions.items():
                    conditions.append(
                        FieldCondition(
                            key=field,
                            match=MatchValue(value=value)
                        )
                    )
                if conditions:
                    qdrant_filter = Filter(must=conditions)
            
            # Execute search
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit,
                query_filter=qdrant_filter,
                with_payload=with_payload,
                with_vectors=with_vectors
            )
            
            # Format results
            formatted = []
            for res in results:
                item = {
                    "id": res.id,
                    "score": res.score,
                }
                if with_payload and res.payload:
                    item["payload"] = res.payload
                formatted.append(item)
            
            return formatted
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def delete_by_filter(self, filter_conditions: Dict) -> bool:
        """Delete points matching filter"""
        try:
            conditions = []
            for field, value in filter_conditions.items():
                conditions.append(
                    FieldCondition(
                        key=field,
                        match=MatchValue(value=value)
                    )
                )
            
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=Filter(must=conditions)
            )
            logger.info(f"Deleted points with filter: {filter_conditions}")
            return True
        except Exception as e:
            logger.error(f"Delete failed: {e}")
            return False
    
    def get_collection_info(self) -> Dict:
        """Get collection information"""
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "points_count": info.points_count,
                "vectors_count": info.vectors_count,
                "status": info.status,
                "vector_size": info.config.params.vectors.size,
                "distance": info.config.params.vectors.distance
            }
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return {}
    
    def scroll(
        self,
        limit: int = 100,
        offset: Optional[int] = None,
        with_payload: bool = True
    ) -> List[Dict]:
        """Scroll through all points in collection"""
        try:
            results = []
            next_offset = offset
            
            while True:
                batch, next_offset = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=limit,
                    offset=next_offset,
                    with_payload=with_payload
                )
                
                for point in batch:
                    results.append({
                        "id": point.id,
                        "payload": point.payload or {}
                    })
                
                if next_offset is None:
                    break
            
            return results
        except Exception as e:
            logger.error(f"Scroll failed: {e}")
            return []


# Singleton instance
_qdrant_instance: Optional[QdrantService] = None

def get_qdrant_service() -> QdrantService:
    """Get or create singleton Qdrant service instance"""
    global _qdrant_instance
    if _qdrant_instance is None:
        _qdrant_instance = QdrantService()
        _qdrant_instance.connect()
    return _qdrant_instance


if __name__ == "__main__":
    # Test connection
    qdrant = get_qdrant_service()
    info = qdrant.get_collection_info()
    print(f"Collection info: {info}")