"""
Embedding Service using Vietnamese_Embedding model
"""
import logging
from typing import List, Union, Optional
from functools import lru_cache
import torch
from sentence_transformers import SentenceTransformer
from config import settings

EMBEDDING_MODEL = settings.EMBEDDING_MODEL
EMBEDDING_DIMENSION = settings.EMBEDDING_DIMENSION

logger = logging.getLogger(__name__)

class EmbeddingService:
    """Service for generating embeddings using Vietnamese_Embedding"""

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model_name = model_name
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Set torch threads for CPU optimization
        if self.device == "cpu":
            torch.set_num_threads(4)  # Use 4 threads for CPU inference
        # Cache for query embeddings (simple dict cache)
        self._query_cache = {}
        logger.info(f"Initializing EmbeddingService on device: {self.device}")

    def load_model(self):
        """Load the embedding model"""
        if self.model is None:
            import time
            logger.info(f"Loading embedding model: {self.model_name}")
            t0 = time.time()
            try:
                self.model = SentenceTransformer(self.model_name, device=self.device)
                load_time = time.time() - t0
                dim = self.model.get_sentence_embedding_dimension()
                logger.info(f"✅ Model loaded in {load_time*1000:.0f}ms. Embedding dimension: {dim}")
                # Warm-up: run dummy inference to avoid first-call latency
                logger.info("Warming up embedding model...")
                t_warm = time.time()
                _ = self.model.encode(
                    ["warm up"],
                    convert_to_numpy=True,
                    normalize_embeddings=True
                )
                warm_time = time.time() - t_warm
                logger.info(f"✅ Embedding warm-up complete in {warm_time*1000:.0f}ms")
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
                raise

    @lru_cache(maxsize=100)
    def encode_query_cached(self, query: str) -> List[float]:
        """Generate embedding for a single query with caching"""
        return self.encode_query(query)
    
    def encode(
        self, 
        texts: Union[str, List[str]], 
        batch_size: int = 32,
        normalize_embeddings: bool = True
    ) -> List[List[float]]:
        """
        Generate embeddings for text(s)
        
        Args:
            texts: Single text or list of texts
            batch_size: Batch size for encoding
            normalize_embeddings: Whether to normalize embeddings to unit length
            
        Returns:
            List of embedding vectors
        """
        if self.model is None:
            self.load_model()
        
        # Convert single text to list
        if isinstance(texts, str):
            texts = [texts]
        
        try:
            # Generate embeddings
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=normalize_embeddings
            )
            
            # Convert to list of lists
            embeddings_list = embeddings.tolist()
            
            # Verify dimension
            if embeddings_list and len(embeddings_list[0]) != EMBEDDING_DIMENSION:
                logger.warning(f"Embedding dimension mismatch: expected {EMBEDDING_DIMENSION}, got {len(embeddings_list[0])}")
            
            return embeddings_list
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    def encode_query(self, query: str) -> List[float]:
        """Generate embedding for a single query (normalized) with caching"""
        if query in self._query_cache:
            logger.debug(f"Embedding cache hit for query: {query[:50]}...")
            return self._query_cache[query]
        embedding = self.encode(query, normalize_embeddings=True)[0]
        self._query_cache[query] = embedding
        return embedding
    
    def encode_documents(self, documents: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple documents"""
        return self.encode(documents, normalize_embeddings=True)
    
    def similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Compute cosine similarity between two embeddings"""
        import numpy as np
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
    
    def unload_model(self):
        """Unload model from memory"""
        if self.model is not None:
            del self.model
            self.model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Embedding model unloaded")


# Singleton instance
_embedding_instance: Optional[EmbeddingService] = None

def get_embedding_service() -> EmbeddingService:
    """Get or create singleton embedding service instance"""
    global _embedding_instance
    if _embedding_instance is None:
        _embedding_instance = EmbeddingService()
    return _embedding_instance


if __name__ == "__main__":
    # Test
    service = EmbeddingService()
    service.load_model()
    
    test_texts = ["Tôi cảm thấy lo lắng về kỳ thi sắp tới", "Tư vấn tâm lý học đường"]
    embeddings = service.encode(test_texts)
    print(f"Generated {len(embeddings)} embeddings with dimension {len(embeddings[0])}")
    print(f"Sample similarity: {service.similarity(embeddings[0], embeddings[1]):.4f}")