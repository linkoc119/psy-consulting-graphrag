"""
Embedding Service using Vietnamese_Embedding model
"""
import logging
from typing import List, Union, Optional
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
        logger.info(f"Initializing EmbeddingService on device: {self.device}")
        
    def load_model(self):
        """Load the embedding model"""
        if self.model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            try:
                self.model = SentenceTransformer(self.model_name, device=self.device)
                logger.info(f"✅ Model loaded. Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
                raise
    
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
        """Generate embedding for a single query (normalized)"""
        return self.encode(query, normalize_embeddings=True)[0]
    
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