"""
Reranker Service using Vietnamese_Reranker model
"""
import logging
from typing import List, Tuple, Union, Optional
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from config import settings

RERANKER_MODEL = settings.RERANKER_MODEL
RERANKER_TOP_K = settings.RERANKER_TOP_K

logger = logging.getLogger(__name__)

class RerankerService:
    """Service for reranking retrieved documents using Vietnamese_Reranker"""
    
    def __init__(self, model_name: str = RERANKER_MODEL, top_k: int = RERANKER_TOP_K):
        self.model_name = model_name
        self.top_k = top_k
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Initializing RerankerService on device: {self.device}")
    
    def load_model(self):
        """Load the reranker model and tokenizer"""
        if self.model is None or self.tokenizer is None:
            logger.info(f"Loading reranker model: {self.model_name}")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
                self.model.to(self.device)
                self.model.eval()
                logger.info("✅ Reranker model loaded")
                # Warm-up: run a dummy inference to avoid first-call latency
                logger.info("Warming up reranker model...")
                _ = self.rerank("warm up", ["dummy document"], [{}])
                logger.info("✅ Reranker warm-up complete")
            except Exception as e:
                logger.error(f"Failed to load reranker model: {e}")
                raise
    
    def rerank(
        self, 
        query: str, 
        documents: List[str], 
        metadata: List[dict] = None
    ) -> List[Tuple[str, float, dict]]:
        """
        Rerank documents based on relevance to query
        
        Args:
            query: The search query
            documents: List of document texts to rerank
            metadata: Optional list of metadata dicts for each document
            
        Returns:
            List of tuples: (document, score, metadata) sorted by score descending
        """
        if self.model is None or self.tokenizer is None:
            self.load_model()
        
        if not documents:
            return []
        
        if metadata is None:
            metadata = [{} for _ in documents]
        
        try:
            # Prepare pairs for cross-encoding
            pairs = [(query, doc) for doc in documents]
            
            # Tokenize
            inputs = self.tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)
            
            # Inference
            with torch.no_grad():
                outputs = self.model(**inputs)
                scores = outputs.logits.squeeze(-1).cpu().numpy()
            
            # If single document, convert to list
            if scores.ndim == 0:
                scores = [float(scores)]
            else:
                scores = scores.tolist()
            
            # Create result tuples
            results = [
                (doc, float(score), meta) 
                for doc, score, meta in zip(documents, scores, metadata)
            ]
            
            # Sort by score descending
            results.sort(key=lambda x: x[1], reverse=True)
            
            # Return top_k
            return results[:self.top_k]
            
        except Exception as e:
            logger.error(f"Error during reranking: {e}")
            # Return original order with default scores if reranking fails
            return [(doc, 0.0, meta) for doc, meta in zip(documents, metadata)]
    
    def rerank_with_graph_context(
        self, 
        query: str, 
        graph_context: List[dict],
        top_k: int = None
    ) -> List[dict]:
        """
        Rerank graph nodes/relationships based on query relevance
        
        Args:
            query: The search query
            graph_context: List of graph entities with 'text' field
            top_k: Override default top_k
            
        Returns:
            Reranked list of graph entities with added 'rerank_score'
        """
        if top_k is None:
            top_k = self.top_k
        
        documents = [item.get('text', '') for item in graph_context]
        metadata = [{k: v for k, v in item.items() if k != 'text'} for item in graph_context]
        
        reranked = self.rerank(query, documents, metadata)
        
        # Merge back with original structure
        results = []
        for doc, score, meta in reranked:
            # Find original item
            for item in graph_context:
                if item.get('text', '') == doc:
                    result = item.copy()
                    result['rerank_score'] = score
                    results.append(result)
                    break
        
        return results
    
    def unload_model(self):
        """Unload model from memory"""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Reranker model unloaded")


# Singleton instance
_reranker_instance: Optional[RerankerService] = None

def get_reranker_service() -> RerankerService:
    """Get or create singleton reranker service instance"""
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = RerankerService()
    return _reranker_instance


if __name__ == "__main__":
    # Test
    reranker = RerankerService()
    reranker.load_model()
    
    query = "Tôi lo lắng về tương lai"
    docs = [
        "Tư vấn tâm lý học đường giúp học sinh giải quyết vấn đề tâm lý.",
        "Triệu chứng lo âu bao gồm tim đập nhanh, khó thở, và cảm giác căng thẳng.",
        "Thuốc an thần kinh có thể được dùng để điều trị các rối loạn lo âu nghiêm trọng."
    ]
    
    results = reranker.rerank(query, docs)
    print("\nReranking results:")
    for i, (doc, score, _) in enumerate(results, 1):
        print(f"{i}. Score: {score:.4f} - {doc[:80]}...")