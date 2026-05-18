"""
Configuration settings for the GraphRAG Psychology Chatbot
Optimized for RTX 3050 4GB VRAM
"""
import os
from typing import List, Optional
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Application settings"""

    # App settings
    APP_NAME: str = "GraphRAG Psychology Chatbot"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # Device settings 
    DEVICE: str = os.getenv("DEVICE", "cpu") 
    EMBEDDING_DEVICE: Optional[str] = os.getenv("EMBEDDING_DEVICE", "cpu")
    RERANKER_DEVICE: Optional[str] = os.getenv("RERANKER_DEVICE", "cpu")

    # LLM settings 
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "ollama")

    # Claude settings
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    ANTHROPIC_API_BASE: str = os.getenv("ANTHROPIC_API_BASE", "http://localhost:20128/v1")
    ANTHROPIC_MODEL: str = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5")

    # Ollama settings
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "qwen2.5:3b-instruct-q6_K") 

    # Common LLM settings
    LLM_TEMPERATURE: float = 0.3  # Giảm xuống 0.3 để câu trả lời ngắn gọn, súc tích và nhanh hơn
    LLM_MAX_TOKENS: int = 512

    # Vector Search settings
    VECTOR_SEARCH_TOP_K: int = 10 
    
    # Reranker Model
    RERANKER_MODEL: str = "AITeamVN/Vietnamese_Reranker"
    RERANKER_TOP_K: int = 3 

    # Embedding Model
    EMBEDDING_MODEL: str = "AITeamVN/Vietnamese_Embedding"
    EMBEDDING_DIMENSION: int = 1024

    # Qdrant Vector DB
    QDRANT_URL: str = os.getenv("QDRANT_URL", "http://qdrant:6333")
    QDRANT_COLLECTION_NAME: str = "psychology_chunks"
    QDRANT_VECTOR_SIZE: int = 1024

    # Neo4j Graph DB
    NEO4J_URI: str = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
    NEO4J_USER: str = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD: str = os.getenv("NEO4J_PASSWORD", "password")

    # GraphRAG settings
    SEVERITY_LEVELS: List[int] = [1, 2, 3, 4, 5]
    TRIAGE_THRESHOLD_HIGH: int = 4
    TRIAGE_THRESHOLD_MEDIUM: int = 3
    TRIAGE_THRESHOLD_LOW: int = 1

    # Chunking settings
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 150

    # API settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "GraphRAG Psychology Chatbot"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore" 

settings = Settings()