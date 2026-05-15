"""
Configuration settings for the GraphRAG Psychology Chatbot
"""
import os
from typing import List
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Application settings"""

    # App settings
    APP_NAME: str = "GraphRAG Psychology Chatbot"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # LLM settings - Support both Ollama and Claude API
    # Use LLM_PROVIDER="ollama" or "claude" to switch
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "ollama")  # "ollama" or "claude"

    # Ollama settings (used only if LLM_PROVIDER=ollama)
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")

    # Claude/Anthropic API settings (used if LLM_PROVIDER=claude)
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    ANTHROPIC_API_BASE: str = os.getenv("ANTHROPIC_API_BASE", "http://localhost:20128/v1")
    ANTHROPIC_MODEL: str = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5")

    # Common LLM settings
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.7"))
    LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "512"))

    # Embedding Model
    EMBEDDING_MODEL: str = "AITeamVN/Vietnamese_Embedding"
    EMBEDDING_DIMENSION: int = 1024  # Actual dimension of the model

    # Reranker Model
    RERANKER_MODEL: str = "AITeamVN/Vietnamese_Reranker"
    RERANKER_TOP_K: int = 5

    # Qdrant Vector DB
    QDRANT_URL: str = "http://qdrant:6333"
    QDRANT_COLLECTION_NAME: str = "psychology_chunks"
    QDRANT_VECTOR_SIZE: int = 1024  # Must match EMBEDDING_DIMENSION

    # Neo4j Graph DB
    NEO4J_URI: str = "bolt://neo4j:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "password"

    # GraphRAG settings
    SEVERITY_LEVELS: List[int] = [1, 2, 3, 4, 5]
    TRIAGE_THRESHOLD_HIGH: int = 4  # severity >= 4 → Crisis/PFA
    TRIAGE_THRESHOLD_MEDIUM: int = 3  # severity 2-3 → Counseling
    TRIAGE_THRESHOLD_LOW: int = 1  # severity 1 → General

    # Chunking settings (used by chunk processor)
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 150

    # API settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "GraphRAG Psychology Chatbot"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()