"""
Application configuration.

All settings are read from environment variables (or a local .env file).
No API keys are required because the LLM runs locally through Ollama.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Vector store
    chroma_dir: str = "./chroma_data"
    collection_name: str = "documents"

    # Embeddings
    embedding_model: str = "all-MiniLM-L6-v2"

    # Chunking
    chunk_strategy: str = "recursive"  # "recursive" or "semantic"
    chunk_size: int = 500
    chunk_overlap: int = 50

    # Semantic chunking
    semantic_threshold: float = 0.5  # cosine distance above which a new chunk starts

    # Retrieval
    top_k: int = 4

    # Reranking (cross-encoder over the bi-encoder candidates)
    rerank_enabled: bool = True
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_candidates: int = 12

    # Ollama (local LLM, no API key needed)
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:latest"
    ollama_timeout_seconds: int = 120

    # CORS — restrict to the origins that actually call this service.
    # Comma-separated list, e.g. "http://localhost:8080,http://localhost:3000"
    allowed_origins: str = "http://localhost:8080"

    # Logging
    log_level: str = "INFO"
    json_logs: bool = False

    # Validation bounds
    max_question_length: int = 2000
    max_top_k: int = 50

    # Ollama generation retries
    ollama_max_retries: int = 2
    ollama_base_backoff_seconds: float = 2.0

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


# Single settings instance used across the app
settings = Settings()
