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
    chunk_size: int = 500
    chunk_overlap: int = 50

    # Retrieval
    top_k: int = 4

    # Ollama (local LLM, no API key needed)
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:latest"
    ollama_timeout_seconds: int = 120

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


# Single settings instance used across the app
settings = Settings()
