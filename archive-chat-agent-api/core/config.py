from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List, Optional
from functools import lru_cache

class Settings(BaseSettings):
    
     # Project settings
    PROJECT_NAME: str = "Archive Chat API"
    API_PREFIX: str = "/api"
    DEBUG: bool = False
    
    # CORS
    CORS_ORIGINS: List[str] = ["*"]
        
    AZURE_OPENAI_ENDPOINT: str
    AZURE_OPENAI_API_KEY: str
    AZURE_OPENAI_DEPLOYMENT_NAME: str
    AZURE_OPENAI_API_VERSION: str
    AZURE_OPENAI_TEXT_EMBEDDING_DEPLOYMENT_NAME: str
    AZURE_STORAGE_CONNECTION_STRING: str
    AZURE_STORAGE_ARCHIVE_CONTAINER_NAME: str
    AZURE_DOCUMENTINTELLIGENCE_ENDPOINT: str
    AZURE_DOCUMENTINTELLIGENCE_API_KEY: str
    PAGE_OVERLAP: int = Field(gt=0)
    MAX_TOKENS: int = Field(gt=0)
    AZURE_AI_SEARCH_SERVICE_ENDPOINT: str
    AZURE_AI_SEARCH_SERVICE_KEY: str
    AZURE_AI_SEARCH_INDEX_NAME: str
    NUM_SEARCH_RESULTS: int = Field(default=5)
    RERANKER_SCORE_THRESHOLD: float = Field(default=1.0, ge=0.0, le=4.0, description="Minimum reranker score threshold for Azure AI Search (0.0-4.0)")
    COSMOS_DATABASE_NAME: str
    COSMOS_CONTAINER_NAME: str
    COSMOS_ENDPOINT: str
    USE_IN_MEMORY_CHAT_HISTORY: bool = False
    USE_SEMANTIC_CHUNKING: bool = False
    WATCHFILES_IGNORE_PATHS: Optional[str] = None   
    
    # Evaluation settings
    EVALUATION_MAX_CONCURRENT: int = Field(default=5, description="Maximum concurrent evaluations for parallel processing")
    AZURE_OPENAI_EVALUATION_DEPLOYMENT_NAME: str = Field(default="gpt-4.1", description="Model deployment for evaluation tasks (separate from RAG pipeline model)")
    EVALUATION_TEMPERATURE: float = Field(default=0.0, ge=0.0, le=2.0, description="Temperature for evaluation model (0.0 = deterministic, higher = more random)")

    # Optional logging settings
    LOG_LEVEL: str = "INFO"

    # Use semantic chunking (bool)
    USE_SEMANTIC_CHUNKING: bool = Field(default=False, description="Enable semantic chunking")
    SEMANTIC_CHUNK_BREAKPOINT_THRESHOLD_AMOUNT: float = Field(default=95.0, description="Semantic chunking breakpoint threshold amount")
    SEMANTIC_CHUNK_MIN_CHUNK_SIZE: int = Field(default=1, description="Semantic chunking minimum chunk size")

    model_config = SettingsConfigDict(
        env_file=".env", 
        case_sensitive=True
    )

    @model_validator(mode='after')
    def check_required_fields(self) -> 'Settings':
        for field_name, field in self.__class__.model_fields.items():
            if field.is_required() and getattr(self, field_name) is None:
                raise ValueError(f"{field_name} environment variable is required")
        return self

@lru_cache()
def get_settings() -> Settings:
    return Settings()

settings = get_settings()