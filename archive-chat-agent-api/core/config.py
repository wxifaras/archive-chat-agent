from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List
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
    AZURE_FOUNDRY_PROJECT_ENDPOINT: str
    AZURE_FOUNDRY_PROJECT_KEY: str
    NUM_SEARCH_RESULTS: int = Field(default=5)
    COSMOS_DATABASE_NAME: str
    COSMOS_CONTAINER_NAME: str
    COSMOS_ENDPOINT: str
    USE_IN_MEMORY_CHAT_HISTORY: bool = False

    # Optional logging settings
    LOG_LEVEL: str = "INFO"

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True)

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