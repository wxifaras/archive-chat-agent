from openai import AzureOpenAI
from core.settings import settings
import json
from typing import List, Dict, Generator

class AzureOpenAIService:
    def __init__(self):
        if not all([
            settings.AZURE_OPENAI_ENDPOINT,
            settings.AZURE_OPENAI_API_KEY,
            settings.AZURE_OPENAI_API_VERSION,
            settings.AZURE_OPENAI_DEPLOYMENT_NAME,
            settings.AZURE_OPENAI_TEXT_EMBEDDING_DEPLOYMENT_NAME
        ]):
            raise ValueError("Required Azure OpenAI settings are missing")

        self.client = AzureOpenAI(
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
            api_key=settings.AZURE_OPENAI_API_KEY,
            api_version=settings.AZURE_OPENAI_API_VERSION,
        )

        self.deployment_name = settings.AZURE_OPENAI_DEPLOYMENT_NAME

def create_embedding(
            self, 
            text: str
        ):

        response = self.client.embeddings.create(
            model=settings.AZURE_OPENAI_TEXT_EMBEDDING_DEPLOYMENT_NAME,
            input=text
        )
        
        embeddings = response.data[0].embedding
        return embeddings