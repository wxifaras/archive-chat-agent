from openai import AsyncAzureOpenAI
from core.settings import settings
from prompts.core_prompts import PROVENANCE_SOURCE_SYSTEM_PROMPT
from typing import List, Dict

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

        self.client = AsyncAzureOpenAI(
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
            api_key=settings.AZURE_OPENAI_API_KEY,
            api_version=settings.AZURE_OPENAI_API_VERSION,
        )

        self.deployment_name = settings.AZURE_OPENAI_DEPLOYMENT_NAME

    async def create_embedding(
            self,
            text: str
        ):

        response = await self.client.embeddings.create(
            model=settings.AZURE_OPENAI_TEXT_EMBEDDING_DEPLOYMENT_NAME,
            input=text
        )
        
        embeddings = response.data[0].embedding
        return embeddings

    async def get_source_from_provenance(
            self,
            provenance_text: str
        ) -> str:

        response = await self.client.beta.chat.completions.create(
            model=self.deployment_name,
            messages=[
                {"role": "system", "content": PROVENANCE_SOURCE_SYSTEM_PROMPT},
                {"role": "user", "content": provenance_text}
            ]
        )

        message_content = response.choices[0].message.content
        return message_content
    
    # This method is used to get a chat response from the Azure OpenAI service using the supplied response format for a structured output response
    async def get_chat_response(
            self,
            messages: List[Dict[str, str]],
            response_format: type
        ):
        response = await self.client.beta.chat.completions.parse(
            model=self.deployment_name,
            messages=messages,
            response_format=response_format
        )

        message_content = response.choices[0].message.parsed

        return message_content
    
    async def get_chat_response_text(self, messages: List[Dict[str, str]]) -> str:
        """
        Gets a simple text response from Azure OpenAI (non-streaming, no structured output)
        """
        response = await self.client.chat.completions.create(
            model=self.deployment_name,
            messages=messages
        )
        
        message_content = response.choices[0].message.content
        
        return message_content