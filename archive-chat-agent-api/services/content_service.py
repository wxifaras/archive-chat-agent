import logging
from typing import IO, List

from pydantic import ValidationError
from services.azure_ai_search_service import AzureAISearchService
from core.settings import settings
from models.email_item import EmailItem, EmailList
from models.chat_request import ChatRequest
from models.chat_response import ChatResponse

azure_search_service = AzureAISearchService()

logger = logging.getLogger(__name__)
logger.setLevel(settings.LOG_LEVEL)

# Console handler (prints to terminal)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

# Formatter
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

ch.setFormatter(formatter)

# Add handler
logger.addHandler(ch)

class ContentService:
    def __init__(self):
        if not all([
            settings.PAGE_OVERLAP
        ]):
            raise ValueError("Required settings are missing")
        
    async def process_content(
            self, 
            document_id: str,
            json_file_name: str,
            json_content: str,
            attachments: List[IO]):
        
            try:
                email_list = EmailList.model_validate_json(json_content)
                email_item = email_list.root[0]
                # index = azure_search_service.create_index()
                # TODO: Create index
                #       Chunk json_content
                #       Extract text from attachments and chunk them
                #       Upload to Azure AI Search
                #       Store uploaded files to Azure Blob Storage
            
            except ValidationError as e:
                logger.error(f"Pydantic validation failed: {e}")

    async def search_content(chat_req: ChatRequest) -> ChatResponse:
        # Implement search logic here
        pass