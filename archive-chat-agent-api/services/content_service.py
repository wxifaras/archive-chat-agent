import logging
import os
from typing import IO, List
import asyncio
from pydantic import ValidationError
import tiktoken
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

                for attachment in attachments:
                    ext = os.path.splitext(attachment.filename)
                    file_type = ext.lstrip('.').lower()
            except ValidationError as e:
                logger.error(f"Pydantic validation failed: {e}")

    async def search_content(chat_req: ChatRequest) -> ChatResponse:
        # Implement search logic here
        pass

    @staticmethod
    def chunk_text(allContent):
        enc = tiktoken.get_encoding("o200k_base")
        pages = allContent.pages
        page_tokens = []
        page_numbers = []

        # Precompute tokens for each page
        for page in pages:
            page_text = " ".join(w['content'] for w in page.get('words', []))
            tokens = enc.encode(page_text)
            page_tokens.append(tokens)
            page_numbers.append(page.get('pageNumber'))

        chunks = []
        
        page_overlap_percent = float(settings.PAGE_OVERLAP) / 100

        for idx, tokens in enumerate(page_tokens):

            prev_tokens = []
            if idx > 0:
                prev = page_tokens[idx - 1]
                prev_len = max(1, int(len(prev) * page_overlap_percent))
                prev_tokens = prev[-prev_len:]

            next_tokens = []
            if idx < len(page_tokens) - 1:
                nxt = page_tokens[idx + 1]
                next_len = max(1, int(len(nxt) * page_overlap_percent))
                next_tokens = nxt[:next_len]

            combined_tokens = prev_tokens + tokens + next_tokens
            combined_page_map = (
                [page_numbers[idx - 1]] * len(prev_tokens) if idx > 0 else []
            ) + [page_numbers[idx]] * len(tokens) + (
                [page_numbers[idx + 1]] * len(next_tokens) if idx < len(page_tokens) - 1 else []
            )

            chunked_text = enc.decode(combined_tokens)
            pages_in_chunk = sorted(set(p for p in combined_page_map if p is not None))

            chunks.append({
                'chunk_tokens': combined_tokens,
                'chunked_text': chunked_text,
                'pages': pages_in_chunk,
                'page_token_map': combined_page_map
            })

        return chunks