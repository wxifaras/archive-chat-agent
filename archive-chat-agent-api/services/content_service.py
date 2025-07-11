import logging
import os
from typing import List
from pydantic import ValidationError
import tiktoken
from fastapi import UploadFile
from services.azure_ai_search_service import AzureAISearchService
from services.azure_doc_intel_service import AzureDocIntelService
from services.azure_storage_service import AzureStorageService
from core.settings import settings
from models.email_item import EmailItem, EmailList
from models.chat_response import ChatResponse
from services.azure_openai_service import AzureOpenAIService

azure_search_service = AzureAISearchService()
azure_doc_intell_service =AzureDocIntelService()
azure_storage_service = AzureStorageService()
azure_openai_service = AzureOpenAIService()

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
            attachments: List[UploadFile]):
        
            try: 
                email_list = EmailList.model_validate_json(json_content)
                email_item = email_list.root[0]
                index = azure_search_service.create_index()

                if len(attachments) == 0:
                    provenance_source = azure_openai_service.get_source_from_provenance(email_item.Provenance)
                    email_item.Provenance_Source = provenance_source

                # Chunking and indexing the text field of the json document
                # Chunking text field by token count
                blob_path= azure_storage_service.upload_file(str(email_item.projectId),json_content, json_file_name)
                if len(email_item.text)==0:
                    email_item.text = "No text content."
                jsonText = self.chunk_json_text(email_item.text)       
                azure_search_service.index_content(jsonText, document_id, email_item, file_name=json_file_name, file_type=".json")

                # Extract text from attachments using Azure Doc Intell and chunk them by page
                attachmentChunks = []
                for attachment in attachments:

                    file_content = await attachment.read()
                    blob_path= azure_storage_service.upload_file(str(email_item.projectId),file_content, attachment.filename)
                    sas_url = azure_storage_service.generate_blob_sas_url(blob_path)
                    file_type = os.path.splitext(attachment.filename)
                    
                    email_item.Provenance_Source = file_type
                    
                    allExtractedContent = azure_doc_intell_service.extract_content(sas_url)
                    attachmentChunks= self.chunk_text(allExtractedContent)

                    azure_search_service.index_content(
                        attachmentChunks, 
                        document_id, 
                        email_item, 
                        file_name=attachment.filename, 
                        file_type=file_type[1],
                        page_number= [chunk['pages'] for chunk in attachmentChunks]
                    )

            except Exception as e:
                logger.error(f"Pydantic validation failed: {e}")

    def search_content(self, user_id: str, session_id: str, message: str) -> ChatResponse:
        # Implement search logic here
        # For now, return a simple response
        return ChatResponse(
            response="This is a placeholder response. Implement your search logic here.",
            metadata={"user_id": user_id, "session_id": session_id, "message": message}
        )

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

    @staticmethod
    def chunk_json_text(json_text_field):
        """
        chunking text field in JSON document
        chunking is by token count
        """
        max_chunk_size = settings.MAX_TOKENS
        overlap_percentage = float(settings.PAGE_OVERLAP) / 100
        overlap_size = int(max_chunk_size * overlap_percentage)
        enc = tiktoken.get_encoding("o200k_base")
        token_ids = enc.encode(json_text_field)
        chunks = []
        i = 0
        while i < len(token_ids):
            chunk_tokens = token_ids[i:i + max_chunk_size]
            chunk_text = enc.decode(chunk_tokens)
            chunks.append({
                'chunked_text': chunk_text,
                'start_token': i,
                'end_token': i + len(chunk_tokens)
            })
            if len(chunk_tokens) < max_chunk_size:
                # If the chunk is smaller than max_chunk_size, it's the last chunk; return early
                return chunks
            i += max_chunk_size - overlap_size
        return chunks