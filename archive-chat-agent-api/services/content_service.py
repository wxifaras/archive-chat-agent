import logging
import os
from typing import List
import uuid
import tiktoken
import time
from fastapi import UploadFile
from services.azure_ai_search_service import AzureAISearchService, SearchResult
from services.azure_doc_intel_service import AzureDocIntelService
from services.azure_storage_service import AzureStorageService
from core.settings import settings
from models.email_item import EmailList
from services.azure_openai_service import AzureOpenAIService
from models.content_conversation import ContentConversation, ConversationResult, ReviewDecision, SearchPromptResponse
from models.chat_history import ChatMessage, Role
from services.in_memory_chat_history_manager import InMemoryChatHistoryManager

azure_search_service = AzureAISearchService()
azure_doc_intell_service =AzureDocIntelService()
azure_storage_service = AzureStorageService()
azure_openai_service = AzureOpenAIService()

if not settings.USE_IN_MEMORY_CHAT_HISTORY:
    from services.chat_history_manager import ChatHistoryManager
    chat_history_manager = ChatHistoryManager()
else:
    chat_history_manager = InMemoryChatHistoryManager()

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

MAX_ATTEMPTS = 3

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
                index = await azure_search_service.create_index()

                if len(attachments) == 0:
                    provenance_source = await azure_openai_service.get_source_from_provenance(email_item.Provenance)
                    email_item.Provenance_Source = provenance_source

                # Chunking and indexing the text field of the json document
                # Chunking text field by token count
                blob_path, uploaded = await azure_storage_service.upload_file_with_dup_check(
                    str(email_item.projectId),
                    json_content,
                    json_file_name
                )

                if not uploaded:
                    logger.warning(f"File {json_file_name} already exists in Azure Storage. Skipping JSON upload.")
                else:   
                    if len(email_item.text)==0:
                        email_item.text = "No text content."

                    jsonText = self.chunk_json_text(email_item.text)       
                    await azure_search_service.index_content(jsonText, document_id, email_item, file_name=json_file_name, file_type="json")

                # Extract text from attachments using Azure Doc Intell and chunk them by page
                attachmentChunks = []
                for attachment in attachments:
                    file_content = await attachment.read()
                    
                    blob_path, uploaded = await azure_storage_service.upload_file_with_dup_check(
                        str(email_item.projectId),
                        file_content,
                        attachment.filename
                    )
                    
                    if not uploaded:
                        logger.warning(f"File {attachment.filename} already exists in Azure Storage. Skipping Attachment upload.")
                        continue

                    sas_url = azure_storage_service.generate_blob_sas_url(blob_path)

                    # Encode the SAS URL to handle special characters                    
                    sas_url = self.encode_sas_url(sas_url)
                    
                    time.sleep(5)

                    # Use original filename for file extension and indexing
                    root, ext = os.path.splitext(attachment.filename)
                    
                    email_item.Provenance_Source = ext[1:]
                    allExtractedContent = await azure_doc_intell_service.extract_content(sas_url)
                    attachmentChunks= self.chunk_text(allExtractedContent)

                    await azure_search_service.index_content(
                        attachmentChunks, 
                        document_id, 
                        email_item, 
                        file_name=attachment.filename,  # Use original filename for indexing
                        file_type=ext[1:],
                        page_number= [chunk['pages'] for chunk in attachmentChunks]
                    )

            except Exception as e:
                logger.error(f"Error processing content: {str(e)}")

    def encode_sas_url(self, sas_url: str) -> str:
        # URL encode the blob name part if it contains special characters
        from urllib.parse import quote
        import re
        
        # Get the set of unsafe characters
        unsafe_chars = re.compile(r'[^A-Za-z0-9\-_.()]')
        
        if unsafe_chars.search(sas_url):
            # Split the URL to isolate the blob name from the query parameters
            if '?' in sas_url:
                base_url, query_params = sas_url.split('?', 1)
            else:
                base_url, query_params = sas_url, ''
            
            # Extract blob name from the URL path (last part after the last /)
            url_parts = base_url.split('/')
            if len(url_parts) > 0:
                blob_name = url_parts[-1]
                # URL encode the blob name with safe characters for Azure blob names
                encoded_blob_name = quote(blob_name, safe='')
                url_parts[-1] = encoded_blob_name
                base_url = '/'.join(url_parts)
            
            # Reconstruct the URL
            sas_url = f"{base_url}?{query_params}" if query_params else base_url
        
        return sas_url

    async def chat_with_content(self, message: str, user_id: str, session_id: str):
        """
        Agentic RAG implementation to return the proper response based on the indexed content.
        """
        try:

            if not session_id:
                session_id = str(uuid.uuid4())
                logger.info(f"Generated new session ID: {session_id}")
                
            # initialize conversation state
            conversation = ContentConversation(
                user_query=message,
                max_attempts=MAX_ATTEMPTS,
                user_id=user_id,
                session_id=session_id
            )

            # Execute conversation workflow
            result = await self.execute_conversation_workflow(conversation)

            return {
                "answer": result.final_answer,
                "citations": result.citations,
                "thought_process": result.thought_process,
                "attempts": result.attempts,
                "search_queries": result.search_queries
            }
        
        except Exception as e:
            logger.error(f"Chat workflow failed: {str(e)}")
            return {
                "answer": f"I encountered the following error processing your question. Please {str(e)}",
                "citations": [],
                "thought_process": [],
                "attempts": 0,
                "search_queries": []
            }
    
    async def execute_conversation_workflow(self, conversation: ContentConversation) -> ConversationResult:
        """Executes the agentic rag workflow"""
        
        # Initialize chat message for user query
        chat_message = ChatMessage(
            user_id= conversation.user_id,
            session_id=conversation.session_id,
            role=Role.USER,
            message=conversation.user_query
        )
                
        chat_history_manager.add_message(chat_message)

        # continue if we have not exceeded max attempts and conversation is not finalized
        while conversation.should_continue():
            # Generate and execute search
            search_query, search_filter = await self.generate_search_query(conversation)
            search_results = await self.execute_search(search_query, search_filter, conversation)
            
            # Review results
            await self.review_search_results(conversation, search_results)

        # Generate final answer by synthesizing vetted results
        final_answer = await self.generate_final_answer(conversation)

        return conversation.to_result(final_answer)

    async def generate_search_query(self, conversation: ContentConversation) -> tuple[str, str]:
        """Generate search query and filter using the LLM based on the conversation history"""

        logger.info(f"Generating search query for attempt {conversation.attempts + 1}")

        from prompts.core_prompts import SEARCH_PROMPT
        
        # Build context more clearly
        context_parts = [f"User Question: {conversation.user_query}"]
        
        if conversation.has_search_history():
            context_parts.append("### Previous Search Attempts ###")
            for i, (search, review) in enumerate(zip(conversation.search_history, conversation.reviews), 1):
                context_parts.append(f"<Attempt {i}>\n")
                context_parts.append(f"   search_query: {search['query']}\n")
                context_parts.append(f"   review: {review}\n")
        
        context = "\n".join(context_parts)
        
        messages = [
            {"role": "system", "content": SEARCH_PROMPT},
            {"role": "user", "content": context}
        ]
        
        try:
            response = await azure_openai_service.get_chat_response(messages, SearchPromptResponse)

            chat_message = ChatMessage(
                user_id=conversation.user_id,
                session_id=conversation.session_id,
                role=Role.ASSISTANT,
                message=response.search_query)
            
            chat_history_manager.add_message(chat_message)

            conversation.add_search_attempt(response.search_query)
            return response.search_query, response.filter
        except Exception as e:
            logger.error(f"Search query generation failed: {str(e)}")
            # Fallback to user query with no filter
            return conversation.user_query, ""
    
    async def execute_search(self, query: str, filter_str: str, conversation: ContentConversation) -> List[SearchResult]:
        """Execute search with proper error handling"""
        try:
            results = await azure_search_service.run_search(
                search_query=query,
                processed_ids=conversation.processed_ids,
                provenance_filter=filter_str if filter_str else None
            )

            conversation.current_results = results
            
            conversation.thought_process.append({
                "step": "retrieve",
                "details": {
                    "user_query": conversation.user_query,
                    "generated_search_query": query,
                    "provenance_filter": filter_str if filter_str else "None",
                    "results_summary": [
                        # The chunk ID may not be super useful here, but it can help track which chunks were returned. If we change the indexing to include source_file, we should use that here instead
                        {"chunk_id": result["chunk_id"]}
                        for result in results
                    ]
                }
            })
            
            return results
            
        except Exception as e:
            logger.error(f"Search execution failed for query '{query}': {str(e)}")
            return []  # Return empty results to continue workflow

    async def review_search_results(self, conversation: ContentConversation, search_results: List[SearchResult]):
        """
        Review search results and determine which are valid/invalid for answering the user's question.
        Uses Azure OpenAI to analyze relevance and make decisions about continuing or finalizing.
        """

        logger.info(f"Reviewing search results.")

        try:
            from prompts.core_prompts import SEARCH_REVIEW_PROMPT

            # Format current search results for review
            current_results_formatted = self.format_search_results_for_review(conversation.current_results)
            
            # Format previously vetted results (don't review these again)
            vetted_results_formatted = self.format_search_results_for_review(conversation.vetted_results)
            
            # Format search history for context
            search_history_formatted = self.format_search_history_for_review(conversation)
            
            # Construct the review prompt with all context
            llm_input = f"""
                User Question: {conversation.user_query}

                <Current Search Results to review>
                {current_results_formatted}
                <end current search results to review>

                <previously vetted results, do not review>
                {vetted_results_formatted}
                <end previously vetted results, do not review>

                <Previous Attempts>
                {search_history_formatted}
                <end Previous Attempts>
                """
            
            messages = [
                {"role": "system", "content": SEARCH_REVIEW_PROMPT},
                {"role": "user", "content": llm_input}
            ]

            # Get review decision from Azure OpenAI
            review_decision = await azure_openai_service.get_chat_response(messages, ReviewDecision)

            conversation.thought_process.append({
                "step": "review",
                "details": {
                    "review_thought_process": review_decision.thought_process,
                    "valid_results": [
                        {
                            "chunk_id": conversation.current_results[idx]["chunk_id"],
                        }
                        for idx in review_decision.valid_results
                    ],
                    "invalid_results": [ 
                        {
                            "chunk_id": conversation.current_results[idx]["chunk_id"]
                        }
                        for idx in review_decision.invalid_results
                    ],
                    "decision": review_decision.decision
                }
            })

            # Validate indices before using them as sometimes Azure OpenAI can return indices like [0,1,2,3,4] when you only have 2 results, causing an index error
            current_results_count = len(conversation.current_results)
            
            # Filter out invalid indices to prevent IndexError
            valid_indices = [idx for idx in review_decision.valid_results if 0 <= idx < current_results_count]
            invalid_indices = [idx for idx in review_decision.invalid_results if 0 <= idx < current_results_count]
            
            # Log warnings if Azure OpenAI returned invalid indices
            if len(valid_indices) != len(review_decision.valid_results):
                invalid_valid_indices = [idx for idx in review_decision.valid_results if idx not in valid_indices]
                logger.warning(f"Azure OpenAI returned invalid valid_results indices: {invalid_valid_indices}. Current results count: {current_results_count}")
            
            if len(invalid_indices) != len(review_decision.invalid_results):
                invalid_invalid_indices = [idx for idx in review_decision.invalid_results if idx not in invalid_indices]
                logger.warning(f"Azure OpenAI returned invalid invalid_results indices: {invalid_invalid_indices}. Current results count: {current_results_count}")

            conversation.reviews.append(review_decision.thought_process)
            conversation.decisions.append(review_decision.decision)

            # add all valid results from this review to the vetted results list of the overall conversation
            for idx in review_decision.valid_results:
                result = conversation.current_results[idx]
                conversation.vetted_results.append(result)
                conversation.processed_ids.add(result["chunk_id"])
            
            # add all invalid results from this review to the discarded results list of the overall conversation
            for idx in review_decision.invalid_results:
                result = conversation.current_results[idx]
                conversation.discarded_results.append(result)
                conversation.processed_ids.add(result["chunk_id"])
            
            # resest the current results to empty for the next search
            conversation.current_results = []
            
        except Exception as e:
            logger.error(f"Search results review failed: {str(e)}")

    def format_search_results_for_review(self, results: List[SearchResult]) -> str:
        """Format search results for the review prompt with clear structure"""
        if not results:
            return "No results available."
        
        output_parts = ["\n=== Search Results ==="]
        for i, result in enumerate(results, 0):
            result_section = [
                f"\nResult #{i}",
                "=" * 80,
                f"Chunk ID: {result.get('chunk_id', 'Unknown')}",
                f"Source: {result.get('pursuit_name', 'Unknown')}",
                "\n--- Content ---",
                result.get('chunk_content', 'No content available'),
                "--- End Content ---"
            ]
            output_parts.extend(result_section)
        
        return "\n".join(output_parts)

    def format_search_history_for_review(self, conversation: ContentConversation) -> str:
        """Format search history for context in the review prompt"""
        if not conversation.search_history:
            return "No previous search attempts."
        
        history_parts = ["\n=== Search History ==="]
        for i, (search, review) in enumerate(zip(conversation.search_history, conversation.reviews), 1):
            history_parts.extend([
                f"<Attempt {i}>",
                f"   Query: {search['query']}",
                f"   Review: {review}",
                "</Attempt>"
            ])
        
        return "\n".join(history_parts)

    async def generate_final_answer(self, conversation: ContentConversation) -> str:
        """Generate final answer using Azure OpenAI with proper error handling"""
        
        logger.info(f"Generating final answer.")
        
        try:
            if not conversation.vetted_results:
                return "I couldn't find relevant information in the content documents to answer your question. Please try rephrasing your question or check if the information exists in the uploaded documents."
            
            # Format vetted results in the same way as review node
            vetted_results_formatted = "\n=== Vetted Results ===\n"
            for i, result in enumerate(conversation.vetted_results, 0):
                result_parts = [
                    f"\nResult #{i}",
                    "=" * 80,
                    f"ID: {result.get('chunk_id')}",
                    "\n<Start Content>",
                    "-" * 80,
                    result.get('chunk_content'),
                    "-" * 80,
                    "<End Content>"
                ]
                vetted_results_formatted += "\n".join(result_parts)

            final_prompt = """Create a comprehensive answer to the user's question using the vetted results."""
            
            llm_input = f"""Create a comprehensive answer to the user's question using the vetted results.

                User Question: {conversation.user_query}

                Vetted Results:
                {vetted_results_formatted}

                Synthesize these results into a clear, complete answer. If there were no vetted results, say you couldn't find any relevant information to answer the question.

                Guidance:
                - Always use valid markdown syntax. Try to use level 1 or level 2 headers for your sections.
                - Cite your sources using the following format: some text <cit>file name - chunk id</cit>, some more text <cit>file name - chunk id> , etc.
                - Only cite sources that are actually used in the answer."""

            chat_history = chat_history_manager.get_history(conversation.session_id)

            # New Messages
            messages = [
                {"role": "system", "content": final_prompt},
                {"role": "user", "content": llm_input}
            ]

            for msg in messages:
                chat_message = ChatMessage(
                    user_id=conversation.user_id,
                    session_id=conversation.session_id,
                    role=Role(msg["role"]),
                    message=msg["content"]
                )

            chat_history_manager.add_message(chat_message)

            for msg in chat_history:
                messages.append({"role": msg.role, "content": msg.message})
            
            final_answer = await azure_openai_service.get_chat_response_text(messages)

            chat_message = ChatMessage(
                user_id=conversation.user_id,
                session_id=conversation.session_id,
                role=Role.ASSISTANT,
                message=final_answer
            )

            chat_history_manager.add_message(chat_message)

            conversation.thought_process.append({
                "step": "response",
                "details": {
                    "final_answer": final_answer
                }
            })

            logger.info(f"Sending final payload.")
            
            return final_answer
            
        except Exception as e:
            logger.error(f"Final answer generation failed: {str(e)}")
            return f"I encountered an error generating the final answer. Error: {str(e)}. Please try rephrasing your question."

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