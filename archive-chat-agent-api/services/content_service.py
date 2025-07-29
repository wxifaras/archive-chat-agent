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
from models.content_conversation import ContentConversation, ConversationResult, ReviewDecision, SearchPromptResponse, NUM_SEARCH_RESULTS
from models.chat_history import ChatMessage, Role
from services.in_memory_chat_history_manager import InMemoryChatHistoryManager
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import AzureOpenAIEmbeddings

azure_search_service = AzureAISearchService()
azure_doc_intell_service = AzureDocIntelService()
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
        
    async def process_existing_blobs(self):
        """Process existing blobs by grouping them by folder and processing each folder as a unit"""
        blob_list = await azure_storage_service.get_blobs()
        
        # Group blobs by folder (project ID)
        folders = {}
        for blob in blob_list:
            # Extract folder name (everything before the first '/')
            if '/' in blob:
                folder_name = blob.split('/')[0]
                file_name = blob.split('/', 1)[1]  # Everything after first '/'
                
                if folder_name not in folders:
                    folders[folder_name] = {
                        'json_files': [],
                        'attachment_files': []
                    }
                
                # Categorize files
                if file_name.endswith('.json'):
                    folders[folder_name]['json_files'].append({
                        'blob_path': blob,
                        'file_name': file_name,
                    })
                else:
                    folders[folder_name]['attachment_files'].append({
                        'blob_path': blob,
                        'file_name': file_name,
                    })
        
        # Process each folder
        for folder_name, files in folders.items():
            try:
                await self.process_folder(folder_name, files)
            except Exception as e:
                logger.error(f"Error processing folder {folder_name}: {e}")
    
    async def process_folder(self, folder_name: str, files: dict):
        """Process a single folder containing JSON and attachment files"""
        json_files = files['json_files']
        attachment_files = files['attachment_files']
        
        if not json_files:
            logger.warning(f"No JSON files found in folder {folder_name}, skipping...")
            return
        if len(json_files) > 1:
            logger.warning(f"Multiple JSON files found in folder {folder_name}, skipping...")
            return
        # Get the JSON file (taking the first one if multiple exist)
        json_file = json_files[0]
        logger.info(f"Processing folder {folder_name} with JSON: {json_file['file_name']} and {len(attachment_files)} attachments")
        document_id = str(uuid.uuid4())
        try:
            # Download JSON content
            json_content = await azure_storage_service.get_blob(json_file['blob_path'])
            json_str = json_content if isinstance(json_content, str) else json_content.decode('utf-8')
            
            # Parse the JSON to get email item for processing
            email_list = EmailList.model_validate_json(json_str)
            email_item = email_list.root[0]
            await azure_search_service.create_index()

            if len(attachment_files) == 0:
                provenance_source = await azure_openai_service.get_source_from_provenance(email_item.Provenance)
                email_item.Provenance_Source = provenance_source
            if len(email_item.text) == 0:
                email_item.text = "No text content."

            # Choose chunking method from settings
            use_semantic_chunking = bool(settings.USE_SEMANTIC_CHUNKING)
            
            if use_semantic_chunking:
                jsonText = self.chunk_semantic_text(email_item.text)
            else:
                jsonText = self.chunk_json_text(email_item.text)

            await azure_search_service.index_content(jsonText, document_id, email_item, file_name=json_file['file_name'], file_type="json")
            # process all attachment files
            for attachment_file in attachment_files:
                try:
                    root, ext = os.path.splitext(attachment_file['file_name'])
                    email_item.Provenance_Source = ext[1:]
                    original_sas_url = azure_storage_service.generate_blob_sas_url(attachment_file['blob_path'])
                    logger.info(f"Original SAS URL: {original_sas_url}")
                    encoded_sas_url = self.encode_sas_url(original_sas_url)
                    logger.info(f"Encoded SAS URL: {encoded_sas_url}")
                    time.sleep(5)
                    allExtractedContent = await azure_doc_intell_service.extract_content(encoded_sas_url)
                    if use_semantic_chunking:
                        attachmentChunks = self.chunk_semantic_text(allExtractedContent.text)
                    else:
                        attachmentChunks = self.chunk_text(allExtractedContent)
                    logger.info(f"Indexing attachment {attachment_file['file_name']} with {len(attachmentChunks)} chunks")
                    await azure_search_service.index_content(
                        attachmentChunks,
                        document_id,
                        email_item,
                        file_name=attachment_file['file_name'],
                        file_type=ext[1:],
                        page_number=[chunk.get('pages', []) for chunk in attachmentChunks]
                    )
                except Exception as e:
                    logger.error(f"Error downloading attachment {attachment_file['blob_path']} with SAS URL {encoded_sas_url}: {e}")
                    continue
            logger.info(f"Successfully processed folder {folder_name}")
        except Exception as e:
            logger.error(f"Error processing folder {folder_name}: {e}")

    async def process_content(
        self,
        document_id: str,
        json_file_name: str,
        json_content: str,
        attachments: List[UploadFile]
    ):
        try:
            email_list = EmailList.model_validate_json(json_content)
            email_item = email_list.root[0]
            index = await azure_search_service.create_index()
            if len(attachments) == 0:
                provenance_source = await azure_openai_service.get_source_from_provenance(email_item.Provenance)
                email_item.Provenance_Source = provenance_source
            use_semantic_chunking = bool(settings.USE_SEMANTIC_CHUNKING)
            blob_path, uploaded = await azure_storage_service.upload_file_with_dup_check(
                str(email_item.projectId),
                json_content,
                json_file_name
            )
            if not uploaded:
                logger.warning(f"File {json_file_name} already exists in Azure Storage. Skipping JSON upload.")
            else:
                if len(email_item.text) == 0:
                    email_item.text = "No text content."
                if use_semantic_chunking:
                    jsonText = self.chunk_semantic_text(email_item.text)
                else:
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
                sas_url = self.encode_sas_url(sas_url)
                time.sleep(5)
                root, ext = os.path.splitext(attachment.filename)
                email_item.Provenance_Source = ext[1:]
                allExtractedContent = await azure_doc_intell_service.extract_content(sas_url)

                if use_semantic_chunking:
                    attachmentChunks = self.chunk_semantic_text(allExtractedContent.content)
                else:
                    attachmentChunks = self.chunk_text(allExtractedContent)
                    
                await azure_search_service.index_content(
                    attachmentChunks,
                    document_id,
                    email_item,
                    file_name=attachment.filename,
                    file_type=ext[1:],
                    page_number=[chunk.get('pages', []) for chunk in attachmentChunks]
                )
        except Exception as e:
            logger.error(f"Error processing content: {str(e)}")

    def encode_sas_url(self, sas_url: str) -> str:
        """
        Properly URL encode Azure blob SAS URLs, handling mixed encoding scenarios.
        Some blob names may be partially encoded (spaces as %20) but missing encoding for # characters.
        """
        from urllib.parse import quote, unquote, urlparse, urlunparse
        
        try:
            logger.info(f"Starting URL encoding for: {sas_url}")
            
            # Check if URL contains unencoded # character (major issue for Azure Document Intelligence)
            if '#' in sas_url and '?' in sas_url:
                # Split at the query string to avoid processing the SAS token
                base_url_part = sas_url.split('?')[0]
                if '#' in base_url_part:
                    logger.warning(f"Found unencoded # character in URL path: {base_url_part}")
            
            # Parse the URL to separate components
            parsed = urlparse(sas_url)
            logger.debug(f"Parsed URL - scheme: {parsed.scheme}, netloc: {parsed.netloc}, path: {parsed.path}")
            
            # Split the path into segments (skip empty first element from leading /)
            path_segments = [segment for segment in parsed.path.split('/') if segment]
            logger.debug(f"Path segments: {path_segments}")
            
            # Process each path segment to handle mixed encoding
            encoded_segments = []
            for i, segment in enumerate(path_segments):
                logger.debug(f"Processing segment {i}: '{segment}'")
                
                # First, decode any existing URL encoding to get the raw text
                # This handles cases where spaces are already encoded as %20
                decoded_segment = unquote(segment)
                logger.debug(f"  After unquote: '{decoded_segment}'")
                
                # Then re-encode the entire segment properly
                # This ensures all special characters including # are encoded
                encoded_segment = quote(decoded_segment, safe='-_.')
                logger.debug(f"  After quote: '{encoded_segment}'")
                
                encoded_segments.append(encoded_segment)
                
                # Log segment transformations for debugging
                if segment != encoded_segment:
                    logger.info(f"Segment encoding: '{segment}' -> '{decoded_segment}' -> '{encoded_segment}'")
            
            # Reconstruct the path
            encoded_path = '/' + '/'.join(encoded_segments) if encoded_segments else parsed.path
            logger.debug(f"Reconstructed path: {encoded_path}")
            
            # Reconstruct the URL with the encoded path
            encoded_url = urlunparse((
                parsed.scheme,
                parsed.netloc, 
                encoded_path,
                parsed.params,
                parsed.query,  # Query parameters (SAS token) should not be re-encoded
                parsed.fragment
            ))
            
            # Final validation - ensure no unencoded # in the path portion
            if '#' in encoded_url and '?' in encoded_url:
                path_portion = encoded_url.split('?')[0]
                if '#' in path_portion:
                    logger.error(f"CRITICAL: Still found unencoded # in final URL path: {path_portion}")
                    # Force encode any remaining # characters in the path
                    fixed_path = path_portion.replace('#', '%23')
                    encoded_url = encoded_url.replace(path_portion, fixed_path)
                    logger.info(f"Applied emergency # encoding fix: {encoded_url}")
            
            # Log the transformation for debugging
            if encoded_url != sas_url:
                logger.info(f"URL encoding applied:")
                logger.info(f"  Original: {sas_url}")
                logger.info(f"  Encoded:  {encoded_url}")
            else:
                logger.info("No URL encoding changes needed")
            
            return encoded_url
            
        except Exception as e:
            logger.error(f"Error encoding SAS URL: {e}. Using original URL: {sas_url}")
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

        await chat_history_manager.add_message(chat_message)

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

        chat_history = await chat_history_manager.get_history(conversation.session_id)
        if chat_history:
            context_parts.append("### Chat History ###")
            for msg in chat_history:
                context_parts.append(f"{msg.role}: {msg.message}")
        
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
            

            await chat_history_manager.add_message(chat_message)
            logger.info(f"Generated search query: {response.search_query}")
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

            # Validate indices against actual results count and filter invalid ones
            current_results_count = len(conversation.current_results)
            valid_indices, invalid_indices = self.validate_and_filter_indices(
                review_decision, current_results_count
            )

            # Override decision if we found mostly valid results and the LLM chooses to finalize (setting at 80% - there are likely more valid results in the search index)
            final_decision = review_decision.decision
            valid_percentage = len(valid_indices) / current_results_count if current_results_count > 0 else 0
            
            if (valid_percentage >= 0.8 and final_decision == "finalize"):
                logger.info(f"Overriding 'finalize' decision: {len(valid_indices)}/{current_results_count} results ({valid_percentage:.1%}) were valid, likely more data available")
                final_decision = "retry"

            conversation.thought_process.append({
                "step": "review",
                "details": {
                    "review_thought_process": review_decision.thought_process,
                    "valid_results": [
                        {
                            "chunk_id": conversation.current_results[idx]["chunk_id"],
                        }
                        for idx in valid_indices  # Use filtered indices to prevent IndexError
                    ],
                    "invalid_results": [ 
                        {
                            "chunk_id": conversation.current_results[idx]["chunk_id"]
                        }
                        for idx in invalid_indices  # Use filtered indices to prevent IndexError
                    ],
                    "llm_decision": review_decision.decision,
                    "final_decision": final_decision,
                    "decision_override": final_decision != review_decision.decision
                }
            })

            conversation.reviews.append(review_decision.thought_process)
            conversation.decisions.append(final_decision)

            # add all valid results from this review to the vetted results list of the overall conversation
            for idx in valid_indices:  # Use filtered indices to prevent IndexError
                result = conversation.current_results[idx]
                conversation.vetted_results.append(result)
                conversation.processed_ids.add(result["chunk_id"])
            
            # add all invalid results from this review to the discarded results list of the overall conversation
            for idx in invalid_indices:  # Use filtered indices to prevent IndexError
                result = conversation.current_results[idx]
                conversation.discarded_results.append(result)
                conversation.processed_ids.add(result["chunk_id"])
            
            # resest the current results to empty for the next search
            conversation.current_results = []
            
        except Exception as e:
            logger.error(f"Search results review failed: {str(e)}")

    def validate_and_filter_indices(self, review_decision: ReviewDecision, actual_results_count: int) -> tuple[List[int], List[int]]:
        """
        Filter indices to only include those that are valid and within range.
        Handles both LLM hallucinations (indices > NUM_SEARCH_RESULTS) and 
        cases where actual results < NUM_SEARCH_RESULTS.
        """
        # Filter valid_results indices
        valid_indices = []
        hallucinated_valid = []
        out_of_range_valid = []
        
        for idx in review_decision.valid_results:
            if idx < 0 or idx >= NUM_SEARCH_RESULTS:
                hallucinated_valid.append(idx)
            elif idx >= actual_results_count:
                out_of_range_valid.append(idx)
            else:
                valid_indices.append(idx)
        
        # Filter invalid_results indices  
        invalid_indices = []
        hallucinated_invalid = []
        out_of_range_invalid = []
        
        for idx in review_decision.invalid_results:
            if idx < 0 or idx >= NUM_SEARCH_RESULTS:
                hallucinated_invalid.append(idx)
            elif idx >= actual_results_count:
                out_of_range_invalid.append(idx)
            else:
                invalid_indices.append(idx)
        
        # Log different types of issues
        if hallucinated_valid:
            logger.warning(f"LLM hallucinated valid_results indices: {hallucinated_valid} (outside range [0, {NUM_SEARCH_RESULTS-1}])")
        if hallucinated_invalid:
            logger.warning(f"LLM hallucinated invalid_results indices: {hallucinated_invalid} (outside range [0, {NUM_SEARCH_RESULTS-1}])")
            
        if out_of_range_valid:
            logger.info(f"Valid indices beyond actual results (ignored): {out_of_range_valid}. Actual results: {actual_results_count}")
        if out_of_range_invalid:
            logger.info(f"Invalid indices beyond actual results (ignored): {out_of_range_invalid}. Actual results: {actual_results_count}")
        
        logger.info(f"Filtered indices - Valid: {valid_indices}, Invalid: {invalid_indices}")
        return valid_indices, invalid_indices

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
                f"File Name: {result.get('file_name', 'Unknown')}",
                "\n--- Content ---",
                result.get('chunk_content', 'No content available'),
                "--- End Content ---"
            ]
            
            # Include provenance if available and not empty
            if result.get('provenance') and result.get('provenance').strip():
                result_section.extend([
                    "\n--- Provenance ---",
                    result.get('provenance'),
                    "--- End Provenance ---"
                ])
            
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
                    f"File Name: {result.get('file_name', 'Unknown')}",
                    "\n<Start Content>",
                    "-" * 80,
                    result.get('chunk_content'),
                    "-" * 80,
                    "<End Content>"
                ]
                
                # Include provenance if available
                if result.get('provenance') and result.get('provenance').strip():
                    result_parts.extend([
                        "\n<Start Provenance>",
                        "-" * 80,
                        result.get('provenance'),
                        "-" * 80,
                        "<End Provenance>"
                    ])
                
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

            chat_history = await chat_history_manager.get_history(conversation.session_id)

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

            await chat_history_manager.add_message(chat_message)

            for msg in chat_history:
                messages.append({"role": msg.role, "content": msg.message})
            
            final_answer = await azure_openai_service.get_chat_response_text(messages)

            chat_message = ChatMessage(
                user_id=conversation.user_id,
                session_id=conversation.session_id,
                role=Role.ASSISTANT,
                message=final_answer
            )

            await chat_history_manager.add_message(chat_message)

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
    def chunk_semantic_text(text, breakpoint_threshold_type="percentile", breakpoint_threshold_amount=80.0, min_chunk_size=100):
        """
        Semantic chunking using LangChain's SemanticChunker with explicit Azure OpenAI credentials.
        """
        try:
            if not text or not text.strip():
                return []
                
            embeddings = AzureOpenAIEmbeddings(
                api_key=settings.AZURE_OPENAI_API_KEY,
                deployment=settings.AZURE_OPENAI_TEXT_EMBEDDING_DEPLOYMENT_NAME,
                model=settings.AZURE_OPENAI_TEXT_EMBEDDING_DEPLOYMENT_NAME,
                azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,  
            )

            text_splitter = SemanticChunker(
                embeddings,
                breakpoint_threshold_type=breakpoint_threshold_type,
                breakpoint_threshold_amount=breakpoint_threshold_amount,
                min_chunk_size=min_chunk_size
            )
            
            docs = text_splitter.create_documents([text])
            
            chunks = []
            seen_content = set()  # Track seen content to prevent duplicates
            
            for i, doc in enumerate(docs):
                chunk_text = doc.page_content.strip()
                
                # Skip empty chunks
                if not chunk_text:
                    continue
                    
                # Create a hash of the content to check for duplicates
                content_hash = hash(chunk_text.lower())
                if content_hash in seen_content:
                    logger.warning(f"Skipping duplicate semantic chunk: {chunk_text[:100]}...")
                    continue
                    
                seen_content.add(content_hash)
                
                chunks.append({
                    'chunked_text': chunk_text,
                    'chunk_index': i,
                    'chunk_size': len(chunk_text),
                    'chunk_method': 'semantic'
                })
                
            logger.info(f"Created {len(chunks)} semantic chunks from text of length {len(text)}")
            return chunks
            
        except Exception as e:
            import traceback
            logger.error(f"Semantic chunking failed: {e}")
            traceback.print_exc()
            # Fallback to token-based chunking
            logger.info("Falling back to token-based chunking")
            return ContentService.chunk_json_text(text)

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