import uuid
from azure.search.documents.indexes.aio import SearchIndexClient
from azure.search.documents.aio import SearchClient
from azure.search.documents.indexes.models import SearchIndex, SearchField, VectorSearch, VectorSearchProfile, HnswAlgorithmConfiguration, SemanticSearch, SemanticConfiguration, SemanticPrioritizedFields, SemanticField, AzureOpenAIVectorizer, AzureOpenAIVectorizerParameters
from models.email_item import EmailItem
from services.azure_openai_service import AzureOpenAIService
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.models import VectorizedQuery
from azure.search.documents.indexes.models import (
    SimpleField,
    SearchFieldDataType,
    SearchableField,
    SearchField,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
    SemanticConfiguration,
    SemanticPrioritizedFields,
    SemanticField,
    SemanticSearch,
    SearchIndex,
    KnowledgeAgent,
    KnowledgeAgentAzureOpenAIModel,
    KnowledgeAgentTargetIndex,
    KnowledgeAgentRequestLimits,
    AzureOpenAIVectorizerParameters
)
from azure.search.documents.agent import KnowledgeAgentRetrievalClient
from azure.search.documents.agent.models import KnowledgeAgentRetrievalRequest, KnowledgeAgentMessage, KnowledgeAgentMessageTextContent, KnowledgeAgentIndexParams
from azure.ai.projects import AIProjectClient
from azure.ai.agents.models import FunctionTool, ToolSet, ListSortOrder, AgentsNamedToolChoice, AgentsNamedToolChoiceType, FunctionName
from azure.search.documents.agent import KnowledgeAgentRetrievalClient
from azure.identity import DefaultAzureCredential
from prompts.core_prompts import AGENTIC_RETRIEVAL_PROMPT
import logging
from typing import List, Set, Optional, TypedDict
from pydantic import BaseModel, Field
from core.settings import settings

K_NEAREST_NEIGHBORS = 30

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

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

# Type Definitions
class SearchResult(TypedDict):
    chunk_id: str
    chunk_content: str
    file_name: str
    source_pages: int
    provenance: str
    reranker_score: float

class AzureAISearchService:
    def __init__(self): 
        if not all([
            settings.AZURE_AI_SEARCH_SERVICE_ENDPOINT,
            settings.AZURE_AI_SEARCH_SERVICE_KEY,
            settings.AZURE_AI_SEARCH_INDEX_NAME
        ]):
            raise ValueError("Required Azure AI Search settings are missing")

        self.search_index_client = SearchIndexClient(settings.AZURE_AI_SEARCH_SERVICE_ENDPOINT, AzureKeyCredential(settings.AZURE_AI_SEARCH_SERVICE_KEY))
        self.search_client = SearchClient(settings.AZURE_AI_SEARCH_SERVICE_ENDPOINT, settings.AZURE_AI_SEARCH_INDEX_NAME, AzureKeyCredential(settings.AZURE_AI_SEARCH_SERVICE_KEY))
        self.openai_service = AzureOpenAIService()

        # Initialize agentic retrieval components as None - will be set up asynchronously
        self.agent = None
        self.project_client = None
        self.retrieval_agent = None
        self.agentic_retrieval_client = None
        self.thread = None
        self.retrieval_results = {}
        self.agentic_setup_complete = False

    async def setup_agentic_retrieval(self):
        """Setup agentic retrieval components asynchronously"""
        if self.agentic_setup_complete:
            return
            
        try:
            # Create the knowledge agent
            self.agent = KnowledgeAgent(
                name="retrieval-agent",
                models=[
                    KnowledgeAgentAzureOpenAIModel(
                        azure_open_ai_parameters=AzureOpenAIVectorizerParameters(
                            resource_url=settings.AZURE_OPENAI_ENDPOINT,
                            deployment_name=settings.AZURE_OPENAI_DEPLOYMENT_NAME,
                            model_name=settings.AZURE_OPENAI_DEPLOYMENT_NAME,
                            api_key=settings.AZURE_OPENAI_API_KEY
                        )
                    )
                ],
                target_indexes=[
                    KnowledgeAgentTargetIndex(
                        index_name=settings.AZURE_AI_SEARCH_INDEX_NAME,
                        default_reranker_threshold=2.5
                    )
                ],
                request_limits=KnowledgeAgentRequestLimits(
                    max_output_size=10000
                )
            )

            await self.search_index_client.create_or_update_agent(self.agent)
            logger.info("Knowledge agent 'retrieval-agent' created or updated successfully")

            # Create project client
            self.project_client = AIProjectClient(
                endpoint=settings.AZURE_FOUNDRY_PROJECT_ENDPOINT, 
                credential=DefaultAzureCredential()
            )

            list(self.project_client.agents.list_agents())

            self.retrieval_agent = self.project_client.agents.create_agent(
                model=settings.AZURE_OPENAI_DEPLOYMENT_NAME,
                name="retrieval-agent",
                instructions=AGENTIC_RETRIEVAL_PROMPT
            )

            print(f"AI agent 'retrieval-agent' created or updated successfully")

            # Create retrieval client
            self.agentic_retrieval_client = KnowledgeAgentRetrievalClient(
                endpoint=settings.AZURE_AI_SEARCH_SERVICE_ENDPOINT, 
                agent_name='retrieval-agent', 
                credential=AzureKeyCredential(settings.AZURE_AI_SEARCH_SERVICE_KEY)
            )

            # Create thread
            if self.project_client:
                self.thread = self.project_client.agents.threads.create()
                logger.info("Agent thread created successfully")

            self.agentic_setup_complete = True
            logger.info("Agentic retrieval setup completed successfully")

        except Exception as e:
            logger.error(f"Failed to setup agentic retrieval: {e}")
            self.agentic_setup_complete = False
            raise RuntimeError(f"Agentic retrieval setup failed: {e}") from e

    def agentic_retrieval(self) -> str:
        if not self.project_client or not self.thread:
            raise RuntimeError("Agentic retrieval is not properly configured")
        
        try:
            # Take the last 5 messages in the conversation
            messages = self.project_client.agents.messages.list(self.thread.id, limit=5, order=ListSortOrder.DESCENDING)
            # Reverse the order so the most recent message is last
            messages = list(messages)
            messages.reverse()
            
            if not self.agentic_retrieval_client:
                raise RuntimeError("Agentic retrieval client is not initialized")
                
            retrieval_result = self.agentic_retrieval_client.retrieve(
                retrieval_request=KnowledgeAgentRetrievalRequest(
                    messages=[KnowledgeAgentMessage(role=msg["role"], content=[KnowledgeAgentMessageTextContent(text=msg.content[0].text)]) for msg in messages if msg["role"] != "system"],
                    target_index_params=[KnowledgeAgentIndexParams(index_name=settings.AZURE_AI_SEARCH_INDEX_NAME, reranker_threshold=2.5)]
                )
            )

            # Associate the retrieval results with the last message in the conversation
            last_message = messages[-1]
            self.retrieval_results[last_message.id] = retrieval_result

            # Return the grounding response to the agent
            return retrieval_result.response[0].content[0].text
        except Exception as e:
            logger.error(f"Error in agentic retrieval: {e}")
            raise

    async def create_index(self) -> str:
        try:
            await self.search_index_client.get_index(settings.AZURE_AI_SEARCH_INDEX_NAME)
            logger.info(f"{settings.AZURE_AI_SEARCH_INDEX_NAME} index already exists")
            return settings.AZURE_AI_SEARCH_INDEX_NAME
        except:
            pass

        fields = [
            SimpleField(name="document_id", type=SearchFieldDataType.String, filterable=True),
            SimpleField(name="projectId", type=SearchFieldDataType.String, filterable=True),
            SimpleField(name="Author", type=SearchFieldDataType.String, filterable=True),
            SimpleField(name="Email_Subject", type=SearchFieldDataType.String, filterable=True),
            SimpleField(name="Received_Date", type=SearchFieldDataType.DateTimeOffset, filterable=True),
            SimpleField(name="Key_Topics", type=SearchFieldDataType.String, filterable=True),
            SimpleField(name="Email_body", type=SearchFieldDataType.String, filterable=True),
            SimpleField(name="Email_ID", type=SearchFieldDataType.String, filterable=True),
            SimpleField(name="URL_Index", type=SearchFieldDataType.String, filterable=True),
            SimpleField(name="URL_Type", type=SearchFieldDataType.String, filterable=True),
            SimpleField(name="Force_Scraper", type=SearchFieldDataType.Boolean, filterable=True),
            SimpleField(name="crawledLink", type=SearchFieldDataType.String, filterable=True),
            SimpleField(name="links", type=SearchFieldDataType.Collection(SearchFieldDataType.String),
            filterable=False,
            sortable=False,
            facetable=False),
            SimpleField(name="allLinks", type=SearchFieldDataType.Collection(SearchFieldDataType.String),
            filterable=False,
            sortable=False,
            facetable=False),
            SimpleField(name="level", type=SearchFieldDataType.Int32, filterable=True),
            SimpleField(name="status", type=SearchFieldDataType.String, filterable=True),
            SimpleField(name="createdOn", type=SearchFieldDataType.DateTimeOffset, filterable=True),
            SimpleField(name="jobDomain", type=SearchFieldDataType.String, filterable=True),
            SimpleField(name="Source", type=SearchFieldDataType.String, filterable=True),
            SimpleField(name="Client_Exposure", type=SearchFieldDataType.Int32, filterable=True),
            SimpleField(name="POV_Rating", type=SearchFieldDataType.Int32, filterable=True),
            SimpleField(name="Comments", type=SearchFieldDataType.String, filterable=True),
            SimpleField(name="Timestamp", type=SearchFieldDataType.String, filterable=True),
            SearchableField(name="Provenance", type=SearchFieldDataType.String, searchable=True, retrievable=True),
            SimpleField(name="Provenance_Source", type=SearchFieldDataType.String, filterable=True),
            SimpleField(name="chunk_id", type=SearchFieldDataType.String, filterable=True, key=True),
            SimpleField(name="file_name", type=SearchFieldDataType.String, filterable=True),
            SimpleField(name="file_type", type=SearchFieldDataType.String, filterable=True),
            SimpleField(name="page_number", type=SearchFieldDataType.Collection(SearchFieldDataType.Int32),
            filterable=False,
            sortable=False,
            facetable=False),
            SearchableField(name="chunk_content", type=SearchFieldDataType.String, searchable=True, retrievable=True),
            SearchField(
                name="chunk_content_vector",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                vector_search_dimensions=3072,
                vector_search_profile_name="archive-vector-config",                
            )
        ]

        vector_search = VectorSearch(
            algorithms=[ HnswAlgorithmConfiguration(name="archive-vector-config", kind="hnsw", parameters={"m":4, "efConstruction":400}) ],
            profiles=[ VectorSearchProfile(name="archive-vector-config", algorithm_configuration_name="archive-vector-config", vectorizer_name="archive-vectorizer") ],
            vectorizers=[ AzureOpenAIVectorizer(
                vectorizer_name="archive-vectorizer",
                parameters=AzureOpenAIVectorizerParameters(
                    resource_url=settings.AZURE_OPENAI_ENDPOINT,
                    deployment_name=settings.AZURE_OPENAI_TEXT_EMBEDDING_DEPLOYMENT_NAME,
                    model_name=settings.AZURE_OPENAI_TEXT_EMBEDDING_DEPLOYMENT_NAME,
                    api_key=settings.AZURE_OPENAI_API_KEY
                )
            )]
        )

        semantic_config = SemanticConfiguration(
            name="semantic-config",
            prioritized_fields=SemanticPrioritizedFields(
                content_fields=[
                    SemanticField(field_name="chunk_content"),
                    SemanticField(field_name="Provenance")
                ],
                title_field=SemanticField(field_name="projectId")
            )
        )
        
        # Create semantic search with default configuration
        semantic_search = SemanticSearch(
            configurations=[semantic_config],
            default_configuration_name="semantic-config"
        )
        
        idx = SearchIndex(
            name=settings.AZURE_AI_SEARCH_INDEX_NAME,
            fields=fields,
            vector_search=vector_search,
            semantic_search=semantic_search
        )

        result = await self.search_index_client.create_or_update_index(idx)
        logger.info(f"Created index: {result.name}")
        return result.name

    async def index_content(self, 
                      chunks: list[str], 
                      document_id: str, 
                      email_item: EmailItem, 
                      file_type: str, 
                      file_name: Optional[str] = None, 
                      page_number: List[str] = None):
   
        documents = []
        for idx, chunk in enumerate(chunks):
            embedding = await self.openai_service.create_embedding(chunk['chunked_text'])
            chunk_id = str(uuid.uuid4())
            if page_number is None:
                page_number = []
            page = page_number[idx] if idx < len(page_number) else []

            if file_type == "json":
                data={
                    "document_id": document_id,
                    "chunk_id": chunk_id,
                    "projectId": str(email_item.projectId),
                    "file_name": file_name,
                    "file_type": file_type,
                    "chunk_content": chunk['chunked_text'],
                    "chunk_content_vector": embedding,
                    "page_number": page,
                    "Author": str(email_item.Author) if email_item.Author is not None else None,
                    "Email_Subject": str(email_item.Email_Subject) if email_item.Email_Subject is not None else None,
                    "Received_Date": email_item.Received_Date if email_item.Received_Date is not None else None,
                    "Key_Topics": str(email_item.Key_Topics) if email_item.Key_Topics is not None else None,
                    "Email_body": str(email_item.Email_body) if email_item.Email_body is not None else None,
                    "Provenance": str(email_item.Provenance) if email_item.Provenance is not None else None,
                    "Email_ID": str(email_item.Email_ID) if email_item.Email_ID is not None else None,
                    "URL_Index": str(email_item.URL_Index) if email_item.URL_Index is not None else None,
                    "URL_Type": str(email_item.URL_Type) if email_item.URL_Type is not None else None,
                    "Force_Scraper": email_item.Force_Scraper, 
                    "crawledLink": str(email_item.crawledLink) if email_item.crawledLink is not None else None,
                    "links": list(map(str, email_item.links)) if getattr(email_item, "links", None) else [],
                    "allLinks": getattr(email_item, "allLinks", None),
                    "status": str(email_item.status) if email_item.status is not None else None,
                    "createdOn": email_item.createdOn,
                    "jobDomain": str(email_item.jobDomain) if email_item.jobDomain is not None else None,
                    "Source": str(email_item.Source) if email_item.Source is not None else None,
                    "Client_Exposure": int(email_item.Client_Exposure) if email_item.Client_Exposure is not None else None,
                    "POV_Rating": int(email_item.POV_Rating) if email_item.POV_Rating is not None else None,
                    "level": int(email_item.level) if email_item.level is not None else None,
                    "Comments": str(email_item.Comments) if email_item.Comments is not None else None,
                    "Timestamp": str(email_item.Timestamp) if email_item.Timestamp is not None else None,
                    "Provenance_Source": str(email_item.Provenance_Source) if email_item.Provenance_Source is not None else None,
                }
                data = {k: v for k, v in data.items() if v is not None}
                documents.append(data)
            else:
                documents.append({
                    "document_id": document_id,
                    "chunk_id": chunk_id,
                    "projectId": str(email_item.projectId),
                    "file_name": file_name,
                    "file_type": file_type,
                    "chunk_content": chunk['chunked_text'],
                    "chunk_content_vector": embedding,
                    "page_number": page,
                    "Provenance_Source": str(email_item.Provenance_Source) if email_item.Provenance_Source is not None else None
                })

        result = await self.search_client.upload_documents(documents=documents)
        uploaded = [str(r.key) for r in result if r.succeeded]
        failed = [str(r.key) for r in result if not r.succeeded]

        if failed:
            logger.error(f"Failed to upload chunks: {failed} for File {file_name}.")
        else:
            logger.info(f"Successfully uploaded {len(uploaded)} chunks for File {file_name}.")

        return uploaded
    
    async def run_search(
            self,
            search_query: str,
            processed_ids: Set[str],
            provenance_filter: str | None = None,
        ) -> List[SearchResult]:
        """
        Perform a search using Azure Cognitive Search with both semantic and vector queries.
        """
        query_vector = await self.openai_service.create_embedding(search_query)
        vector_query = VectorizedQuery(
            vector=query_vector,
            k_nearest_neighbors=K_NEAREST_NEIGHBORS,
            fields="chunk_content_vector",
        )
        filter_parts = []
        if processed_ids:
            ids_string = ','.join(processed_ids)
            filter_parts.append(f"not search.in(chunk_id, '{ids_string}')")
        
        # Add provenance filter if provided
        if provenance_filter:
            filter_parts.append(f"({provenance_filter})")
        
        filter_str = " and ".join(filter_parts) if filter_parts else None

        results = await self.search_client.search(
            search_text=search_query,
            vector_queries=[vector_query],
            filter=filter_str,
            select=["chunk_id", "chunk_content", "file_name", "Provenance"],
            top=settings.NUM_SEARCH_RESULTS,
            query_type="semantic",
            semantic_configuration_name="semantic-config"
        )

        search_results = []
        async for result in results:
            search_result: SearchResult = {
                "chunk_id": result["chunk_id"],
                "chunk_content": result["chunk_content"],
                "reranker_score": result["@search.reranker_score"],
                "file_name": result.get("file_name", ""),
                "provenance": result.get("Provenance", ""),
            }
            search_results.append(search_result)

        return search_results
    
    async def run_agentic_retrieval(
            self,
            search_query: str,
            processed_ids: Set[str],
            provenance_filter: str | None = None,
        ) -> List[SearchResult]:
        """
        Perform a search using Azure Search's Agentic Retrieval feature
        """
        # Ensure agentic retrieval is set up
        if not self.agentic_setup_complete:
            await self.setup_agentic_retrieval()

        try:
            functions = FunctionTool({self.agentic_retrieval})
            toolset = ToolSet()
            toolset.add(functions)
            self.project_client.agents.enable_auto_function_calls(toolset)

            message = self.project_client.agents.messages.create(
                thread_id=self.thread.id,
                role="user",
                content=search_query
            )

            run = self.project_client.agents.runs.create_and_process(
                thread_id=self.thread.id,
                agent_id=self.retrieval_agent.id,
                tool_choice=AgentsNamedToolChoice(type=AgentsNamedToolChoiceType.FUNCTION, function=FunctionName(name="agentic_retrieval")),
                toolset=toolset
            )
            
            if run.status == "failed":
                logger.error(f"Agentic retrieval failed for query: {search_query}")
                raise RuntimeError(f"Run failed: {run.last_error}")
                
            output = self.project_client.agents.messages.get_last_message_text_by_role(thread_id=self.thread.id, role="assistant").text.value

            formatted_output = output.replace('.', '\n')
            logger.info(f"Agent response: {formatted_output}")
            
            # You'll need to parse the output and convert it to SearchResult format
            # This is a placeholder - you'll need to implement the parsing logic
            # based on the actual format returned by your agentic retrieval
            search_results = []
            # TODO: Parse output and create SearchResult objects
            
            return search_results
            
        except Exception as e:
            logger.error(f"Error in agentic retrieval: {e}")
