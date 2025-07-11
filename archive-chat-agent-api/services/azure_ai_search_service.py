import uuid
from azure.search.documents.indexes.models import SearchIndex, SearchField, VectorSearch, VectorSearchProfile, HnswAlgorithmConfiguration, SemanticSearch, SemanticConfiguration, SemanticPrioritizedFields, SemanticField, AzureOpenAIVectorizer, AzureOpenAIVectorizerParameters
from models.email_item import EmailItem
from services.azure_openai_service import AzureOpenAIService
from azure.search.documents.indexes import SearchIndexClient
from azure.core.credentials import AzureKeyCredential 
from azure.search.documents import SearchClient
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
    SearchIndex
)
import logging
from typing import List, Set, Optional
from pydantic import BaseModel, Field
from core.settings import settings

NUM_SEARCH_RESULTS = 5
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

    def create_index(self) -> str:
        try:
            self.search_index_client.get_index(settings.AZURE_AI_SEARCH_INDEX_NAME)
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
            SimpleField(name="Provenance", type=SearchFieldDataType.String, filterable=True),
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
                content_fields=[SemanticField(field_name="chunk_content")],
                title_field=SemanticField(field_name="projectId")
            )
        )
        
        idx = SearchIndex(
            name=settings.AZURE_AI_SEARCH_INDEX_NAME,
            fields=fields,
            vector_search=vector_search,
            semantic_search=SemanticSearch(configurations=[semantic_config])
        )

        result = self.search_index_client.create_or_update_index(idx)
        logger.info(f"Created index: {result.name}")
        return result.name

    def index_content(self, 
                      chunks: list[str], 
                      document_id: str, 
                      email_item: EmailItem, 
                      file_type: str, 
                      file_name: Optional[str] = None, 
                      page_number: List[str] = None):
   
        chunkedContent = []
        for idx, chunk in enumerate(chunks):
            embedding = self.openai_service.create_embedding(chunk['chunked_text'])
            chunk_id = str(uuid.uuid4())
            if page_number is None:
                page_number = []
            page = page_number[idx] if idx < len(page_number) else []

            if file_type == ".json":
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
                    "Provenance_Source": None
                }
                data = {k: v for k, v in data.items() if v is not None}
                chunkedContent.append(data)
            else:
                chunkedContent.append({
                    "document_id": document_id,
                    "chunk_id": chunk_id,
                    "projectId": str(email_item.projectId),
                    "file_name": file_name,
                    "file_type": file_type,
                    "chunk_content": chunk['chunked_text'],
                    "chunk_content_vector": embedding,
                    "page_number": page
                })

        result = self.search_client.upload_documents(documents=chunkedContent)
        uploaded = [str(r.key) for r in result if r.succeeded]
        failed = [str(r.key) for r in result if not r.succeeded]

        if failed:
            logger.error(f"Failed to upload chunks: {failed}")
        else:
            logger.info(f"Successfully uploaded {len(uploaded)} chunks.")
        
        return uploaded