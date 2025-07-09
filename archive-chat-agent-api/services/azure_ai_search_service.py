import uuid
from azure.search.documents.indexes.models import SearchIndex, SearchField, VectorSearch, VectorSearchProfile, HnswAlgorithmConfiguration, SemanticSearch, SemanticConfiguration, SemanticPrioritizedFields, SemanticField, AzureOpenAIVectorizer, AzureOpenAIVectorizerParameters
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

    def create_index(self) -> str:
        try:
            self.search_index_client.get_index(settings.AZURE_AI_SEARCH_INDEX_NAME)
            logger.info(f"{settings.AZURE_AI_SEARCH_INDEX_NAME} index already exists")
            return settings.AZURE_AI_SEARCH_INDEX_NAME
        except:
            pass

        fields = [
            SimpleField(name="document_id", type=SearchFieldDataType.String, filterable=True),
            SimpleField(name="text", type=SearchFieldDataType.String, filterable=True),
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
                retrievable=False
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