
from azure.identity import DefaultAzureCredential
from azure.cosmos import CosmosClient, PartitionKey
from core.settings import settings

class CosmosDBService:
    def __init__(self):
        if not all([
            settings.COSMOS_ENDPOINT,
            settings.COSMOS_CONTAINER_NAME,
            settings.COSMOS_DATABASE_NAME
        ]):
            raise ValueError("Required Azure Cosmos DB settings are missing")
        
        credential = DefaultAzureCredential()
        self.client = CosmosClient(url=settings.COSMOS_ENDPOINT, credential=credential)
        self.database = self.client.get_database_client(settings.COSMOS_DATABASE_NAME)
        self.container = self.database.get_container_client(settings.COSMOS_CONTAINER_NAME)

    def upsert_item(self, item: dict):
        # Must include 'session_id' in the item
        upsert_item = self.container.upsert_item(item)
        return upsert_item

    def query_items(self, query: str, parameters=None, partition_key=None, **kwargs):
        return list(self.container.query_items(
            query=query,
            parameters=parameters or [],
            partition_key=partition_key,
            **kwargs
        ))
