from azure.identity.aio import DefaultAzureCredential
from azure.cosmos.aio import CosmosClient
from core.settings import settings

class CosmosDBService:
    def __init__(self):
        if not all([
            settings.COSMOS_ENDPOINT,
            settings.COSMOS_CONTAINER_NAME,
            settings.COSMOS_DATABASE_NAME
        ]):
            raise ValueError("Required Azure Cosmos DB settings are missing")
        
    async def upsert_item(self, item: dict):
        async with DefaultAzureCredential() as credential:
            async with CosmosClient(url=settings.COSMOS_ENDPOINT, credential=credential) as client:
                database =  client.get_database_client(settings.COSMOS_DATABASE_NAME)
                container = database.get_container_client(settings.COSMOS_CONTAINER_NAME)
                upsert_item = await container.upsert_item(item)
                return upsert_item

    async def query_items(self, query: str, parameters=None, partition_key=None, **kwargs):
         async with DefaultAzureCredential() as credential:
            async with CosmosClient(url=settings.COSMOS_ENDPOINT, credential=credential) as client:
                database = client.get_database_client(settings.COSMOS_DATABASE_NAME)
                container = database.get_container_client(settings.COSMOS_CONTAINER_NAME)
                items_iter = container.query_items(
                    query=query,
                    parameters=parameters or [],
                    partition_key=partition_key,
                    **kwargs
                )
                results = []
                async for item in items_iter:
                    results.append(item)
                return results