from azure.storage.blob import BlobSasPermissions, generate_blob_sas
from azure.storage.blob.aio import BlobServiceClient
from azure.core.exceptions import ResourceNotFoundError
from core.settings import settings
import datetime
import logging

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

class AzureStorageService:
    def __init__(self):
        if not all([
            settings.AZURE_STORAGE_CONNECTION_STRING,
            settings.AZURE_STORAGE_ARCHIVE_CONTAINER_NAME
        ]):
            raise ValueError("Required Azure Storage settings are missing")
        
        self.blob_service_client = BlobServiceClient.from_connection_string(settings.AZURE_STORAGE_CONNECTION_STRING)
        self.container_client = self.blob_service_client.get_container_client(settings.AZURE_STORAGE_ARCHIVE_CONTAINER_NAME)

    def generate_blob_sas_url(self, blob_name: str) -> str:
        start_time = datetime.datetime.now(datetime.timezone.utc)
        expiry_time = start_time + datetime.timedelta(days=1)
        account_key = self.get_account_key_from_conn_str(settings.AZURE_STORAGE_CONNECTION_STRING)

        sas_token = generate_blob_sas(
            account_name=self.blob_service_client.account_name,
            container_name=self.container_client.container_name,
            blob_name=blob_name,
            account_key=account_key,
            permission=BlobSasPermissions(read=True),
            start=start_time,
            expiry=expiry_time,
        )

        blob_url = f"https://{self.blob_service_client.account_name}.blob.core.windows.net/{self.container_client.container_name}/{blob_name}?{sas_token}"
        return blob_url
    
    @staticmethod
    def get_account_key_from_conn_str(conn_str: str) -> str:
        """Extract the AccountKey from the connection string."""
        for part in conn_str.split(";"):
            if part.startswith("AccountKey="):
                return part.split("=", 1)[1]
        raise ValueError("AccountKey not found in connection string.")
    
    async def get_blobs(self) -> list[str]:
        blobs = []
        async for blob in self.container_client.list_blobs():
            blobs.append(blob.name)
        return blobs

    async def get_blob(self, blob_path: str) -> str:
        blob_client = self.container_client.get_blob_client(blob_path)

        try:
            data = await(await blob_client.download_blob()).readall()
            return data.decode("utf-8")
        except ResourceNotFoundError:
            logger.error(f"{blob_path}' not found.")
            return ""

    async def upload_file_with_dup_check(
        self,
        folder_name: str,
        content: bytes | str,
        blob_name: str,
    ) -> tuple[str, bool]:

        # Normalize the blob path
        blob_path = f"{folder_name.rstrip('/')}/{blob_name.lstrip('/')}"

        # Ensure content is bytes
        if isinstance(content, str):
            file_bytes = content.encode('utf-8')
        else:
            file_bytes = content

        # Get a client and upload
        blob_client = self.container_client.get_blob_client(blob_path)

        #Check if the blob already exists
        if await blob_client.exists():
            logger.info(f"Blob '{blob_path}' already exists; Skipping upload.")
            return blob_path, False

        await blob_client.upload_blob(file_bytes, overwrite=False)
        return blob_path, True
