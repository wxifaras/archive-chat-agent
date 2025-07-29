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