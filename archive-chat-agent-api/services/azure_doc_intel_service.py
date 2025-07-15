from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest
from azure.ai.documentintelligence.aio import DocumentIntelligenceClient
from core.settings import settings

class AzureDocIntelService:
    def __init__(self):
        if not all([
            settings.AZURE_DOCUMENTINTELLIGENCE_ENDPOINT,
            settings.AZURE_DOCUMENTINTELLIGENCE_API_KEY
        ]):
            raise ValueError("Required Azure Document Intelligence settings are missing")

        self.document_intelligence_client = DocumentIntelligenceClient(
            endpoint=settings.AZURE_DOCUMENTINTELLIGENCE_ENDPOINT, credential=AzureKeyCredential(settings.AZURE_DOCUMENTINTELLIGENCE_API_KEY)
        )

    async def extract_content(self, url: str):
        poller = await self.document_intelligence_client.begin_analyze_document(
            "prebuilt-layout", AnalyzeDocumentRequest(url_source=url)
        )
        
        result = await poller.result()
        return result