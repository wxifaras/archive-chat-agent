import uuid
import azure.functions as func
import logging

from pydantic import ValidationError
from core.settings import settings
from services.content_service import ContentService
from models.chat_request import ChatRequest

app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)

content_service = ContentService()

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

@app.route(route="upload_content", methods=["POST"])
def upload_content(req: func.HttpRequest) -> func.HttpResponse:
    logger.info("Processing upload_content request")

    try:
        doc_id = req.form.get("document_id") or req.params.get("document_id")
        if not doc_id:
            doc_id = str(uuid.uuid4())
            logger.warning(f"No document_id provided — generated new GUID: {doc_id}")

        json_file = req.files.get("json")
        if not json_file:
            return func.HttpResponse("Missing required field 'json'.", status_code=400)

        attachments = req.files.getlist("attachment")
        json_str = json_file.stream.read().decode("utf-8")
        
        content_service.process_content(
            document_id=doc_id,
            json_file_name=json_file.filename,
            json_content=json_str,
            attachments=attachments
        )

        return func.HttpResponse(f"Processed content for doc_id: {doc_id}", status_code=200)

    except Exception as e:
        logger.error(f"Error processing upload_content request: {e}", exc_info=True)
        return func.HttpResponse("Invalid form-data format", status_code=400)
    
@app.route(route="chat", methods=["GET"])
def chat(req: func.HttpRequest) -> func.HttpResponse:
    logger.info("Processing chat request")
    try:
        body = req.get_json()
    except ValueError:
        return func.HttpResponse("Invalid JSON body", status_code=400)

    try:
        chat_request = ChatRequest.model_validate(body)
        chat_response = content_service.search_content(chat_request)
    except ValidationError as e:
        logger.error(f"Validation error: {e}")
        return func.HttpResponse(e.json(), status_code=422)

    logger.info(f"Chat request received: {chat_request.json()}")

    return func.HttpResponse("Chat request processed successfully.", status_code=200)