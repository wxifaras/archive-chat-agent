import uuid
import logging
import asyncio
from typing import Annotated, Optional, List
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import ValidationError
from core.settings import settings
from services.content_service import ContentService
from models.chat_request import ChatRequest
from models.chat_response import ChatResponse

router = APIRouter()

content_service = ContentService()

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

@router.post("/upload_content")
async def upload_content(
    document_id: Annotated[str, Form(...)],
    json_file: Annotated[UploadFile, File(...)],
    attachments: Optional[List[UploadFile]] = File(default=None)
):
    logger.info("Processing upload_content request")

    try:
        if not document_id:
            document_id = str(uuid.uuid4())
            logger.warning(f"No document_id provided — generated new GUID: {document_id}")

        if not json_file:
            logger.error("Missing required field 'json'")
            raise HTTPException(status_code=400, detail="Missing required field 'json'")

        json_content = await json_file.read()
        json_str = json_content.decode("utf-8")
        
        await content_service.process_content(
            document_id=document_id,
            json_file_name=json_file.filename,
            json_content=json_str,
            attachments=attachments
        )

        return {
            "message": f"Processed content for JSON file: {json_file.filename}",
            "document_id": document_id
        }

    except Exception as e:
        logger.error(f"Error processing upload_content request: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))
    
@router.post("/chat", response_model=ChatResponse)
async def chat(
    user_id: Annotated[str, Form(...)],
    session_id: Annotated[str, Form(...)],
    message: Annotated[str, Form(...)]
):
    logger.info(f"Processing chat request: {message}")

    try:
        chat_request = ChatRequest(
            User_Id=user_id,
            Session_Id=session_id,
            Message=message
        )

        logger.info(f"Chat request received: {chat_request.model_dump_json()}")

        chat_response = content_service.search_content(chat_request)
    except ValidationError as e:
        logger.error(f"Validation error: {e}")
        return {"error": str(e)}
    except Exception as e:
        logger.error(f"Error processing chat request: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    return chat_response