import uuid
import logging
from typing import Annotated, Optional, List
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import ValidationError
from core.settings import settings
from services.content_service import ContentService
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
    attachments: List[UploadFile] = File(default=[])
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
    
@router.post("/chat")
async def chat(
    user_id: Annotated[str, Form(...)],
    session_id: Annotated[str, Form(...)],
    message: Annotated[str, Form(...)]
):
    try:
        logger.info(f"Received chat request from user_id: {user_id}, session_id: {session_id}")

        if not user_id or not session_id or not message:
            raise HTTPException(status_code=400, detail="Missing required fields: user_id, session_id, or message")

        chat_response = await content_service.chat_with_content(user_id=user_id, session_id=session_id, message=message)
    except ValidationError as e:
        logger.error(f"Validation error: {e}")
        return {"error": str(e)}
    except Exception as e:
        logger.error(f"Error processing chat request: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    return chat_response