from pydantic import BaseModel
from typing import Optional

class UploadRequest(BaseModel):
    document_id: Optional[str] = None
    # Note: Files (json_file, attachments) cannot be included in Pydantic models
    # They must be handled separately as UploadFile parameters
