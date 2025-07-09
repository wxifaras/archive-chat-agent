from pydantic import BaseModel

class ChatResponse(BaseModel):
    response: str
    metadata: dict