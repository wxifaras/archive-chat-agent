from pydantic import BaseModel

class ChatRequest(BaseModel):
    User_Id: str
    Session_Id: str
    Message: str