import uuid
import logging
from services.cosmos_db_service import CosmosDBService
from models.chat_history import ChatMessage
from core.settings import settings

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

class ChatHistoryManager:
    def __init__(self):
        pass

    def add_message(self, msg_in: ChatMessage):
        data = msg_in.model_dump(mode="json")
        
        # Ensure 'id' is set if absent
        if not data.get("id"):
            data["id"] = str(uuid.uuid4())

        CosmosDBService().upsert_item(data)
        logger.info(f"Message added to history: {data['id']} for session {data['session_id']}")

    def get_history(self, session_id: str) -> list[ChatMessage]:
        items = CosmosDBService().query_items(
            query="SELECT * FROM c WHERE c.session_id=@sid ORDER BY c.timestamp ASC",
            parameters=[{"name": "@sid", "value": session_id}],
            partition_key=session_id       
        )

        logger.info(f"Retrieved {len(items)} messages for session {session_id}")
        return [ChatMessage.model_validate(item) for item in items]