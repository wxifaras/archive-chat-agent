import uuid
import logging
import threading
from typing import Dict, List
from models.chat_history import ChatMessage
from core.settings import settings

logger = logging.getLogger(__name__)
logger.setLevel(settings.LOG_LEVEL)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)

class InMemoryChatHistoryManager:
    def __init__(self):
        # session_id → list of message dicts
        self._store: Dict[str, List[dict]] = {}
        # Lock to protect concurrent access
        self._lock = threading.Lock()

    def add_message(self, msg_in: ChatMessage):
        data = msg_in.model_dump(mode="json")
        if not data.get("id"):
            data["id"] = str(uuid.uuid4())
        session = data["session_id"]

        with self._lock:
            bucket = self._store.setdefault(session, [])
            bucket.append(data)

        logger.info(f"(Message added: {data['id']} for session {session}")

    def get_history(self, session_id: str) -> List[ChatMessage]:
        with self._lock:
            items = list(self._store.get(session_id, []))
        # Sort by timestamp to ensure chronological order
        items.sort(key=lambda x: x.get("timestamp"))
        logger.info(f"(Retrieved {len(items)} messages for session {session_id}")
        return [ChatMessage.model_validate(item) for item in items]