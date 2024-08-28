from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime

@dataclass
class Message:
    role: str
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Optional[dict] = None

class ChatHistory:
    def __init__(self):
        self.messages: List[Message] = []

    def add_message(self, role: str, content: str, metadata: Optional[dict] = None):
        message = Message(role=role, content=content, metadata=metadata)
        self.messages.append(message)

    def get_last_n_messages(self, n: int) -> List[Message]:
        return self.messages[-n:]

    def get_messages_by_role(self, role: str) -> List[Message]:
        return [msg for msg in self.messages if msg.role == role]

    def clear(self):
        self.messages.clear()

    def to_dict_list(self) -> List[dict]:
        return [
            {
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp.isoformat(),
                "metadata": msg.metadata
            }
            for msg in self.messages
        ]

    @classmethod
    def from_dict_list(cls, dict_list: List[dict]) -> 'ChatHistory':
        chat_history = cls()
        for msg_dict in dict_list:
            chat_history.add_message(
                role=msg_dict["role"],
                content=msg_dict["content"],
                metadata=msg_dict.get("metadata")
            )
        return chat_history

    def __len__(self):
        return len(self.messages)

    def __str__(self):
        return "\n".join([f"{msg.role}: {msg.content}" for msg in self.messages])