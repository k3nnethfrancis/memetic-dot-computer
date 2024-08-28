from pydantic import BaseModel
from typing import List, Optional, Union, Tuple

# create a chat message object that is a dictionary of role and content and tools
class ChatMessage(BaseModel):
    role: str
    content: str

# create a chat history object that is a list of chat messages (which are dictionaries of role, content, and tools)
class ChatHistory(BaseModel):
    messages: List[ChatMessage]