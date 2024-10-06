from abc import ABC, abstractmethod
from typing import List, AsyncGenerator
from cognition.models.chat_models import ChatMessage

class BaseLLM(ABC):
    @abstractmethod
    async def generate(self, messages: List[ChatMessage]) -> str:
        pass

    @abstractmethod
    async def stream(self, messages: List[ChatMessage]) -> AsyncGenerator[str, None]:
        pass

    @abstractmethod
    async def function_call(self, function_name: str, function_args: dict) -> str:
        pass